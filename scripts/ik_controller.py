"""GUI-safe arm-only differential IK controller for the NHC12 + Robotiq demo.

The operator target represents the user-selected grasp-centre TCP, not the
wrist flange.  PhysX has no Jacobian row for a cosmetic Xform, so the solver
uses the Robotiq ``base_link_0`` rigid-body row and shifts its linear
Jacobian to the TCP point.  Only arm DOFs 0-5 are ever commanded; gripper
DOFs 6-11 are observed for acceptance evidence but never written.

Public entry points are ``ik_follow_start()``, ``ik_follow_start_async()``,
``ik_follow_status()``, ``ik_follow_stop()``, and
``ik_follow_subscription_count()``.
"""

from __future__ import annotations

import asyncio
import builtins
import json
import math
import traceback

import numpy as np

import carb
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.core.api.world import World
from isaacsim.core.experimental.prims import Articulation, XformPrim
import isaacsim.core.experimental.utils.backend as backend_utils


ARTICULATION_PATH = "/World/ArmWithHandOnly"
HAND_LINK_NAME = "base_link_0"
HAND_LINK_PATH = (
    "/World/ArmWithHandOnly/Robotiq_2F_85/Robotiq_2F_85/base_link"
)
TCP_PRIM_PATH = f"{HAND_LINK_PATH}/tcp"
TARGET_PRIM_PATH = "/World/IKTarget"
ARM_JOINT_NAMES = [
    "joint_1_s",
    "joint_2_l",
    "joint_3_u",
    "joint_4_r",
    "joint_5_b",
    "joint_6_t",
]
GRIPPER_JOINT_NAMES = [
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_knuckle_joint",
]
EXPECTED_ARM_DOF_INDICES = [0, 1, 2, 3, 4, 5]
EXPECTED_GRIPPER_DOF_INDICES = [6, 7, 8, 9, 10, 11]

DAMPING = 0.01
NULL_DAMPING = 0.05
# Keep the arm target slew slow enough that the explicitly uncontrolled
# gripper does not receive a large inertial impulse during a GUI drag.
MAX_JOINT_DELTA_PER_STEP = 0.02
POSTURE_GAIN = 0.02
MAX_POSTURE_DELTA_PER_STEP = 0.005
POS_WEIGHT = 1.0
ROT_WEIGHT_MIN = 0.08
ROT_WEIGHT_MAX = 0.6
ROT_WEIGHT_RAMP_SCALE = 0.06
SIGMA_THRESHOLD = 0.05
LAMBDA_MAX = 0.3

CLEAN_ARM_POSE_RAD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
CLEAN_ARM_POSE_METADATA_KEY = "demo_clean_arm_pose_rad"
_REGISTRY_ATTR = "_arm310d_ik_follow_controller_registry"
FABRIC_MANIPULATOR_EXTENSION_NAME = "omni.kit.manipulator.prim.fabric"


def _get_registry() -> dict:
    if not hasattr(builtins, _REGISTRY_ATTR):
        setattr(
            builtins,
            _REGISTRY_ATTR,
            {
                "instance": None,
                "start_task": None,
                "last_start_result": None,
            },
        )
    registry = getattr(builtins, _REGISTRY_ATTR)
    registry.setdefault("instance", None)
    registry.setdefault("start_task", None)
    registry.setdefault("last_start_result", None)
    return registry


def _configured_clean_arm_pose() -> np.ndarray:
    try:
        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            raw = stage.GetRootLayer().customLayerData.get(
                CLEAN_ARM_POSE_METADATA_KEY
            )
            if isinstance(raw, str):
                values = json.loads(raw)
                if len(values) == len(ARM_JOINT_NAMES) and all(
                    math.isfinite(float(value)) for value in values
                ):
                    return np.asarray(values, dtype=np.float64)
    except Exception:
        pass
    return np.asarray(CLEAN_ARM_POSE_RAD, dtype=np.float64)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _axis_angle_error(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    q_err = _quat_mul(q_target, _quat_conjugate(q_current))
    if q_err[0] < 0.0:
        q_err = -q_err
    w = float(np.clip(q_err[0], -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    sin_half = math.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < 1e-8 or angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return (q_err[1:4] / sin_half) * angle


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def _steps_per_update() -> int:
    """Physics steps (1/60 s) per Kit update: 1 with the default timeline; 2 when time codes/s and
    /persistent/simulation/minFrameRate are both 30. The per-step limits below are per 1/60 s of sim time (11/08)."""
    tc = omni.timeline.get_timeline_interface().get_time_codes_per_seconds()
    mfr = carb.settings.get_settings().get("/persistent/simulation/minFrameRate") or 60
    return max(1, int(round(60.0 * min(1.0 / tc, 1.0 / mfr))))


def _manipulability_damping(sigma_min: float, lambda_base: float) -> float:
    if sigma_min >= SIGMA_THRESHOLD:
        return lambda_base
    frac = 1.0 - (sigma_min / SIGMA_THRESHOLD)
    return lambda_base + (LAMBDA_MAX - lambda_base) * (frac**2)


def _adaptive_rot_weight(pos_err_norm: float) -> float:
    if pos_err_norm >= ROT_WEIGHT_RAMP_SCALE:
        return ROT_WEIGHT_MIN
    frac = 1.0 - (pos_err_norm / ROT_WEIGHT_RAMP_SCALE)
    return ROT_WEIGHT_MIN + (ROT_WEIGHT_MAX - ROT_WEIGHT_MIN) * frac


def _get_fabric_world_poses(prim: XformPrim):
    """Read the runtime pose seen by the Fabric viewport/manipulator.

    Isaac Sim 6 uses the Fabric prim manipulator while Fabric Scene Delegate
    is enabled.  Reading the default USD backend here would make the solver
    observe a different target state from the Cube the operator is dragging.
    """
    with backend_utils.use_backend("fabric", raise_on_fallback=True):
        return prim.get_world_poses()


def _ensure_fabric_manipulator_ready() -> dict:
    """Enable the official Fabric gizmo accessor when FSD is active."""
    settings = carb.settings.get_settings()
    if not settings.get_as_bool("/app/useFabricSceneDelegate"):
        raise RuntimeError(
            "Fabric Scene Delegate is disabled; this controller requires the "
            "viewport and IK target to share the Fabric runtime state"
        )

    manager = omni.kit.app.get_app().get_extension_manager()
    enabled_before = manager.is_extension_enabled(
        FABRIC_MANIPULATOR_EXTENSION_NAME
    )
    enabled_id = manager.get_enabled_extension_id(
        FABRIC_MANIPULATOR_EXTENSION_NAME
    )
    if not enabled_before:
        candidates = [
            item
            for item in manager.get_extensions()
            if item.get("name") == FABRIC_MANIPULATOR_EXTENSION_NAME
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "unable to resolve one installed Fabric manipulator extension: "
                f"{candidates}"
            )
        enabled_id = candidates[0]["id"]
        if not manager.set_extension_enabled_immediate(enabled_id, True):
            raise RuntimeError(
                f"failed to enable Fabric manipulator extension: {enabled_id}"
            )

    from omni.kit.manipulator.prim.core import get_prim_data_accessor_registry

    accessors = get_prim_data_accessor_registry().get_data_accessors_func()
    if "FABRIC" not in accessors:
        raise RuntimeError(
            f"Fabric manipulator accessor did not register: {list(accessors)}"
        )
    return {
        "fabric_scene_delegate": True,
        "fabric_manipulator_extension": enabled_id,
        "fabric_manipulator_enabled_on_start": not enabled_before,
        "registered_manipulator_accessors": sorted(accessors),
    }


class IKFollowController:
    """Weighted single-stack DLS controller for the six NHC12 arm joints."""

    def __init__(self) -> None:
        self._sub = None
        self._art = None
        self._hand = None
        self._tcp = None
        self._target = None
        self._arm_dof_indices = None
        self._gripper_dof_indices = None
        self._hand_jac_row = None
        self._dof_lo = None
        self._dof_hi = None
        self._clean_arm_targets = None
        self._q_cmd = None
        self._last_measured_arm = None
        self._posture_ref = None
        self._gripper_positions_at_start = None
        self._running = False
        self._error_count = 0
        self._last_error = None
        self._step_count = 0
        self._restore_requested = False
        self._command_call_count = 0
        self._commanded_dof_indices_seen = set()

    def start(self) -> dict:
        if self._running:
            return {"status": "already_running"}

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no active USD stage")
        for path in (HAND_LINK_PATH, TCP_PRIM_PATH, TARGET_PRIM_PATH):
            if not stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"required prim missing: {path}")

        art = Articulation(ARTICULATION_PATH)
        if not art.is_physics_tensor_entity_initialized():
            raise RuntimeError(
                "articulation tensor entity is not initialized after async runtime preparation"
            )
        dof_names = list(art.dof_names)
        link_names = list(art.link_names)
        link_paths = list(art.link_paths[0])

        missing = [name for name in ARM_JOINT_NAMES if name not in dof_names]
        if missing:
            raise RuntimeError(f"arm DOFs missing on {ARTICULATION_PATH}: {missing}")
        missing_gripper = [
            name for name in GRIPPER_JOINT_NAMES if name not in dof_names
        ]
        if missing_gripper:
            raise RuntimeError(f"gripper DOFs missing: {missing_gripper}")

        arm_dof_indices = [dof_names.index(name) for name in ARM_JOINT_NAMES]
        gripper_dof_indices = [
            dof_names.index(name) for name in GRIPPER_JOINT_NAMES
        ]
        if arm_dof_indices != EXPECTED_ARM_DOF_INDICES:
            raise RuntimeError(
                f"arm DOF order changed: {arm_dof_indices}; refusing to command"
            )
        if gripper_dof_indices != EXPECTED_GRIPPER_DOF_INDICES:
            raise RuntimeError(
                f"gripper DOF order changed: {gripper_dof_indices}; refusing to start"
            )

        hand_link_index = link_paths.index(HAND_LINK_PATH)
        if link_names[hand_link_index] != HAND_LINK_NAME:
            raise RuntimeError(
                f"hand link identity mismatch: {link_names[hand_link_index]} at {HAND_LINK_PATH}"
            )
        hand_jac_row = hand_link_index - 1
        jac_shape = art.get_jacobian_matrices().numpy().shape
        if jac_shape[1] != len(link_names) - 1 or jac_shape[2:] != (6, 12):
            raise RuntimeError(
                f"unverified Jacobian shape {jac_shape} for {len(link_names)} links"
            )

        dof_lo, dof_hi = art.get_dof_limits()
        lo = dof_lo.numpy()[0]
        hi = dof_hi.numpy()[0]
        clean = _configured_clean_arm_pose()
        arm_lo = lo[arm_dof_indices]
        arm_hi = hi[arm_dof_indices]
        if np.any(clean < arm_lo) or np.any(clean > arm_hi):
            raise RuntimeError(f"configured clean pose violates limits: {clean.tolist()}")

        hand = XformPrim(HAND_LINK_PATH)
        tcp = XformPrim(TCP_PRIM_PATH)
        target = XformPrim(TARGET_PRIM_PATH)
        tcp_p, tcp_q = _get_fabric_world_poses(tcp)
        target_p, target_q = _get_fabric_world_poses(target)

        q_all = art.get_dof_positions().numpy()[0]
        live_arm_targets = q_all[arm_dof_indices].copy()
        self._art = art
        self._hand = hand
        self._tcp = tcp
        self._target = target
        self._arm_dof_indices = arm_dof_indices
        self._gripper_dof_indices = gripper_dof_indices
        self._hand_jac_row = hand_jac_row
        self._dof_lo = lo
        self._dof_hi = hi
        self._clean_arm_targets = clean
        self._q_cmd = live_arm_targets
        self._last_measured_arm = live_arm_targets.copy()
        self._posture_ref = clean.copy()
        self._gripper_positions_at_start = q_all[gripper_dof_indices].copy()
        self._error_count = 0
        self._last_error = None
        self._step_count = 0
        self._restore_requested = False
        self._command_call_count = 0
        self._commanded_dof_indices_seen = set()

        self._sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="arm310d_ik_follow_controller_update"
        )
        self._running = True
        _get_registry()["instance"] = self
        return {
            "status": "started",
            "articulation_path": ARTICULATION_PATH,
            "arm_dof_indices": arm_dof_indices,
            "gripper_dof_indices": gripper_dof_indices,
            "hand_link_name": HAND_LINK_NAME,
            "hand_link_path": HAND_LINK_PATH,
            "hand_jac_row": hand_jac_row,
            "tcp_prim_path": TCP_PRIM_PATH,
            "tcp_pos": tcp_p.numpy()[0].tolist(),
            "tcp_quat_wxyz": tcp_q.numpy()[0].tolist(),
            "target_pos": target_p.numpy()[0].tolist(),
            "target_quat_wxyz": target_q.numpy()[0].tolist(),
            "runtime_pose_backend": "fabric",
            "target_written_on_start": False,
            "clean_arm_pose_rad": clean.tolist(),
        }

    def stop(self, restore: bool = True) -> dict:
        if self._sub is not None:
            self._sub.unsubscribe()
            self._sub = None
        restore_error = None
        if restore and self._art is not None and self._arm_dof_indices is not None:
            try:
                self._command_arm_targets(self._clean_arm_targets)
                self._restore_requested = True
            except Exception as exc:
                restore_error = f"{type(exc).__name__}: {exc}"
        self._running = False
        registry = _get_registry()
        if registry.get("instance") is self:
            registry["instance"] = None
        result = {
            "status": "stopped",
            "restore_requested": bool(restore and restore_error is None),
            "commanded_dof_indices": list(self._arm_dof_indices or []),
            "command_call_count": self._command_call_count,
            "commanded_dof_indices_seen": sorted(self._commanded_dof_indices_seen),
            "last_measured_arm_positions": (
                self._last_measured_arm.tolist()
                if self._last_measured_arm is not None
                else None
            ),
            "last_commanded_arm_targets": (
                self._q_cmd.tolist() if self._q_cmd is not None else None
            ),
        }
        if restore_error is not None:
            result["restore_error"] = restore_error
        return result

    def status(self) -> dict:
        gripper_current = None
        if self._art is not None and self._gripper_dof_indices is not None:
            try:
                q_all = self._art.get_dof_positions().numpy()[0]
                gripper_current = q_all[self._gripper_dof_indices].tolist()
            except Exception:
                pass
        return {
            "status": "running" if self._running else "stopped",
            "running": self._running,
            "step_count": self._step_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "arm_command_indices": list(self._arm_dof_indices or []),
            "gripper_observation_indices": list(self._gripper_dof_indices or []),
            "gripper_positions_at_start": (
                self._gripper_positions_at_start.tolist()
                if self._gripper_positions_at_start is not None
                else None
            ),
            "gripper_positions_current": gripper_current,
            "restore_requested": self._restore_requested,
            "command_call_count": self._command_call_count,
            "commanded_dof_indices_seen": sorted(self._commanded_dof_indices_seen),
            "last_measured_arm_positions": (
                self._last_measured_arm.tolist()
                if self._last_measured_arm is not None
                else None
            ),
            "last_commanded_arm_targets": (
                self._q_cmd.tolist() if self._q_cmd is not None else None
            ),
        }

    def _command_arm_targets(self, targets: np.ndarray) -> None:
        """Single guarded write path; rejects any index outside arm DOFs 0-5."""
        indices = list(self._arm_dof_indices or [])
        if indices != EXPECTED_ARM_DOF_INDICES:
            raise RuntimeError(
                f"unsafe command indices {indices}; expected {EXPECTED_ARM_DOF_INDICES}"
            )
        self._command_call_count += 1
        self._commanded_dof_indices_seen.update(indices)
        self._art.set_dof_position_targets(targets, dof_indices=indices)

    def _on_update(self, _event) -> None:
        if not self._running:
            return
        try:
            if not omni.timeline.get_timeline_interface().is_playing():
                return
            if not self._art.is_physics_tensor_entity_initialized():
                return

            target_p, target_q = _get_fabric_world_poses(self._target)
            tcp_p, tcp_q = _get_fabric_world_poses(self._tcp)
            hand_p, _hand_q = _get_fabric_world_poses(self._hand)
            target_p = target_p.numpy()[0]
            target_q = target_q.numpy()[0]
            tcp_p = tcp_p.numpy()[0]
            tcp_q = tcp_q.numpy()[0]
            hand_p = hand_p.numpy()[0]

            q_all = self._art.get_dof_positions().numpy()[0]
            measured_arm = q_all[self._arm_dof_indices].copy()
            if not np.all(np.isfinite(measured_arm)):
                raise RuntimeError(
                    f"non-finite measured arm position: {measured_arm.tolist()}"
                )
            self._last_measured_arm = measured_arm

            pos_err = target_p - tcp_p
            rot_err = _axis_angle_error(tcp_q, target_q)

            j_body = self._art.get_jacobian_matrices().numpy()[
                0, self._hand_jac_row
            ][:, self._arm_dof_indices]
            r_world = tcp_p - hand_p
            j_tcp = j_body.copy()
            j_tcp[0:3, :] = j_body[0:3, :] - _skew(r_world) @ j_body[3:6, :]

            e6 = np.concatenate([pos_err, rot_err])
            rot_weight = _adaptive_rot_weight(float(np.linalg.norm(pos_err)))
            weights = np.array(
                [POS_WEIGHT, POS_WEIGHT, POS_WEIGHT, rot_weight, rot_weight, rot_weight]
            )
            j_weighted = j_tcp * weights[:, None]
            e_weighted = e6 * weights
            sigma_min = float(np.linalg.svd(j_weighted, compute_uv=False)[-1])
            lambda_task = _manipulability_damping(sigma_min, DAMPING)
            lambda_null = _manipulability_damping(sigma_min, NULL_DAMPING)

            n_arm = len(self._arm_dof_indices)
            identity_task = np.eye(6)
            task_matrix = j_weighted @ j_weighted.T + (
                lambda_task**2
            ) * identity_task
            j_pinv = j_weighted.T @ np.linalg.solve(task_matrix, identity_task)
            dq_task = j_pinv @ e_weighted

            null_matrix = j_weighted @ j_weighted.T + (
                lambda_null**2
            ) * identity_task
            j_pinv_null = j_weighted.T @ np.linalg.solve(
                null_matrix, identity_task
            )
            null_projector = np.eye(n_arm) - j_pinv_null @ j_weighted
            k = _steps_per_update()                     # sim frames per update: per-step limits scale with it
            posture_pull = POSTURE_GAIN * k * (self._posture_ref - measured_arm)
            dq_null = null_projector @ posture_pull
            null_norm = float(np.linalg.norm(dq_null))
            if null_norm > MAX_POSTURE_DELTA_PER_STEP * k:
                dq_null *= MAX_POSTURE_DELTA_PER_STEP * k / null_norm

            dq = dq_task + dq_null
            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > MAX_JOINT_DELTA_PER_STEP * k:
                dq *= MAX_JOINT_DELTA_PER_STEP * k / dq_norm

            # Closed-loop bounded position update.  Always base this tick's
            # target on measured articulation state, never on the previous
            # Python command.  This prevents command windup when Kit update
            # callbacks run faster than PhysX can realize position targets.
            new_pos = measured_arm + dq
            arm_lo = self._dof_lo[self._arm_dof_indices]
            arm_hi = self._dof_hi[self._arm_dof_indices]
            new_pos = np.clip(new_pos, arm_lo, arm_hi)
            self._q_cmd = new_pos
            self._command_arm_targets(new_pos)

            self._step_count += 1
            self._last_error = {
                "pos_error_norm_m": float(np.linalg.norm(pos_err)),
                "rot_error_norm_rad": float(np.linalg.norm(rot_err)),
                "sigma_min_weighted": sigma_min,
                "damping": lambda_task,
                "tcp_offset_world_m": r_world.tolist(),
                "measured_command_gap_norm_rad": float(
                    np.linalg.norm(new_pos - measured_arm)
                ),
            }
        except Exception as exc:
            self._error_count += 1
            self._last_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            self.stop(restore=True)


async def _ensure_runtime_ready() -> dict:
    manipulator = _ensure_fabric_manipulator_ready()
    await omni.kit.app.get_app().next_update_async()
    world = World.instance() or World(stage_units_in_meters=1.0)
    initialized_context = False
    reset_runtime = False
    if getattr(world, "_physics_context", None) is None:
        await world.initialize_simulation_context_async()
        initialized_context = True
    if getattr(world, "physics_sim_view", None) is None:
        await world.reset_async()
        reset_runtime = True
    elif not omni.timeline.get_timeline_interface().is_playing():
        await world.play_async()
    await omni.kit.app.get_app().next_update_async()
    return {
        "initialized_context": initialized_context,
        "reset_runtime": reset_runtime,
        "physics_context_valid": getattr(world, "_physics_context", None)
        is not None,
        "physics_sim_view_valid": getattr(world, "physics_sim_view", None)
        is not None,
        "timeline_playing": omni.timeline.get_timeline_interface().is_playing(),
        **manipulator,
    }


async def ik_follow_start_async() -> dict:
    registry = _get_registry()
    existing = registry.get("instance")
    if existing is not None:
        existing.stop(restore=False)
    runtime = await _ensure_runtime_ready()
    if not runtime["physics_context_valid"] or not runtime["physics_sim_view_valid"]:
        raise RuntimeError(f"runtime initialization failed: {runtime}")
    controller = IKFollowController()
    result = controller.start()
    result["runtime"] = runtime
    registry["last_start_result"] = result
    return result


async def _scheduled_start_wrapper() -> dict:
    registry = _get_registry()
    try:
        result = await ik_follow_start_async()
        registry["last_start_result"] = result
        return result
    except Exception as exc:
        result = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        registry["last_start_result"] = result
        return result
    finally:
        registry["start_task"] = None


def ik_follow_start() -> dict:
    """GUI-safe entry point; schedules async runtime preparation and start."""
    registry = _get_registry()
    task = registry.get("start_task")
    if task is not None and not task.done():
        return {"status": "already_scheduled"}
    task = asyncio.ensure_future(_scheduled_start_wrapper())
    registry["start_task"] = task
    return {"status": "scheduled"}


def ik_follow_stop() -> dict:
    registry = _get_registry()
    task = registry.get("start_task")
    if task is not None and not task.done():
        task.cancel()
        registry["start_task"] = None
    existing = registry.get("instance")
    if existing is None:
        return {"status": "not_running"}
    return existing.stop(restore=True)


def ik_follow_status() -> dict:
    registry = _get_registry()
    task = registry.get("start_task")
    if task is not None and not task.done():
        return {"status": "starting", "running": False}
    existing = registry.get("instance")
    if existing is None:
        last = registry.get("last_start_result")
        if isinstance(last, dict) and last.get("status") == "error":
            return {"status": "error", "running": False, "last_start_result": last}
        return {"status": "not_running", "running": False}
    return existing.status()


def ik_follow_subscription_count() -> int:
    registry = _get_registry()
    existing = registry.get("instance")
    if existing is None or existing._sub is None:
        return 0
    return 1
