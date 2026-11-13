"""Pure quaternion math (wxyz order) for wrist yaw alignment. No simulator
dependency; testable with plain python3.

Composition order: quat_multiply(q_yaw, q_base) is a world-frame
(extrinsic) rotation, not a body-frame one. Getting this backwards does not
raise an error -- it silently rotates the gripper the wrong way.
"""

import math

TARGET_QUAT_WXYZ = (
    -4.4930672515874903e-07,
    0.7071068286895752,
    0.7071067094802856,
    4.984278234587691e-07,
)


def quat_multiply(q1, q2):
    """Hamilton product of two quaternions, wxyz order.

    Args:
        q1: Quaternion (w, x, y, z), applied second.
        q2: Quaternion (w, x, y, z), applied first.

    Returns:
        Quaternion (w, x, y, z): q1 * q2.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_down_with_yaw(yaw_deg):
    """Tool-down orientation rotated by yaw_deg about the world Z axis.

    Args:
        yaw_deg: Additional rotation about world Z, in degrees.

    Returns:
        Quaternion (w, x, y, z).
    """
    half = math.radians(yaw_deg) / 2.0
    q_yaw = (math.cos(half), 0.0, 0.0, math.sin(half))
    return quat_multiply(q_yaw, TARGET_QUAT_WXYZ)


def quat_angle_deg(q1, q2):
    """Angle in degrees between two quaternion orientations.

    Both inputs are normalized before comparing: TARGET_QUAT_WXYZ is a
    single manual measurement, not an exact unit quaternion (norm off by
    ~1.7e-8) -- skipping normalization would report a spurious ~0.03 deg
    angle between an orientation and itself.

    Args:
        q1: Quaternion (w, x, y, z).
        q2: Quaternion (w, x, y, z).

    Returns:
        Angle in degrees. Uses abs(dot) to handle the quaternion double
        cover (q and -q represent the same orientation).
    """
    n1 = math.sqrt(sum(a * a for a in q1))
    n2 = math.sqrt(sum(a * a for a in q2))
    dot = sum(a * b for a, b in zip(q1, q2)) / (n1 * n2)
    return math.degrees(2.0 * math.acos(min(1.0, abs(dot))))


def wrap_to_symmetry(yaw_deg, period_deg):
    """Fold yaw_deg into the equivalent angle within one symmetry period.

    Args:
        yaw_deg: Angle in degrees to fold.
        period_deg: The object's symmetry period in degrees -- always
            required, never a fixed constant (90 for square-like objects,
            180 for elongated ones).

    Returns:
        Angle in degrees, within (-period_deg/2, +period_deg/2].
    """
    half = period_deg / 2.0
    r = yaw_deg % period_deg
    if r > half:
        r -= period_deg
    return r
