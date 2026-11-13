#!/usr/bin/env bash
# 在本資料夾建立 YOLO 推論用的獨立 Python 環境（perception/.venv），不動 Isaac Sim 內建 python。
# 用法：bash perception/setup_yolo_env.sh（本專案 YOLO：LOCK=requirements.basket.lock.txt bash perception/setup_yolo_env.sh）   完成後把 perception/config.json 的 yolo_python 改成 perception/.venv/bin/python
set -e
cd "$(dirname "$0")"
UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}
"$UV" venv .venv --python 3.12
"$UV" pip install --python .venv/bin/python -r "${LOCK:-requirements.lock.txt}" \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
.venv/bin/python -c "import ultralytics, torch; print('OK', ultralytics.__version__, torch.__version__, torch.cuda.is_available())"
