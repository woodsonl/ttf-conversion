#!/bin/bash
# Convert a bailing_hybrid checkpoint on the first CUDA device, detached and resumable. Installs the CUDA build of
# torch into .venv when the present build has no CUDA, runs the converter tests, then starts (or resumes) the run.
#   scripts/bailing_gpu_run.sh HF_DIR DATA.bin OUT LOG [converter args]      (absolute paths)
#   scripts/bailing_gpu_run.sh status OUT LOG                                 (progress of a run)
cd "$(dirname "$0")/.." && export PYTHONPATH=.:../ttf-ternary TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
if [ "$1" = status ]; then
  date; grep -E "^[0-9:]+ (start|layer|held)" "$3" | tail -n 6 | cut -c1-700; ls "$2" | grep -c "pt$"
  pgrep -af "bailing[.]convert" | cut -c1-80; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  tail -n 3 "$3" | cut -c1-300; exit 0
fi
HF=$1; DATA=$2; OUT=$3; LOG=$4; shift 4
pgrep -f "bailing[.]convert .* $OUT( |\$)" > /dev/null && { echo "a run into $OUT is active"; exit 1; }
$PY -c "import torch,sys;sys.exit(not torch.cuda.is_available())" 2>/dev/null || {
  V=$($PY -c "import torch;print(torch.__version__.split('+')[0])")
  .venv/bin/pip uninstall -y -q torch && .venv/bin/pip install -q "torch==$V" --index-url https://pypi.org/simple; }
$PY -c "import torch;assert torch.cuda.is_available();print(torch.__version__, torch.cuda.get_device_name(0))" || exit 1
$PY -m pytest -q tests/test_bailing_convert.py tests/test_bailing_fold.py | tail -n 2
[ "${PIPESTATUS[0]}" = 0 ] || { echo "tests failed, not started"; exit 1; }
nohup setsid $PY -m ttfconv.bailing.convert "$HF" "$DATA" "$OUT" --device cuda "$@" >> "$LOG" 2>&1 < /dev/null &
echo "started, log $LOG"
