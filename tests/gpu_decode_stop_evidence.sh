#!/usr/bin/env bash
# Hardware-level decode-stop evidence on a dedicated H20B GPU.
#
# One independent SGLang instance on GPU 2 (the shared GPU 0/1 backends are
# never touched) serves two rounds:
#
# Round 1 - CUPTI kernel timeline. SGLang's own torch.profiler (Kineto, which
#           uses the CUPTI Activity API) records the scheduler process while the
#           abort probe runs. A clock calibration maps the probe's monotonic
#           markers onto the trace; chrome-analyze counts decode-step spans and
#           CUDA kernels before the abort and proves a kernel-free idle window
#           after the abort frame.
# Round 2 - DCGM hardware profiling counters (fields 1001/1002/1004/1005)
#           sampled at 100 ms while a second probe runs, classified into
#           baseline / decode / teardown / post-abort by the probe markers.
#
# Why not nsys: nsys 2023.4 (image default) and 2026.4.1 both stop delivering
# CUPTI events for SGLang's spawned scheduler once it enters the steady-state
# event loop on this CUDA 13 forward-compat stack; Kineto runs inside that
# process and is not affected. The failed nsys traces are kept under
# nsys-failed/ with a note.
#
# All text logs are capped at 64 KiB. Run on the H20B host: $0 <repo-checkout>.
set -euo pipefail

SRC=${1:?usage: $0 <repo-checkout-dir>}
CONTAINER=linyu-dev
GPU=2
PORT=43102
MODEL=/root/omniinfer-model-validation-20261111-k3/models/omni
CONT_WORK=/root/umminfer-phase6-gpu-evidence
PROBE=$CONT_WORK/gpu_abort_evidence.py
EV=/tmp/umminfer-phase6-gpu-evidence
LOG_CAP=65536

mkdir -p "$EV/cupti" "$EV/dcgm" "$EV/tests"
: > "$EV/commands.log"
log_cmd() { printf '+ %s\n' "$*" | tee -a "$EV/commands.log"; }

cap() {  # keep only the last LOG_CAP bytes of a text file
  local f=$1
  if [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt "$LOG_CAP" ]; then
    tail -c "$LOG_CAP" "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  fi
}

dexec() { docker exec "$CONTAINER" bash -lc "$*"; }
dexec_env="LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:\$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$GPU"

log_cmd "docker cp $SRC/tests/gpu_abort_evidence.py $CONTAINER:$PROBE"
docker cp "$SRC/tests/gpu_abort_evidence.py" "$CONTAINER:$PROBE"
cp "$SRC/tests/gpu_abort_evidence.py" "$EV/tests/"
dexec "mkdir -p $CONT_WORK/cupti $CONT_WORK/dcgm"

{
  echo "date=$(date -Is)"
  echo "host=$(hostname)"
  echo "container=$CONTAINER"
  echo "gpu=$GPU port=$PORT model=$MODEL"
  echo "dcgmi=$(dcgmi --version 2>&1 | head -1)"
  echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "sglang=$(dexec 'python3 -c "import sglang;print(sglang.__version__)" 2>/dev/null')"
  echo "torch=$(dexec 'python3 -c "import torch;print(torch.__version__)" 2>/dev/null')"
  echo "compat_lib=$(dexec 'readlink -f /usr/local/cuda-13.0/compat/libcuda.so.1')"
  echo "gpu_state:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
} > "$EV/environment.txt" 2>&1

SERVER_FLAGS=(--model-path "$MODEL" --served-model-name qwen3-omni
  --host 0.0.0.0 --port "$PORT" --mem-fraction-static=0.90
  --disable-cuda-graph --disable-fast-image-processor --grammar-backend=none
  --enable-multimodal --log-level info)

cleanup() {
  # Bracket pattern so pkill -f does not match its own command line.
  dexec "pkill -f 'sglang.launch_server.*--port ${PORT%?}[${PORT: -1}]' 2>/dev/null || true"
}
trap cleanup EXIT

# --------------------------------------------------------------- instance
echo "== start independent SGLang on GPU $GPU port $PORT =="
log_cmd "launch sglang.launch_server (CUDA_VISIBLE_DEVICES=$GPU)"
dexec "cd $CONT_WORK && $dexec_env nohup python3 -I -m sglang.launch_server \
  ${SERVER_FLAGS[*]} >$CONT_WORK/server.boot.log 2>&1 &"
for _ in $(seq 1 120); do
  if docker exec "$CONTAINER" bash -lc \
       "python3 -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5)\"" \
       >/dev/null 2>&1; then break; fi
  sleep 5
done
docker exec "$CONTAINER" bash -lc \
  "python3 -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=10)\"" \
  >/dev/null
echo "server healthy"

# ------------------------------------------------- round 1: CUPTI/Kineto
echo "== round 1: SGLang torch.profiler CUPTI kernel timeline =="
log_cmd "python3 $PROBE calibrate"
dexec "$dexec_env python3 -I $PROBE calibrate \
  --out $CONT_WORK/cupti/clock-calibration.json \
  --trace $CONT_WORK/cupti/clock-trace.json" | tee "$EV/cupti/calibrate.stdout"

log_cmd "POST /start_profile"
dexec "curl -s -X POST http://127.0.0.1:$PORT/start_profile \
  -H 'Content-Type: application/json' \
  -d '{\"output_dir\":\"$CONT_WORK/cupti\",\"with_stack\":false,\"record_shapes\":false}'" \
  | tee "$EV/cupti/start_profile.stdout"
log_cmd "python3 $PROBE probe (under torch.profiler)"
dexec "python3 -I $PROBE probe --base-url http://127.0.0.1:$PORT \
  --out $CONT_WORK/cupti/probe.json" | tee "$EV/cupti/probe.stdout"
log_cmd "POST /stop_profile"
dexec "curl -s -X POST http://127.0.0.1:$PORT/stop_profile" \
  | tee "$EV/cupti/stop_profile.stdout"
TRACE=$(dexec "ls -t $CONT_WORK/cupti/*.trace.json.gz | head -1")
echo "trace=$TRACE" | tee -a "$EV/commands.log"

log_cmd "python3 $PROBE chrome-analyze"
dexec "python3 -I $PROBE chrome-analyze --trace $TRACE \
  --probe-json $CONT_WORK/cupti/probe.json \
  --calibration $CONT_WORK/cupti/clock-calibration.json \
  --out $CONT_WORK/cupti/chrome-analyze.json" \
  | tee "$EV/cupti/chrome-analyze.stdout"

# ------------------------------------------------- round 2: DCGM counters
echo "== round 2: DCGM hardware profiling counters =="
log_cmd "dcgmi dmon -i $GPU -e 1001,1002,1004,1005 -d 100"
echo 'epoch,GRACT,SMACT,TENSO,DRAMA' > "$EV/dcgm/dmon.csv"
( dcgmi dmon -i "$GPU" -e 1001,1002,1004,1005 -d 100 -c 120 \
    | while read -r a b c d e f; do
        [ "$a" = "GPU" ] || continue
        printf '%s,%s,%s,%s,%s\n' "$(date +%s.%N)" "$c" "$d" "$e" "$f"
      done >> "$EV/dcgm/dmon.csv" ) &
DMON_PID=$!
sleep 3
log_cmd "python3 $PROBE probe (under DCGM sampling)"
dexec "python3 -I $PROBE probe --base-url http://127.0.0.1:$PORT \
  --out $CONT_WORK/dcgm/probe.json" | tee "$EV/dcgm/probe.stdout"
wait "$DMON_PID"

docker cp "$CONTAINER:$CONT_WORK/dcgm/probe.json" "$EV/dcgm/probe.json"
log_cmd "python3 $PROBE dcgm-analyze"
python3 -I "$SRC/tests/gpu_abort_evidence.py" dcgm-analyze \
  --dcgmi-csv "$EV/dcgm/dmon.csv" --probe-json "$EV/dcgm/probe.json" \
  --out "$EV/dcgm/dcgm-analyze.json" | tee "$EV/dcgm/dcgm-analyze.stdout"

# --------------------------------------------------------------- collect
cleanup
trap - EXIT
for f in probe.json chrome-analyze.json clock-calibration.json; do
  docker cp "$CONTAINER:$CONT_WORK/cupti/$f" "$EV/cupti/$f" 2>/dev/null || true
done
docker cp "$CONTAINER:$TRACE" "$EV/cupti/$(basename "$TRACE")"
docker cp "$CONTAINER:$CONT_WORK/server.boot.log" "$EV/server.log" 2>/dev/null || true
for f in $(find "$EV" -type f \( -name "*.log" -o -name "*.stdout" \
  -o -name "*.csv" \)); do cap "$f"; done

( cd "$EV" && find . -type f ! -name SHA256SUMS -print0 | sort -z \
    | xargs -0 sha256sum > SHA256SUMS )
echo "evidence: $EV"
grep -H . "$EV/cupti/chrome-analyze.stdout" "$EV/dcgm/dcgm-analyze.stdout"
