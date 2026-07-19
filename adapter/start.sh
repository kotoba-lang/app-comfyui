#!/usr/bin/env bash
# Boot adapter (+ optional LiteLLM in front). Usage:
#   start.sh                # adapter only
#   start.sh --with-litellm # adapter + LiteLLM :4000
#
# Env overrides:
#   COMFY_URL           (default http://127.0.0.1:8188)
#   ANIMAGINE_PORT      (default 8001)
#   LITELLM_PORT        (default 4000)
#   LLM_BACKEND_URL     (optional, mid-tier LLM passthrough for /v1/chat/completions)
#   COMFY_CHECKPOINT    (default animagine-xl-4.0.safetensors)
#   ANIMATEDIFF_MOTION_MODULE, SVD_CHECKPOINT, WAN5B_MODEL,
#   MUSICGEN_MODEL, STABLE_AUDIO_MODEL, SBV2_DEFAULT_MODEL,
#   XTTS_DEFAULT_MODEL

set -euo pipefail
cd "$(dirname "$0")"

export ANIMAGINE_PORT="${ANIMAGINE_PORT:-8001}"
LITELLM_PORT="${LITELLM_PORT:-4000}"
WITH_LITELLM=0
for arg in "$@"; do
  case "$arg" in
    --with-litellm) WITH_LITELLM=1 ;;
  esac
done

cleanup() {
  echo ""
  echo "[stop] terminating..."
  kill "${BACK_PID:-}" "${PROXY_PID:-}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[boot] adapter on :${ANIMAGINE_PORT}"
uv run python server.py &
BACK_PID=$!

for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:${ANIMAGINE_PORT}/health" >/dev/null 2>&1; then
    echo "[boot] adapter ready"
    break
  fi
  sleep 1
done

if [ "$WITH_LITELLM" = "1" ]; then
  echo "[boot] litellm proxy on :${LITELLM_PORT}"
  uv run litellm --config litellm_config.yaml --port "${LITELLM_PORT}" --host 127.0.0.1 &
  PROXY_PID=$!
fi

cat <<EOF

Ready:
  Direct  http://127.0.0.1:${ANIMAGINE_PORT}/v1/{images,videos,audio,chat}/...
EOF
if [ "$WITH_LITELLM" = "1" ]; then
  cat <<EOF
  LiteLLM http://127.0.0.1:${LITELLM_PORT}/v1/... (Bearer sk-local-master) — images only
EOF
fi

cat <<EOF

Example:
  curl -sS http://127.0.0.1:${ANIMAGINE_PORT}/v1/images/generations \\
    -H "Content-Type: application/json" \\
    -d '{"prompt":"1girl, masterpiece","size":"1024x1024","steps":25}' \\
    | jq -r '.data[0].b64_json' | base64 -d > /tmp/out.png && open /tmp/out.png

Ctrl-C to stop.
EOF

wait
