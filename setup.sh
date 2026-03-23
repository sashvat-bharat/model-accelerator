#!/usr/bin/env bash
# setup.sh — Environment bootstrap for model-accelerator
# Arch Linux / any Linux with Python 3.10+
#
# Usage:
#   bash setup.sh             # basic CPU install
#   bash setup.sh --cuda      # with CUDA (NVIDIA GPU) support
#   bash setup.sh --metal     # macOS Metal (Apple Silicon)

set -euo pipefail

CUDA=false
METAL=false

for arg in "$@"; do
  case $arg in
    --cuda)   CUDA=true ;;
    --metal)  METAL=true ;;
  esac
done

echo "======================================================"
echo "  model-accelerator — environment setup"
echo "======================================================"
echo ""

# ── 1. Python venv ─────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "[setup] Creating virtual environment…"
  python -m venv .venv
else
  echo "[setup] .venv already exists — skipping creation"
fi

# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip --quiet

# ── 2. llama-cpp-python ────────────────────────────────────
echo ""
echo "[setup] Installing llama-cpp-python…"

if $CUDA; then
  echo "        → CUDA (cuBLAS) build"
  CMAKE_ARGS="-DGGML_CUDA=on" \
    pip install llama-cpp-python --force-reinstall --no-cache-dir
elif $METAL; then
  echo "        → Metal build"
  CMAKE_ARGS="-DGGML_METAL=on" \
    pip install llama-cpp-python --force-reinstall --no-cache-dir
else
  echo "        → CPU-only build (pass --cuda or --metal for GPU)"
  pip install llama-cpp-python --quiet
fi

# ── 3. Other deps ──────────────────────────────────────────
echo ""
echo "[setup] Installing other dependencies…"
pip install -r requirements.txt --quiet

# ── 4. models/ directory ───────────────────────────────────
mkdir -p models
touch models/.gitkeep

echo ""
echo "======================================================"
echo "  Done!  Activate the env with:"
echo "    source .venv/bin/activate"
echo ""
echo "  Quick-start:"
echo "    python cli.py load TheBloke/Mistral-7B-v0.1-GGUF \\"
echo "                  --alias mistral7b"
echo "    python cli.py run  mistral7b --prompt 'Hello, who are you?'"
echo "    python cli.py chat mistral7b"
echo "======================================================"
