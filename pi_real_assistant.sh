#!/usr/bin/env bash
set -euo pipefail

ARCH="$(uname -m)"
echo "Detected architecture: $ARCH"

if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" && "$ARCH" != "armv7l" && "$ARCH" != "armhf" ]]; then
  echo "This script is intended for Raspberry Pi ARM boards."
  echo "Your board is: $ARCH"
  exit 1
fi

if [[ "$ARCH" == "armv7l" || "$ARCH" == "armhf" ]]; then
  echo "Warning: 32-bit ARM is limited. This will be much slower and may not run large models well."
  echo "Best option: use a Pi 4/5 64-bit OS, or a stronger machine."
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required on the Pi. Please run this with a user that can sudo."
  exit 1
fi

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  git \
  curl \
  python3 \
  python3-pip \
  ca-certificates

if ! command -v git >/dev/null 2>&1; then
  echo "git did not install correctly"
  exit 1
fi

mkdir -p "$HOME/src"
cd "$HOME/src"
if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
fi
cd llama.cpp
make -j"$(nproc)"

MODEL_URL="https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf"
MODEL_FILE="$HOME/src/llama.cpp/SmolLM2-360M-Instruct-Q4_K_M.gguf"

if [ ! -f "$MODEL_FILE" ]; then
  echo "Downloading model: $MODEL_URL"
  curl -L --fail -o "$MODEL_FILE" "$MODEL_URL"
fi

cat <<'EOF'

Real local assistant is ready.

Run this on your Pi:

cd "$HOME/src/llama.cpp"
./main -m "$HOME/src/llama.cpp/SmolLM2-360M-Instruct-Q4_K_M.gguf" \
  -n 128 \
  --color \
  -i \
  -p "You are a helpful local assistant. Keep answers concise, clear, and practical."

If you want a simpler setup:

curl -fsSL https://ollama.com/install.sh | sh
ollama run smollm2:360m

Important:
- Pi Zero / 32-bit ARM: not practical for a real assistant
- Pi 4/5 64-bit: this is the best local option
- This is the actual working assistant path; the NumPy version is only a prototype.
EOF
