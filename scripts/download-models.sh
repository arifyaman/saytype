#!/usr/bin/env bash
# Downloads the Silero VAD model and the streaming Zipformer English model
# into models/.
#
# Note: k2-fsa reorganizes model artifacts over time. If a download 404s,
# check https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models for the
# current sherpa-onnx-streaming-zipformer-en-* filename and update MODEL_DIR
# below (the daemon auto-detects any sherpa-onnx-streaming-zipformer-* dir).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$REPO_DIR/models"
BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models

MODEL_DIR="sherpa-onnx-streaming-zipformer-en-2023-06-26"

mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

fetch() {
    if command -v wget >/dev/null 2>&1; then
        wget -O "$2" "$1"
    else
        curl -L --fail -o "$2" "$1"
    fi
}

if [ ! -f silero_vad.onnx ]; then
    echo "Downloading silero_vad.onnx..."
    fetch "$BASE/silero_vad.onnx" silero_vad.onnx
else
    echo "silero_vad.onnx already present, skipping."
fi

if [ ! -f "$MODEL_DIR/tokens.txt" ]; then
    echo "Downloading $MODEL_DIR.tar.bz2..."
    fetch "$BASE/$MODEL_DIR.tar.bz2" "$MODEL_DIR.tar.bz2"
    tar xjf "$MODEL_DIR.tar.bz2"
    rm -f "$MODEL_DIR.tar.bz2"
else
    echo "$MODEL_DIR already present, skipping."
fi

echo
echo "Models in $MODELS_DIR:"
ls -la
echo
echo "Done. Install the user service with: scripts/install-user-service.sh"
