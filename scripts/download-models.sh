#!/usr/bin/env bash
# Downloads the Silero VAD model plus the ASR models into models/:
#
#   - Moonshine v2 tiny (English, quantized) - preferred by the daemon,
#     batch per segment, outputs casing + punctuation.
#   - Streaming Zipformer (English) - fallback, and the only backend with
#     live partials (needed for mvp2 live typing).
#
# Both can coexist; the daemon prefers Moonshine when present.
#
# Note: k2-fsa reorganizes model artifacts over time. If a download 404s,
# check https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models for the
# current filename and update the dir names below (the daemon auto-detects
# any dir matching the expected layout).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$REPO_DIR/models"
BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models

MOONSHINE_DIR="sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27"
ZIPFORMER_DIR="sherpa-onnx-streaming-zipformer-en-2023-06-26"

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

for MODEL_DIR in "$MOONSHINE_DIR" "$ZIPFORMER_DIR"; do
    if [ ! -f "$MODEL_DIR/tokens.txt" ]; then
        echo "Downloading $MODEL_DIR.tar.bz2..."
        fetch "$BASE/$MODEL_DIR.tar.bz2" "$MODEL_DIR.tar.bz2"
        tar xjf "$MODEL_DIR.tar.bz2"
        rm -f "$MODEL_DIR.tar.bz2"
    else
        echo "$MODEL_DIR already present, skipping."
    fi
done

echo
echo "Models in $MODELS_DIR:"
ls -la
echo
echo "Done. Install the user service with: scripts/install-user-service.sh"
