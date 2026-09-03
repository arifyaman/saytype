#!/usr/bin/env bash
# Downloads ASR models into models/.
#
# Default (no arguments): the daemon's default backend stack
#   - Silero VAD (always required - supplies utterance boundaries)
#   - Nemotron Speech Streaming EN 0.6B (560 ms chunk) - the daemon's
#     preferred backend (mvp2 default). Truly streaming (live partials),
#     trained on ~530k h, natively cased + punctuated. 560 ms is the
#     accuracy/latency sweet spot; 80/160/1120 ms chunk variants exist.
#   - Online punctuation/casing (English) - small (7 MB); restores denser
#     casing + punctuation on the streaming backends.
#
# Optional extra backends (all can coexist; the daemon prefers
# Nemotron > Zipformer > Moonshine and picks what it finds):
#   --zipformer   Streaming Zipformer EN - live partials, all-caps output
#                 handled by the punct model; ~110 MB of weights.
#   --moonshine   Moonshine v2 tiny EN (quantized) - batch per utterance
#                 (no live typing), outputs its own casing + punctuation.
#   --all         Download both optional backends in addition to the default.
#
# Note: k2-fsa reorganizes model artifacts over time. If a download 404s,
# check https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models (ASR)
# and .../tag/punctuation-models (punctuation) for the current filename and
# update the dir names below (the daemon auto-detects any dir matching the
# expected layout).
set -euo pipefail

WANT_ZIPFORMER=0
WANT_MOONSHINE=0

usage() {
    cat <<'EOF'
Usage: scripts/download-models.sh [OPTIONS]

Downloads the default backend stack (Silero VAD + Nemotron streaming
EN 0.6B + online punctuation model) into models/.

Options:
  --zipformer   Also download the streaming Zipformer EN model
  --moonshine   Also download the Moonshine v2 tiny EN model (batch)
  --all         Download both optional backends
  -h, --help    Show this help
EOF
}

for arg in "$@"; do
    case "$arg" in
        --zipformer) WANT_ZIPFORMER=1 ;;
        --moonshine) WANT_MOONSHINE=1 ;;
        --all)       WANT_ZIPFORMER=1; WANT_MOONSHINE=1 ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; usage >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$REPO_DIR/models"
BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
PUNCT_BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models

NEMOTRON_DIR="sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
MOONSHINE_DIR="sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27"
ZIPFORMER_DIR="sherpa-onnx-streaming-zipformer-en-2023-06-26"
PUNCT_DIR="sherpa-onnx-online-punct-en-2024-08-06"

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

ASR_DIRS=("$NEMOTRON_DIR")
[ "$WANT_ZIPFORMER" = 1 ] && ASR_DIRS+=("$ZIPFORMER_DIR")
[ "$WANT_MOONSHINE" = 1 ] && ASR_DIRS+=("$MOONSHINE_DIR")

for MODEL_DIR in "${ASR_DIRS[@]}"; do
    if [ ! -f "$MODEL_DIR/tokens.txt" ]; then
        echo "Downloading $MODEL_DIR.tar.bz2..."
        fetch "$BASE/$MODEL_DIR.tar.bz2" "$MODEL_DIR.tar.bz2"
        tar xjf "$MODEL_DIR.tar.bz2"
        rm -f "$MODEL_DIR.tar.bz2"
    else
        echo "$MODEL_DIR already present, skipping."
    fi
done

# Online punctuation/casing model: only the int8 model + BPE vocab are needed
# (the fp32 model.onnx is ~29MB and no faster here).
if [ ! -f "$PUNCT_DIR/bpe.vocab" ] || [ ! -f "$PUNCT_DIR/model.int8.onnx" ]; then
    echo "Downloading $PUNCT_DIR.tar.bz2..."
    fetch "$PUNCT_BASE/$PUNCT_DIR.tar.bz2" "$PUNCT_DIR.tar.bz2"
    tar xjf "$PUNCT_DIR.tar.bz2"
    rm -f "$PUNCT_DIR.tar.bz2"
    # Keep only what the daemon loads.
    rm -f "$PUNCT_DIR/model.onnx"
else
    echo "$PUNCT_DIR already present, skipping."
fi

echo
echo "Models in $MODELS_DIR:"
ls -la
echo
echo "Done. Install the user service with: scripts/install-user-service.sh"
