#!/usr/bin/env bash
# Builds the release binary, installs it to ~/.local/bin, and enables the
# saytype systemd user service (restarting it so the new binary runs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# cargo (via rustup) is not on PATH in non-interactive shells.
if ! command -v cargo >/dev/null 2>&1 && [ -x "$HOME/.cargo/bin/cargo" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi
command -v cargo >/dev/null 2>&1 || {
    echo "error: cargo not found. Install Rust (https://rustup.rs) or add it to PATH." >&2
    exit 1
}

cd "$REPO_DIR"

echo ">> Building (release)..."
cargo build --release

BIN="$HOME/.local/bin/saytype"
mkdir -p "$(dirname "$BIN")"
cp -f target/release/saytype "$BIN"
echo ">> Installed $BIN"

# Toggle helper for the GNOME custom shortcut (full path: user-service and
# custom-shortcut environments do not guarantee ~/.local/bin on PATH).
TOGGLE="$HOME/.local/bin/saytype-toggle"
install -m 0755 "$SCRIPT_DIR/saytype-toggle" "$TOGGLE"
echo ">> Installed $TOGGLE"

if [ ! -f "$REPO_DIR/models/silero_vad.onnx" ]; then
    echo ">> Models missing - downloading..."
    "$SCRIPT_DIR/download-models.sh"
fi

mkdir -p "$HOME/.config/systemd/user"
sed -e "s|@REPO@|$REPO_DIR|g" -e "s|@BIN@|$BIN|g" \
    "$REPO_DIR/systemd/saytype.service" \
    > "$HOME/.config/systemd/user/saytype.service"
echo ">> Installed ~/.config/systemd/user/saytype.service"

systemctl --user daemon-reload
systemctl --user enable saytype.service
# `enable --now` would not restart an already-running (old) daemon.
systemctl --user restart saytype.service

echo
echo ">> saytype service status:"
systemctl --user status saytype.service --no-pager || true
echo
echo ">> Live logs: journalctl --user -u saytype -f"
echo
echo ">> Bind a key in GNOME Settings -> Keyboard -> Custom Shortcuts:"
echo "     Name:    SayType dictation"
echo "     Command: $TOGGLE"
echo "     Key:     e.g. <Ctrl><Shift>Space"
