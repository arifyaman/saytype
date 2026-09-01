#!/usr/bin/env bash
# Sets up ydotool + ydotoool for text injection.
#
# Steps:
#   1. Install ydotool via apt (needs sudo).
#   2. Grant /dev/uinput access to your user (input group + udev rule).
#   3. Install ydotoold as a systemd *user* service (socket lands in your
#      XDG_RUNTIME_DIR, reachable by the saytype daemon without extra env).
#
# Re-login (or reboot) is required after the group change takes effect.
set -euo pipefail

USER_NAME="${SUDO_USER:-$USER}"

if ! command -v ydotool >/dev/null 2>&1; then
    echo ">> Installing ydotool (sudo password may be requested)..."
    sudo apt-get install -y ydotool
fi
echo ">> ydotool: $(command -v ydotool)"

# --- /dev/uinput access -------------------------------------------------
if [ ! -e /dev/uinput ]; then
    echo ">> WARNING: /dev/uinput does not exist. Load the module:"
    echo "   sudo modprobe uinput"
else
    if [ "$(stat -c '%G' /dev/uinput)" != "input" ]; then
        echo ">> Installing udev rule so /dev/uinput belongs to group 'input'..."
        sudo tee /etc/udev/rules.d/99-ydotool.rules >/dev/null <<'EOF'
KERNEL=="uinput", GROUP="input", MODE="0660"
EOF
        sudo udevadm control --reload-rules || true
        sudo udevadm trigger || true
    fi
    if ! id -nG "$USER_NAME" | tr ' ' '\n' | grep -qx input; then
        echo ">> Adding $USER_NAME to group 'input' (re-login required)..."
        sudo usermod -aG input "$USER_NAME"
    fi
fi

# --- ydotoold as a user service ------------------------------------------
YDOTOOLD="$(command -v ydotoold || true)"
if [ -z "$YDOTOOLD" ]; then
    echo ">> ydotoold not found on PATH; skipping user service install."
    exit 0
fi

mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/ydotool-user.service" <<EOF
[Unit]
Description=ydotoold (user) - fake input device daemon for ydotool
After=graphical-session.target

[Service]
Type=simple
ExecStart=$YDOTOOLD
Restart=on-failure
RestartSec=2

[Install]
WantedBy=graphical-session.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now ydotool-user.service
echo ">> ydotool-user.service started."

# --- round trip test ------------------------------------------------------
cat <<'EOF'

>> Next steps:
   1. Re-login to your GNOME session (so the 'input' group applies).
   2. Focus a text editor, then run:
        ydotool type "hello saytype"
      and confirm the text appears.
   3. Then install the daemon: scripts/install-user-service.sh
EOF
