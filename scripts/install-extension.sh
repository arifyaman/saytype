#!/usr/bin/env bash
# Install or uninstall the SayType HUD GNOME Shell extension.
#
#   scripts/install-extension.sh             copy + enable
#   scripts/install-extension.sh --uninstall  disable + remove
#
# A running gnome-shell only scans the extension directories at startup,
# so a freshly copied extension is pre-registered in
# org.gnome.shell enabled-extensions and loads on the next shell start
# (log out/in, or Alt+F2 -> r on X11). Re-running the script after a shell
# restart also enables it live via gnome-extensions.
set -euo pipefail

UUID="saytype@saytype.local"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_SRC="$SCRIPT_DIR/../extension/$UUID"
EXT_DEST="$HOME/.local/share/gnome-shell/extensions/$UUID"
GSET_SCHEMA="org.gnome.shell"
GSET_KEY="enabled-extensions"

# Rewrite the enabled-extensions gsetting list, add/remove $1.
set_enabled_list() {
    local op="$1" uuid="$2"
    gsettings set "$GSET_SCHEMA" "$GSET_KEY" "$(
        gsettings get "$GSET_SCHEMA" "$GSET_KEY" | python3 -c '
import sys
uuid = sys.argv[1]
cur = eval(sys.stdin.read().strip() or "[]")
if sys.argv[2] == "add" and uuid not in cur:
    cur.append(uuid)
if sys.argv[2] == "remove":
    cur = [x for x in cur if x != uuid]
print("[" + ", ".join(repr(x) for x in cur) + "]")
' "$uuid" "$op")"
}

if [[ ! -d "$EXT_SRC" ]]; then
    echo "extension source not found: $EXT_SRC" >&2
    exit 1
fi

if [[ "${1:-}" == "--uninstall" ]]; then
    gnome-extensions disable "$UUID" 2>/dev/null || true
    set_enabled_list remove "$UUID"
    rm -rf "$EXT_DEST"
    echo "uninstalled $UUID"
    exit 0
fi

mkdir -p "$EXT_DEST"
cp -rf "$EXT_SRC"/* "$EXT_DEST"/
if [[ -d "$EXT_DEST/schemas" ]]; then
    glib-compile-schemas "$EXT_DEST/schemas"
fi
set_enabled_list add "$UUID"

if gnome-extensions enable "$UUID" 2>/dev/null; then
    echo "installed and enabled $UUID (live)"
else
    cat <<EOF
Installed $UUID and pre-enabled it for the next shell start.
Restart gnome-shell to load it: log out and back in, or
Alt+F2 -> r on an X11 session.
EOF
fi
