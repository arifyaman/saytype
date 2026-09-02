// SayType HUD - focus-free dictation overlay for the saytype daemon.
//
// While dictation is active it draws a dim overlay across all monitors and
// a pill near the mouse pointer (which it tracks), so the HUD is visible on
// any screen in a multi-monitor setup. It is a Clutter actor on the shell
// stage, not a window, so it never steals focus and works the same on X11
// and Wayland sessions. All state arrives over
// the session bus:
//
//   io.saytype.Dictate / io.saytype.Dictate1
//     signals:  StateChanged(String), SegmentTranscribed(String)
//     methods:  Toggle(), Stop()
//
// The daemon drains the pipeline (final SegmentTranscribed) before
// sending StateChanged("Idle"), so the last segment is always visible.
//
// ESM-first extension (GNOME 46): no `imports` module loader, no `Me`
// global - Main comes from the shell's main.js module and the extension
// dir from the Extension base class. API usage follows what gnome-shell
// 46 itself and the official gnome-shell-extensions do in this build
// (inline St styles, add_child, Main.layoutManager chrome,
// Gio.DBusProxy.makeProxyWrapper, GLib.Source.remove).

'use strict';

import Clutter from 'gi://Clutter';
import GLib from 'gi://GLib';
import Gio from 'gi://Gio';
import Pango from 'gi://Pango';
import St from 'gi://St';

import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';

const BUS_NAME = 'io.saytype.Dictate';
const OBJECT_PATH = '/io/saytype/Dictate';
const INTERFACE_NAME = 'io.saytype.Dictate1';

const SayTypeInterface = `
<node>
  <interface name="${INTERFACE_NAME}">
    <method name="Toggle" />
    <method name="Stop" />
    <signal name="StateChanged">
      <arg type="s" name="state" />
    </signal>
    <signal name="SegmentTranscribed">
      <arg type="s" name="text" />
    </signal>
  </interface>
</node>`;
const SayTypeProxy = Gio.DBusProxy.makeProxyWrapper(SayTypeInterface);

const MAX_TEXT_WIDTH_PX = 560;
const CURSOR_GAP = 28; // pill offset from the pointer, below it
const EDGE_MARGIN = 8; // keep the pill off the monitor edges
const TRACK_INTERVAL_MS = 120; // pointer tracking cadence while recording
const IDLE_LABEL = 'Listening\u2026';
const MIC_TEXT = '\u{1F3A4}'; // microphones
const OVERLAY_STYLE = 'background-color: rgba(0, 0, 0, 0.35);';

const PILL_STYLE = `
  background-color: rgba(28, 28, 30, 0.94);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 20px;
  padding: 7px 16px;
  spacing: 10px;
`;
const MIC_STYLE = 'font-size: 18px;';
const TEXT_STYLE = `
  color: #f2f2f2;
  font-size: 15px;
  max-width: ${MAX_TEXT_WIDTH_PX}px;
`;

function SayTypeHUD() {
  this._init();
}

SayTypeHUD.prototype = {
  _init() {
    this._proxy = null;
    this._conns = [];
    this._nameConn = 0;
    this._layoutConnId = 0;
    this._repositionId = 0;
    this._trackId = 0;
    this._pulseId = 0;
    this._pulseDimmed = false;
    this._text = '';
    this._recording = false;

    this._mic = new St.Label({
      text: MIC_TEXT,
      style: MIC_STYLE,
      y_align: Clutter.ActorAlign.CENTER,
    });

    this._label = new St.Label({
      text: IDLE_LABEL,
      style: TEXT_STYLE,
      y_align: Clutter.ActorAlign.CENTER,
    });
    this._label.clutter_text.ellipsize = Pango.EllipsizeMode.END;

    this._pill = new St.BoxLayout({
      vertical: false,
      style: PILL_STYLE,
      reactive: true,
    });
    this._pill.add_child(this._mic);
    this._pill.add_child(this._label);
    this._pill.hide();
    this._pill.connect('button-press-event', () => {
      this._callToggle();
      return Clutter.EVENT_STOP;
    });

    // Full-screen dim shown while recording. Non-reactive so it never
    // blocks mouse or keyboard input; it is only a visual cue. Added to
    // top chrome BEFORE the pill so the pill (added later) draws on top.
    // St.Actor is not in this build's curated St typelib, so use
    // St.Widget, as the shell itself does for its own stage actors.
    this._overlay = new St.Widget({
      style: OVERLAY_STYLE,
      reactive: false,
    });
    this._overlay.hide();
    Main.layoutManager.addTopChrome(this._overlay);

    Main.layoutManager.addTopChrome(this._pill);
    this._layoutOverlay();
    // Follow the panel on monitor / workarea changes. Main.panel is a JS
    // class without C-signal connectivity from here, so use the display's
    // workareas-changed signal (the shell's own Panel does the same).
    this._layoutConnId = global.display.connect('workareas-changed',
      () => {
        this._layoutOverlay();
        this._queueReposition();
      });

    this._initDBus();
  },

  async _initDBus() {
    let proxy;
    try {
      proxy = new SayTypeProxy(Gio.DBus.session, BUS_NAME, OBJECT_PATH);
    } catch (e) {
      log(`saytype-hud: cannot create D-Bus proxy: ${e.message}`);
      return;
    }
    this._proxy = proxy;

    this._conns.push(proxy.connectSignal('StateChanged',
      (p, sender, [state]) => {
        this._setState(state === 'Recording');
      }));
    this._conns.push(proxy.connectSignal('SegmentTranscribed',
      (p, sender, [text]) => {
        this._appendText(text);
      }));

    // The proxy reports loss of the name owner (daemon died) so a stale
    // pill gets cleared even if the daemon crashes.
    this._nameConn = proxy.connect('g-name-loss', () => {
      log('saytype-hud: daemon left the bus, hiding HUD');
      this._setState(false);
    });
  },

  destroy() {
    if (this._proxy) {
      for (const conn of this._conns) {
        try {
          this._proxy.disconnectSignal(conn);
        } catch (e) {
          /* already gone */
        }
      }
      if (this._nameConn) {
        try {
          this._proxy.disconnect(this._nameConn);
        } catch (e) {
          /* already gone */
        }
      }
    }
    if (this._layoutConnId)
      global.display.disconnect(this._layoutConnId);
    if (this._repositionId)
      GLib.Source.remove(this._repositionId);
    this._stopPulse();
    this._stopTracking();
    if (this._pill.get_parent())
      Main.layoutManager.removeChrome(this._pill);
    this._pill.destroy();
    if (this._overlay.get_parent())
      Main.layoutManager.removeChrome(this._overlay);
    this._overlay.destroy();
  },

  _setState(recording) {
    if (this._recording === recording)
      return;
    this._recording = recording;
    this._text = '';
    this._updateLabel();
    if (recording) {
      this._layoutOverlay();
      this._overlay.show();
      this._pill.show();
      this._startPulse();
      this._startTracking();
      this._queueReposition();
    } else {
      this._overlay.hide();
      this._pill.hide();
      this._stopPulse();
      this._stopTracking();
    }
  },

  _appendText(text) {
    if (!this._recording)
      return;
    this._text = this._text ? this._text + ' ' + text : text;
    this._updateLabel();
    this._queueReposition();
  },

  _updateLabel() {
    this._label.set_text(this._text || IDLE_LABEL);
  },

  _layoutOverlay() {
    const mons = Main.layoutManager.monitors;
    if (!mons.length)
      return;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const m of mons) {
      minX = Math.min(minX, m.x);
      minY = Math.min(minY, m.y);
      maxX = Math.max(maxX, m.x + m.width);
      maxY = Math.max(maxY, m.y + m.height);
    }
    this._overlay.set_position(Math.round(minX), Math.round(minY));
    this._overlay.set_size(Math.round(maxX - minX), Math.round(maxY - minY));
  },

  // Place the pill near the pointer, on the monitor that contains it, so a
  // multi-monitor user always sees it on the screen they are working on.
  _reposition() {
    if (!this._recording)
      return;
    const [px, py] = global.get_pointer();
    let mon = Main.layoutManager.monitors.find(m =>
      px >= m.x && px < m.x + m.width &&
      py >= m.y && py < m.y + m.height);
    if (!mon)
      mon = Main.layoutManager.primaryMonitor;
    let [w, h] = this._pill.get_size();
    if (!w)
      [w, h] = [200, 40];
    let x = Math.round(px - w / 2);
    x = Math.max(mon.x + EDGE_MARGIN,
      Math.min(x, mon.x + mon.width - w - EDGE_MARGIN));
    let y = py + CURSOR_GAP;
    if (y + h > mon.y + mon.height - EDGE_MARGIN)
      y = py - CURSOR_GAP - h;
    y = Math.max(mon.y + EDGE_MARGIN,
      Math.min(y, mon.y + mon.height - h - EDGE_MARGIN));
    this._pill.set_position(Math.round(x), Math.round(y));
  },

  _startTracking() {
    this._stopTracking();
    this._trackId = GLib.timeout_add(GLib.PRIORITY_DEFAULT,
      TRACK_INTERVAL_MS, () => {
        this._reposition();
        return true;
      });
  },

  _stopTracking() {
    if (this._trackId) {
      GLib.Source.remove(this._trackId);
      this._trackId = 0;
    }
  },

  _queueReposition() {
    if (this._repositionId)
      return;
    this._repositionId = GLib.timeout_add(
      GLib.PRIORITY_DEFAULT,
      10,
      () => {
        this._repositionId = 0;
        this._reposition();
        return false;
      });
  },

  _startPulse() {
    this._stopPulse();
    this._pulseDimmed = false;
    this._mic.opacity = 255;
    this._pulseId = GLib.timeout_add(GLib.PRIORITY_DEFAULT, 650, () => {
      this._pulseDimmed = !this._pulseDimmed;
      this._mic.ease({
        opacity: this._pulseDimmed ? 100 : 255,
        duration: 600,
        mode: Clutter.AnimationMode.EASE_IN_OUT,
      });
      return true;
    });
  },

  _stopPulse() {
    if (this._pulseId) {
      GLib.Source.remove(this._pulseId);
      this._pulseId = 0;
    }
    if (this._mic)
      this._mic.opacity = 255;
  },

  _callToggle() {
    if (!this._proxy)
      return;
    this._proxy.ToggleAsync().catch(e => {
      log(`saytype-hud: Toggle failed (daemon running?): ${e.message}`);
    });
  },
};

export default class SayTypeHudExtension extends Extension {
  enable() {
    log('saytype-hud: enabling');
    this._saytypeHud = new SayTypeHUD();
  }

  disable() {
    log('saytype-hud: disabling');
    if (this._saytypeHud) {
      this._saytypeHud.destroy();
      this._saytypeHud = null;
    }
  }
}
