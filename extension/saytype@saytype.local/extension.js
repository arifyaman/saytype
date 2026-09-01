// SayType HUD - focus-free dictation overlay for the saytype daemon.
//
// Draws a pill below the top panel while dictation is active. It is a
// Clutter actor on the shell stage, not a window, so it never steals focus
// and works the same on X11 and Wayland sessions. All state arrives over
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
const PILL_TOP_GAP = 12;
const IDLE_LABEL = 'Listening\u2026';
const MIC_TEXT = '\u{1F3A4}'; // microphones

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
    this._panelAllocId = 0;
    this._repositionId = 0;
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

    Main.layoutManager.addTopChrome(this._pill);
    // Follow the panel on monitor / workarea changes. Main.panel is a JS
    // class without C-signal connectivity from here, so use the display's
    // workareas-changed signal (the shell's own Panel does the same).
    this._panelAllocId = global.display.connect('workareas-changed',
      () => this._queueReposition());

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
    if (this._panelAllocId)
      global.display.disconnect(this._panelAllocId);
    if (this._repositionId)
      GLib.Source.remove(this._repositionId);
    this._stopPulse();
    if (this._pill.get_parent())
      Main.layoutManager.removeChrome(this._pill);
    this._pill.destroy();
  },

  _setState(recording) {
    if (this._recording === recording)
      return;
    this._recording = recording;
    this._text = '';
    this._updateLabel();
    if (recording) {
      this._pill.show();
      this._startPulse();
      this._queueReposition();
    } else {
      this._pill.hide();
      this._stopPulse();
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

  _reposition() {
    if (!this._recording)
      return;
    const mon = Main.layoutManager.primaryMonitor;
    let [w] = this._pill.get_size();
    if (!w)
      w = 200;
    const panelH = Main.panel ? Main.panel.height : 32;
    const x = Math.round(mon.x + (mon.width - w) / 2);
    const y = Math.round(mon.y + panelH + PILL_TOP_GAP);
    this._pill.set_position(x, y);
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
