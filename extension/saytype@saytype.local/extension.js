// SayType HUD - focus-free dictation overlay for the saytype daemon.
//
// While dictation is active it draws a dim overlay across all monitors and
// a pill near the mouse pointer (which it tracks), so the HUD is visible on
// any screen in a multi-monitor setup - this positioning is deliberately
// the same design as the original MVP1 HUD (see `git show
// main:extension/saytype@saytype.local/extension.js`), which worked fine;
// a detour through a fixed bottom-anchored redesign turned out to fix
// nothing the daemon wasn't already doing right and cost the pointer-
// following UX, so it was reverted. The one real change from the original
// is wrapping instead of truncating (see `_label` below) - the original
// also cropped long text to "..." at this same width, it just never came
// up until dictation got longer. It is a Clutter actor on the shell
// stage, not a window, so it never steals keyboard focus and works the
// same on X11 and Wayland sessions. All state arrives over the session
// bus. The HUD consumes/calls only these members of the interface (the
// full D-Bus API, incl. Toggle()/Stop() used by the custom hotkey, is
// documented in AGENTS.md):
//
//   io.saytype.Dictate / io.saytype.Dictate1
//     signals:  StateChanged(String), TranscriptUpdated(String)
//     methods:  EraseWord(), UndoErase()
//
// TranscriptUpdated carries the daemon's authoritative visible transcript
// (committed finals plus the live partial, mid-dictation erasures applied,
// casing/punctuation restored); the pill renders it verbatim. While
// recording, the Left and Right arrow keys are grabbed as global
// keybindings: Left erases the last word (EraseWord), Right restores it
// (UndoErase). The grab is added only for the duration of the session and
// removed the moment it stops, so the arrow keys behave normally (text
// cursor movement, etc.) the rest of the time; no mouse input is used.
// Dictation itself is started/stopped with a separate custom hotkey.
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
import Meta from 'gi://Meta';
import Pango from 'gi://Pango';
import Shell from 'gi://Shell';
import St from 'gi://St';

import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';

const BUS_NAME = 'io.saytype.Dictate';
const OBJECT_PATH = '/io/saytype/Dictate';
const INTERFACE_NAME = 'io.saytype.Dictate1';

const SayTypeInterface = `
<node>
  <interface name="${INTERFACE_NAME}">
    <method name="EraseWord" />
    <method name="UndoErase" />
    <signal name="StateChanged">
      <arg type="s" name="state" />
    </signal>
    <signal name="TranscriptUpdated">
      <arg type="s" name="text" />
    </signal>
  </interface>
</node>`;
const SayTypeProxy = Gio.DBusProxy.makeProxyWrapper(SayTypeInterface);

const CURSOR_GAP = 28; // pill offset from the pointer, below it
const EDGE_MARGIN = 8; // keep the pill off the monitor edges
const TRACK_INTERVAL_MS = 120; // pointer tracking cadence while recording
const IDLE_LABEL = 'Listening\u2026';
const MIC_ICON_NAME = 'audio-input-microphone-symbolic'; // Adwaita/system icon theme, always present
const MIC_ICON_SIZE = 20; // px, sized up with the bigger initial pill/text
const OVERLAY_STYLE = 'background-color: rgba(0, 0, 0, 0.35);';
const ERASE_KEYBINDING = 'erase-word-keybinding';
const UNDO_KEYBINDING = 'undo-erase-keybinding';

const PILL_STYLE = `
  background-color: rgba(28, 28, 30, 0.94);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 22px;
  padding: 12px 18px;
  spacing: 12px;
`;
const MIC_STYLE = 'color: #f2f2f2;';
const TEXT_STYLE = `
  color: #f2f2f2;
  font-size: 17px;
`;

function SayTypeHUD(settings) {
  this._init(settings);
}

SayTypeHUD.prototype = {
  _init(settings) {
    this._settings = settings;
    this._keybindingsActive = false;
    this._proxy = null;
    this._conns = [];
    this._nameConn = 0;
    this._layoutConnId = 0;
    this._repositionId = 0;
    this._trackId = 0;
    this._pulseId = 0;
    this._pulseDimmed = false;
    // The daemon's authoritative visible transcript (committed finals
    // plus the live partial, erasures applied, casing/punctuation
    // restored). Rendered verbatim in the pill.
    this._transcript = '';
    this._recording = false;

    this._mic = new St.Icon({
      icon_name: MIC_ICON_NAME,
      icon_size: MIC_ICON_SIZE,
      style: MIC_STYLE,
      y_align: Clutter.ActorAlign.CENTER,
    });

    this._label = new St.Label({
      text: IDLE_LABEL,
      style: TEXT_STYLE,
      y_align: Clutter.ActorAlign.CENTER,
    });
    // Wrap onto multiple lines instead of truncating, and never stop
    // rendering once one line's worth fills (both reproduced live):
    // explicitly disable single-line mode - it is not guaranteed off by
    // default for every St.Label in this build, and single-line-mode
    // silently discards (not ellipsizes, not clips-with-indicator - just
    // stops rendering) any content past the first line regardless of
    // `line_wrap`. The actual wrap width is computed and (re)applied
    // dynamically per update in `_resizePill`, not fixed here.
    this._label.clutter_text.single_line_mode = false;
    this._label.clutter_text.line_wrap = true;
    this._label.clutter_text.line_wrap_mode = Pango.WrapMode.WORD_CHAR;
    this._label.clutter_text.ellipsize = Pango.EllipsizeMode.NONE;

    this._pill = new St.BoxLayout({
      vertical: false,
      style: PILL_STYLE,
    });
    this._pill.add_child(this._mic);
    this._pill.add_child(this._label);
    this._pill.hide();
    // Belt-and-suspenders on top of the polling in `_startTracking`/
    // `_queueReposition` below: also reposition the instant Clutter
    // finishes computing this actor's real post-layout size (fires after
    // a text change grows/shrinks the wrapped line count), so a
    // repositioning read of `get_size()` is never racing a relayout that
    // has not happened yet.
    this._pill.connect('notify::allocation', () => this._reposition());

    // Full-screen dim shown while recording, purely visual (not an input
    // surface: erase/undo is driven by the Left/Right arrow keybindings
    // below, not mouse input). Added to top chrome BEFORE the pill so the
    // pill (added later) draws on top. St.Actor is not in this build's
    // curated St typelib, so use St.Widget, as the shell itself does for
    // its own stage actors.
    this._overlay = new St.Widget({
      style: OVERLAY_STYLE,
      reactive: false,
    });
    this._overlay.hide();
    Main.layoutManager.addTopChrome(this._overlay);

    Main.layoutManager.addTopChrome(this._pill);
    // `_resizePill` queries theme-node-dependent preferred sizes
    // (padding, etc. from CSS), which requires the actor to actually be
    // in the stage - calling it before `addTopChrome` above caused a real
    // crash ("st_widget_get_theme_node called on the widget which is not
    // in the stage" -> NULL pointer -> an unhandled promise rejection
    // that killed `_initDBus`, silently taking the whole D-Bus connection
    // down with it - not merely a cosmetic glitch). Must run after.
    this._resizePill();
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
    this._conns.push(proxy.connectSignal('TranscriptUpdated',
      (p, sender, [text]) => {
        this._setTranscript(text);
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
    this._removeKeybindings();
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
    this._transcript = '';
    this._updateLabel();
    if (recording) {
      this._layoutOverlay();
      this._overlay.show();
      this._pill.show();
      this._startPulse();
      this._startTracking();
      this._queueReposition();
      this._addKeybindings();
    } else {
      this._overlay.hide();
      this._pill.hide();
      this._stopPulse();
      this._stopTracking();
      this._removeKeybindings();
    }
  },

  // Grab the Left/Right arrow keys globally only while recording, so
  // erase/undo works no matter which window has focus (or none) without
  // permanently stealing arrow-key text navigation the rest of the time.
  _addKeybindings() {
    if (this._keybindingsActive)
      return;
    this._keybindingsActive = true;
    Main.wm.addKeybinding(
      ERASE_KEYBINDING,
      this._settings,
      Meta.KeyBindingFlags.NONE,
      Shell.ActionMode.ALL,
      () => this._callErase());
    Main.wm.addKeybinding(
      UNDO_KEYBINDING,
      this._settings,
      Meta.KeyBindingFlags.NONE,
      Shell.ActionMode.ALL,
      () => this._callUndo());
  },

  _removeKeybindings() {
    if (!this._keybindingsActive)
      return;
    this._keybindingsActive = false;
    Main.wm.removeKeybinding(ERASE_KEYBINDING);
    Main.wm.removeKeybinding(UNDO_KEYBINDING);
  },

  // The daemon's authoritative visible transcript (committed finals plus
  // the live partial, erasures applied, casing/punctuation restored).
  // Replaces whatever the pill currently shows.
  _setTranscript(text) {
    if (!this._recording)
      return;
    this._transcript = text;
    this._updateLabel();
    this._queueReposition();
  },

  // Grows the label onto more lines as text wraps (proven correct with
  // targeted diagnostics: `label.get_size()` height reliably tracks the
  // wrapped content, 19px/38px/57px for 1/2/3 lines). What *doesn't*
  // reliably grow on its own: the pill (`St.BoxLayout`) that contains it -
  // it settles to a height once and then never re-requests its natural
  // size again for the rest of the session, no matter how tall the label
  // gets, silently clipping/mispositioning the overflow. Rather than rely
  // on Clutter's automatic parent relayout (empirically stuck here), the
  // pill's height is forced explicitly from the label's own
  // (already-correct) preferred height every update.
  _updateLabel() {
    this._label.set_text(this._transcript || IDLE_LABEL);
    this._resizePill();
  },

  // Sizes the pill to fit its content exactly, both dimensions, on every
  // text update - not just height (see the comment on `_updateLabel`),
  // also width: growing the pill only as wide as the current text
  // actually needs (up to half the width of the monitor it is currently
  // showing on - re-evaluated on every call via `_currentMonitor`, so it
  // is never fixed to whichever screen it first appeared on) so a short
  // phrase (or the idle "Listening..." label) gets a narrow pill, and
  // only once real content would exceed that cap does it stop growing
  // wider and start wrapping onto more lines instead.
  _resizePill() {
    const clutterText = this._label.clutter_text;
    const maxTextWidth = Math.round(this._currentMonitor().width / 2);
    // Natural (unwrapped, single-line) width first: temporarily clear any
    // previously forced width so the query reflects the current text,
    // not the last computed size.
    clutterText.set_width(-1);
    const [, naturalWidth] = clutterText.get_preferred_width(-1);
    const effectiveWidth = Math.min(naturalWidth, maxTextWidth);
    clutterText.set_width(effectiveWidth);
    const [, labelHeight] = clutterText.get_preferred_height(effectiveWidth);
    const [, micWidth] = this._mic.get_preferred_width(-1);
    const [, micHeight] = this._mic.get_preferred_height(-1);
    const horizontalPadding = 36; // PILL_STYLE: `padding: 12px 18px` (left+right)
    const verticalPadding = 24; // PILL_STYLE: `padding: 12px 18px` (top+bottom)
    const spacing = 12; // PILL_STYLE: `spacing: 12px` (between mic and label)
    this._pill.set_width(effectiveWidth + micWidth + spacing + horizontalPadding);
    this._pill.set_height(Math.max(labelHeight, micHeight) + verticalPadding);
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

  // The monitor the pointer currently sits on (falling back to primary),
  // shared by `_reposition` (where the pill goes) and `_resizePill` (how
  // wide it is allowed to grow before wrapping - half of that monitor's
  // width): both need "whichever screen the pill is showing on right
  // now", re-evaluated live, not fixed to wherever it first appeared.
  _currentMonitor() {
    const [px, py] = global.get_pointer();
    const mon = Main.layoutManager.monitors.find(m =>
      px >= m.x && px < m.x + m.width &&
      py >= m.y && py < m.y + m.height);
    return mon || Main.layoutManager.primaryMonitor;
  },

  // Place the pill near the pointer, on the monitor that contains it, so a
  // multi-monitor user always sees it on the screen they are working on.
  _reposition() {
    if (!this._recording)
      return;
    const [px, py] = global.get_pointer();
    const mon = this._currentMonitor();
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

  // Erase the last visible word (Left arrow key while recording).
  _callErase() {
    if (!this._proxy)
      return;
    this._proxy.EraseWordAsync().catch(e => {
      log(`saytype-hud: EraseWord failed (daemon running?): ${e.message}`);
    });
  },

  // Restore the last erased word (Right arrow key while recording).
  _callUndo() {
    if (!this._proxy)
      return;
    this._proxy.UndoEraseAsync().catch(e => {
      log(`saytype-hud: UndoErase failed (daemon running?): ${e.message}`);
    });
  },
};

export default class SayTypeHudExtension extends Extension {
  enable() {
    log('saytype-hud: enabling');
    this._saytypeHud = new SayTypeHUD(this.getSettings());
  }

  disable() {
    log('saytype-hud: disabling');
    if (this._saytypeHud) {
      this._saytypeHud.destroy();
      this._saytypeHud = null;
    }
  }
}
