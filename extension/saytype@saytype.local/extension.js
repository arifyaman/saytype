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

const { St, Clutter, Gio, GLib, Pango, Main } = imports.gi;

const BUS_NAME = 'io.saytype.Dictate';
const OBJECT_PATH = '/io/saytype/Dictate';
const INTERFACE_NAME = 'io.saytype.Dictate1';

const MAX_TEXT_CHARS = 64;
const PILL_TOP_GAP = 12;
const IDLE_LABEL = 'Listening\u2026';

function SayTypeHUD() {
  this._init();
}

SayTypeHUD.prototype = {
  _init() {
    this._bus = Gio.DBusConnection.getDefault();
    this._signalSub = 0;
    this._nameSub = 0;
    this._repositionId = 0;
    this._pulseAnim = null;
    this._text = '';
    this._recording = false;

    this._label = new St.Label({ text: IDLE_LABEL });
    this._label.set_ellipsize(Pango.EllipsizeMode.END);
    this._label.set_max_width_chars(MAX_TEXT_CHARS);
    this._label.add_style_class('saytype-pill-text');

    // Bundled icon: the distro themes here have no plain microphone
    // symbolic, so we ship our own and register Me.dir/icons in enable().
    this._mic = new St.Icon({
      icon_name: 'saytype-mic-symbolic',
      pixel_size: 20,
    });
    this._mic.add_style_class('saytype-mic');

    this._pill = new St.BoxLayout({
      orientation: St.Orientation.HORIZONTAL,
      spacing: 10,
    });
    this._pill.add_style_class('saytype-pill');
    this._pill.add_actor(this._mic);
    this._pill.add_actor(this._label);
    this._pill.set_visible(false);
    this._pill.connect('button-press-event', () => {
      this._callToggle();
      return Clutter.EVENT_STOP;
    });

    Main.uiGroup.add_actor(this._pill);
    this._layoutChangedId = Main.layout.connect('changed', () => this._reposition());

    this._signalSub = this._bus.signal_subscribe(
      BUS_NAME,
      OBJECT_PATH,
      INTERFACE_NAME,
      null,
      Gio.DBusSignalFlags.NONE,
      (conn, sender, path, iface, signal, params) => {
        this._onSignal(signal, params);
      });

    // Detect the daemon leaving the bus so a stale pill gets cleared.
    this._nameSub = this._bus.signal_subscribe(
      null,
      null,
      'org.freedesktop.DBus',
      'NameOwnerChanged',
      Gio.DBusSignalFlags.NONE,
      (conn, sender, path, iface, signal, params) => {
        const [name, _oldOwner, newOwner] = params.deep_unpack();
        if (name === BUS_NAME && newOwner === null) {
          log('saytype-hud: daemon left the bus, hiding HUD');
          this._setState(false);
        }
      });
  },

  destroy() {
    if (this._signalSub)
      this._bus.signal_unsubscribe(this._signalSub);
    if (this._nameSub)
      this._bus.signal_unsubscribe(this._nameSub);
    if (this._layoutChangedId)
      Main.layout.disconnect(this._layoutChangedId);
    if (this._repositionId) {
      GLib.source_remove(this._repositionId);
      this._repositionId = 0;
    }
    this._stopPulse();
    this._pill.destroy();
  },

  _onSignal(signal, params) {
    if (signal === 'StateChanged') {
      const [state] = params.deep_unpack();
      this._setState(state === 'Recording');
    } else if (signal === 'SegmentTranscribed') {
      const [text] = params.deep_unpack();
      this._appendText(text);
    }
  },

  _setState(recording) {
    if (this._recording === recording)
      return;
    this._recording = recording;
    this._text = '';
    this._updateLabel();
    this._pill.set_visible(recording);
    if (recording) {
      this._startPulse();
      this._queueReposition();
    } else {
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
    if (!this._pill.get_visible())
      return;
    const mon = global.get_primary_monitor();
    const alloc = this._pill.get_allocation();
    const x = Math.round(mon.x + (mon.width - alloc.width) / 2);
    const y = Math.round(mon.y + Main.layout.panelHeight + PILL_TOP_GAP);
    this._pill.set_position(x, y);
  },

  _queueReposition() {
    if (this._repositionId)
      return;
    this._repositionId = GLib.timeout_add(
      GLib.PRIORITY_DEFAULT,
      30,
      () => {
        this._repositionId = 0;
        this._reposition();
        return GLib.SourceRemove.REMOVE;
      });
  },

  _startPulse() {
    this._stopPulse();
    this._mic.opacity = 255;
    this._pulseAnim = this._mic.new_animation('opacity', {
      duration: 650,
      'from': 255,
      'to': 90,
      mode: Clutter.AnimationMode.EASE_IN_OUT,
      autoreverse: true,
      loop: true,
    });
  },

  _stopPulse() {
    if (this._pulseAnim) {
      this._pulseAnim.stop();
      this._pulseAnim = null;
    }
    this._mic.opacity = 255;
  },

  _callToggle() {
    try {
      this._bus.call(
        BUS_NAME,
        OBJECT_PATH,
        INTERFACE_NAME,
        'Toggle',
        new GLib.Variant('()', []),
        null,
        Gio.DBusCallFlags.NONE,
        -1,
        null,
        (conn, result) => {
          try {
            this._bus.call_finish(result);
          } catch (e) {
            log(`saytype-hud: Toggle failed (daemon running?): ${e.message}`);
          }
        });
    } catch (e) {
      log(`saytype-hud: cannot reach bus: ${e.message}`);
    }
  },
};

function enable() {
  log('saytype-hud: enabling');
  const stage = St.stage_get_default();
  const iconTheme = St.ThemeContext.get_for_stage(stage).get_icon_theme();
  iconTheme.addPath(`${Me.dir}/icons`);
  this._saytypeHud = new SayTypeHUD();
}

function disable() {
  log('saytype-hud: disabling');
  if (this._saytypeHud) {
    this._saytypeHud.destroy();
    this._saytypeHud = null;
  }
}
