# saytype

Background speech-to-text dictation for GNOME.
Toggle it with a hotkey, speak naturally, and your words get typed into
whatever window has keyboard focus.
A focus-free HUD pill (a GNOME Shell extension) appears below the top panel
while dictating and shows the running transcript.
There is no per-app integration: text is injected as synthesized key events
via ydotool, so it works in any application.

## How it works

Each dictation session runs a pipeline inside the `saytype` daemon:

    PipeWire mic capture -> Silero VAD -> sherpa-onnx ASR -> ydotool typing

- **Daemon** (`saytype`): a Rust binary running as a systemd user service.
  Owns the audio pipeline, the models, and the D-Bus service.
  ASR uses sherpa-onnx with two interchangeable backends, auto-detected from
  `models/`: Moonshine v2 (preferred; batch per segment, outputs casing and
  punctuation) or streaming Zipformer (fallback; the only backend able to
  produce live partials, needed for future real-time typing).
- **HUD** (`extension/saytype@saytype.local/`): a GNOME Shell extension that
  lives inside gnome-shell.
  It is a plain D-Bus client - no audio, no models.
  Because it is shell-stage artwork rather than a window, it never steals
  focus and behaves identically on X11 and Wayland.
  While recording it shows a pill (mic + accumulated transcript); clicking
  the pill toggles dictation.
- **IPC**: the session D-Bus bus.
  Bus name `io.saytype.Dictate`, path `/io/saytype/Dictate`, interface
  `io.saytype.Dictate1`.
  Methods `Toggle()` / `Stop()`; signals `StateChanged(String)` and
  `SegmentTranscribed(String)`.
  The daemon always emits the final `SegmentTranscribed` before
  `StateChanged("Idle")`.

## Requirements

- GNOME 46 (developed and tested on Ubuntu 24.04, X11 or Wayland)
- A microphone set as the default audio input (check with `wpctl status`)
- `ydotool` + `ydotoold` (installed by `scripts/setup-ydotool.sh`)
- Models in `models/` (downloaded by `scripts/download-models.sh`)

## Install

```sh
scripts/setup-ydotool.sh          # one-time: apt ydotool, udev rule, ydotoold service
scripts/download-models.sh        # Silero VAD + Moonshine v2 + Zipformer into models/
scripts/install-user-service.sh   # release build, install binary + saytype-toggle, enable service
scripts/install-extension.sh      # install + enable the HUD extension
```

After `install-extension.sh`, restart the GNOME shell once so it loads the
extension: Alt+F2, type `r`, press Enter (X11), or log out and back in
(Wayland).

Bind the dictation hotkey in GNOME Settings -> Keyboard -> Custom Shortcuts
with the command `~/.local/bin/saytype-toggle`.

## Usage

- Hit the hotkey (or run `~/.local/bin/saytype-toggle`): the pill appears and
  dictation starts.
  Transcribed segments are typed into the focused app and accumulate in the
  pill.
- Hit the hotkey again - or click the pill - to stop.
- If the daemon is not running, the pill clicks are ignored; check
  `journalctl --user -u saytype`.

Offline model checks (16 kHz mono WAV, e.g. from `models/*/test_wavs/`):

```sh
saytype --transcribe <wav>
saytype --vad-test <wav>
```

## Development

```sh
cargo build --release        # binary at target/release/saytype
cargo test                   # unit tests (AudioRing, capitalize_first)
cargo run                    # run the daemon manually (needs models/ + session bus)
journalctl --user -u saytype -f
gdbus monitor --session --dest io.saytype.Dictate --object-path /io/saytype/Dictate
```

After changing the extension, redeploy and restart the shell:

```sh
node --check extension/saytype@saytype.local/extension.js
scripts/install-extension.sh
```

Architecture, data flow, and hard-won gotchas live in `AGENTS.md`.
The original MVP spec is `gnome-dictation-mvp-guide.md`; the next iteration
(live/real-time typing) is planned in `mvp2-plan.md`.
