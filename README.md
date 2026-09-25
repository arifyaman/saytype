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
  ASR uses sherpa-onnx with three interchangeable backends, auto-detected
  from `models/`: NVIDIA Nemotron Speech Streaming EN 0.6B (preferred; truly
  streaming, live partials, native casing/punctuation, ~530k h training),
  streaming Zipformer (streaming fallback; all-caps output post-processed by
  an optional online punctuation model), and Moonshine v2 (batch per
  segment). Live partials power the real-time typing: text is typed into the
  target app word-by-word as it is recognized.
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
  Methods `Toggle()` / `Stop()`; signals `StateChanged(String)`,
  `PartialTranscribed(String)` (live hypothesis, streaming backend only) and
  `SegmentTranscribed(String)`.
  The daemon always emits the final `SegmentTranscribed` before
  `StateChanged("Idle")`, and every stage of the stop path is bounded, so
  that final signal always arrives (a wedged capture thread cannot hang the
  stop).

## Requirements

- GNOME 46 (developed and tested on Ubuntu 24.04, X11 or Wayland)
- A microphone set as the default audio input (check with `wpctl status`)
- `ydotool` + `ydotoold` (installed by `scripts/setup-ydotool.sh`)
- Models in `models/` (downloaded by `scripts/download-models.sh`); the location can be overridden with the `SAYTYPE_MODELS_DIR` environment variable

## Install

```sh
scripts/setup-ydotool.sh          # one-time: apt ydotool, udev rule, ydotoold service
scripts/download-models.sh        # default stack into models/ (VAD + Nemotron streaming + punct)
                                  #   add --zipformer / --moonshine / --all for the other backends
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

## Model selection

All three ASR backends can coexist in `models/`; the daemon auto-detects
whatever is present and, with the default `auto` selection, picks Nemotron
streaming, then Zipformer streaming, then Moonshine (batch).
`scripts/download-models.sh` fetches only the default stack (Nemotron) by
default; add `--zipformer`, `--moonshine` or `--all` to fetch the other
backends.

To force a backend, add `--asr <name>` to `ExecStart` in
`~/.config/systemd/user/saytype.service` and run
`systemctl --user daemon-reload && systemctl --user restart saytype`:

- `auto` (default) - best available, in the order above
- `streaming` - best streaming backend (Nemotron, then Zipformer)
- `zipformer` / `moonshine` - force that specific backend

The streaming backends (Nemotron, Zipformer) give live typing: words land
in the target app as they are recognized, and the HUD pill shows a live
partial. The offline backend (Moonshine) transcribes each finished
utterance in one pass - one typed chunk per utterance, no live partials.
Only these three dir layouts are auto-detected (detection lives in
`src/asr.rs`); other sherpa-onnx offline models (Whisper, Parakeet TDT,
Canary, SenseVoice, ...) are not picked up today.

## Development

```sh
cargo build --release        # binary at target/release/saytype
cargo test                   # unit tests (transcript incl. empty-final edge cases + mixed erase/undo order across the commit boundary + erase-boundary no-ops and shortening-final erase survival + position-tracking across a word-replacing final, injector incl. clipboard fallback + deferred-paste pipeline + leading-dash end-of-options typing, audio format conversion + malformed-rate resampling guard, VAD AudioRing + gated real-model Silero detection, ASR model detection + polish + backend fallback + empty/short-input guard + real-model streaming round trip, ONNX envelope validation (corrupt/truncated model = clean re-download error, never a process abort) across VAD/ASR loads, daemon stable-target logic + toggle debounce + all three typing-mode injector sequences (live and final-only against a faked ydotool, incl. a dead-ydotool failure-resilience pass and Live-mode mid-dictation erase/undo editing the live buffer) + gated real-model streaming and batch pipeline tasks + engine lifecycle (idle no-ops, double-start guard, drain order), CLI --asr value parsing (incl. repeated-flag last-wins) + typing-mode flag parsing + models-dir resolution precedence) + tests/cli.rs binary-level checks (wrong-rate/missing WAV rejected before any model load + CLI exit-code contract: misuse exits 2 incl. a missing/empty --asr value; repeated --asr flags are last-wins, not an error, --help exits 0, empty --paste-test exits 0 headlessly)
cargo fmt --check            # verify rustfmt-clean (run `cargo fmt` to fix)
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
