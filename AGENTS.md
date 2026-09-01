# AGENTS.md

Quick orientation for coding agents (and humans) working on saytype.
Spec lives in `gnome-dictation-mvp-guide.md`; the next iteration is planned in `mvp2-plan.md`.

## What this is

saytype is a background speech-to-text dictation daemon for GNOME. Toggle it
with a hotkey, speak naturally, and the words get typed into whatever window
has keyboard focus. There is no per-app integration: text is injected as
synthesized key events via ydotool. A GNOME Shell extension draws a
focus-free HUD pill (recording state + transcript) while dictating.

Pipeline per dictation session:

    PipeWire mic capture -> Silero VAD -> sherpa-onnx ASR (Moonshine v2 or streaming Zipformer, EN) -> ydotool typing

## Processes and IPC

- `saytype` - the daemon. Runs as a systemd **user** service (`saytype.service`).
  Owns the audio pipeline, models, and the D-Bus service. It is the only
  saytype process; the daemon no longer spawns a UI.
- The HUD is a **GNOME Shell extension** (`extension/saytype@saytype.local/`).
  It runs inside gnome-shell (gjs), not in our process: a plain D-Bus client
  with no audio and no models. It draws a pill below the top panel while
  recording; being shell-stage artwork it never steals focus and behaves
  identically on X11 and Wayland. Installed with
  `scripts/install-extension.sh`.
- IPC is the session D-Bus bus:
  - Bus name `io.saytype.Dictate`, path `/io/saytype/Dictate`, interface `io.saytype.Dictate1`
  - Methods: `Toggle()`, `Stop()`
  - Signals: `StateChanged(String)`, `SegmentTranscribed(String)`

## File map

| File | What to find there |
|---|---|
| `src/main.rs` | CLI dispatch (daemon / `--transcribe <wav>` / `--vad-test <wav>`), models-dir resolution, logging |
| `src/daemon.rs` | D-Bus service, `Engine` state machine (Idle/Recording), per-session pipeline tasks (`vad_task`, `asr_task`, `injector_task`) |
| `src/audio.rs` | PipeWire capture on a dedicated OS thread; S16LE 16 kHz mono in, f32 frames out via mpsc |
| `src/vad.rs` | Silero VAD wrapper, `VadParams` defaults, `AudioRing` context-padding buffer + unit tests |
| `src/asr.rs` | `Asr` dual backend (Moonshine `OfflineRecognizer` preferred, streaming Zipformer `OnlineRecognizer` fallback), model auto-detection, batch `transcribe()` |
| `src/injector.rs` | ydotool typing (`type_text`), `capitalize_first` MVP punctuation stand-in + test |
| `extension/saytype@saytype.local/` | GNOME Shell extension HUD (gjs, ESM-first): D-Bus proxy client, top-center pill (inline St styles, emoji mic), `metadata.json` |
| `models/` | `silero_vad.onnx` + ASR model dirs (Moonshine v2 preferred, streaming zipformer fallback; gitignored; `scripts/download-models.sh`) |
| `systemd/saytype.service` | Unit template with `@REPO@`/`@BIN@` placeholders |
| `scripts/setup-ydotool.sh` | One-time: apt ydotool, udev rule for `/dev/uinput`, `ydotoold` user service |
| `scripts/download-models.sh` | Downloads the VAD + Zipformer models into `models/` |
| `scripts/install-user-service.sh` | Release build, install binary + `saytype-toggle` to `~/.local/bin`, install/enable service |
| `scripts/install-extension.sh` | Copy the HUD extension into `~/.local/share/gnome-shell/extensions/` + enable it (`--uninstall` to reverse) |
| `scripts/saytype-toggle` | `dbus-send` Toggle wrapper; bind this to a GNOME custom shortcut |
| `gnome-dictation-mvp-guide.md` | Original MVP spec + decisions |
| `mvp2-plan.md` | Next-iteration plan (live/real-time typing) |

## Pipeline data flow (daemon.rs)

One task per stage, chained with tokio mpsc channels, spawned per session:

    audio thread -> frames -> vad_task -> segs -> asr_task -> texts -> injector_task -> ydotool

State/segment events flow back over an unbounded channel into `run()`, which
emits the D-Bus signals. The injector is a single consumer, so ydotool calls
are strictly serialized. On stop the daemon drains the pipeline (final
`SegmentTranscribed`) **before** emitting `StateChanged("Idle")`; consumers
may rely on that order.

## Gotchas (read before touching code)

1. **zbus methods do not run on tokio.** `#[interface]` methods are dispatched
   on zbus's internal executor; any tokio API (`spawn`, `spawn_blocking`,
   `time::`) panics there. Forward a command to the engine loop task instead
   (see `Dictate::toggle` in daemon.rs).
2. **PipeWire is not thread-safe.** The whole capture stack (mainloop, context,
   core, stream) lives on one dedicated OS thread in `audio.rs`. `pod_bytes`
   from the FFI `pw_stream_connect` must outlive the stream.
3. **VAD trims segments to the detected speech boundaries**, so the first
   word's attack and the last word's tail get clipped. Segments are re-padded
   from the `AudioRing` (raw audio, absolute sample indexes): 300 ms pre /
   800 ms post (`PRE_PAD_MS`/`POST_PAD_MS` in vad.rs). Do not "simplify" the
   ring away.
4. **ASR is batch-per-segment in v1 (no live partials).** Zipformer path:
   fresh `OnlineStream`, feed the whole padded segment, `input_finished()`,
   decode loop, `get_result()`, `enable_endpoint = false`, greedy search.
   Moonshine path: fresh offline stream, feed the segment, single
   `decode()`, `get_result()`. Both: CPU provider, thread count from config.
5. **Models are auto-detected, not hardcoded.** `asr.rs` scans `models/` and
   prefers a **Moonshine v2** dir (`encoder_model.*` + `decoder_model_merged.*`
   as `.onnx`/`.ort` + `tokens.txt`, 16 kHz input) over a
   `sherpa-onnx-streaming-zipformer-*` dir (encoder/decoder/joiner
   `*epoch*.onnx` + `tokens.txt`, non-int8 variants preferred). Both can
   coexist; remove one dir to switch. Only the zipformer backend can produce
   live partials (mvp2); Moonshine is batch-per-segment but outputs proper
   casing and punctuation.
6. **The HUD is a GNOME Shell extension, not a process.** It runs inside
   gnome-shell (gjs). Keep it small and defensive: a JS fault there can take
   the whole shell down. The shell does **not** hot-load changed user
   extensions - after installing or changing the JS, restart the shell (log
   out/in, or Alt+F2 `r` on X11; the restart is in-process, so the shell PID
   stays the same). The extension only consumes the D-Bus signals; do not add
   side channels.
7. **GNOME 46's ESM GJS bindings expose a reduced, differently-named API
   surface.** Do not copy API calls from generic GJS docs or pre-45
   extensions; verify each against the shell's own code
   (`strings /usr/lib/gnome-shell/libshell-14.so`) or a known-good 46
   extension (gnome-shell-extensions repo, speech2text-extension). Hard-won
   specifics:
   - The St typelib (`/usr/lib/gnome-shell/St-14.typelib`) is **curated**:
     `St.Label` has only `text`/`clutter_text` (no `set_ellipsize` /
     `set_max_width_chars` - use `label.clutter_text.ellipsize = ...` plus CSS
     `max-width`); no `add_actor` (use `add_child`); no `set_visible` /
     `get_visible` (use `hide()` / `show()`); no `new_animation` with
     `autoreverse` / `loop` (use `.ease({...})` + a repeating
     `GLib.timeout_add` returning `true`).
   - GI keeps C snake_case for statics (`Gio.DBusConnection.get_default()`,
     not `getDefault()`).
   - D-Bus: use `Gio.DBusProxy.makeProxyWrapper(XML)` +
     `new Proxy(Gio.DBus.session, busName, objectPath)` +
     `proxy.connectSignal(name, (p, sender, [args]) => ...)` +
     `proxy.MethodAsync()`. Raw `signal_subscribe` takes a different arg
     order than the C API in this build (sender, interface, member, path,
     rule, flags, cb).
   - `Main.panel` is a JS class: its C-base signals can't be connected from
     extensions (use `global.display.connect('workareas-changed', ...)`);
     reading `Main.panel.height` is fine.
   - Chrome registration: `Main.layoutManager.addTopChrome(actor)` /
     `Main.layoutManager.removeChrome(actor)`; source removal via
     `GLib.Source.remove(id)`. Inline `style:` on St actors works.
8. **Build env**: `.cargo/config.toml` sets `LIBCLANG_PATH` and
   `BINDGEN_EXTRA_CLANG_ARGS` (bindgen needs them for the pipewire/sherpa-onnx
   crates). Non-interactive shells don't have `~/.cargo/bin` on PATH.
9. **Audio source matters.** The system default input must be the real
   microphone (on this machine: UMC202HD, mic on the left channel). A wrong
   default (e.g. a webcam's IEC958 line) yields garbage. Check with
   `wpctl status` / set with `wpctl set-default <node>`.

## Build, run, test

```sh
cargo build --release        # release binary at target/release/saytype
cargo test                   # unit tests (AudioRing, capitalize_first)
cargo run                    # run the daemon manually (needs models/ + session bus)

scripts/install-user-service.sh   # build + install + enable the user service
scripts/install-extension.sh      # install + enable the HUD extension
journalctl --user -u saytype -f   # live daemon logs
~/.local/bin/saytype-toggle       # what the GNOME hotkey calls

# HUD extension: syntax-check, redeploy, then restart the shell to reload
node --check extension/saytype@saytype.local/extension.js
scripts/install-extension.sh      # Alt+F2 -> r (X11) or log out/in afterwards

# watch the D-Bus signals directly (handy when the journal is noisy)
gdbus monitor --session --dest io.saytype.Dictate --object-path /io/saytype/Dictate

# offline model checks (16 kHz mono WAV, e.g. models/*/test_wavs/)
saytype --transcribe <wav>
saytype --vad-test <wav>
```

End-to-end check: toggle, speak a sentence, verify the text lands in the
focused app, the HUD pill shows the transcript, and `journalctl` shows
`VAD segment complete` + `transcribed ...`.

## Conventions

- Rust 2021, `anyhow` for error handling, `tracing` for logging
  (`RUST_LOG=saytype=info` in the unit).
- Keep the doc comments on non-obvious invariants (several are hard-won; the
  `audio.rs`/`vad.rs`/`daemon.rs` headers explain why things are the way they are).
- The daemon (systemd unit) and the HUD extension (gnome-extensions) deploy
  separately. Keep the D-Bus API in sync between them when changing either
  side.
