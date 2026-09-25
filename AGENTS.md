# AGENTS.md

Quick orientation for coding agents (and humans) working on saytype.

## What this is

saytype is a background speech-to-text dictation daemon for GNOME. Toggle it
with a hotkey, speak naturally, and the words get typed into whatever window
has keyboard focus. There is no per-app integration: text is injected as
synthesized key events via ydotool. A GNOME Shell extension draws a
focus-free HUD pill (recording state + transcript) while dictating.

Pipeline per dictation session:

    PipeWire mic capture -> Silero VAD -> sherpa-onnx ASR (streaming Zipformer, EN; Moonshine v2 batch fallback) -> ydotool typing

## Processes and IPC

- `saytype` - the daemon. Runs as a systemd **user** service (`saytype.service`).
  Owns the audio pipeline, models, and the D-Bus service. It is the only
  saytype process; the daemon no longer spawns a UI.
- The HUD is a **GNOME Shell extension** (`extension/saytype@saytype.local/`).
  It runs inside gnome-shell (gjs), not in our process: a plain D-Bus client
  with no audio and no models. While recording it draws a dim overlay across
  all monitors plus a pill that follows the mouse pointer (polled every
  120ms, so it is visible on any screen in a multi-monitor setup - this is
  the original MVP1 design, restored after a detour through a fixed
  bottom-anchored redesign that turned out to fix nothing real; see the
  file header) and renders the daemon's authoritative `TranscriptUpdated`
  verbatim. The pill is sized explicitly, both dimensions, on every text
  update (`_resizePill`) rather than trusting Clutter's automatic parent
  relayout, which does not reliably happen for this actor in this build
  (root-caused live with targeted diagnostics - see the file header):
  it starts narrow for a short phrase (or the idle "Listening..." label)
  and grows horizontally up to a cap - half the width of whichever
  monitor it is currently showing on, re-evaluated live from the pointer
  position on every update via `_currentMonitor()`, not fixed to wherever
  it first appeared - before it starts wrapping onto more lines, so a
  full sentence always fits, never truncated; being shell-stage artwork it never steals keyboard
  focus and behaves identically on X11 and Wayland. While recording, the Left and
  Right arrow keys are grabbed as global keybindings (GSettings schema
  `org.gnome.shell.extensions.saytype`, keys `erase-word-keybinding` /
  `undo-erase-keybinding`) for mid-dictation word erase/undo (Left erases,
  Right restores -> `EraseWord`/`UndoErase`); the grab is added on
  `StateChanged("Recording")` and removed on `StateChanged("Idle")`, so the
  arrow keys behave normally (text cursor movement, etc.) the rest of the
  time. No mouse input is used. Start/stop stays the custom hotkey.
  Installed with `scripts/install-extension.sh` (also compiles the
  extension's GSettings schema).
- IPC is the session D-Bus bus:
  - Bus name `io.saytype.Dictate`, path `/io/saytype/Dictate`, interface `io.saytype.Dictate1`
  - Methods: `Toggle()`, `Stop()`, `EraseWord()`, `UndoErase()`
  - `Toggle()` debounces: a repeat call within 350ms of the last accepted
    one is ignored. GNOME custom shortcuts fire again on X11 key
    auto-repeat if the hotkey is held even slightly too long - each repeat
    spawns a new `saytype-toggle` process/D-Bus call, and without this
    guard a burst of repeats flips Recording/Idle rapidly and can land
    back on the wrong state (looked like "the HUD doesn't close"; also
    corrupted the HUD's transcript, reset on every state flip). A
    deliberate double-toggle by a human is essentially never this fast.
  - Signals: `StateChanged(String)`, `PartialTranscribed(String)`,
    `SegmentTranscribed(String)`, `TranscriptUpdated(String)`
  - `PartialTranscribed` is a live hypothesis for the utterance in progress
    (streaming backend only); `SegmentTranscribed` is the committed final.
  - `EraseWord()`/`UndoErase()` (mid-dictation, driven by the extension's
    Left/Right arrow-key global keybindings, active only while recording)
    pop/restore the last visible word of the running session (no-op when
    idle). `TranscriptUpdated(String)` carries the full visible transcript
    (committed finals + live partial, erasures applied, casing/punctuation
    restored); the HUD renders it verbatim and the daemon re-sends it after
    every change (partial, final, erase, undo).

## File map

| File | What to find there |
|---|---|
| `src/main.rs` | CLI dispatch (daemon / `--transcribe <wav>` / `--vad-test <wav>` / `--stream-test <wav>`, all take `--asr`), models-dir resolution, logging |
| `src/daemon.rs` | D-Bus service (`Toggle`/`Stop`/`EraseWord`/`UndoErase` + signals), `Engine` state machine (Idle/Recording), per-session pipeline: `stream_task` (streaming) or `vad_task`+`asr_task` (batch), `injector_task` (owns the session's `Transcript`, applies erase/undo, emits `TranscriptUpdated`); `EngineEvent`/`AsrOutput`/`InjectorInput` |
| `src/audio.rs` | PipeWire capture on a dedicated OS thread; S16LE 16 kHz mono in, f32 frames out via mpsc; pure conversion helpers `pcm_to_mono_f32` / `resample_linear` + unit tests |
| `src/vad.rs` | Silero VAD wrapper, `detected()` in-progress probe, `VadParams` defaults, `AudioRing` context-padding buffer (batch path) + unit tests |
| `src/asr.rs` | `Asr` tri-backend (Nemotron streaming `OnlineRecognizer` preferred, Zipformer `OnlineRecognizer` fallback, Moonshine `OfflineRecognizer` batch), `BackendSelection`/`AsrKind`, `StreamingSession` (feed/partial/commit), batch `transcribe()`, `polish()` (lowercase + strip punct + online punct) output policy, optional `OnlinePunctuation` model auto-detect, + unit tests for the model-dir detection helpers (`find_moonshine`/`find_nemotron`/`resolve_zipformer_paths`/`find_online_punct` + filename matching/preference rules) and `polish()` |
| `src/injector.rs` | ydotool typing (`type_text`, `backspaces`), `diff` prefix-diff, `capitalize_first` MVP punctuation stand-in + tests |
| `src/transcript.rs` | Pure `Transcript` state machine: single source of truth for the visible text (committed finals + live partial) with mid-dictation word erase/undo. Committed text is a token-slot sequence (permanently erased words are gaps, pending erasures hide a slot) so a restored word returns to its original position; partial erasures track token positions across decoder revisions and stay restorable for the whole utterance in progress regardless of how much the live hypothesis grows tick to tick, becoming permanent only at a genuine utterance boundary (this utterance's final showing content no partial ever did, or the next utterance's first partial); first visible word per segment is capitalized at render time; `display()` (HUD) + `buffer_target()` (target buffer) + unit tests |
| `extension/saytype@saytype.local/` | GNOME Shell extension HUD (gjs, ESM-first): D-Bus proxy client, pointer-following word-wrapping pill + dim overlay (inline St styles, symbolic mic icon) rendering the authoritative `TranscriptUpdated` in full (no truncation); while recording, global Left/Right arrow-key keybindings (added/removed with the session) drive erase/undo (`EraseWord`/`UndoErase`); `metadata.json`, `schemas/org.gnome.shell.extensions.saytype.gschema.xml` |
| `models/` | `silero_vad.onnx` + ASR model dirs (Nemotron streaming preferred, Zipformer + Moonshine v2 fallbacks) + optional online punct dir; gitignored; default install has only the Nemotron stack |
| `systemd/saytype.service` | Unit template with `@REPO@`/`@BIN@` placeholders |
| `scripts/setup-ydotool.sh` | One-time: apt ydotool, udev rule for `/dev/uinput`, `ydotoold` user service |
| `scripts/download-models.sh` | Downloads VAD + Nemotron streaming + online punct into `models/` (default); `--zipformer` / `--moonshine` / `--all` fetch the other backends |
| `scripts/install-user-service.sh` | Release build, install binary + `saytype-toggle` to `~/.local/bin`, install/enable service |
| `scripts/install-extension.sh` | Copy the HUD extension into `~/.local/share/gnome-shell/extensions/` + enable it (`--uninstall` to reverse) |
| `scripts/saytype-toggle` | `dbus-send` Toggle wrapper; bind this to a GNOME custom shortcut |

## Pipeline data flow (daemon.rs)

One task per stage, chained with tokio mpsc channels, spawned per session.
Two shapes depending on the active backend:

Streaming (Nemotron EN 0.6B 560 ms by default, Zipformer fallback):

    audio thread -> frames -> stream_task (VAD + one continuous OnlineStream)
                 -> AsrOutput (Partial | Final) -> injector_task -> ydotool

The VAD only supplies **commit boundaries**: a finalized segment (0.8 s
pause) finalizes the current utterance (`commit()`), and live partials are
emitted on a 150 ms tick while `vad.detected()`.

**Casing/punctuation** (streaming path): every partial/final is passed
through `polish()` - lowercased, existing punctuation stripped, then run
through the optional online punct model (`OnlinePunctuation`, auto-detected
in `models/`) which restores casing + punctuation. The Zipformer emits
all-caps and needs this always; Nemotron is natively cased/punctuated and
is only polished when the punct model is present (denser punctuation, and
the strip step prevents double-marking), otherwise its native output is
kept. Moonshine (batch) output is never post-processed.

**Typing mode** (`TypingMode`, default `Deferred`): controls *when* the
injector types the transcript into the target app; the HUD always shows
the live transcript immediately regardless of mode via `TranscriptUpdated`.

- `Deferred` (default, no flag): nothing is typed while the session is
  recording. The injector only tracks the `Transcript` and updates the
  HUD; the whole accumulated transcript (with any mid-dictation erase/undo
  already applied) is typed into the target app exactly **once**, right
  after the session stops. This is what makes erase/undo safe to use: they
  edit the pending HUD-only text, never a live buffer in some other app.
- `Live` (`--live-typing`, the old mvp2 default): the injector types a
  partial's prefix into the target app once it has survived
  `STABILITY_PARTIALS` (2) consecutive partials, corrects the tail with
  `backspaces` + retype as the decoder revises it, and each `Final`
  converges the buffer to the committed text (prefix-diff via
  `injector::diff`) then resets. Erase/undo also edit this live-typed
  buffer immediately (backspaces into whatever real app has focus).
- `FinalOnly` (`--no-live-typing`, the MVP1 behavior): only committed
  finals are typed, immediately, one chunk per utterance; no partial
  typing. Erase/undo still edit the live-typed buffer immediately, same as
  `Live`.

Batch (Moonshine):

    audio thread -> frames -> vad_task -> segs -> asr_task -> AsrOutput -> injector_task -> ydotool

State/partial/segment events flow back over an unbounded channel into
`run()`, which emits the D-Bus signals. The injector is a single consumer,
so ydotool calls are strictly serialized. On stop the daemon drains the
pipeline (final `SegmentTranscribed`) **before** emitting
`StateChanged("Idle")`; consumers may rely on that order.

## Gotchas (read before touching code)

1. **zbus methods do not run on tokio.** `#[interface]` methods are dispatched
   on zbus's internal executor; any tokio API (`spawn`, `spawn_blocking`,
   `time::`) panics there. Forward a command to the engine loop task instead
   (see `Dictate::toggle` in daemon.rs).
2. **PipeWire is not thread-safe.** The whole capture stack (mainloop, context,
   core, stream) lives on one dedicated OS thread in `audio.rs`. `pod_bytes`
   from the FFI `pw_stream_connect` must outlive the stream.
3. **VAD trims segments to the detected speech boundaries**, so the first
   word's attack and the last word's tail get clipped. **Batch (Moonshine)
   path only**: segments are re-padded from the `AudioRing` (raw audio,
   absolute sample indexes): 300 ms pre / 800 ms post
   (`PRE_PAD_MS`/`POST_PAD_MS` in vad.rs). Do not "simplify" the ring away
   while the batch path exists. The **streaming path does not use the
   ring**: the `OnlineStream` sees every sample live, and the 0.8 s pause
   that finalizes a VAD segment is exactly the trailing silence the decoder
   needs to commit the last word, so no padding is required.
4. **Streaming ASR is one continuous session per dictation session**
   (`stream_task` in daemon.rs, `StreamingSession` in asr.rs). The
   `OnlineStream` is created once, fed every frame in order, `partial()`
   mid-stream returns the live hypothesis, and at each VAD commit boundary
   `commit()` runs `input_finished()`, a decode loop, `get_result()`, then
   `reset()` for the next utterance. `enable_endpoint = false` - VAD
   finalization is the only commit boundary. `OnlineRecognizer`/`OnlineStream`
   are `Send + Sync` (single-object C library), so the session lives inside
   the tokio task. The **Moonshine path** is still batch-per-segment: fresh
   offline stream, feed the padded segment, single `decode()`,
   `get_result()`. Both: CPU provider, thread count from config.
 5. **Models are auto-detected, not hardcoded.** `asr.rs` scans `models/` and
    (for `auto`) prefers a **Nemotron streaming** dir
    (`sherpa-onnx-nemotron-speech-streaming-en-*`: encoder/decoder/joiner
    `*.onnx` + `tokens.txt`, int8 variants preferred; loaded with
    `model_type = "nemo_transducer"`) over a **streaming Zipformer** dir
    (`sherpa-onnx-streaming-zipformer-*`: encoder/decoder/joiner
    `*epoch*.onnx` + `tokens.txt`, non-int8 variants preferred) over a
    **Moonshine v2** dir (`encoder_model.*` + `decoder_model_merged.*` as
    `.onnx`/`.ort` + `tokens.txt`, 16 kHz input). All can coexist; `--asr
    streaming|zipformer|moonshine` forces a backend. Both streaming backends
    produce live partials (mvp2); Moonshine is batch-per-segment. A separate
    **online punct** dir (`sherpa-onnx-online-punct-*`:
    `model.int8.onnx`/`model.onnx` + `bpe.vocab`) is auto-detected and, when
    present, restores casing + punctuation on the streaming path (the
    Zipformer emits all-caps and needs it always; Nemotron is only polished
    for denser punctuation). It is optional - without it, Zipformer text is
    lowercased but unpunctuated, and Nemotron text keeps its native
    casing/punctuation.
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
    microphone. A wrong default (e.g. a webcam's IEC958 line or a virtual
    device) yields garbage - some mics also only deliver audio on one
    channel, which is fine as long as it is the mic's real channel. Check
    with `wpctl status` / set with `wpctl set-default <node>`.

## Build, run, test

```sh
cargo build --release        # release binary at target/release/saytype
cargo test                   # unit tests (transcript, injector, audio format conversion, VAD AudioRing, ASR model detection + polish, daemon stable-target logic, config)
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
saytype --transcribe <wav>            # batch check (auto backend)
saytype --vad-test <wav>
saytype --stream-test <wav>           # streaming replay: partials + finals + RTF
saytype --paste-test "<text>"         # injector check: clipboard set + ydotool paste, timed
```

End-to-end check: toggle, speak a sentence, verify the HUD pill shows the
accumulating transcript **word-by-word while speaking**, wrapped in full
(never cropped to "..."), toggle off, and confirm the whole corrected
sentence lands in the focused app **in one paste** (not typed out
character by character) and the HUD closes promptly. `journalctl` should
show `VAD segment complete` + `streaming commit: ...` while recording and
a fast `dictation session stopping` -> `stopped` gap (well under a second)
on stop.

## Conventions

- Rust 2021, `anyhow` for error handling, `tracing` for logging
  (`RUST_LOG=saytype=info` in the unit).
- Keep the doc comments on non-obvious invariants (several are hard-won; the
  `audio.rs`/`vad.rs`/`daemon.rs` headers explain why things are the way they are).
- The daemon (systemd unit) and the HUD extension (gnome-extensions) deploy
  separately. Keep the D-Bus API in sync between them when changing either
  side.
