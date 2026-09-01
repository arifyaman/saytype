# GNOME Speech-to-Text Dictation Daemon — MVP Implementation Guide

## 0. Purpose & scope of this document

This is an engineer-facing guide to build a **functional MVP** of a background speech-to-text dictation tool for GNOME on Linux (CPU-only ASR, English only). It specifies architecture, interfaces, module boundaries, exact commands, and acceptance criteria for each phase — it deliberately does **not** contain filled-in source code. The intent is that an engineer can implement each phase directly from this spec, phase by phase, with each phase independently testable before moving to the next.

### Locked-in decisions for this MVP

| Decision | Choice | Why |
|---|---|---|
| Language / stack | Rust — `gtk4`/`libadwaita`, `zbus`, `pipewire`, `sherpa-onnx` (official crate), `tokio` | Concurrency-heavy always-on daemon; native bindings exist for every component |
| ASR model | sherpa-onnx streaming Zipformer2, English (`csukuangfj/sherpa-onnx-zipformer-small-en-2023-06-26`, or `-large-en-2023-06-26` for higher accuracy) | Mature, CPU-friendly, prebuilt, well-documented |
| Text injector | `ydotool` (via `/dev/uinput`) | Simplest to get working locally, no sandboxing constraints since this isn't packaged yet |
| Trigger | GNOME Settings custom keyboard shortcut → D-Bus call into the daemon | Simpler than validating the `GlobalShortcuts` portal on day one |
| Distribution | `systemd --user` service, run from a personal repo, no packaging | Matches "run manually for now" |
| Typing granularity | **One typed chunk per detected pause** (VAD-segment level), not word-by-word | Avoids the partial-hypothesis-revision problem entirely for v1 |

Everything above is a deliberate MVP simplification. Section 12 lists what's explicitly deferred.

---

## 1. System overview

```
Hotkey (GNOME custom shortcut)
        │  D-Bus: Toggle()
        ▼
Background daemon (systemd --user service, state: Idle / Recording)
        │                              │
        ▼                              ▼
Audio capture (PipeWire,        UI popup (GTK4/libadwaita,
16kHz mono PCM)                 shown on Recording, hidden on Idle)
        │
        ▼
VAD (sherpa-onnx Silero VAD) — detects utterance boundaries
        │  on segment complete
        ▼
ASR (sherpa-onnx streaming Zipformer2) — transcribes the finalized segment
        │  finalized text
        ▼
Injector (ydotool, serialized queue) — types into whatever has focus
```

The UI's live label and the injector both consume the same "segment transcribed" event, but only the injector's output is irreversible — the UI can always redraw freely.

---

## 2. Target environment & assumptions

- Ubuntu 24.04+ / Debian 12+ (or equivalent), GNOME 45+
- Wayland session (default on modern GNOME). If your session turns out to be Xorg, note it during Phase 0 — the injector story is actually simpler there (XTest), but this guide assumes Wayland throughout.
- Rust via `rustup`, stable channel
- PipeWire already running (default on these distros)
- `ydotool` + `ydotoold` installed separately (not a Rust dependency)

---

## 3. Project structure

```
gnome-dictate/
├── Cargo.toml
├── src/
│   ├── main.rs        # entrypoint: starts tokio runtime, wires modules together
│   ├── daemon.rs       # state machine + D-Bus service implementation
│   ├── audio.rs        # PipeWire capture stream, runs on its own thread
│   ├── vad.rs           # wraps sherpa-onnx Silero VAD, emits segment boundaries
│   ├── asr.rs            # wraps sherpa-onnx streaming recognizer
│   ├── injector.rs        # ydotool wrapper + serialized typing queue
│   └── ui.rs                # GTK4/libadwaita window + widgets
├── models/                   # downloaded sherpa-onnx model files (gitignored)
├── scripts/
│   ├── setup-ydotool.sh       # installs/configures ydotoold + permissions
│   └── install-user-service.sh
└── systemd/
    └── gnome-dictate.service
```

A single binary crate is enough for MVP — don't split into a workspace yet.

---

## 4. Dependencies

| Crate | Purpose | Notes |
|---|---|---|
| `sherpa-onnx` | Streaming Zipformer2 ASR **and** Silero VAD | Official bindings; downloads a prebuilt native lib automatically unless `SHERPA_ONNX_LIB_DIR` is set |
| `zbus` | D-Bus service (daemon) + client (UI, trigger) | Async, pairs naturally with `tokio` |
| `gtk4` | UI toolkit bindings | Via the `gtk-rs` project |
| `libadwaita` | Adwaita widgets/styling | Same family as `gtk4` |
| `pipewire` | Microphone capture | Runs its own mainloop — see Phase 2 note |
| `tokio` | Async runtime | Ties daemon state machine, D-Bus, and channels together |

No crate is needed for `ydotool` itself — shell out to the installed binary via `std::process::Command` for the MVP; that's simpler than talking to its socket protocol directly.

---

## 5. Phase 0 — Environment spike tests (do this before writing app code)

Each of these should work standalone, outside your own code, before you build on top of it.

**0.1 — ydotool round-trip**
```bash
sudo apt install ydotool   # or build from source
ydotoold &                 # or install as its own systemd service
# ensure your user can reach /dev/uinput (input group or udev rule)
ydotool type "hello world"
```
Focus a text editor first — confirm the exact string appears, including capitalization and punctuation.

**0.2 — PipeWire capture**
```bash
pw-record --channels 1 --rate 16000 test.wav
# speak for a few seconds, Ctrl+C, then:
paplay test.wav   # or any player
```
Confirm clean audio at correct volume with no dropouts.

**0.3 — sherpa-onnx model + CLI validation**
```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-small-en-2023-06-26.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-small-en-2023-06-26.tar.bz2
```
Check the current release page for the exact current filename — model artifacts get reorganized over time. Then, using a prebuilt `sherpa-onnx-microphone` binary (from a sherpa-onnx release, or built from source):
```bash
./sherpa-onnx-microphone \
  --encoder=./sherpa-onnx-streaming-zipformer-small-en-2023-06-26/encoder-epoch-99-avg-1.onnx \
  --decoder=./sherpa-onnx-streaming-zipformer-small-en-2023-06-26/decoder-epoch-99-avg-1.onnx \
  --joiner=./sherpa-onnx-streaming-zipformer-small-en-2023-06-26/joiner-epoch-99-avg-1.onnx \
  --tokens=./sherpa-onnx-streaming-zipformer-small-en-2023-06-26/tokens.txt \
  --model-type=zipformer2
```
Speak into the mic and confirm live text output with acceptable accuracy and latency on your actual CPU. This is your ground-truth check before any Rust code touches the model — if accuracy or speed feels wrong here, it's a model/config problem, not a code problem.

**0.4 — Hotkey reaches a command reliably**

In GNOME Settings → Keyboard → Custom Shortcuts, bind a key combo to something trivial first, e.g. `notify-send "hotkey fired"`. Confirm it fires reliably before wiring it to your daemon.

---

## 6. Phase 1 — Daemon skeleton (trigger → state machine)

**D-Bus interface spec**

| | |
|---|---|
| Bus name | `com.<you>.Dictate` (session bus) |
| Object path | `/com/<you>/Dictate` |
| Interface | `com.<you>.Dictate1` |
| Method | `Toggle() -> ()` |
| Method | `Stop() -> ()` |
| Signal | `StateChanged(state: String)` — `"Idle"` \| `"Recording"` |
| Signal | `SegmentTranscribed(text: String)` — added once ASR is wired up in Phase 3, but reserve it now |

**State machine**: two states, `Idle` and `Recording`.
- `Toggle()` while `Idle` → start audio capture pipeline, emit `StateChanged("Recording")`.
- `Toggle()` while `Recording` → stop pipeline (flush any in-flight segment through VAD/ASR/injector first), emit `StateChanged("Idle")`.

**systemd user service** (`~/.config/systemd/user/gnome-dictate.service`): standard `[Unit]`/`[Service]`/`[Install]` shape — `ExecStart` pointing at your built binary, `Type=simple`. Enable with:
```bash
systemctl --user enable --now gnome-dictate.service
journalctl --user -u gnome-dictate -f   # for live logs while testing
```

**Update the GNOME custom shortcut** from Phase 0.4 to call your daemon instead, e.g. via `dbus-send --session --dest=com.<you>.Dictate --type=method_call /com/<you>/Dictate com.<you>.Dictate1.Toggle` (or a tiny dedicated helper binary if you'd rather not depend on `dbus-send` being present).

**Phase done when**: pressing the hotkey twice toggles state and you can watch the transitions in `journalctl`. No audio touches the system yet.

---

## 7. Phase 2 — Audio capture + VAD

- PipeWire stream format: 16-bit mono PCM at 16kHz — this is what the ASR model expects, so capture at this format directly rather than resampling later.
- **Integration note**: PipeWire's mainloop is callback-driven, not async-native. Run it on its own dedicated OS thread and forward captured frames into your `tokio` runtime via an `mpsc` channel — don't try to drive PipeWire's loop from inside `tokio::spawn`.
- VAD: use `sherpa-onnx`'s built-in `VoiceActivityDetector` (Silero VAD) — feed it the same 16kHz mono frames. It reports segment boundaries, so you don't need to hand-roll energy-based silence detection.
- Data flow: PipeWire thread → channel → VAD consumer task → on "segment complete," hand the buffered audio segment onward to Phase 3.

**Phase done when**: for a few test utterances with natural pauses, logged segment boundaries look sane (each segment roughly matches one spoken sentence, typically 1–4 seconds).

---

## 8. Phase 3 — ASR integration

- Use the `sherpa-onnx` crate's recognizer against the model files downloaded in Phase 0.3.
- Because you're transcribing once per finalized VAD segment (not continuously streaming), you can run the recognizer in a simple "feed the whole segment, get the final text back" mode — no need to wire up incremental partial-hypothesis polling for MVP.
- Start with the **small** English model for lower CPU load and faster iteration. The **large** variant is a drop-in swap later (same file shape, same code path) if accuracy needs improving.

**Phase done when**: for a handful of test segments, transcribed text is reasonably accurate, and per-segment latency (segment-end to text-ready) is logged and sits within a few hundred milliseconds on your CPU.

---

## 9. Phase 4 — Text injection

- Wrap `ydotool type "<text>"` via `std::process::Command`, one call per finalized, transcribed segment.
- **Serialize strictly**: route finalized segments through a single-consumer `mpsc` channel so `ydotool` calls never overlap, even if segment N+1 finishes transcribing before segment N finishes typing.
- For MVP, join segments with a space and capitalize the first letter of each — a cheap stand-in for real punctuation, explicitly not real punctuation restoration (that's a post-MVP item).

**Phase done when**: dictating 3–4 consecutive sentences into a plain text editor, with natural pauses between them, produces correctly ordered text with no drops, duplicates, or interleaving.

---

## 10. Phase 5 — Minimal UI

- Create the GTK4/libadwaita window once at startup; keep it alive but hidden (withdrawn, not destroyed) between uses, so toggling is instant.
- Widgets: one record/stop button, one label showing the most recently transcribed segment (purely informational — updating this label never affects what's typed).
- The UI is a D-Bus client of the daemon: it calls the same `Toggle()` method the hotkey uses, and subscribes to `StateChanged` and `SegmentTranscribed` to update its own visibility and label.

**Phase done when**: the full loop works — press hotkey, window appears, speak a sentence, see it in the label *and* typed into the previously focused window, press hotkey again, window hides.

---

## 11. End-to-end MVP acceptance checklist

- [ ] Daemon starts on login via `systemd --user`, sits near 0% CPU while idle
- [ ] Hotkey reliably toggles recording from a cold start
- [ ] A sentence spoken with a natural pause types into the previously focused window within ~1s of the pause
- [ ] Multiple consecutive sentences type in the correct order
- [ ] Toggling off mid-utterance doesn't crash the daemon or leave it stuck
- [ ] Re-triggering after a full stop works repeatedly without restarting the service

---

## 12. Explicitly out of scope for this MVP

- Live word-by-word typing (the stability-buffer approach discussed separately) — next iteration
- Real punctuation restoration / spoken voice commands ("period", "new line", "stop dictation")
- `RemoteDesktop` / `GlobalShortcuts` portals — using the ydotool + custom-shortcut path instead
- Focus-change safety checks (typing into the wrong window if focus shifts mid-dictation)
- Pluggable ASR backend interface / swapping in Moonshine
- Non-English support
- Any packaging (Flatpak, `.deb`)

---

## 13. Troubleshooting notes

- **`ydotool` "permission denied" on `/dev/uinput`**: confirm your user is in the `input` group (or the udev rule is applied) and that you've re-logged-in since the group change.
- **`ydotoold` socket not found from inside the systemd service**: environment variables like `YDOTOOL_SOCKET` set in your interactive shell are *not* automatically inherited by a `systemd --user` unit — set them explicitly in the service file's `Environment=` if you rely on a non-default socket path.
- **Custom shortcut doesn't reach the daemon**: check whether your session is actually Wayland or Xorg (`echo $XDG_SESSION_TYPE`) — this affects nothing here directly, but is worth knowing if other pieces behave unexpectedly later.
- **PipeWire capture silent or wrong device**: use `pw-cli ls Node` or `wpctl status` to confirm which device is default and that it's actually your microphone.

---

## 14. Reference material

- sherpa-onnx pretrained models: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
- sherpa-onnx model releases: https://github.com/k2-fsa/sherpa-onnx/releases
- Official Rust API docs: https://docs.rs/sherpa-onnx
- ydotool: https://github.com/ReimuNotMoe/ydotool
- gtk4-rs book: https://gtk-rs.org/gtk4-rs/stable/latest/book/
- zbus docs: https://docs.rs/zbus
