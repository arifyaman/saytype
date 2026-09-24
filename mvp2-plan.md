# MVP2 plan: live (real-time) typing via continuous streaming

## Current status (update as work progresses - this is the resume point)

- Spike: done 2026-09-03 (see results below).
- Phase 1: **done 2026-09-03, deployed** (daemon + HUD, both verified).
  `stream_task` in daemon.rs (VAD + one continuous `StreamingSession`),
  `AsrOutput` (Partial|Final) channel, `injector_task` relays partials as
  `EngineEvent::PartialTranscribed` -> D-Bus `PartialTranscribed` signal.
  Batch (Moonshine) pipeline kept intact and verified. `--asr
  auto|streaming|moonshine` accepted by the daemon and all offline
  subcommands; `auto` prefers streaming. HUD pill shows committed finals
  + live partial tail. AGENTS.md updated. Verified: cargo test,
  `--stream-test` seam run, daemon toggle start/stop on both backends,
  D-Bus introspection (3 signals), shell restarted with extension
  reloaded (no JS errors).
- Phase 2: **done 2026-09-03, deployed** (daemon; no HUD change). Live
  typing in `injector_task`: `LiveTyping` state machine types the
  `STABILITY_PARTIALS` (2) -stable prefix of partials into the target app,
  corrects the tail with `backspaces()` + retype, and each `Final`
  converges the buffer to the committed text then resets. `backspaces(n)`
  + pure `diff(typed, new)` in injector.rs (unit-tested). `--no-live-typing`
  flag restores finals-only (MVP1) typing. Live typing is ON by default
   for the streaming backend. Verified: 14 unit tests (diff, stability
   buffer, transform), session start/stop with live typing on,
   `--stream-test` regression.
- Punctuation/casing: **done 2026-09-03, deployed.** `asr.rs`
  auto-detects a `sherpa-onnx-online-punct-*` dir (7 MB int8 CNN-BiLSTM,
  ~2 ms/sentence) and, for the streaming backends, runs raw output through
  `polish()` (lowercase + strip existing punctuation + `OnlinePunctuation`),
  restoring casing + punctuation. Moonshine output is untouched.
- **Backend upgrade to Nemotron**: **done 2026-09-03, deployed.** Web
  research (2026 local ASR landscape: Nemotron, Parakeet, Qwen3-ASR,
  Whisper) found NVIDIA **Nemotron Speech Streaming EN 0.6B** (Jan 2026) - a
  truly streaming FastConformer+RNNT model trained on ~530k h (vs
  Zipformer's ~1k h LibriSpeech), 6.93% WER while streaming, native
  casing/punctuation, and it ships pre-converted for sherpa-onnx
  (encoder/decoder/joiner + tokens, `model_type = "nemo_transducer"`).
  Verified it loads with our exact crate (sherpa-onnx 1.13.6) and streams
  live partials: RTF ~0.1 on this CPU (4 threads; 8 is the sweet spot, 16
  regresses), 632 MB int8. Now the **default** (`auto` and `--asr streaming`
  prefer it); `--asr zipformer`/`moonshine` force the others. Output
  policy: Zipformer always polished; Nemotron polished only when the punct
  model is present (else native output kept); strip-before-punct prevents
  double punctuation. `scripts/download-models.sh` fetches the 560 ms chunk
  package (80/160/1120 ms variants also exist - 160 ms = snappier first
  word, 1120 ms = best accuracy). Verified: `--stream-test` on all three
  backends, `--transcribe` on all three, seam run (final == last partial,
  no double commas), session start/stop (backend: Nemotron), 14 unit tests.
- Mid-dictation word erase/undo (mouse buttons): **done 2026-09-24**
  (final step for the running install: redeploy binary + extension and
  restart the user service / shell). While a session is live, a LEFT
  mouse press on the HUD overlay erases the last visible word from both
  the typed buffer and the HUD; a RIGHT press restores the last erased
  word. Both are LIFO, so multiple presses erase/undo one word at a time.
  A newly transcribed word makes pending erasures permanent: the erased
  word is gone for good. `src/transcript.rs` is the pure state machine:
  committed text is a token-slot sequence (gaps for permanent erasures,
  hidden slots for pending ones) so a restore returns a word to its
  original position, and partial erasures track token positions across
  decoder revisions. The injector task owns the transcript and converges
  the typed buffer after every partial/final/erase/undo; the D-Bus
  surface gained `EraseWord()` / `UndoErase()` (no-op when idle) and the
  authoritative `TranscriptUpdated(String)` signal, which the HUD renders
  verbatim (the recording overlay is the mouse input surface). Verified:
  cargo test (43 tests: 25 transcript, 10 live-typing incl. 4
  post-erase stability cases, 5 injector, 3 vad), clippy clean, isolated
  D-Bus smoke test under `dbus-run-session` (the installed daemon holds
  the real bus name), `node --check` + GNOME-46 API-surface verification
  of the extension (no live shell available in this environment).
- Phase 3 (next): stability N / tick tuning, stats logging (corrections
  per segment, speech-to-first-word latency), docs.
- Outstanding: E2E with real voice (needs the user) - toggle, speak,
  watch words land live in a focused app with occasional corrections,
  HUD tail, journal, and LEFT/RIGHT mouse erase/undo while speaking.
  A WAV-injection E2E was deliberately not attempted (risk of
  disturbing the user's finicky mic/default-input setup).
- Env notes: repo is on a FUSE mount that reports every file executable -
  commit content only, never mode changes; `git -c safe.directory=...`
  needed (dubious ownership). ydotoold runs as user service
  `ydotool-user.service` (virtual device /dev/input/event22). The daemon
  can silently lose its D-Bus name (zbus 4 does not reconnect) - check
  `busctl --user list | grep saytype`, restart the unit if missing.
  E2E speech test needs the user's voice; until then verify with
  `--stream-test` + daemon toggle start/stop + `gdbus monitor`.

Goal: while the user is still talking, the text appears in the focused
application (terminal, browser text-area, ...) and settles in real time.
Today nothing appears in the target app until the speaker pauses (>= 0.8 s),
which makes dictation feel dead.

This replaces the MVP's deliberate "one typed chunk per detected pause"
(granularity decision in `gnome-dictation-mvp-guide.md`).

## Architecture: one continuous stream per session, VAD commits boundaries

The first draft of this plan kept the MVP1 pipeline shape: wait for a VAD
segment to finalize, then run a per-segment streaming session (500 ms
pre-feed, `last_fed_index` bookkeeping, post-pad sliced from the ring at
finish). MVP2 instead makes the whole dictation session **one continuous
stream**:

- A single `OnlineStream` lives for the entire session and consumes every
  audio frame - speech, pauses, everything. No per-segment session setup,
  no pre-feed, no `last_fed_index`, no ring slicing.
- The Silero VAD no longer supplies audio to ASR. It only supplies **commit
  boundaries**: a finalized VAD segment (0.8 s of trailing silence) means
  the utterance is done, so the stream finalizes (`input_finished` +
  drained decode), the final text is emitted, and the stream `reset()`s and
  keeps consuming.
- The 30 s `AudioRing` and the `PRE_PAD_MS`/`POST_PAD_MS` re-padding exist
  only because the batch path must reconstitute context the VAD trimmed
  away. The streaming decoder has never lost context, so the padding
  machinery is retired from the streaming pipeline. It stays on the
  batch/Moonshine path, which keeps working as today.
- "Chunk processing" (one offline transcribe per pause) survives only as
  the Moonshine batch backend. The streaming backend never batches.

Why this is better than per-segment sessions:

- Every sample reaches ASR exactly once, in order, as it arrives. There is
  no double-feed/missed-sample bookkeeping.
- The decoder sees the true attack of the first word (it has been
  consuming silence since session start) and the true tail (the 0.8 s
  pause is decoded live), so the first/last-word clipping that the padding
  solved goes away structurally.
- Commit boundaries come from the VAD tuning we already have
  (`min_silence_duration`). Pauses under 0.8 s (breaths inside a sentence)
  do not commit, which is the desired granularity. The decoder's own
  endpointing (`is_endpoint`) stays disabled; VAD is the only boundary
  source.
- Stream state can never interleave two utterances.

## Why it is feasible (verified against sherpa-onnx 1.13.6)

- `OnlineStream` accepts waveform chunks indefinitely. `get_result()`
  mid-stream returns the current partial hypothesis; after
  `input_finished()` and draining `is_ready()`/`decode()`, `get_result()`
  returns the final text for the utterance;
  `OnlineRecognizer::reset(&stream)` returns the stream to a clean state
  for the next utterance. This is the standard endpoint workflow of the
  crate, and all of it is present in the Rust bindings.
- VAD semantics, checked in `voice-activity-detector.cc`:
  `detected()` is `start_ != -1`, i.e. true from speech onset until the
  full `min_silence_duration` (0.8 s) has elapsed; the segment is queued at
  exactly that point and `detected()` drops. Finalization and the
  `detected()` drop happen inside the same `accept_waveform` call, so a
  task that feeds a frame and then drains `take_segment()` sees the commit
  boundary on the same tick.
- ydotool sends arbitrary key events in addition to `ydotool type`, which
  is all "correction" requires. Exact backspace syntax is verified in the
  spike.
- Prerequisite for live streaming: the **streaming Zipformer backend** must
  be active. Moonshine is an offline model and can never produce partials;
  it remains the batch backend (see Backend selection).

## What the user will see

The target app shows the sentence growing while speaking; the last word or
two occasionally changes (backspace + retype) as the decoder revises its
partial. When the 0.8 s pause ends an utterance, the text is committed and
settled. The saytype HUD pill shows the **accumulating session
transcript**: committed segments plus the live partial tail (partials
replace the tail, finals append it).

Caveats accepted for MVP2: visible corrections, no focus-drift protection.
(Casing + punctuation now restored by the online punct model - see status
above - so the old "no punctuation / `capitalize_first` stand-in" caveat is
gone for the streaming backend.)

## Design

### 1. ASR (asr.rs)

- `Asr::new` gains a backend selection parameter (`auto` | `streaming` |
  `zipformer` | `moonshine`). `auto` prefers a Nemotron streaming dir (the
  mvp2 default), then a streaming Zipformer dir, then Moonshine; `streaming`
  picks the best streaming backend (Nemotron over Zipformer); an explicit
  choice errors clearly when its model dir is missing. The daemon default is
  `auto`.
- `Asr` exposes its kind (`Nemotron` | `Zipformer` | `Moonshine`) and
  `is_streaming()` so the engine wires the matching pipeline and gates live
  typing on the streaming backends.
- Streaming backend: a thin per-session stream wrapper around the shared
  `OnlineRecognizer`:
  - `new()` - fresh `OnlineStream`.
  - `feed(samples)` - `accept_waveform`, then decode while `is_ready`.
  - `partial() -> String` - mid-stream `get_result`.
  - `commit() -> String` - `input_finished()`, drain decode,
    `get_result()`, then `reset()`.
- The batch `transcribe()` stays for `--transcribe` (both backends,
  unchanged).

### 2. Pipeline (daemon.rs)

**Streaming mode** (zipformer backend): one task replaces `vad_task` +
`asr_task`.

    audio thread -> frames -> stream_task -> asr_out channel -> injector_task

`stream_task` owns the `Vad` and the stream session:

    session start: stream = asr.streaming_session()
    for each frame:
        vad.feed(frame)
        stream.feed(frame)            // every sample, exactly once
        if let Some(seg) = vad.take_segment():   // 0.8 s pause committed
            if seg.len() >= MIN_BLIP_SAMPLES:
                text = stream.commit()
                if !text.is_empty(): send(AsrOutput::Final(text))
            // blip: drop, stream already reset inside commit
        if vad.detected() and tick(150 ms) and text changed:
            send(AsrOutput::Partial(stream.partial()))
    channel closed (session stop): vad.flush(), drain take_segment(),
    commit + send final, exit

- `AsrOutput` is a new enum channel (`Partial(String)` | `Final(String)`),
  replacing the `segs` + `texts` channels in streaming mode.
- `vad_task`/`asr_task`/`AudioRing`/padding stay as-is for **batch mode**
  (Moonshine backend); the engine wires whichever pipeline matches the
  active backend at session start.
- Decode cost: the streaming decode is incremental (one 32 ms chunk at a
  time) and runs inline in `stream_task`. If the spike shows decode can
  fall behind real time on 4 threads, move the feed+decode hop to
  `spawn_blocking` (decide then; do not preempt it).
- Stop path keeps today's drain order: capture stops, `stream_task`
  flushes its trailing utterance (finals), `injector_task` drains, only
  then `StateChanged("Idle")`.

### 3. Partial emission and live typing (injector.rs)

- Cadence: fixed 150 ms tick, emit only when the text actually changed
  (avoids no-op D-Bus signals; simplest correct rule).
- New engine event `PartialTranscribed(String)` and matching D-Bus signal;
  `SegmentTranscribed` stays the committed-final event. The injector task
  is the single producer of both (it emits a signal only after the text
  has been processed), preserving today's invariant that signals reflect
  what was typed and that finals precede `StateChanged("Idle")`.
- **Live typing (phase 2)**: the injector keeps `typed: String` - exactly
  what it has put in the target buffer for the current segment. For each
  new text (stable prefix of a partial, or the final):
  - `c` = length of the longest common prefix of `typed` and `new`.
  - If `c == typed.len()`, type `new[c..]` (pure extension, the common
    case). Otherwise send `typed.len() - c` BackSpaces, then type
    `new[c..]`. Set `typed = new`.
  - On `Final`, the same diff runs against the final text, so the buffer
    always converges to the committed transcription (the final can differ
    from the last partial - the decoder commits with trailing context).
    Then `typed` resets to empty.
  - `capitalize_first` applies to the first partial of a segment (first
    char capitalized once, kept by the diff); the space-before-chunk rule
    applies to the first partial of every segment after the first.
- **Stability buffer** (mitigates correction churn): type only the prefix
  that is common to the last N partials (default N = 2, ~300 ms); the
  unstable tail waits. Finals ignore stability and always converge. N and
  the tick are constants, tuned in phase 3.
- `injector.rs` gains `backspaces(n)` (ydotool key event; exact syntax and
  per-key latency verified in the spike) and a pure `diff(typed, new) ->
  (backspaces, to_type)` function with unit tests.
- **Batch mode** (Moonshine) keeps the current injector behavior exactly:
  one `type_text` per finalized segment, no partials, no backspaces.

### 4. D-Bus

- Add signal `PartialTranscribed(String)` to `io.saytype.Dictate1`. No
  method changes. The daemon and the extension deploy separately, so land
  them together when the signal does.
- `AGENTS.md` IPC section and the extension's interface XML get the new
  signal.

### 5. HUD (extension/saytype@saytype.local/extension.js)

- Keep `_committed` (finals appended, space-joined) and `_partial`
  (current segment tail). Pill text = `_committed` + (if `_partial`)
  " " + `_partial`.
- `PartialTranscribed` sets `_partial` (replaces the tail);
  `SegmentTranscribed` appends to `_committed` and clears `_partial`.
- `StateChanged("Recording")` resets both; `StateChanged("Idle")` keeps
  the last transcript visible (unchanged behavior).

### 6. Backend selection and models

- All ASR models can coexist in `models/`. `auto` now prefers the Nemotron
  streaming dir (the mvp2 default - see status), then Zipformer, then
  Moonshine; `--asr streaming` picks the best streaming backend; `--asr
  zipformer`/`moonshine` force a specific backend (Moonshine = MVP1 batch
  behavior, no live typing). The systemd unit stays on the default.
- The Zipformer emits all-caps, unpunctuated text; the optional
  `sherpa-onnx-online-punct-*` model (auto-detected) lowercases + strips +
  restores casing/punctuation on the streaming path. Nemotron is natively
  cased/punctuated and is polished only when that model is present (denser
  punctuation; the strip step prevents double-marking). Moonshine output is
  never post-processed.
- `--transcribe` and `--vad-test` keep using the auto selection;
  `--transcribe` on a streaming backend still works (batch feed of the whole
  file through the online recognizer, then the same output policy).

### 7. Offline streaming check (main.rs)

- New `--stream-test <wav>` (16 kHz mono): replays a file through the same
  stream logic headlessly - feed in 32 ms chunks, print each changed
  partial with its timestamp, print committed finals at VAD boundaries,
  print total committed text at the end. This is the regression harness
  for the streaming path (what `--transcribe` is for the batch path) and
  the tool for watching partial evolution and seam behavior without a
  live mic.

## Non-goals (stay out of MVP2)

- Spoken voice commands ("period", "new line"). (Lightweight punctuation +
  casing restoration is now in, via the online punct model - see status.)
- Focus-change safety (corrections scatter if focus moves mid-sentence).
- IBus input-method architecture with native pre-edit text (endgame
  option; much larger project).
- Live partials on the Moonshine backend (impossible; offline model).
- Multi-language, model switching at runtime, packaging.

## Risks and mitigations

- **Backspace churn / visual flicker** - streaming partials revise the
  tail frequently (and the punct model re-places commas/periods as context
  grows, adding its own revisions). Mitigation: stability buffer (type
  only the prefix stable across N ~300 ms partials); make live typing
  toggleable (`--no-live-typing`, falls back to typing only finals =
  phase-1 behavior) so the HUD-partial experience is always available
  without corrections in the target app.
- **Autocomplete interference** (terminal completion, browser typeahead,
  editor auto-indent) reacting to rapid type/backspace. Document as a
  known wart; plain terminals and most text-areas are fine.
- **Focus drift** while speaking - out of scope; document "keep the
  target window focused".
- **Continuous decode cost while idle** - resolved by the spike: RTF
  0.036-0.042 on 4 threads, so decoding idle silence costs ~4% of one
  core. No idle gate is built. (If a much weaker machine shows a
  problem later, the fallback is gating feed on `vad.detected()` with a
  ~500 ms pre-roll buffer covering the onset.)
- **Seam artifacts** - `commit()` + `reset()` boundaries could in theory
  drop or duplicate a word at segment edges (the final can re-derive the
  tail differently from the last partial). `--stream-test` on recorded
  utterances with natural pauses is the check; the final-convergence diff
  guarantees the typed buffer always ends on the committed text.
- **ydotool backspace semantics** (key syntax, per-key latency, whether
  repeated key events need spacing) - resolved in the spike before
  building on it.
- **Partial latency** - the first word should appear well under 0.5 s
  after onset; measured in the spike, targeted in phase 3.

## Spike results (done 2026-09-03)

The spike shipped as `--stream-test` (it doubles as the permanent
regression harness) plus direct ydotool/uinput checks:

- **ydotool backspaces**: `ydotool key --repeat N Backspace --delay 0
  --repeat-delay <ms>` produced exactly N press/release pairs (EV_KEY
  code 14) on the uinput event node; per-key spacing ~0.1 ms at the
  uinput level with `--repeat-delay 0`. The default 100 ms lead-in delay
  must be zeroed (`--delay 0`). A running `ydotoold` is required; here it
  runs as the user service `ydotool-user.service` (the apt 0.1.8 daemon
  works, it just had no unit in this session).
- **Decode cost**: RTF 0.036-0.042 on 4 CPU threads (feed + commit
  time over 17-25 s of audio). Continuous decode while idle is
  negligible; no idle gate is needed and decode runs inline in the task
  (no `spawn_blocking`).
- **Partial evolution**: words appear in order, ~one per 0.3 s on the
  bundled recordings; on clean studio audio the final always equals the
  last partial (no rewrites). First word appears ~0.1-0.4 s after
  speech onset. Real speech will show tail corrections - that is what
  the stability buffer is for.
- **Seam**: two concatenated utterances through one session
  (commit + reset) gave two exact commits with no duplicated or dropped
  words at the boundary.
- **Tail**: the decoder needs ~0.8 s of trailing silence to commit the
  last word. A file ending abruptly loses it ("IN HE" instead of "IN
  HEAVEN"); with 0.8 s of appended silence the final matches the
  reference exactly. In live operation this silence is exactly the VAD
  pause (`min_silence_duration`) streamed before the commit, so the
  property holds structurally - no padding code needed.

## Phases

1. **Spike** - done (see results above).
2. **Phase 1 - streaming pipeline + live partials in the HUD:**
   `stream_task`, `AsrOutput`, `PartialTranscribed` signal, HUD tail,
   backend selection, `--stream-test`. Target-app typing unchanged
   (committed chunk per pause, no live typing). Shippable on its own: the
   user gets visible progress and a working streaming backend.
3. **Phase 2 - live typing:** injector diff + backspaces + stability
   buffer + final convergence + `--no-live-typing`. End-to-end: speak ->
   text lands word-by-word with corrections.
4. **Phase 3 - polish:** stability N and cadence tuning, stats logging
   (corrections/segment, speech-to-first-partial latency, decode RTF),
   docs (AGENTS.md pipeline + gotchas, `--help`).

## Verification

- `cargo test` green (new: diff/backspaces unit tests, existing
  `AudioRing`/`capitalize_first` stay for the batch path).
- `--transcribe` and `--vad-test` unchanged (regression).
- `--stream-test` on recorded utterances: partials appear incrementally,
  committed finals match the `--transcribe` of the same file closely,
  no duplicated or dropped words at segment boundaries.
- Manual E2E: speak a paragraph with natural pauses into (a) a terminal
  prompt, (b) a browser text-area. Check: words appear while talking,
  corrections settle, committed text matches the offline transcription of
  the same utterance (record alongside with `pw-record`), HUD pill shows
  committed + live tail.
- `journalctl --user -u saytype -f` shows partial cadence, commit
  boundaries, correction counts, and finalization per segment.

## Open questions (decide during implementation)

- ~~Idle-decode gate~~ - resolved: not needed (spike RTF 0.036-0.042).
- Whether `PartialTranscribed` should be emitted also in batch mode with
  the final text as the single "partial" (keeps HUD code paths uniform) -
  default: no, batch mode emits finals only.
- Stability buffer default N (2 vs 3) and tick (150 vs 200 ms) - set in
  phase 3 from measured correction rates.
- `--no-live-typing` as flag vs env vs config - flag is fine, matches the
  existing CLI style.
