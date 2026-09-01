# MVP2 plan: live (real-time) typing

Goal: while the user is still talking, the text appears in the focused
application (terminal, browser text-area, ...) and updates/corrects in real
time. Today nothing appears in the target app until the speaker pauses
(>= 0.8 s), which makes dictation feel dead.

This replaces the MVP's deliberate "one typed chunk per detected pause"
(granularity decision in `gnome-dictation-mvp-guide.md`, "live word-by-word
typing" listed as next iteration).

## Why this is feasible

- The Zipformer model is a **streaming** model: `OnlineStream` accepts
  waveform chunks and `get_result()` mid-stream returns the current partial
  hypothesis. No model or crate change needed (verified against
  sherpa-onnx 1.13.6).
- `VoiceActivityDetector::detected()` reports a segment in progress, so the
  pipeline knows when speech is live without waiting for the pause.
- The typing layer (ydotool) can already send arbitrary key events, including
  `BackSpace`, which is all that "correction" requires.
- Prerequisite: the **streaming Zipformer backend** must be active. A
  Moonshine v2 model dir in `models/` is preferred by `asr.rs` and is
  batch-only, so for mvp2 it must be removed (or backend selection added).

## What the user will see

The target app shows the sentence growing while speaking; the last word or two
occasionally changes (backspace + retype) as the model revises its partial.
The final text is settled when the pause ends the segment. This matches
Dragon-style live dictation. The saytype HUD pill also shows the live
partial.

Caveats accepted for MVP2: visible corrections, no punctuation (MVP's
`capitalize_first` stand-in stays), no focus-drift protection.

## Design

### 1. Streaming ASR session (asr.rs)

Add a per-segment streaming session on top of the shared `OnlineRecognizer`:

- `start()` - create a fresh `OnlineStream`.
- `feed(samples: &[f32])` - `accept_waveform`, then decode while
  `is_ready(&stream)`.
- `partial() -> String` - `get_result` on the live stream (mid-stream =
  partial hypothesis).
- `finish() -> String` - `input_finished()`, final decode, `get_result`
  (final text).

Keep the existing batch `transcribe()` for `--transcribe` (it can be
reimplemented as start/feed/finish on the same pieces).

### 2. Pipeline restructure (daemon.rs)

Today: `vad_task` waits for a finalized VAD segment, then sends the padded
segment to a batch `asr_task`.

MVP2: the ASR stream for a segment is alive **during** the speech.

- `vad_task` keeps pushing frames into the `AudioRing` and the VAD.
- When `vad.detected()` flips true (segment in progress), start an ASR
  session and pre-feed it `ring.slice(now - 500 ms, now)` to cover the
  pre-roll (500 ms > the 300 ms pre-pad we want; a little extra leading
  silence is harmless).
- Track `last_fed_index` (absolute sample index). On every subsequent frame
  batch feed `ring.slice(last_fed_index, now)` - no double-feeding, no gaps.
- When the segment finalizes (`take_segment()` -> `(samples, start)`), feed
  up to `start + len + POST_PAD_MS` (the ring always has it: the pause is
  800 ms and the ring holds 30 s), then `finish()`.
- Segments never overlap with the current VAD config, so one live session at
  a time; if a finalize is pending when a new `detected()` arrives, finish
  the old one first (defensive, should not happen).

The batch `asr_task`/`segs` channel is replaced by this single
streaming-ASR task (vad_task and asr work can be merged or kept as two tasks
sharing the ring - decide during implementation; keep the ring owned by the
VAD side).

### 3. Partial emission + live typing

- The ASR task emits a partial every ~150-200 ms (cadence: decide between
  fixed interval vs "only when text changed" - see Open questions).
- New engine event: `PartialTranscribed(String)` -> new D-Bus signal of the
  same name. `SegmentTranscribed` stays as the final.
- **Injector prefix-diff** (the core of live typing). The injector task
  keeps `typed: String` - exactly what it has put in the target buffer for
  the current segment. For each new text `new` (partial or final):
  - `c` = length of the longest common prefix of `typed` and `new`.
  - If `c == typed.len()` -> just type `new[c..]` (pure extension, the
    common case).
  - Otherwise -> send `typed.len() - c` BackSpaces, then type `new[c..]`.
  - Set `typed = new`.
  - Reset `typed` to empty at segment start; the space-before-chunk rule and
    `capitalize_first` apply to the first partial of a segment (first char
    capitalized once, kept by the diff).
- `injector.rs` gains `backspaces(n: u32)` (ydotool key event for BackSpace;
  verify the exact key code/syntax in the spike - x11 keycode vs evdev code).
- Finalization: `finish()` text goes through the same diff, so the target
  buffer always converges to the final transcription (the final can differ
  from the last partial - the decoder commits with trailing context).

### 4. HUD (extension/saytype@saytype.local/extension.js)

The HUD is the GNOME Shell extension (the GTK popup is gone).
- Subscribe to `PartialTranscribed` and update the pill text.
- On `SegmentTranscribed` refresh with the final (optionally flash).
- The pill shows the **accumulating session transcript** (decided; partials
  replace the current-segment tail, finals append it).

### 5. D-Bus

- Add signal `PartialTranscribed(String)` to `io.saytype.Dictate1`.
  No method changes. The daemon and the extension deploy separately, so
  update both together when the signal lands.

## Non-goals (stay out of MVP2)

- Punctuation restoration / voice commands ("period", "new line").
- Focus-change safety (corrections scatter if focus moves mid-sentence).
- IBus input-method architecture with native pre-edit text (endgame option;
  much larger project).
- Multi-language, model switching, packaging.

## Risks and mitigations

- **Backspace churn / visual flicker** - streaming partials revise the tail
  frequently. Mitigation: stability buffer - only type a suffix that has been
  unchanged across N consecutive partials (~300 ms); tune N. Make live typing
  toggleable (`--no-live-typing` or env flag) to fall back to MVP1 behavior.
- **Autocomplete interference** (terminal completion, browser typeahead,
  editor auto-indent) reacting to rapid type/backspace. Document as a known
  wart; plain terminals and most text-areas are fine.
- **Focus drift** while speaking - out of scope; document "keep the target
  window focused".
- **Decode cost per partial** - streaming decode is incremental and cheap on
  4 CPU threads; measure in the spike anyway.
- **ydotool backspace semantics** (key code, timing, per-key latency) -
  resolve in the spike before building on it.

## Phases

1. **Spike (small):** confirm `ydotool` can send N BackSpaces reliably;
   measure per-partial decode latency; hand-drive a live ASR session from a
   recording to see partial evolution (what the corrections will look like).
2. **Phase 1 - live partials in the HUD only:** streaming ASR session,
    `PartialTranscribed` signal, live pill text. Target-app typing unchanged
    (final chunk at pause). Shippable on its own: user gets visible progress.
3. **Phase 2 - live typing:** injector prefix-diff + backspaces +
   finalization. End-to-end: speak -> text lands word-by-word.
4. **Phase 3 - polish:** stability buffer + cadence tuning, stats logging
   (corrections/segment, speech-to-first-word latency), fallback flag.

## Verification

- `cargo test` + `--transcribe` + `--vad-test` remain green (regression).
- Manual E2E: speak a paragraph with natural pauses into (a) a terminal
  prompt, (b) a browser text-area. Check: words appear while talking,
  corrections settle, final text matches the offline `--transcribe` of the
  same utterance (record alongside with `pw-record`).
- `journalctl --user -u saytype -f` shows partials, correction counts, and
  finalization per segment.

## Open questions (decide during implementation)

- Partial cadence: fixed 150-200 ms interval vs only-when-changed vs token
  boundary. (Fixed interval is simplest; only-when-changed avoids no-op
  D-Bus signals.)
- Whether pre-roll should be the fixed 500 ms or exactly `PRE_PAD_MS` plus a
  small margin (needs the in-progress segment's start index, which the crate
  does not expose - the fixed window is the practical choice).
