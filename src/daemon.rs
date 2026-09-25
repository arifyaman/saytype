//! D-Bus service and dictation engine.
//!
//! State machine: Idle <-> Recording.
//!
//! While Recording, one of two pipelines runs depending on the active ASR
//! backend:
//!
//! - **Streaming** (Zipformer, mvp2 default):
//!   PipeWire thread -> frames -> stream task (VAD + one continuous
//!   OnlineStream) -> AsrOutput channel -> injector task (single consumer,
//!   ydotool). The VAD only supplies commit boundaries: a finalized segment
//!   (0.8 s pause) finalizes the current utterance; live partials are
//!   emitted on a fixed tick.
//! - **Batch** (Moonshine):
//!   PipeWire thread -> frames -> VAD task (ring-padded segments) -> ASR
//!   task -> AsrOutput channel -> injector task. No partials.
//!
//! The injector owns the session's `Transcript` (see `transcript.rs`), the
//! single source of truth for what the target buffer holds and what the HUD
//! shows. Mid-dictation word erase/undo arrives as `EraseWord()` / `UndoErase()`
//! D-Bus calls (mouse buttons), is forwarded through the engine to the
//! injector, and both the typed buffer and the `TranscriptUpdated` signal
//! follow the transcript.
//!
//! All state/signal transitions are reported to D-Bus through `EngineEvent`s,
//! pumped by the daemon task that owns the `SignalContext`.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::sync::{mpsc, oneshot};
use zbus::{interface, SignalContext};

use crate::asr::{Asr, AsrKind, BackendSelection};
use crate::audio::{self, CaptureHandle};
use crate::injector;
use crate::transcript::Transcript;
use crate::vad::{AudioRing, Vad, VadConfig, POST_PAD_MS, PRE_PAD_MS};

pub const BUS_NAME: &str = "io.saytype.Dictate";
pub const OBJECT_PATH: &str = "/io/saytype/Dictate";
pub const INTERFACE_NAME: &str = "io.saytype.Dictate1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineEvent {
    StateChanged(String),
    /// Live partial hypothesis for the utterance in progress (streaming
    /// backend only; replaced by the next partial, superseded by the final).
    PartialTranscribed(String),
    /// Final text for a committed utterance.
    SegmentTranscribed(String),
    /// The full visible transcript (committed finals plus the live partial,
    /// mid-dictation erasures applied) that the HUD shows verbatim.
    TranscriptUpdated(String),
}

/// Text produced by the ASR half of the pipeline, consumed by the injector.
/// `Partial` is a live hypothesis (revised by later partials and superseded
/// by the `Final`); `Final` is the committed text for one utterance.
#[derive(Debug, Clone)]
pub enum AsrOutput {
    Partial(String),
    Final(String),
}

/// What the injector consumes: ASR outputs from the pipeline plus user
/// commands (mouse-button erase/undo) forwarded by the engine. One channel
/// keeps the ordering between typed text and edits strictly serialized.
#[derive(Debug)]
enum InjectorInput {
    Asr(AsrOutput),
    /// Erase the last visible word (left mouse press while recording).
    Erase,
    /// Restore the last erased word (right mouse press while recording).
    Undo,
}

/// Commands sent from the D-Bus methods to the engine loop.
#[derive(Debug, Clone, Copy)]
pub enum EngineCmd {
    Toggle,
    Stop,
    Erase,
    Undo,
}

/// Minimum gap between two accepted `Toggle()` calls. GNOME custom
/// shortcuts fire again on X11 key auto-repeat if the hotkey is held even
/// slightly past the initial repeat delay - each repeat spawns a whole new
/// `saytype-toggle` process and D-Bus call, which without this guard could
/// flood the engine with a burst of toggles (observed: 8 calls in ~4s from
/// a single physical press) that flip Recording/Idle repeatedly and can
/// land back on the wrong state - looking like "the HUD doesn't close".
/// A human deliberately toggling twice is essentially never this fast; a
/// hardware auto-repeat burst is always this fast. Debouncing here (at the
/// D-Bus method, closest to the input source) rejects repeats before they
/// ever reach the engine queue, regardless of what is bound to the hotkey.
const TOGGLE_DEBOUNCE: Duration = Duration::from_millis(350);

pub struct Dictate {
    cmd_tx: mpsc::UnboundedSender<EngineCmd>,
    last_toggle: std::sync::Mutex<Option<std::time::Instant>>,
}

/// D-Bus interface. The methods must not do real work here: zbus dispatches
/// them on its own internal executor, which is not a tokio runtime, so any
/// tokio API (spawn, spawn_blocking, time) would panic. Instead we just
/// forward a command to the engine loop task, which runs on the tokio runtime.
#[interface(name = "io.saytype.Dictate1")]
impl Dictate {
    async fn toggle(&mut self) {
        let now = std::time::Instant::now();
        let mut last = self.last_toggle.lock().expect("last_toggle mutex poisoned");
        if let Some(prev) = *last {
            if now.duration_since(prev) < TOGGLE_DEBOUNCE {
                tracing::debug!("ignoring Toggle(): debounced (hotkey auto-repeat?)");
                return;
            }
        }
        *last = Some(now);
        drop(last);
        let _ = self.cmd_tx.send(EngineCmd::Toggle);
    }

    async fn stop(&mut self) {
        let _ = self.cmd_tx.send(EngineCmd::Stop);
    }

    /// Erase the last visible word of the running dictation session
    /// (left mouse press). No-op when idle.
    async fn erase_word(&mut self) {
        let _ = self.cmd_tx.send(EngineCmd::Erase);
    }

    /// Restore the last erased word of the running dictation session
    /// (right mouse press). No-op when idle or when nothing is pending.
    async fn undo_erase(&mut self) {
        let _ = self.cmd_tx.send(EngineCmd::Undo);
    }

    #[zbus(signal)]
    async fn state_changed(ctxt: &SignalContext<'_>, state: String) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn partial_transcribed(ctxt: &SignalContext<'_>, text: String) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn segment_transcribed(ctxt: &SignalContext<'_>, text: String) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn transcript_updated(ctxt: &SignalContext<'_>, text: String) -> zbus::Result<()>;
}

impl Dictate {
    pub fn new(cmd_tx: mpsc::UnboundedSender<EngineCmd>) -> Self {
        Self {
            cmd_tx,
            last_toggle: std::sync::Mutex::new(None),
        }
    }
}

/// Sequentially process engine commands on the tokio runtime.
async fn engine_loop(mut engine: Engine, mut cmd_rx: mpsc::UnboundedReceiver<EngineCmd>) {
    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            EngineCmd::Toggle => {
                if engine.is_recording() {
                    engine.stop().await;
                } else if let Err(e) = engine.start().await {
                    tracing::error!("failed to start dictation: {e:?}");
                }
            }
            EngineCmd::Stop => engine.stop().await,
            EngineCmd::Erase => engine.erase_last().await,
            EngineCmd::Undo => engine.undo_last().await,
        }
    }
}

pub struct Engine {
    vad_config: VadConfig,
    asr: std::sync::Arc<Asr>,
    events: mpsc::UnboundedSender<EngineEvent>,
    /// How/when transcribed text reaches the target app (default
    /// `Deferred`; see [`TypingMode`]).
    typing_mode: TypingMode,
    session: Option<Session>,
}

/// How/when transcribed text is injected into the target app during a
/// session. Independent of what the HUD shows: `TranscriptUpdated` always
/// carries the live transcript (partials, finals, erase/undo applied)
/// regardless of mode, since the HUD has its own focus-free overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypingMode {
    /// Type partials into the target app as soon as they are stable (see
    /// `STABILITY_PARTIALS`), corrected with backspaces as the decoder
    /// revises the tail; each `Final` converges the buffer to the
    /// committed text. Mid-dictation erase/undo edit the already-typed
    /// buffer live, via backspaces, in the target app - so an erase/undo
    /// touches whatever real app currently has focus while you dictate.
    Live,
    /// Type only committed finals, immediately, one chunk per utterance
    /// (no partial typing; the MVP1 behavior). Erase/undo still edit the
    /// already-typed buffer live, same as `Live`.
    FinalOnly,
    /// Type nothing into the target app while the session is recording:
    /// only the HUD overlay shows the live transcript, so speaking and
    /// correcting with erase/undo never touches whatever real app has
    /// focus. The whole accumulated (post-erase/undo) transcript is typed
    /// into the target app exactly once, when the session stops. This is
    /// the default: it is what makes erase/undo safe to use without
    /// racing a live buffer in some other application.
    Deferred,
}

struct Session {
    capture: CaptureHandle,
    exited_rx: oneshot::Receiver<()>,
    // The ASR half of the pipeline. Streaming mode is one task (VAD +
    // continuous stream); batch mode keeps the two-task VAD/ASR split.
    pipeline: PipelineTasks,
    inj_task: tokio::task::JoinHandle<()>,
    /// Into the injector, for mid-session user commands (erase/undo).
    inj_tx: mpsc::Sender<InjectorInput>,
}

enum PipelineTasks {
    Streaming {
        stream: tokio::task::JoinHandle<()>,
    },
    Batch {
        vad: tokio::task::JoinHandle<()>,
        asr: tokio::task::JoinHandle<()>,
    },
}

impl PipelineTasks {
    /// Wait for the pipeline to finish draining (bounded, like stop() did
    /// per-task before the split).
    async fn wait(self) {
        match self {
            PipelineTasks::Streaming { stream } => {
                let _ = tokio::time::timeout(Duration::from_secs(60), stream).await;
            }
            PipelineTasks::Batch { vad, asr } => {
                let _ = tokio::time::timeout(Duration::from_secs(5), vad).await;
                let _ = tokio::time::timeout(Duration::from_secs(60), asr).await;
            }
        }
    }
}

impl Engine {
    pub async fn load(
        models_dir: &Path,
        events: mpsc::UnboundedSender<EngineEvent>,
        selection: BackendSelection,
        typing_mode: TypingMode,
    ) -> Result<Self> {
        crate::vad::check_model_file(&models_dir.join("silero_vad.onnx"))?;
        crate::asr::check_models_dir(models_dir)?;

        let asr = tokio::task::spawn_blocking({
            let dir = models_dir.to_path_buf();
            move || -> Result<Asr> {
                let start = std::time::Instant::now();
                let asr = Asr::new(&dir, 4, selection)
                    .with_context(|| format!("loading ASR model from {:?}", dir))?;
                tracing::info!("ASR model loaded in {:?}", start.elapsed());
                Ok(asr)
            }
        })
        .await
        .context("ASR load task panicked")?
        .context("loading ASR model")?;

        Ok(Self {
            vad_config: VadConfig::from_models_dir(models_dir),
            asr: std::sync::Arc::new(asr),
            events,
            typing_mode,
            session: None,
        })
    }

    pub fn is_recording(&self) -> bool {
        self.session.is_some()
    }

    /// Erase the last visible word of the running session (left mouse
    /// press). No-op when idle; the injector applies it and relays the
    /// updated transcript.
    pub async fn erase_last(&self) {
        if let Some(session) = &self.session {
            let _ = session.inj_tx.send(InjectorInput::Erase).await;
        }
    }

    /// Restore the last erased word of the running session (right mouse
    /// press). No-op when idle or when nothing is pending.
    pub async fn undo_last(&self) {
        if let Some(session) = &self.session {
            let _ = session.inj_tx.send(InjectorInput::Undo).await;
        }
    }

    pub async fn start(&mut self) -> Result<()> {
        anyhow::ensure!(!self.is_recording(), "already recording");

        let (frames_tx, frames_rx) = mpsc::channel(128);
        let (out_tx, out_rx) = mpsc::channel::<InjectorInput>(16);
        let (exited_tx, exited_rx) = oneshot::channel::<()>();
        // A second handle for the engine, to forward user commands to the
        // injector while the session runs.
        let inj_tx = out_tx.clone();

        let capture = audio::start_capture(frames_tx, exited_tx);

        let pipeline = match self.asr.kind() {
            AsrKind::Nemotron | AsrKind::Zipformer => {
                let vad = Vad::new(&self.vad_config)?;
                let asr = self.asr.clone();
                let handle = tokio::spawn(stream_task(vad, asr, frames_rx, out_tx));
                PipelineTasks::Streaming { stream: handle }
            }
            AsrKind::Moonshine => {
                let (segs_tx, segs_rx) = mpsc::channel(16);
                let vad = Vad::new(&self.vad_config)?;
                // 30s of lookback: covers max segment (20s) + padding comfortably.
                let ring = AudioRing::new(30 * audio::SAMPLE_RATE as usize);
                let vad_handle = tokio::spawn(vad_task(vad, ring, frames_rx, segs_tx));
                let asr = self.asr.clone();
                let asr_handle = tokio::spawn(asr_task(asr, segs_rx, out_tx));
                PipelineTasks::Batch {
                    vad: vad_handle,
                    asr: asr_handle,
                }
            }
        };

        let events = self.events.clone();
        let streaming = self.asr.is_streaming();
        let inj_task = tokio::spawn(injector_task(out_rx, events, self.typing_mode, streaming));

        self.session = Some(Session {
            capture,
            exited_rx,
            pipeline,
            inj_task,
            inj_tx,
        });

        let _ = self
            .events
            .send(EngineEvent::StateChanged("Recording".into()));
        tracing::info!(
            "dictation session started (backend: {:?})",
            self.asr.kind()
        );
        Ok(())
    }

    pub async fn stop(&mut self) {
        let Some(session) = self.session.take() else {
            return;
        };

        tracing::info!("dictation session stopping");
        session.capture.request_stop();

        // Wait for the capture thread to exit; it drops its frames sender.
        let _ = tokio::time::timeout(Duration::from_secs(5), session.exited_rx).await;

        // Audio gone: the pipeline flushes its trailing utterance and exits.
        session.pipeline.wait().await;
        // Drop our own sender clone before awaiting the injector: it also
        // holds a clone (for mid-session erase/undo commands), so the
        // injector's receive loop only sees the channel close once both
        // the pipeline's sender (dropped above) and this one are gone.
        // Keeping it alive across the join would deadlock the drain until
        // the timeout, delaying the Idle StateChanged (and the HUD hiding)
        // by the full duration on every stop.
        drop(session.inj_tx);
        // Injector drains the remaining outputs. Every subprocess call inside
        // it (ydotool, xclip/wl-copy) is now individually bounded (see
        // `injector::SUBPROCESS_TIMEOUT`, 5s), so this is a safety net for
        // something unforeseen, not the primary bound - it used to be 30s,
        // and a hang anywhere in the injector (observed live: the one-shot
        // deferred paste) meant the HUD stayed visibly stuck on screen for
        // the whole 30s before `StateChanged("Idle")` could be sent.
        let _ = tokio::time::timeout(Duration::from_secs(12), session.inj_task).await;
        session.capture.join();

        let _ = self.events.send(EngineEvent::StateChanged("Idle".into()));
        tracing::info!("dictation session stopped");
    }
}

/// VAD consumer: feed frames, emit finalized (context-padded) segments.
async fn vad_task(
    vad: Vad,
    mut ring: AudioRing,
    mut frames_rx: mpsc::Receiver<Vec<f32>>,
    seg_tx: mpsc::Sender<Vec<f32>>,
) {
    while let Some(frame) = frames_rx.recv().await {
        ring.push(&frame);
        vad.feed(&frame);
        drain_vad(&vad, &ring, &seg_tx).await;
    }
    // Audio source ended: flush any trailing speech.
    vad.flush();
    drain_vad(&vad, &ring, &seg_tx).await;
}

async fn drain_vad(vad: &Vad, ring: &AudioRing, seg_tx: &mpsc::Sender<Vec<f32>>) {
    let pre = (PRE_PAD_MS * audio::SAMPLE_RATE / 1000) as i64;
    let post = (POST_PAD_MS * audio::SAMPLE_RATE / 1000) as i64;
    while let Some((samples, start)) = vad.take_segment() {
        // Skip trivially short blips (~100ms).
        if samples.len() < audio::SAMPLE_RATE as usize / 10 {
            continue;
        }
        let secs = samples.len() as f32 / audio::SAMPLE_RATE as f32;
        // The VAD trims segments to the detected speech boundaries, which
        // clips the first word's attack and leaves the last word no trailing
        // context. The ring holds the raw audio, so slice real context on
        // both ends: [start - pre, start + len + post).
        let begin = start - pre;
        let end = start + samples.len() as i64 + post;
        let padded = ring.slice(begin, end);
        tracing::info!(
            "VAD segment complete: {:.2}s speech, {:.2}s sent to ASR (+{PRE_PAD_MS}ms pre, +{POST_PAD_MS}ms post)",
            secs,
            padded.len() as f32 / audio::SAMPLE_RATE as f32
        );
        if seg_tx.send(padded).await.is_err() {
            return;
        }
    }
}

/// ASR consumer (batch backend): transcribe each finalized segment.
async fn asr_task(
    asr: std::sync::Arc<Asr>,
    mut seg_rx: mpsc::Receiver<Vec<f32>>,
    out_tx: mpsc::Sender<InjectorInput>,
) {
    while let Some(samples) = seg_rx.recv().await {
        let audio_secs = samples.len() as f32 / audio::SAMPLE_RATE as f32;
        let asr = asr.clone();
        let (text, elapsed) = tokio::task::spawn_blocking(move || asr.transcribe(&samples))
            .await
            .expect("ASR task panicked");
        tracing::info!(
            "transcribed {:.2}s of audio in {:?}: {:?}",
            audio_secs,
            elapsed,
            text
        );
        if text.trim().is_empty() {
            continue;
        }
        if out_tx.send(InjectorInput::Asr(AsrOutput::Final(text))).await.is_err() {
            return;
        }
    }
}

/// Cadence for emitting live partials while an utterance is in progress.
const PARTIAL_TICK_MS: u64 = 150;

/// Streaming pipeline (mvp2): one continuous ASR stream for the whole
/// session. Every frame is fed to both the VAD and the stream, in order.
/// The VAD no longer supplies audio - only commit boundaries: when a segment
/// finalizes (0.8 s of trailing silence), the stream finalizes the current
/// utterance, emits its text, and resets for the next one. Live partials are
/// emitted on a fixed tick while speech is in progress.
async fn stream_task(
    vad: Vad,
    asr: std::sync::Arc<Asr>,
    mut frames_rx: mpsc::Receiver<Vec<f32>>,
    out_tx: mpsc::Sender<InjectorInput>,
) {
    let Some(session) = asr.streaming_session() else {
        tracing::error!("stream_task started without the streaming backend");
        return;
    };
    let mut last_partial = String::new();
    let mut last_tick = Instant::now();

    while let Some(frame) = frames_rx.recv().await {
        vad.feed(&frame);
        session.feed(&frame);

        // A finalized segment means the utterance is done: finalize the
        // stream (the 0.8 s pause was already fed live, which is exactly
        // the trailing silence the decoder needs for the last word), emit,
        // and reset for the next utterance.
        while let Some((seg, _)) = vad.take_segment() {
            let secs = seg.len() as f32 / audio::SAMPLE_RATE as f32;
            let text = session.commit();
            tracing::info!(
                "VAD segment complete: {:.2}s, streaming commit: {:?}",
                secs,
                text
            );
            last_partial.clear();
            // Same blip guard as the batch path (~100 ms): the stream was
            // still committed (reset), but no text is emitted.
            if seg.len() < audio::SAMPLE_RATE as usize / 10 || text.is_empty() {
                continue;
            }
            if out_tx.send(InjectorInput::Asr(AsrOutput::Final(text))).await.is_err() {
                return;
            }
        }

        let now = Instant::now();
        if now.duration_since(last_tick) >= Duration::from_millis(PARTIAL_TICK_MS) {
            last_tick = now;
            if vad.detected() {
                let partial = session.partial();
                if !partial.is_empty() && partial != last_partial {
                    last_partial = partial.clone();
                    if out_tx
                        .send(InjectorInput::Asr(AsrOutput::Partial(partial)))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
            }
        }
    }

    // Audio source ended: flush any trailing speech, like the batch path.
    vad.flush();
    while let Some((seg, _)) = vad.take_segment() {
        let text = session.commit();
        last_partial.clear();
        if seg.len() < audio::SAMPLE_RATE as usize / 10 || text.is_empty() {
            continue;
        }
        tracing::info!("flushed trailing utterance: {:?}", text);
        if out_tx.send(InjectorInput::Asr(AsrOutput::Final(text))).await.is_err() {
            return;
        }
    }
}

/// How many consecutive partials a prefix must survive before it is typed
/// into the target app. The anti-churn buffer for live typing: decoder tails
/// are revised often, so only text stable for ~N * PARTIAL_TICK_MS is typed;
/// the unstable tail waits (and is corrected with backspaces if it was).
const STABILITY_PARTIALS: usize = 2;

/// Live-typing state: what has actually been put in the target buffer and
/// the recent partial history that decides when a prefix is stable enough
/// to type. The text itself (casing, spaces, erasures) is owned by the
/// session's [`Transcript`]; `typed` always equals the full buffer content.
struct LiveTyping {
    /// Exactly what has been put in the target buffer (transformed):
    /// the committed text plus the current utterance's stable prefix.
    typed: String,
    /// The most recent `STABILITY_PARTIALS` raw partials of the utterance
    /// in progress.
    history: std::collections::VecDeque<String>,
}

impl LiveTyping {
    fn new() -> Self {
        Self {
            typed: String::new(),
            history: std::collections::VecDeque::new(),
        }
    }

    fn push_partial(&mut self, text: &str) {
        self.history.push_back(text.to_string());
        if self.history.len() > STABILITY_PARTIALS {
            self.history.pop_front();
        }
    }

    /// The full text the target buffer should hold while the utterance is
    /// in progress: the committed text plus the longest prefix that (a) is
    /// stable across all recent partials and (b) is still visible after
    /// mid-dictation erasures, transformed by the transcript. `None` until
    /// there are enough partials for a non-empty stable prefix (the caller
    /// then skips typing rather than churn the buffer).
    fn stable_target(&self, transcript: &Transcript) -> Option<String> {
        if self.history.len() < STABILITY_PARTIALS {
            return None;
        }
        let mut it = self.history.iter();
        let mut stable = it.next()?.clone();
        for h in it {
            let c = stable
                .chars()
                .zip(h.chars())
                .take_while(|(a, b)| a == b)
                .count();
            let byte_off: usize = stable.chars().take(c).map(|ch| ch.len_utf8()).sum();
            stable.truncate(byte_off);
            if stable.is_empty() {
                return None;
            }
        }
        // An erase may have removed a word from the visible partial, so a
        // raw stable prefix can reach past the visible text: truncate to a
        // prefix of the visible partial (and drop any dangling space).
        let visible = transcript.visible_partial();
        let c = stable
            .chars()
            .zip(visible.chars())
            .take_while(|(a, b)| a == b)
            .count();
        let byte_off: usize = stable.chars().take(c).map(|ch| ch.len_utf8()).sum();
        stable.truncate(byte_off);
        let stable = stable.trim_end();
        if stable.is_empty() {
            return None;
        }
        Some(transcript.buffer_target(stable))
    }

    /// Bring the target buffer from `typed` to `target` with the minimal
    /// edit (trailing backspaces + append), and record it as typed.
    async fn apply(&mut self, target: &str) -> std::io::Result<()> {
        let (backspaces, to_type) = injector::diff(&self.typed, target);
        if backspaces > 0 {
            injector::backspaces(backspaces as u32).await?;
        }
        if !to_type.is_empty() {
            injector::type_text(to_type).await?;
        }
        self.typed = target.to_string();
        Ok(())
    }

    /// A segment committed: the stability buffer starts fresh for the next
    /// utterance. `typed` is kept: it now holds the whole committed text,
    /// the baseline the next segment's prefix extends.
    fn commit_segment(&mut self) {
        self.history.clear();
    }
}

/// Injector: single consumer, strictly serialized ydotool calls.
///
/// Owns the session's [`Transcript`], the single source of truth for what
/// the target buffer holds and what the HUD shows. `typing_mode` (see
/// [`TypingMode`]) controls *when* that text reaches the target app; the
/// HUD, via `TranscriptUpdated`, always reflects it immediately regardless
/// of mode. In `Deferred` mode (the default) nothing is typed while the
/// loop runs; the whole final transcript is typed once, after the loop
/// ends (`out_rx` closed: the session is stopping and every sender,
/// including mid-session erase/undo commands, has been dropped).
async fn injector_task(
    mut out_rx: mpsc::Receiver<InjectorInput>,
    events: mpsc::UnboundedSender<EngineEvent>,
    typing_mode: TypingMode,
    streaming: bool,
) {
    let mut live = LiveTyping::new();
    let mut transcript = Transcript::new();
    let deferred = typing_mode == TypingMode::Deferred;
    while let Some(input) = out_rx.recv().await {
        match input {
            InjectorInput::Asr(AsrOutput::Partial(text)) => {
                let _ = events.send(EngineEvent::PartialTranscribed(text.clone()));
                transcript.feed_partial(&text);
                if typing_mode == TypingMode::Live {
                    live.push_partial(&text);
                    if let Some(target) = live.stable_target(&transcript) {
                        if let Err(e) = live.apply(&target).await {
                            // A transient ydotool failure must not kill the
                            // session; the final will (re)converge the buffer.
                            tracing::error!("live typing failed: {e}");
                        }
                    }
                }
                let _ = events.send(EngineEvent::TranscriptUpdated(transcript.display()));
            }
            InjectorInput::Asr(AsrOutput::Final(text)) => {
                if streaming {
                    transcript.feed_final(&text);
                } else {
                    transcript.feed_batch_final(&text);
                }
                let event = if deferred {
                    // Nothing is typed yet; the segment is only committed
                    // to the transcript. The whole thing is typed once, at
                    // session stop.
                    text
                } else {
                    // Convergence: type whatever makes the buffer exactly
                    // the committed text (an extension in the common case;
                    // backspaces + retype when the final differs from the
                    // last partial), then reset for the next segment.
                    let target = transcript.buffer_target("");
                    let ok = live.apply(&target).await.is_ok();
                    if !ok {
                        tracing::error!("ydotool failed to type final: {text:?}");
                    }
                    if ok {
                        text
                    } else {
                        format!("[type failed] {text}")
                    }
                };
                live.commit_segment();
                let _ = events.send(EngineEvent::SegmentTranscribed(event));
                let _ = events.send(EngineEvent::TranscriptUpdated(transcript.display()));
            }
            InjectorInput::Erase => {
                if transcript.erase_last() {
                    apply_transcript_change(&mut live, &transcript, &events, typing_mode, "erased")
                        .await;
                }
            }
            InjectorInput::Undo => {
                if transcript.undo_last() {
                    apply_transcript_change(
                        &mut live, &transcript, &events, typing_mode, "restored",
                    )
                    .await;
                }
            }
        }
    }
    if deferred {
        // Session stopping: paste the whole accumulated (post-erase/undo)
        // transcript exactly once (clipboard + Ctrl+V, not a simulated
        // keystroke-per-character type: it lands instantly instead of
        // visibly "typing itself out", and never fires the target app's
        // per-keystroke handlers for text the user never watched arrive).
        // By now every utterance has committed (the pipeline flushes its
        // trailing speech before this task's senders are dropped), so
        // `display()` is pure committed text.
        let target = transcript.display();
        if let Err(e) = injector::paste_text(&target).await {
            tracing::error!("deferred paste failed: {e}");
        }
    }
}

/// After a mid-dictation erase/undo: converge the target buffer to the
/// transcript's current visible state (the stable prefix of the live
/// partial when there is one, else the committed text) and relay the
/// display to the HUD. In `Deferred` mode nothing is typed yet, so only
/// the HUD is updated.
async fn apply_transcript_change(
    live: &mut LiveTyping,
    transcript: &Transcript,
    events: &mpsc::UnboundedSender<EngineEvent>,
    typing_mode: TypingMode,
    what: &str,
) {
    if typing_mode != TypingMode::Deferred {
        let target = live
            .stable_target(transcript)
            .unwrap_or_else(|| transcript.buffer_target(""));
        if let Err(e) = live.apply(&target).await {
            tracing::error!("buffer convergence after {what} word failed: {e}");
        }
    }
    tracing::info!("{what} word, transcript: {:?}", transcript.display());
    let _ = events.send(EngineEvent::TranscriptUpdated(transcript.display()));
}

/// Start the daemon: load models, register the D-Bus service, pump events.
pub async fn run(
    models_dir: &Path,
    selection: BackendSelection,
    typing_mode: TypingMode,
) -> Result<()> {
    let (events_tx, mut events_rx) = mpsc::unbounded_channel::<EngineEvent>();
    let engine = Engine::load(models_dir, events_tx, selection, typing_mode).await?;
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<EngineCmd>();

    // The engine loop owns the Engine and runs on this tokio runtime.
    tokio::spawn(engine_loop(engine, cmd_rx));

    let conn = zbus::Connection::session()
        .await
        .context("connecting to session bus")?;
    conn.request_name(BUS_NAME)
        .await
        .context("requesting bus name")?;

    conn.object_server()
        .at(OBJECT_PATH, Dictate::new(cmd_tx))
        .await
        .context("registering D-Bus interface")?;

    let iface_ref = conn
        .object_server()
        .interface::<_, Dictate>(OBJECT_PATH)
        .await
        .context("getting interface ref for signal emission")?;
    let signal_ctx = iface_ref.signal_context().clone();

    tracing::info!("daemon ready, serving {BUS_NAME} at {OBJECT_PATH} ({INTERFACE_NAME})");

    // Pump engine events onto the bus.
    while let Some(event) = events_rx.recv().await {
        match event {
            EngineEvent::StateChanged(state) => {
                if let Err(e) = Dictate::state_changed(&signal_ctx, state).await {
                    tracing::error!("emitting StateChanged failed: {e}");
                }
            }
            EngineEvent::PartialTranscribed(text) => {
                if let Err(e) = Dictate::partial_transcribed(&signal_ctx, text).await {
                    tracing::error!("emitting PartialTranscribed failed: {e}");
                }
            }
            EngineEvent::SegmentTranscribed(text) => {
                if let Err(e) = Dictate::segment_transcribed(&signal_ctx, text).await {
                    tracing::error!("emitting SegmentTranscribed failed: {e}");
                }
            }
            EngineEvent::TranscriptUpdated(text) => {
                if let Err(e) = Dictate::transcript_updated(&signal_ctx, text).await {
                    tracing::error!("emitting TranscriptUpdated failed: {e}");
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{make_fake_tool, model_lock, repo_models_dir, EnvPatch};

    /// Push the same partial into both the transcript and the stability
    /// buffer, as the injector does on each tick.
    fn tick(live: &mut LiveTyping, t: &mut Transcript, text: &str) {
        t.feed_partial(text);
        live.push_partial(text);
    }

    #[test]
    fn first_segment_target_capitalized_without_leading_space() {
        let mut live = LiveTyping::new();
        let mut t = Transcript::new();
        tick(&mut live, &mut t, "hello world");
        tick(&mut live, &mut t, "hello world");
        assert_eq!(live.stable_target(&t), Some("Hello world".to_string()));
    }

    #[test]
    fn later_segments_get_a_space_prefix() {
        let mut live = LiveTyping::new();
        let mut t = Transcript::new();
        t.feed_final("hello world");
        tick(&mut live, &mut t, "next");
        tick(&mut live, &mut t, "next");
        assert_eq!(live.stable_target(&t), Some("Hello world Next".to_string()));
    }

    #[test]
    fn stable_target_needs_enough_partials() {
        let mut live = LiveTyping::new();
        let mut t = Transcript::new();
        tick(&mut live, &mut t, "the");
        assert_eq!(live.stable_target(&t), None);
        tick(&mut live, &mut t, "the");
        assert_eq!(live.stable_target(&t), Some("The".to_string()));
    }

    #[test]
    fn stable_target_is_common_prefix() {
        let mut live = LiveTyping::new();
        let mut t = Transcript::new();
        tick(&mut live, &mut t, "their");
        tick(&mut live, &mut t, "there");
        // Common prefix "the", capitalized, no space (first segment).
        assert_eq!(live.stable_target(&t), Some("The".to_string()));
    }

    #[test]
    fn stable_target_rolls_forward() {
        let mut live = LiveTyping::new();
        let mut t = Transcript::new();
        tick(&mut live, &mut t, "their");
        tick(&mut live, &mut t, "there");
        assert_eq!(live.stable_target(&t), Some("The".to_string()));
        // Next partial: "there" + "them" -> common prefix "the".
        tick(&mut live, &mut t, "them");
        assert_eq!(live.stable_target(&t), Some("The".to_string()));
        // "them" + "them" -> "them".
        tick(&mut live, &mut t, "them");
        assert_eq!(live.stable_target(&t), Some("Them".to_string()));
    }

    #[test]
    fn stable_target_none_when_prefix_diverges_immediately() {
        let mut live = LiveTyping::new();
        let mut t = Transcript::new();
        tick(&mut live, &mut t, "the");
        tick(&mut live, &mut t, "you");
        // No common prefix -> nothing stable to type yet.
        assert_eq!(live.stable_target(&t), None);
    }

    #[test]
    fn stable_prefix_truncated_to_visible_after_erase() {
        let mut live = LiveTyping::new();
        let mut t = Transcript::new();
        tick(&mut live, &mut t, "the quick brown");
        tick(&mut live, &mut t, "the quick brown");
        assert_eq!(live.stable_target(&t), Some("The quick brown".to_string()));
        // Left mouse press erases the last word; the stability buffer still
        // holds pre-erase partials, so the target must be truncated to the
        // visible partial ("the quick").
        assert!(t.erase_last());
        assert_eq!(live.stable_target(&t), Some("The quick".to_string()));
    }

    #[test]
    fn stable_prefix_never_retires_a_permanently_erased_word() {
        let mut live = LiveTyping::new();
        let mut t = Transcript::new();
        tick(&mut live, &mut t, "the quick brown");
        tick(&mut live, &mut t, "the quick brown");
        assert!(t.erase_last()); // "brown" pending
        // A new word arrives: "brown" is gone for good.
        tick(&mut live, &mut t, "the quick brown fox");
        tick(&mut live, &mut t, "the quick brown fox");
        // The raw stable prefix "the quick brown fox" must not retype
        // "brown"; only what is visible may be typed.
        assert_eq!(live.stable_target(&t), Some("The quick".to_string()));
    }

    #[test]
    fn fully_erased_partial_types_nothing() {
        let mut live = LiveTyping::new();
        let mut t = Transcript::new();
        tick(&mut live, &mut t, "one");
        tick(&mut live, &mut t, "one");
        assert!(t.erase_last());
        assert_eq!(live.stable_target(&t), None);
        assert_eq!(t.buffer_target(""), "");
    }

    #[test]
    fn final_converges_buffer_to_committed_text() {
        let mut live = LiveTyping::new();
        let mut t = Transcript::new();
        tick(&mut live, &mut t, "the quick");
        tick(&mut live, &mut t, "the quick");
        assert_eq!(live.stable_target(&t), Some("The quick".to_string()));
        // The final extends the committed text; the typed buffer must end
        // up exactly at it.
        t.feed_final("the quick brown");
        assert_eq!(t.buffer_target(""), "The quick brown");
    }

    /// The documented X11 hotkey auto-repeat guard: a burst of Toggle() calls
    /// (one per physical key repeat) must produce exactly one engine command,
    /// and repeats must not extend the window past the last *accepted* toggle.
    #[tokio::test]
    async fn toggle_debounces_auto_repeat_bursts() {
        let (tx, mut rx) = mpsc::unbounded_channel::<EngineCmd>();
        let mut d = Dictate::new(tx);
        // First press: accepted.
        d.toggle().await;
        // Key auto-repeat: the same physical press re-fires the shortcut
        // well inside the debounce window - both must be dropped.
        d.toggle().await;
        d.toggle().await;
        assert!(matches!(rx.try_recv(), Ok(EngineCmd::Toggle)));
        assert!(rx.try_recv().is_err());
        // Past the window (anchored to the accepted toggle, not the dropped
        // repeats) a deliberate toggle is accepted again.
        tokio::time::sleep(TOGGLE_DEBOUNCE + Duration::from_millis(10)).await;
        d.toggle().await;
        assert!(matches!(rx.try_recv(), Ok(EngineCmd::Toggle)));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn stable_target_multibyte_prefix_lands_on_char_boundary() {
        let mut live = LiveTyping::new();
        let mut t = Transcript::new();
        tick(&mut live, &mut t, "café");
        tick(&mut live, &mut t, "caffè");
        // The common prefix "caf" ends just before the two-byte \u{e}: the
        // char count must be converted to a byte offset on a char boundary
        // (a wrong offset would panic on the truncate), and the result is
        // capitalized as the first segment's text.
        assert_eq!(live.stable_target(&t), Some("Caf".to_string()));
    }

    /// The default (Deferred) injector task, end to end: partials, a final,
    /// and mid-dictation erasures drive exactly the D-Bus event sequence the
    /// daemon emits. Because the user erases every word before stopping, the
    /// one-shot stop paste is a no-op (no subprocess), so this runs in any
    /// environment, including headless CI.
    #[tokio::test]
    async fn deferred_injector_task_emits_the_documented_event_sequence() {
        let (out_tx, out_rx) = mpsc::channel::<InjectorInput>(8);
        let (ev_tx, mut ev_rx) = mpsc::unbounded_channel::<EngineEvent>();
        let handle = tokio::spawn(injector_task(
            out_rx,
            ev_tx,
            TypingMode::Deferred,
            true,
        ));

        out_tx
            .send(InjectorInput::Asr(AsrOutput::Partial("hello world".into())))
            .await
            .unwrap();
        out_tx
            .send(InjectorInput::Asr(AsrOutput::Partial("hello world".into())))
            .await
            .unwrap();
        out_tx
            .send(InjectorInput::Asr(AsrOutput::Final("hello world".into())))
            .await
            .unwrap();
        out_tx.send(InjectorInput::Erase).await.unwrap();
        out_tx.send(InjectorInput::Erase).await.unwrap();
        // Close the input: the session is stopping, so the (now empty)
        // transcript is pasted once - a no-op here.
        drop(out_tx);
        handle.await.unwrap();

        let mut events = Vec::new();
        while let Ok(ev) = ev_rx.try_recv() {
            events.push(ev);
        }
        assert_eq!(
            events,
            vec![
                EngineEvent::PartialTranscribed("hello world".into()),
                EngineEvent::TranscriptUpdated("hello world".into()),
                EngineEvent::PartialTranscribed("hello world".into()),
                EngineEvent::TranscriptUpdated("hello world".into()),
                EngineEvent::SegmentTranscribed("hello world".into()),
                EngineEvent::TranscriptUpdated("Hello world".into()),
                EngineEvent::TranscriptUpdated("Hello".into()),
                EngineEvent::TranscriptUpdated(String::new()),
            ]
        );
    }

    /// The Live-mode (old mvp2 default) injector task, end to end, against a
    /// faked `ydotool` on a sandboxed PATH: pins the documented typing
    /// behavior - nothing is typed until a partial has survived
    /// `STABILITY_PARTIALS` consecutive ticks, the stable prefix is typed
    /// (capitalized, first segment) and extended as it grows, a decoder
    /// revision that shortens the stable prefix is corrected with exactly
    /// that many backspaces, the final converges the buffer to the
    /// committed text, and - unlike Deferred mode - no one-shot paste
    /// happens at session stop.
    #[tokio::test]
    async fn live_injector_task_types_stable_prefix_corrects_tail_and_converges_on_final() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        let log = d.join("ydotool.log");
        // The fake ydotool records each invocation (space-joined args) as
        // one line; the injector task serializes its calls and awaits each
        // one, so the lines appear in exactly the order they were sent.
        make_fake_tool(
            d,
            "ydotool",
            &format!("#!/bin/sh\nprintf '%s\\n' \"$*\" >> {}\n", log.display()),
        );
        let _env = EnvPatch::new(d, true).await;

        let (out_tx, out_rx) = mpsc::channel::<InjectorInput>(8);
        let (ev_tx, mut ev_rx) = mpsc::unbounded_channel::<EngineEvent>();
        let handle = tokio::spawn(injector_task(out_rx, ev_tx, TypingMode::Live, true));

        // 1: single partial - below the stability threshold, nothing typed.
        out_tx
            .send(InjectorInput::Asr(AsrOutput::Partial("hello wo".into())))
            .await
            .unwrap();
        // 2: stable prefix "hello wo" across the last two partials -> type it.
        out_tx
            .send(InjectorInput::Asr(AsrOutput::Partial("hello world".into())))
            .await
            .unwrap();
        // 3: stable prefix grows to "hello world" -> extend, no backspaces.
        out_tx
            .send(InjectorInput::Asr(AsrOutput::Partial("hello world".into())))
            .await
            .unwrap();
        // 4: decoder revises the tail ("world" -> "worx"): the stable prefix
        //    shrinks to "hello wor" -> exactly two backspaces, nothing typed.
        out_tx
            .send(InjectorInput::Asr(AsrOutput::Partial("hello worx".into())))
            .await
            .unwrap();
        // 5: the final converges the buffer to the committed text.
        out_tx
            .send(InjectorInput::Asr(AsrOutput::Final("hello world.".into())))
            .await
            .unwrap();
        drop(out_tx);
        handle.await.unwrap();

        // Exactly four ydotool invocations: no pre-stability typing, one
        // extension, one backspace correction, one final convergence - and
        // no `key ctrl+v` paste at stop (Live mode never pastes).
        assert_eq!(
            std::fs::read_to_string(&log).unwrap(),
            "type Hello wo\ntype rld\nkey --repeat 2 --delay 0 --repeat-delay 0 Backspace\ntype ld.\n"
        );

        let mut events = Vec::new();
        while let Ok(ev) = ev_rx.try_recv() {
            events.push(ev);
        }
        assert_eq!(
            events,
            vec![
                EngineEvent::PartialTranscribed("hello wo".into()),
                EngineEvent::TranscriptUpdated("hello wo".into()),
                EngineEvent::PartialTranscribed("hello world".into()),
                EngineEvent::TranscriptUpdated("hello world".into()),
                EngineEvent::PartialTranscribed("hello world".into()),
                EngineEvent::TranscriptUpdated("hello world".into()),
                EngineEvent::PartialTranscribed("hello worx".into()),
                EngineEvent::TranscriptUpdated("hello worx".into()),
                EngineEvent::SegmentTranscribed("hello world.".into()),
                EngineEvent::TranscriptUpdated("Hello world.".into()),
            ]
        );
    }

    /// The FinalOnly-mode (MVP1 behavior) injector task, end to end, against
    /// a faked `ydotool` on a sandboxed PATH: pins the documented typing
    /// behavior - partials are *never* typed, even at the exact moment Live
    /// mode would type them (a partial seen `STABILITY_PARTIALS` times in a
    /// row), each `Final` is typed immediately as one chunk that extends the
    /// committed text, a mid-utterance erase/undo leaves the (finals-only)
    /// buffer untouched while the HUD transcript follows, a committed-word
    /// erase backspaces exactly that word from the buffer (undo retypes it),
    /// and - unlike Deferred mode - no one-shot paste happens at session stop.
    #[tokio::test]
    async fn final_only_injector_task_types_each_final_immediately_and_never_partials() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        let log = d.join("ydotool.log");
        make_fake_tool(
            d,
            "ydotool",
            &format!("#!/bin/sh\nprintf '%s\\n' \"$*\" >> {}\n", log.display()),
        );
        let _env = EnvPatch::new(d, true).await;

        let (out_tx, out_rx) = mpsc::channel::<InjectorInput>(8);
        let (ev_tx, mut ev_rx) = mpsc::unbounded_channel::<EngineEvent>();
        let handle = tokio::spawn(injector_task(out_rx, ev_tx, TypingMode::FinalOnly, true));

        // Two identical partials: in Live mode the second one crosses the
        // stability threshold and gets typed; FinalOnly must not type.
        out_tx
            .send(InjectorInput::Asr(AsrOutput::Partial("hello".into())))
            .await
            .unwrap();
        out_tx
            .send(InjectorInput::Asr(AsrOutput::Partial("hello world".into())))
            .await
            .unwrap();
        // The final is typed immediately, one chunk, capitalized.
        out_tx
            .send(InjectorInput::Asr(AsrOutput::Final("hello world".into())))
            .await
            .unwrap();
        // Two more identical partials: still nothing typed.
        out_tx
            .send(InjectorInput::Asr(AsrOutput::Partial("next thing".into())))
            .await
            .unwrap();
        out_tx
            .send(InjectorInput::Asr(AsrOutput::Partial("next thing".into())))
            .await
            .unwrap();
        // Erasing the live partial word leaves the (finals-only) buffer
        // unchanged; the HUD transcript follows both the erase and the undo.
        out_tx.send(InjectorInput::Erase).await.unwrap();
        out_tx.send(InjectorInput::Undo).await.unwrap();
        // The second final extends the buffer by exactly the new chunk
        // (leading space and all, since the new segment is capitalized).
        out_tx
            .send(InjectorInput::Asr(AsrOutput::Final("next thing".into())))
            .await
            .unwrap();
        // A committed-word erase backspaces exactly that word; undo retypes it.
        out_tx.send(InjectorInput::Erase).await.unwrap();
        out_tx.send(InjectorInput::Undo).await.unwrap();
        drop(out_tx);
        handle.await.unwrap();

        // Exactly four invocations: two final chunks, one backspace
        // correction, one retype - no partial typing anywhere, and no
        // `key ctrl+v` paste at stop (FinalOnly never pastes).
        assert_eq!(
            std::fs::read_to_string(&log).unwrap(),
            "type Hello world\ntype  Next thing\nkey --repeat 6 --delay 0 --repeat-delay 0 Backspace\ntype  thing\n"
        );

        let mut events = Vec::new();
        while let Ok(ev) = ev_rx.try_recv() {
            events.push(ev);
        }
        assert_eq!(
            events,
            vec![
                EngineEvent::PartialTranscribed("hello".into()),
                EngineEvent::TranscriptUpdated("hello".into()),
                EngineEvent::PartialTranscribed("hello world".into()),
                EngineEvent::TranscriptUpdated("hello world".into()),
                EngineEvent::SegmentTranscribed("hello world".into()),
                EngineEvent::TranscriptUpdated("Hello world".into()),
                EngineEvent::PartialTranscribed("next thing".into()),
                EngineEvent::TranscriptUpdated("Hello world next thing".into()),
                EngineEvent::PartialTranscribed("next thing".into()),
                EngineEvent::TranscriptUpdated("Hello world next thing".into()),
                EngineEvent::TranscriptUpdated("Hello world next".into()),
                EngineEvent::TranscriptUpdated("Hello world next thing".into()),
                EngineEvent::SegmentTranscribed("next thing".into()),
                EngineEvent::TranscriptUpdated("Hello world Next thing".into()),
                EngineEvent::TranscriptUpdated("Hello world Next".into()),
                EngineEvent::TranscriptUpdated("Hello world Next thing".into()),
            ]
        );
    }

    /// The streaming pipeline task, end to end, on the real model + real
    /// speech (silently skipped when the repo's `models/` dir has no
    /// streaming backend or its test WAV): feed a 16 kHz utterance in
    /// 512-sample frames slightly slower than real time, so the 150 ms
    /// wall-clock partial tick actually fires while the VAD detects speech
    /// (a burst feed would finish before the first tick), and pin the
    /// documented contract: live `Partial`s are emitted while the utterance
    /// is in progress, all of them before a single `Final` that the VAD
    /// commits at the utterance's trailing silence.
    // The model-serialization lock is a bare unit flag whose whole purpose
    // is to span the test's model load *and* decode phase, so holding it
    // across the awaits is intentional (the sync real-model tests in asr/
    // vad hold it the same way; a tokio mutex would not work there).
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn stream_task_emits_live_partials_then_one_final_for_real_utterance() {
        let _lock = model_lock();
        let Some(models) = repo_models_dir() else {
            return;
        };
        let asr = match Asr::new(&models, 2, BackendSelection::Streaming) {
            Ok(a) => a,
            Err(_) => return,
        };
        // Use the test WAV shipped with whichever streaming model dir is
        // installed (both carry the same "After early nightfall..." file).
        let wav = if let Some(p) = Asr::find_nemotron(&models) {
            p.dir.join("test_wavs/0.wav")
        } else if let Ok(p) = Asr::resolve_zipformer_paths(&models) {
            p.dir.join("test_wavs/0.wav")
        } else {
            return;
        };
        if !wav.is_file() {
            return;
        }
        let vad = match Vad::new(&VadConfig::from_models_dir(&models)) {
            Ok(v) => v,
            Err(_) => return,
        };
        let wave =
            sherpa_onnx::Wave::read(wav.to_str().expect("utf-8 path")).expect("read test wav");
        assert_eq!(wave.sample_rate(), 16000, "fixture must be 16 kHz");
        let frames: Vec<Vec<f32>> = wave.samples().chunks(512).map(|c| c.to_vec()).collect();

        let (frames_tx, frames_rx) = mpsc::channel::<Vec<f32>>(128);
        let (out_tx, mut out_rx) = mpsc::channel::<InjectorInput>(16);
        let handle = tokio::spawn(stream_task(
            vad,
            std::sync::Arc::new(asr),
            frames_rx,
            out_tx,
        ));

        // Pace the first frames (20 ms per 32 ms frame, ~1.6x real time)
        // until the first live partial arrives, then burst the rest: the
        // trailing silence inside the WAV is what finalizes the utterance.
        let mut fed = 0;
        for frame in &frames {
            frames_tx.send(frame.clone()).await.unwrap();
            fed += 1;
            tokio::time::sleep(Duration::from_millis(20)).await;
            if matches!(
                out_rx.try_recv(),
                Ok(InjectorInput::Asr(AsrOutput::Partial(_)))
            ) {
                break;
            }
        }
        for frame in &frames[fed..] {
            frames_tx.send(frame.clone()).await.unwrap();
        }
        // Audio source ended: the task flushes (nothing left, the trailing
        // silence already finalized the utterance) and exits.
        drop(frames_tx);
        handle.await.unwrap();

        let mut outputs = Vec::new();
        while let Ok(input) = out_rx.try_recv() {
            outputs.push(input);
        }
        let final_idx = outputs
            .iter()
            .position(|o| matches!(o, InjectorInput::Asr(AsrOutput::Final(_))));
        let mut partials = 0;
        let mut finals = Vec::new();
        for (idx, input) in outputs.iter().enumerate() {
            match input {
                InjectorInput::Asr(AsrOutput::Partial(_)) => {
                    partials += 1;
                    assert!(
                        final_idx.is_none_or(|f| idx < f),
                        "a partial arrived after the final"
                    );
                }
                InjectorInput::Asr(AsrOutput::Final(f)) => finals.push(f.clone()),
                other => panic!("unexpected input: {other:?}"),
            }
        }
        assert!(
            partials >= 1,
            "no live partials were emitted ({:?} outputs)",
            outputs.len()
        );
        assert_eq!(finals.len(), 1, "exactly one final expected: {finals:?}");
        assert!(
            finals[0].to_lowercase().contains("yellow lamps"),
            "unexpected final: {finals:?}"
        );
    }

    /// The batch pipeline (Moonshine path), end to end, on the real VAD +
    /// real Moonshine model + real speech (silently skipped when `models/`
    /// lacks the Moonshine backend, its VAD, or a 16 kHz test WAV): feed a
    /// 16 kHz utterance in 512-sample frames in one burst (the batch path
    /// has no wall-clock cadence, unlike `stream_task`) through the exact
    /// production wiring - `vad_task` (ring-padded segments) -> `asr_task`
    /// - and pin the documented contract: exactly one `Final` per
    /// utterance, and - unlike the streaming path - never a `Partial`.
    // The model-serialization lock is a bare unit flag whose whole purpose
    // is to span the test's model load *and* decode phase, so holding it
    // across the awaits is intentional (the sync real-model tests in asr/
    // vad hold it the same way; a tokio mutex would not work there).
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn batch_pipeline_emits_one_final_per_utterance_and_no_partials() {
        let _lock = model_lock();
        let Some(models) = repo_models_dir() else {
            return;
        };
        let asr = match Asr::new(&models, 2, BackendSelection::Moonshine) {
            Ok(a) => a,
            Err(_) => return,
        };
        let vad = match Vad::new(&VadConfig::from_models_dir(&models)) {
            Ok(v) => v,
            Err(_) => return,
        };
        // The 16 kHz fixture ships with the streaming model dirs (the
        // Moonshine dir's own test WAV is 24 kHz and would garble the
        // 16 kHz-only VAD), same selection as the streaming pipeline test.
        let wav = if let Some(p) = Asr::find_nemotron(&models) {
            p.dir.join("test_wavs/0.wav")
        } else if let Ok(p) = Asr::resolve_zipformer_paths(&models) {
            p.dir.join("test_wavs/0.wav")
        } else {
            return;
        };
        if !wav.is_file() {
            return;
        }
        let wave =
            sherpa_onnx::Wave::read(wav.to_str().expect("utf-8 path")).expect("read test wav");
        assert_eq!(wave.sample_rate(), 16000, "fixture must be 16 kHz");
        let frames: Vec<Vec<f32>> = wave.samples().chunks(512).map(|c| c.to_vec()).collect();

        // The exact wiring `Engine::start` uses for the batch backend,
        // including the 30 s lookback ring the VAD task slices padding
        // from (max segment 20 s + padding).
        let (frames_tx, frames_rx) = mpsc::channel::<Vec<f32>>(128);
        let (segs_tx, segs_rx) = mpsc::channel::<Vec<f32>>(16);
        let (out_tx, mut out_rx) = mpsc::channel::<InjectorInput>(16);
        let vad_handle = tokio::spawn(vad_task(
            vad,
            AudioRing::new(30 * audio::SAMPLE_RATE as usize),
            frames_rx,
            segs_tx,
        ));
        let asr_handle = tokio::spawn(asr_task(std::sync::Arc::new(asr), segs_rx, out_tx));

        // One burst: the VAD finalizes the utterance on the trailing
        // silence inside the WAV; when the source ends, `vad_task` flushes
        // (nothing left) and exits, closing the segment channel, which
        // ends `asr_task` after it drains.
        for frame in &frames {
            frames_tx.send(frame.clone()).await.unwrap();
        }
        drop(frames_tx);
        vad_handle.await.unwrap();
        asr_handle.await.unwrap();

        let mut finals = Vec::new();
        while let Ok(input) = out_rx.try_recv() {
            match input {
                // The batch path has no live hypotheses: only finals.
                InjectorInput::Asr(AsrOutput::Final(text)) => finals.push(text),
                other => panic!("the batch path never emits partials: {other:?}"),
            }
        }
        assert_eq!(finals.len(), 1, "the WAV holds one utterance: {finals:?}");
        assert!(
            finals[0].to_lowercase().contains("yellow lamps"),
            "unexpected final: {finals:?}"
        );
    }
}

#[cfg(test)]
mod double_erase_regression {
    //! Regression coverage for a real bug: two Left-arrow presses in quick
    //! succession during live dictation could silently erase the wrong
    //! word (or none at all) and desync the undo stack from what was
    //! visibly typed. Root cause: `Transcript::feed_partial` used to treat
    //! any growth in the decoder's live hypothesis as "the user kept
    //! talking" and permanently retired pending erasures on the spot -
    //! but `saytype --stream-test` on a real recording shows the streaming
    //! decoder grows its partial roughly every 500-600ms *continuously*
    //! while speaking, which is well within human key-press cadence. A
    //! press, then one ordinary decoder tick, then a second press would
    //! make the first erasure permanent and retarget the second press at
    //! whatever word just streamed in, not the word the user meant.

    use super::*;

    fn tick(live: &mut LiveTyping, t: &mut Transcript, text: &str) {
        t.feed_partial(text);
        live.push_partial(text);
    }

    #[test]
    fn double_erase_without_intervening_tick() {
        let mut live = LiveTyping::new();
        let mut t = Transcript::new();
        tick(&mut live, &mut t, "the quick brown fox");
        tick(&mut live, &mut t, "the quick brown fox");
        assert_eq!(live.stable_target(&t), Some("The quick brown fox".to_string()));
        assert!(t.erase_last());
        assert!(t.erase_last());
        assert_eq!(t.display(), "the quick");
        assert_eq!(live.stable_target(&t), Some("The quick".to_string()));
    }

    // A VAD auto-commit (0.8s pause) lands between the user's two presses:
    // press 1 erases "fox" (pending, mid-partial), the pause auto-commits
    // the utterance, then press 2 arrives. It must still target "brown".
    #[test]
    fn erase_commit_erase_targets_the_right_word() {
        let mut live = LiveTyping::new();
        let mut t = Transcript::new();
        tick(&mut live, &mut t, "the quick brown fox");
        tick(&mut live, &mut t, "the quick brown fox");
        assert!(t.erase_last()); // "fox" pending-erased
        t.feed_final("the quick brown fox"); // same tokens: no growth
        live.commit_segment();
        assert_eq!(t.display(), "The quick brown");
        assert!(t.erase_last()); // press 2: must erase "brown", not "fox"
        assert_eq!(t.display(), "The quick");
    }

    // The regression itself: a new word streams in between the user's two
    // presses purely from ordinary decoder cadence (not a deliberate new
    // utterance). Whichever word each press actually lands on (that is
    // inherently racy against a live decoder), neither erasure may be
    // silently made permanent, and both must stay undoable, LIFO.
    #[test]
    fn erase_twice_survives_a_decoder_tick_in_between() {
        let mut live = LiveTyping::new();
        let mut t = Transcript::new();
        tick(&mut live, &mut t, "the quick brown fox");
        tick(&mut live, &mut t, "the quick brown fox");
        assert!(t.erase_last()); // press 1: "fox" pending-erased
        assert_eq!(t.display(), "the quick brown");
        // One ordinary decoder tick with a new word, before press 2 lands.
        tick(&mut live, &mut t, "the quick brown fox jumps");
        // "fox" must still be pending (not silently made permanent), still
        // hidden from the now-longer partial.
        assert_eq!(t.display(), "the quick brown jumps");
        assert!(t.erase_last()); // press 2: erases the new tail, "jumps"
        assert_eq!(t.display(), "the quick brown");
        // Both erasures are still restorable, LIFO.
        assert!(t.undo_last());
        assert_eq!(t.display(), "the quick brown jumps");
        assert!(t.undo_last());
        assert_eq!(t.display(), "the quick brown fox jumps");
    }
}
