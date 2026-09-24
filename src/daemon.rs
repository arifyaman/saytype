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

#[derive(Debug, Clone)]
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

pub struct Dictate {
    cmd_tx: mpsc::UnboundedSender<EngineCmd>,
}

/// D-Bus interface. The methods must not do real work here: zbus dispatches
/// them on its own internal executor, which is not a tokio runtime, so any
/// tokio API (spawn, spawn_blocking, time) would panic. Instead we just
/// forward a command to the engine loop task, which runs on the tokio runtime.
#[interface(name = "io.saytype.Dictate1")]
impl Dictate {
    async fn toggle(&mut self) {
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
        Self { cmd_tx }
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
    /// Live typing of partials into the target app (off via
    /// `--no-live-typing`; only meaningful for the streaming backend).
    live_typing: bool,
    session: Option<Session>,
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
        live_typing: bool,
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
            live_typing,
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
        // Live typing only makes sense with a streaming backend (they alone
        // produce partials); the batch backend always types finals only.
        let live_typing = self.live_typing && streaming;
        let inj_task = tokio::spawn(injector_task(out_rx, events, live_typing, streaming));

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
        // Injector drains the remaining outputs.
        let _ = tokio::time::timeout(Duration::from_secs(30), session.inj_task).await;
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
/// the target buffer holds and what the HUD shows. With `live_typing`
/// (streaming backend without `--no-live-typing`): partials are typed into
/// the target app as soon as they are stable (see `STABILITY_PARTIALS`),
/// corrected with backspaces when the decoder revises the tail; each
/// `Final` converges the buffer to the committed text. Without it (batch
/// backend, or `--no-live-typing`): only finals are typed - one chunk per
/// committed utterance, the MVP1 behavior. Mid-dictation erase/undo
/// commands (mouse buttons) apply to the transcript and converge the
/// buffer. The raw ASR text is always relayed to the HUD, and the visible
/// transcript (erasures applied) is re-sent after every change.
async fn injector_task(
    mut out_rx: mpsc::Receiver<InjectorInput>,
    events: mpsc::UnboundedSender<EngineEvent>,
    live_typing: bool,
    streaming: bool,
) {
    let mut live = LiveTyping::new();
    let mut transcript = Transcript::new();
    while let Some(input) = out_rx.recv().await {
        match input {
            InjectorInput::Asr(AsrOutput::Partial(text)) => {
                let _ = events.send(EngineEvent::PartialTranscribed(text.clone()));
                transcript.feed_partial(&text);
                if live_typing {
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
                // Convergence: type whatever makes the buffer exactly the
                // committed text (an extension in the common case;
                // backspaces + retype when the final differs from the last
                // partial), then reset for the next segment.
                if streaming {
                    transcript.feed_final(&text);
                } else {
                    transcript.feed_batch_final(&text);
                }
                let target = transcript.buffer_target("");
                let ok = live.apply(&target).await.is_ok();
                if !ok {
                    tracing::error!("ydotool failed to type final: {text:?}");
                }
                live.commit_segment();
                let event = if ok {
                    text
                } else {
                    format!("[type failed] {text}")
                };
                let _ = events.send(EngineEvent::SegmentTranscribed(event));
                let _ = events.send(EngineEvent::TranscriptUpdated(transcript.display()));
            }
            InjectorInput::Erase => {
                if transcript.erase_last() {
                    apply_transcript_change(&mut live, &transcript, &events, "erased").await;
                }
            }
            InjectorInput::Undo => {
                if transcript.undo_last() {
                    apply_transcript_change(&mut live, &transcript, &events, "restored").await;
                }
            }
        }
    }
}

/// After a mid-dictation erase/undo: converge the target buffer to the
/// transcript's current visible state (the stable prefix of the live
/// partial when there is one, else the committed text) and relay the
/// display to the HUD.
async fn apply_transcript_change(
    live: &mut LiveTyping,
    transcript: &Transcript,
    events: &mpsc::UnboundedSender<EngineEvent>,
    what: &str,
) {
    let target = live
        .stable_target(transcript)
        .unwrap_or_else(|| transcript.buffer_target(""));
    if let Err(e) = live.apply(&target).await {
        tracing::error!("buffer convergence after {what} word failed: {e}");
    }
    tracing::info!("{what} word, transcript: {:?}", transcript.display());
    let _ = events.send(EngineEvent::TranscriptUpdated(transcript.display()));
}

/// Start the daemon: load models, register the D-Bus service, pump events.
pub async fn run(
    models_dir: &Path,
    selection: BackendSelection,
    live_typing: bool,
) -> Result<()> {
    let (events_tx, mut events_rx) = mpsc::unbounded_channel::<EngineEvent>();
    let engine = Engine::load(models_dir, events_tx, selection, live_typing).await?;
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
}
