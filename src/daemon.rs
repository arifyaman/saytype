//! D-Bus service and dictation engine.
//!
//! State machine: Idle <-> Recording.
//!
//! While Recording the pipeline is:
//!   PipeWire thread -> frames channel -> VAD task -> segments channel
//!   -> ASR task -> texts channel -> injector task (single consumer, ydotool).
//!
//! All state/signal transitions are reported to D-Bus through `EngineEvent`s,
//! pumped by the daemon task that owns the `SignalContext`.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::{mpsc, oneshot};
use zbus::{interface, SignalContext};

use crate::asr::Asr;
use crate::audio::{self, CaptureHandle};
use crate::injector;
use crate::vad::{AudioRing, Vad, VadConfig, POST_PAD_MS, PRE_PAD_MS};

pub const BUS_NAME: &str = "io.saytype.Dictate";
pub const OBJECT_PATH: &str = "/io/saytype/Dictate";
pub const INTERFACE_NAME: &str = "io.saytype.Dictate1";

#[derive(Debug, Clone)]
pub enum EngineEvent {
    StateChanged(String),
    SegmentTranscribed(String),
}

/// Commands sent from the D-Bus methods to the engine loop.
#[derive(Debug, Clone, Copy)]
pub enum EngineCmd {
    Toggle,
    Stop,
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

    #[zbus(signal)]
    async fn state_changed(ctxt: &SignalContext<'_>, state: String) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn segment_transcribed(ctxt: &SignalContext<'_>, text: String) -> zbus::Result<()>;
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
        }
    }
}

pub struct Engine {
    vad_config: VadConfig,
    asr: std::sync::Arc<Asr>,
    events: mpsc::UnboundedSender<EngineEvent>,
    session: Option<Session>,
}

struct Session {
    capture: CaptureHandle,
    exited_rx: oneshot::Receiver<()>,
    vad_task: tokio::task::JoinHandle<()>,
    asr_task: tokio::task::JoinHandle<()>,
    inj_task: tokio::task::JoinHandle<()>,
}

impl Engine {
    pub async fn load(
        models_dir: &Path,
        events: mpsc::UnboundedSender<EngineEvent>,
    ) -> Result<Self> {
        crate::vad::check_model_file(&models_dir.join("silero_vad.onnx"))?;
        crate::asr::check_models_dir(models_dir)?;

        let asr = tokio::task::spawn_blocking({
            let dir = models_dir.to_path_buf();
            move || -> Result<Asr> {
                let start = std::time::Instant::now();
                let asr = Asr::new(&dir, 4)
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
            session: None,
        })
    }

    pub fn is_recording(&self) -> bool {
        self.session.is_some()
    }

    pub async fn start(&mut self) -> Result<()> {
        anyhow::ensure!(!self.is_recording(), "already recording");

        let (frames_tx, frames_rx) = mpsc::channel(128);
        let (segs_tx, segs_rx) = mpsc::channel(16);
        let (texts_tx, texts_rx) = mpsc::channel(16);
        let (exited_tx, exited_rx) = oneshot::channel::<()>();

        let capture = audio::start_capture(frames_tx, exited_tx);

        let vad = Vad::new(&self.vad_config)?;
        // 30s of lookback: covers max segment (20s) + padding comfortably.
        let ring = AudioRing::new(30 * audio::SAMPLE_RATE as usize);
        let vad_task = tokio::spawn(vad_task(vad, ring, frames_rx, segs_tx));

        let asr = self.asr.clone();
        let asr_task = tokio::spawn(asr_task(asr, segs_rx, texts_tx));

        let events = self.events.clone();
        let inj_task = tokio::spawn(injector_task(texts_rx, events));

        self.session = Some(Session {
            capture,
            exited_rx,
            vad_task,
            asr_task,
            inj_task,
        });

        let _ = self
            .events
            .send(EngineEvent::StateChanged("Recording".into()));
        tracing::info!("dictation session started");
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

        // Audio gone: the VAD task flushes its trailing segment and exits.
        let _ = tokio::time::timeout(Duration::from_secs(5), session.vad_task).await;
        // ASR drains the remaining segments.
        let _ = tokio::time::timeout(Duration::from_secs(60), session.asr_task).await;
        // Injector drains the remaining texts.
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

/// ASR consumer: transcribe each finalized segment.
async fn asr_task(
    asr: std::sync::Arc<Asr>,
    mut seg_rx: mpsc::Receiver<Vec<f32>>,
    text_tx: mpsc::Sender<String>,
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
        if text_tx.send(text).await.is_err() {
            return;
        }
    }
}

/// Injector: single consumer, strictly serialized ydotool calls.
async fn injector_task(
    mut text_rx: mpsc::Receiver<String>,
    events: mpsc::UnboundedSender<EngineEvent>,
) {
    let mut first = true;
    while let Some(text) = text_rx.recv().await {
        let mut to_type = injector::capitalize_first(&text);
        if !first {
            to_type.insert(0, ' ');
        }
        first = false;
        match injector::type_text(&to_type).await {
            Ok(()) => {
                let _ = events.send(EngineEvent::SegmentTranscribed(text));
            }
            Err(e) => {
                tracing::error!("ydotool failed to type {:?}: {e}", text);
                let _ = events.send(EngineEvent::SegmentTranscribed(format!(
                    "[type failed] {text}"
                )));
            }
        }
    }
}

/// Start the daemon: load models, register the D-Bus service, pump events.
pub async fn run(models_dir: &Path) -> Result<()> {
    let (events_tx, mut events_rx) = mpsc::unbounded_channel::<EngineEvent>();
    let engine = Engine::load(models_dir, events_tx).await?;
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
            EngineEvent::SegmentTranscribed(text) => {
                if let Err(e) = Dictate::segment_transcribed(&signal_ctx, text).await {
                    tracing::error!("emitting SegmentTranscribed failed: {e}");
                }
            }
        }
    }
    Ok(())
}
