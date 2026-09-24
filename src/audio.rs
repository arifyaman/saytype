//! PipeWire microphone capture.
//!
//! PipeWire's mainloop is callback-driven and its objects are not thread-safe, so the
//! whole capture pipeline (mainloop, context, core, stream) runs on one dedicated OS
//! thread. Captured frames (mono f32 at 16 kHz) are forwarded across a tokio mpsc
//! channel into the rest of the daemon.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use pipewire as pw;
use pw::{
    context::ContextRc, keys, main_loop::MainLoopRc, properties::properties, spa,
    stream::StreamBox, stream::StreamFlags,
};
use spa::param::audio::{AudioFormat, AudioInfoRaw};
use spa::param::format::{MediaSubtype, MediaType};
use spa::param::format_utils;
use spa::param::ParamType;
use spa::pod::serialize::PodSerializer;
use spa::pod::{Object as PodObject, Value as PodValue};
use spa::utils::Direction;
use tokio::sync::{mpsc, oneshot};

pub const SAMPLE_RATE: i32 = 16000;

pub fn start_capture(
    frame_tx: mpsc::Sender<Vec<f32>>,
    exited_tx: oneshot::Sender<()>,
) -> CaptureHandle {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = stop.clone();
    let join = std::thread::Builder::new()
        .name("saytype-audio".into())
        .spawn(move || {
            let res = run_capture(&stop_thread, frame_tx);
            tracing::info!("audio capture thread exiting");
            let _ = exited_tx.send(());
            let _ = res;
        })
        .expect("failed to spawn audio thread");
    CaptureHandle {
        stop,
        join: Some(join),
    }
}

impl CaptureHandle {
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Block until the capture thread has fully shut down.
    pub fn join(mut self) {
        if let Some(h) = self.join.take() {
            let _ = h.join();
        }
    }
}

pub struct CaptureHandle {
    stop: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
}

fn run_capture(stop: &Arc<AtomicBool>, frame_tx: mpsc::Sender<Vec<f32>>) -> Result<(), String> {
    pw::init();

    let mainloop = MainLoopRc::new(None).map_err(|e| format!("mainloop: {e}"))?;
    let context = ContextRc::new(&mainloop, None).map_err(|e| format!("context: {e}"))?;
    let core = context
        .connect_rc(None)
        .map_err(|e| format!("connect: {e}"))?;

    let props = properties! {
        *keys::MEDIA_TYPE => "Audio",
        *keys::MEDIA_CATEGORY => "Capture",
        *keys::MEDIA_ROLE => "Dictation",
    };
    let stream =
        StreamBox::new(&core, "saytype-capture", props).map_err(|e| format!("stream: {e}"))?;

    let ud = StreamUserData {
        format: None,
        stop: stop.clone(),
    };
    let ml = mainloop.clone();
    let tx = frame_tx.clone();
    let _listener = stream
        .add_local_listener_with_user_data(ud)
        .param_changed(|_stream, ud, id, param| {
            if id != ParamType::Format.as_raw() {
                return;
            }
            let Some(param) = param else {
                return;
            };
            let Ok((media_type, media_subtype)) = format_utils::parse_format(param) else {
                return;
            };
            if media_type != MediaType::Audio || media_subtype != MediaSubtype::Raw {
                return;
            }
            let mut info = AudioInfoRaw::new();
            if info.parse(param).is_ok() {
                ud.format = Some(info);
                tracing::info!(
                    "capture negotiated: format={:?} rate={} channels={}",
                    info.format(),
                    info.rate(),
                    info.channels()
                );
            }
        })
        .process(move |stream, ud| {
            if ud.stop.load(Ordering::SeqCst) {
                ml.quit();
                return;
            }
            let Some(format) = ud.format else {
                return;
            };
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }
            let data = &mut datas[0];
            let chunk_bytes = data.chunk().size() as usize;
            let Some(bytes) = data.data() else {
                return;
            };
            let n = chunk_bytes.min(bytes.len());
            let samples = pcm_to_mono_f32(&format, &bytes[..n]);
            if !samples.is_empty() && tx.blocking_send(samples).is_err() {
                // VAD consumer gone (session stopped); shut the loop down.
                ml.quit();
            }
        })
        .register()
        .map_err(|e| format!("listener: {e}"))?;

    // The serialized param pod must outlive the stream's parameter negotiation.
    let pod_bytes = connect_capture_stream(&stream)?;

    tracing::info!("audio capture loop running");
    mainloop.run();
    drop(pod_bytes);

    tracing::info!("audio capture loop finished");
    Ok(())
}

struct StreamUserData {
    format: Option<AudioInfoRaw>,
    stop: Arc<AtomicBool>,
}

/// Connect the capture stream. We call the FFI directly because the crate's
/// `Stream::connect` takes `&mut [Pod]` (owned pods) while the only public pod
/// constructors return references, which is not expressible in the type system.
/// Returns the serialized pod bytes; the caller must keep them alive.
fn connect_capture_stream(stream: &StreamBox) -> Result<Vec<u8>, String> {
    let mut info = AudioInfoRaw::new();
    info.set_format(AudioFormat::S16LE);
    info.set_rate(SAMPLE_RATE as u32);
    info.set_channels(1);

    let obj = PodObject {
        type_: spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: ParamType::EnumFormat.as_raw(),
        properties: info.into(),
    };
    let cursor = std::io::Cursor::new(Vec::new());
    let values: Vec<u8> = PodSerializer::serialize(cursor, &PodValue::Object(obj))
        .map_err(|e| format!("pod serialize: {e:?}"))?
        .0
        .into_inner();
    // `values` must outlive the FFI call; it holds the serialized spa_pod.
    let pod_ptr = values.as_ptr() as *const pw::spa::sys::spa_pod;
    let params: [*const pw::spa::sys::spa_pod; 1] = [pod_ptr];

    let flags = StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS | StreamFlags::RT_PROCESS;
    let rc = unsafe {
        pw::sys::pw_stream_connect(
            stream.as_raw_ptr(),
            Direction::Input.as_raw(),
            pw::constants::ID_ANY,
            flags.bits(),
            // `pw_stream_connect` takes `pw_stream_flags` (a u32) and
            // `*mut *const spa_pod`.
            params.as_ptr().cast_mut(),
            params.len() as u32,
        )
    };
    if rc < 0 {
        return Err(format!(
            "pw_stream_connect failed: {}",
            std::io::Error::from_raw_os_error(-rc)
        ));
    }
    Ok(values)
}

fn pcm_to_mono_f32(format: &AudioInfoRaw, bytes: &[u8]) -> Vec<f32> {
    let channels = format.channels().max(1) as usize;
    let rate = format.rate();
    let data = match format.format() {
        AudioFormat::S16LE => {
            let n_frames = bytes.len() / 2 / channels;
            (0..n_frames)
                .map(|i| {
                    let v: i16 =
                        i16::from_le_bytes([bytes[i * channels * 2], bytes[i * channels * 2 + 1]]);
                    v as f32 / i16::MAX as f32
                })
                .collect::<Vec<f32>>()
        }
        AudioFormat::F32LE => {
            let n_frames = bytes.len() / 4 / channels;
            (0..n_frames)
                .map(|i| {
                    f32::from_le_bytes([
                        bytes[i * channels * 4],
                        bytes[i * channels * 4 + 1],
                        bytes[i * channels * 4 + 2],
                        bytes[i * channels * 4 + 3],
                    ])
                })
                .collect::<Vec<f32>>()
        }
        _ => return Vec::new(),
    };
    if rate == SAMPLE_RATE as u32 {
        return data;
    }
    resample_linear(&data, rate as i32, SAMPLE_RATE)
}

fn resample_linear(input: &[f32], from: i32, to: i32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let out_len = ((input.len() as f64) / ratio) as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos as usize;
            if idx + 1 < input.len() {
                let frac = (pos - idx as f64) as f32;
                input[idx] * (1.0f32 - frac) + input[idx + 1] * frac
            } else {
                input.get(idx).copied().unwrap_or(0.0)
            }
        })
        .collect()
}
