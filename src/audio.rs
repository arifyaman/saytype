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

/// Linearly resample `input` from rate `from` to rate `to`.
///
/// A non-positive rate is malformed (e.g. a negotiated format that parsed
/// with the default rate 0). Without a valid source rate the samples cannot
/// be placed on the target grid, so emit nothing; a zero source rate would
/// otherwise divide by zero, saturate the output length to `usize::MAX`,
/// and panic the capture thread on the "capacity overflow" allocation.
fn resample_linear(input: &[f32], from: i32, to: i32) -> Vec<f32> {
    if from <= 0 || to <= 0 {
        return Vec::new();
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a negotiated-format fixture the way the param-changed callback
    /// does.
    fn info(format: AudioFormat, rate: u32, channels: u32) -> AudioInfoRaw {
        let mut i = AudioInfoRaw::new();
        i.set_format(format);
        i.set_rate(rate);
        i.set_channels(channels);
        i
    }

    fn s16le(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn f32le(samples: &[f32]) -> Vec<u8> {
        samples.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], eps: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= eps, "expected {e} +/- {eps}, got {a}");
        }
    }

    // --- pcm_to_mono_f32: decoding ---

    #[test]
    fn s16le_mono_decodes_normalized() {
        let fmt = info(AudioFormat::S16LE, 16000, 1);
        let out = pcm_to_mono_f32(&fmt, &s16le(&[0, 16383, -16384, 32767]));
        // Normalized against i16::MAX: 32767 is exactly 1.0.
        assert_close(&out, &[0.0, 0.5, -0.5, 1.0], 1e-3);
    }

    #[test]
    fn s16le_stereo_takes_first_channel() {
        // Interleaved frames (L0,R0),(L1,R1): only the L (first) channel is
        // decoded. Some mics put their signal on one channel only, and this
        // is the channel the daemon's capture node is known to use.
        let fmt = info(AudioFormat::S16LE, 16000, 2);
        let out = pcm_to_mono_f32(&fmt, &s16le(&[0, 1000, 2000, -3000]));
        assert_close(&out, &[0.0, 2000.0 / i16::MAX as f32], 1e-6);
    }

    #[test]
    fn f32le_mono_passthrough() {
        let fmt = info(AudioFormat::F32LE, 16000, 1);
        let out = pcm_to_mono_f32(&fmt, &f32le(&[0.25, -0.5, 1.0]));
        assert_close(&out, &[0.25, -0.5, 1.0], 0.0);
    }

    #[test]
    fn f32le_stereo_takes_first_channel() {
        let fmt = info(AudioFormat::F32LE, 16000, 2);
        let out = pcm_to_mono_f32(&fmt, &f32le(&[0.1, 0.9, 0.2, 0.8]));
        assert_close(&out, &[0.1, 0.2], 0.0);
    }

    #[test]
    fn unsupported_format_returns_empty() {
        // Neither S32LE nor an unknown format is decoded; the daemon then
        // just sees no audio instead of garbage.
        let s32 = info(AudioFormat::S32LE, 16000, 1);
        assert!(pcm_to_mono_f32(&s32, &[0; 16]).is_empty());
        let unknown = info(AudioFormat::Unknown, 16000, 1);
        assert!(pcm_to_mono_f32(&unknown, &[0; 16]).is_empty());
    }

    #[test]
    fn empty_bytes_return_empty() {
        let fmt = info(AudioFormat::S16LE, 16000, 1);
        assert!(pcm_to_mono_f32(&fmt, &[]).is_empty());
    }

    #[test]
    fn trailing_partial_frame_is_dropped() {
        // 5 bytes hold 2 complete i16 frames plus 1 stray byte; the stray
        // byte must not be read as a (corrupt) frame.
        let fmt = info(AudioFormat::S16LE, 16000, 1);
        let bytes = s16le(&[1000, 2000]);
        let mut bytes = bytes;
        bytes.push(0xFF);
        let out = pcm_to_mono_f32(&fmt, &bytes);
        assert_close(
            &out,
            &[1000.0 / i16::MAX as f32, 2000.0 / i16::MAX as f32],
            1e-6,
        );
    }

    #[test]
    fn zero_channels_clamped_to_mono() {
        // A format that (malformed or not) reports 0 channels must not
        // divide by zero: it is treated as mono.
        let fmt = info(AudioFormat::S16LE, 16000, 0);
        let out = pcm_to_mono_f32(&fmt, &s16le(&[1000]));
        assert_close(&out, &[1000.0 / i16::MAX as f32], 1e-6);
    }

    // --- pcm_to_mono_f32: rate conversion ---

    #[test]
    fn s16le_8k_is_resampled_to_16k() {
        let fmt = info(AudioFormat::S16LE, 8000, 1);
        let out = pcm_to_mono_f32(&fmt, &s16le(&[0, 8192, 16384, 24576]));
        // 4 samples at 8 kHz -> 8 at 16 kHz, linearly interpolated.
        assert_close(
            &out,
            &[0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.75],
            1e-3,
        );
    }

    // --- resample_linear: direct ---

    #[test]
    fn resample_same_rate_is_a_copy() {
        let out = resample_linear(&[0.5, -0.25, 1.0], 16000, 16000);
        assert_close(&out, &[0.5, -0.25, 1.0], 0.0);
    }

    #[test]
    fn resample_empty_input() {
        assert!(resample_linear(&[], 48000, 16000).is_empty());
    }

    #[test]
    fn resample_nonpositive_rates_return_empty() {
        // A format that parses with the default rate 0 used to make
        // `from as f64 / to as f64` a zero divisor: the output length
        // saturated to usize::MAX and the collect panicked with
        // "capacity overflow". Non-positive rates are malformed, so the
        // frame is dropped like an unsupported format instead.
        assert!(resample_linear(&[1.0, 2.0, 3.0], 0, 16000).is_empty());
        assert!(resample_linear(&[1.0, 2.0, 3.0], 16000, 0).is_empty());
        assert!(resample_linear(&[1.0, 2.0, 3.0], -1, 16000).is_empty());
        // Both rates zero with no input: empty in, empty out.
        assert!(resample_linear(&[], 0, 0).is_empty());
    }

    #[test]
    fn resample_single_sample_upsampled_holds_last_value() {
        // Two output positions both map into the single input sample.
        let out = resample_linear(&[7.0], 8000, 16000);
        assert_close(&out, &[7.0, 7.0], 0.0);
    }

    #[test]
    fn resample_48k_to_16k_picks_every_third() {
        // Integer ratio with zero fraction: exact picks, no averaging.
        let out = resample_linear(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 48000, 16000);
        assert_close(&out, &[0.0, 3.0], 0.0);
    }

    #[test]
    fn resample_non_integer_ratio_tracks_ramp() {
        // 44.1 kHz -> 16 kHz: ratio 2.75625. A linear ramp must come out
        // as the same ramp sampled at the new grid (linear interpolation
        // of a linear function is exact up to f32 rounding).
        let ramp: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let out = resample_linear(&ramp, 44100, 16000);
        let ratio = 44100.0 / 16000.0;
        assert_eq!(out.len(), (100.0 / ratio) as usize); // 36
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - i as f32 * ratio as f32).abs() <= 0.01,
                "sample {i}: expected ~{}, got {}",
                i as f32 * ratio as f32,
                v
            );
        }
    }
}
