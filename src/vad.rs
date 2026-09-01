use anyhow::{bail, Context, Result};
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};
use std::path::Path;

const SAMPLE_RATE: i32 = 16000;
const VAD_BUFFER_SECONDS: f32 = 300.0;

/// Context padding (ms) added around each VAD segment before it reaches ASR.
/// The VAD trims segments to the detected speech boundaries: only ~70 ms of
/// pre-roll survives, and the end is cut at the first dip below the VAD's
/// negative threshold - which lands mid-word whenever a word has a soft tail,
/// amputating the rest of that word. Both paddings are sliced from real
/// captured audio (see `AudioRing`): pre-padding restores the first word's
/// attack, post-padding reaches past the mid-word dip to capture the last
/// word's tail plus the trailing silence the decoder needs to commit it.
pub const PRE_PAD_MS: i32 = 300;
pub const POST_PAD_MS: i32 = 800;

/// Rolling buffer of recent audio, addressable by absolute input-sample index.
/// Lets the daemon slice the real audio around a VAD segment (which is only
/// reported after a pause, once its pre-context would otherwise be discarded).
#[derive(Debug)]
pub struct AudioRing {
    buf: Vec<f32>,
    total: i64,
}

impl AudioRing {
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0.0; capacity],
            total: 0,
        }
    }

    pub fn push(&mut self, samples: &[f32]) {
        let cap = self.buf.len();
        for s in samples {
            self.buf[(self.total as usize) % cap] = *s;
            self.total += 1;
        }
    }

    /// Return audio over absolute sample range `[start, end)`, zero-filling any
    /// part that precedes the start of capture (session just began).
    pub fn slice(&self, start: i64, end: i64) -> Vec<f32> {
        let len = (end - start).max(0) as usize;
        let mut out = vec![0.0f32; len];
        if len == 0 {
            return out;
        }
        let lo = start.max(0);
        let hi = end.min(self.total);
        for i in lo..hi {
            out[(i - start) as usize] = self.buf[(i as usize) % self.buf.len()];
        }
        out
    }
}

#[derive(Debug, Clone)]
pub struct VadParams {
    pub threshold: f32,
    pub min_silence_duration: f32,
    pub min_speech_duration: f32,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_silence_duration: 0.8,
            min_speech_duration: 0.25,
        }
    }
}

pub struct VadConfig {
    pub model_path: String,
    pub params: VadParams,
}

impl VadConfig {
    pub fn from_models_dir(models_dir: &Path) -> Self {
        Self {
            model_path: models_dir
                .join("silero_vad.onnx")
                .to_string_lossy()
                .into_owned(),
            params: VadParams::default(),
        }
    }
}

pub struct Vad {
    inner: VoiceActivityDetector,
}

impl Vad {
    pub fn new(config: &VadConfig) -> Result<Self> {
        let vad_config = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(config.model_path.clone()),
                threshold: config.params.threshold,
                min_silence_duration: config.params.min_silence_duration,
                min_speech_duration: config.params.min_speech_duration,
                window_size: 512,
                // Hard bound on segment length so the daemon's lookback ring
                // (30s) always covers segment + padding. Natural pauses split
                // segments far earlier via min_silence_duration.
                max_speech_duration: 20.0,
            },
            sample_rate: SAMPLE_RATE,
            num_threads: 1,
            ..Default::default()
        };
        let inner = VoiceActivityDetector::create(&vad_config, VAD_BUFFER_SECONDS)
            .with_context(|| format!("failed to create VAD from {:?}", config.model_path))?;
        Ok(Self { inner })
    }

    pub fn feed(&self, samples: &[f32]) {
        self.inner.accept_waveform(samples);
    }

    /// Return the next finalized segment as `(samples, start)`, where `start`
    /// is the absolute input-sample index of the segment's first sample. The
    /// start lets the daemon fetch real context audio around the segment from
    /// an `AudioRing` (see `PRE_PAD_MS` / `POST_PAD_MS`).
    pub fn take_segment(&self) -> Option<(Vec<f32>, i64)> {
        let seg = self.inner.front()?;
        let samples = seg.samples().to_vec();
        let start = seg.start() as i64;
        self.inner.pop();
        Some((samples, start))
    }

    pub fn flush(&self) {
        self.inner.flush();
    }
}

pub fn check_model_file(path: &Path) -> Result<()> {
    if !path.is_file() {
        bail!(
            "VAD model not found at {:?}. Run scripts/download-models.sh first.",
            path
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::AudioRing;

    #[test]
    fn slice_returns_pushed_range() {
        let mut ring = AudioRing::new(1000);
        let data: Vec<f32> = (0..500).map(|i| i as f32).collect();
        ring.push(&data);
        let got = ring.slice(100, 200);
        assert_eq!(got.len(), 100);
        assert_eq!(got[0], 100.0);
        assert_eq!(got[99], 199.0);
    }

    #[test]
    fn slice_zero_fills_before_capture_start() {
        let mut ring = AudioRing::new(1000);
        let data: Vec<f32> = (0..500).map(|i| i as f32).collect();
        ring.push(&data);
        // Range begins 50 samples before capture started: first 50 are zeros.
        let got = ring.slice(-50, 50);
        assert_eq!(got.len(), 100);
        assert!(got[..50].iter().all(|&v| v == 0.0));
        assert_eq!(got[50], 0.0);
        assert_eq!(got[99], 49.0);
    }

    #[test]
    fn wraparound_keeps_most_recent_window() {
        let mut ring = AudioRing::new(100);
        // Push 3x capacity; only the last 100 (indices 200..300) remain valid.
        let data: Vec<f32> = (0..300).map(|i| i as f32).collect();
        ring.push(&data);
        let got = ring.slice(250, 300);
        assert_eq!(got.len(), 50);
        assert_eq!(got[0], 250.0);
        assert_eq!(got[49], 299.0);
        // Oldest retained index is total - cap = 200.
        let oldest = ring.slice(200, 201);
        assert_eq!(oldest[0], 200.0);
    }
}
