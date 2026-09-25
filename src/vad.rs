use anyhow::{bail, Context, Result};
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};
use std::path::Path;

const SAMPLE_RATE: i32 = 16000;
// The VAD's internal speech buffer only ever holds one in-progress segment
// (the buffer is emptied when a segment is popped), and `max_speech_duration`
// caps a segment at 20 s. 3x that headroom is enough; 300 s wasted ~19 MB
// per session for nothing.
const VAD_BUFFER_SECONDS: f32 = 60.0;

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
    /// `capacity` is clamped to at least 1: a zero-length buffer would make
    /// the index arithmetic in `push`/`slice` divide by zero and panic.
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0.0; capacity.max(1)],
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

    /// Return audio over absolute sample range `[start, end)`. Any part that
    /// is not available - before the start of capture, past the latest
    /// sample, or older than the ring's retained window - is zero-filled
    /// (never stale wrapped-around data).
    pub fn slice(&self, start: i64, end: i64) -> Vec<f32> {
        let len = (end - start).max(0) as usize;
        let mut out = vec![0.0f32; len];
        if len == 0 {
            return out;
        }
        // Oldest index still held in the ring; anything older is gone.
        let lo = start.max(0).max(self.total - self.buf.len() as i64);
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

    /// True while a segment is in progress: speech has started and the
    /// `min_silence_duration` of trailing silence that finalizes the segment
    /// has not elapsed yet.
    pub fn detected(&self) -> bool {
        self.inner.detected()
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
    fn slice_zero_fills_end_past_capture() {
        let mut ring = AudioRing::new(1000);
        let data: Vec<f32> = (0..500).map(|i| i as f32).collect();
        ring.push(&data);
        // Range reaches 100 samples past the latest: tail is zeros. This is
        // the POST_PAD case at the very end of a session.
        let got = ring.slice(450, 600);
        assert_eq!(got.len(), 150);
        assert_eq!(got[0], 450.0);
        assert_eq!(got[49], 499.0);
        assert!(got[50..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn slice_zero_fills_out_of_window_not_stale() {
        let mut ring = AudioRing::new(100);
        // Push 3x capacity; indices 0..200 have scrolled out of the ring.
        let data: Vec<f32> = (0..300).map(|i| i as f32).collect();
        ring.push(&data);
        // A range partly outside the retained window must zero-fill the
        // stale part, not return wrapped-around (new) samples from the same
        // buffer slots.
        let got = ring.slice(150, 250);
        assert_eq!(got.len(), 100);
        assert!(got[..50].iter().all(|&v| v == 0.0));
        assert_eq!(got[50], 200.0);
        assert_eq!(got[99], 249.0);
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

    #[test]
    fn slice_zero_length_and_inverted_ranges_are_empty() {
        let mut ring = AudioRing::new(100);
        let data: Vec<f32> = (0..50).map(|i| i as f32).collect();
        ring.push(&data);
        assert!(ring.slice(10, 10).is_empty());
        assert!(ring.slice(30, 10).is_empty());
    }

    #[test]
    fn slice_on_unused_ring_zero_fills() {
        // A fresh ring holds no samples at all: any range is zero-filled.
        let ring = AudioRing::new(100);
        let got = ring.slice(0, 10);
        assert_eq!(got.len(), 10);
        assert!(got.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn slice_entirely_before_capture_is_all_zeros() {
        let mut ring = AudioRing::new(100);
        let data: Vec<f32> = (0..50).map(|i| i as f32).collect();
        ring.push(&data);
        // Both endpoints before the first sample: nothing is available.
        let got = ring.slice(-20, -5);
        assert_eq!(got.len(), 15);
        assert!(got.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn slice_end_exactly_at_total_has_no_tail_padding() {
        let mut ring = AudioRing::new(100);
        let data: Vec<f32> = (0..50).map(|i| i as f32).collect();
        ring.push(&data);
        // `end == total` is the boundary: the whole range is real audio,
        // no zero tail (unlike the POST_PAD case, which reaches past it).
        let got = ring.slice(40, 50);
        assert_eq!(got.len(), 10);
        assert_eq!(got[0], 40.0);
        assert_eq!(got[9], 49.0);
    }

    #[test]
    fn slice_start_past_total_is_all_zeros() {
        let mut ring = AudioRing::new(100);
        let data: Vec<f32> = (0..50).map(|i| i as f32).collect();
        ring.push(&data);
        let got = ring.slice(60, 80);
        assert_eq!(got.len(), 20);
        assert!(got.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn slice_exact_full_retained_window() {
        let mut ring = AudioRing::new(100);
        let data: Vec<f32> = (0..150).map(|i| i as f32).collect();
        ring.push(&data);
        // The retained window is exactly indices 50..150: every sample in
        // the range is real (no zero fill on either side).
        let got = ring.slice(50, 150);
        assert_eq!(got.len(), 100);
        assert_eq!(got[0], 50.0);
        assert_eq!(got[99], 149.0);
    }

    #[test]
    fn push_empty_slice_is_a_noop() {
        let mut ring = AudioRing::new(100);
        ring.push(&[]);
        // Nothing was recorded: the ring still behaves like a fresh one.
        assert!(ring.slice(0, 10).iter().all(|&v| v == 0.0));
        ring.push(&[7.0]);
        assert_eq!(ring.slice(0, 1), vec![7.0]);
    }

    #[test]
    fn zero_capacity_ring_stays_safe() {
        // Degenerate construction must not panic in push/slice; it behaves
        // like a one-sample ring (only the newest sample is addressable).
        let mut ring = AudioRing::new(0);
        ring.push(&[1.0, 2.0, 3.0]);
        // Only the newest sample (absolute index 2) is retained (oldest
        // retained index is total - cap = 3 - 1 = 2).
        assert_eq!(ring.slice(2, 3), vec![3.0]);
        let got = ring.slice(0, 3);
        assert_eq!(got[0], 0.0); // index 0 fell out of the 1-sample window
        assert_eq!(got[1], 0.0); // index 1 fell out too
        assert_eq!(got[2], 3.0); // the newest sample is retained
    }
}
