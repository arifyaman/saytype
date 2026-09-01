//! ASR backend.
//!
//! Two model families are supported behind one interface, auto-detected from
//! `models/`:
//!
//! - **Moonshine v2** (preferred when present): a
//!   `sherpa-onnx-moonshine-*` dir with `encoder_model.*` +
//!   `decoder_model_merged.*` (`.onnx` or `.ort`) + `tokens.txt`. Runs as an
//!   `OfflineRecognizer` - batch per finalized segment. Outputs proper casing
//!   and punctuation.
//! - **Streaming Zipformer** (fallback): a `sherpa-onnx-streaming-zipformer-*`
//!   dir with encoder/decoder/joiner `*epoch*.onnx` + `tokens.txt`. Runs as an
//!   `OnlineRecognizer`; this is the only backend that can produce live
//!   partials (required for mvp2 live typing).
//!
//! Both models can coexist in `models/`; Moonshine wins. Remove its directory
//! (or the zipformer one) to switch.

use anyhow::{bail, Context, Result};
use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OnlineRecognizer, OnlineRecognizerConfig,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

const SAMPLE_RATE: i32 = 16000;

pub struct Asr {
    backend: Backend,
}

enum Backend {
    Streaming(OnlineRecognizer),
    Moonshine(OfflineRecognizer),
}

impl Asr {
    pub fn new(models_dir: &Path, num_threads: i32) -> Result<Self> {
        if let Some(paths) = Self::find_moonshine(models_dir) {
            tracing::info!("ASR backend: moonshine v2 in {:?}", paths.dir);
            let mut config = OfflineRecognizerConfig::default();
            config.model_config.moonshine.encoder = Some(paths.encoder);
            config.model_config.moonshine.merged_decoder = Some(paths.merged_decoder);
            config.model_config.tokens = Some(paths.tokens);
            config.model_config.num_threads = num_threads;
            config.model_config.provider = Some("cpu".to_string());
            let recognizer = OfflineRecognizer::create(&config)
                .context("failed to create Moonshine OfflineRecognizer (check model files)")?;
            return Ok(Self {
                backend: Backend::Moonshine(recognizer),
            });
        }

        let paths = Self::resolve_zipformer_paths(models_dir)?;
        tracing::info!("ASR backend: streaming zipformer in {:?}", paths.dir);
        let mut config = OnlineRecognizerConfig::default();
        config.model_config.transducer.encoder = Some(paths.encoder);
        config.model_config.transducer.decoder = Some(paths.decoder);
        config.model_config.transducer.joiner = Some(paths.joiner);
        config.model_config.tokens = Some(paths.tokens);
        config.model_config.num_threads = num_threads;
        config.model_config.provider = Some("cpu".to_string());
        config.decoding_method = Some("greedy_search".to_string());
        config.enable_endpoint = false;

        let recognizer = OnlineRecognizer::create(&config)
            .context("failed to create OnlineRecognizer (check model files)")?;
        Ok(Self {
            backend: Backend::Streaming(recognizer),
        })
    }

    /// Find a Moonshine v2 model dir: `encoder_model*` +
    /// `decoder_model_merged*` (`.onnx` or `.ort`) + `tokens.txt`.
    fn find_moonshine(models_dir: &Path) -> Option<MoonshinePaths> {
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(models_dir)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();

        for dir in &dirs {
            let Some(encoder) = find_model(dir, "encoder_model") else {
                continue;
            };
            let Some(merged_decoder) = find_model(dir, "decoder_model_merged") else {
                continue;
            };
            let tokens = dir.join("tokens.txt");
            if tokens.is_file() {
                return Some(MoonshinePaths {
                    dir: dir.clone(),
                    encoder: encoder.to_string_lossy().into(),
                    merged_decoder: merged_decoder.to_string_lossy().into(),
                    tokens: tokens.to_string_lossy().into(),
                });
            }
        }
        None
    }

    fn resolve_zipformer_paths(models_dir: &Path) -> Result<ZipformerPaths> {
        // Model artifacts get reorganized between releases, so instead of
        // hardcoding exact filenames we scan for a directory that contains an
        // encoder/decoder/joiner ONNX triple plus tokens.txt.
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(models_dir)
            .with_context(|| format!("reading models dir {:?}", models_dir))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_dir()
                    && p.file_name()
                        .map(|n| {
                            n.to_string_lossy()
                                .starts_with("sherpa-onnx-streaming-zipformer")
                        })
                        .unwrap_or(false)
            })
            .collect();
        dirs.sort();

        for dir in &dirs {
            let Some(encoder) = find_zipformer_onnx(dir, "encoder") else {
                continue;
            };
            let Some(decoder) = find_zipformer_onnx(dir, "decoder") else {
                continue;
            };
            let Some(joiner) = find_zipformer_onnx(dir, "joiner") else {
                continue;
            };
            let tokens = dir.join("tokens.txt");
            if tokens.is_file() {
                return Ok(ZipformerPaths {
                    dir: dir.clone(),
                    encoder: encoder.to_string_lossy().into(),
                    decoder: decoder.to_string_lossy().into(),
                    joiner: joiner.to_string_lossy().into(),
                    tokens: tokens.to_string_lossy().into(),
                });
            }
        }

        bail!(
            "no ASR model found under {:?}. Expected a sherpa-onnx-moonshine-* \
             directory (encoder_model.*, decoder_model_merged.*, tokens.txt) or a \
             sherpa-onnx-streaming-zipformer-* directory (encoder/decoder/joiner \
             *epoch*.onnx, tokens.txt). Run scripts/download-models.sh first.",
            models_dir
        )
    }

    pub fn transcribe(&self, samples: &[f32]) -> (String, std::time::Duration) {
        let start = Instant::now();
        let text = match &self.backend {
            Backend::Streaming(recognizer) => {
                let stream = recognizer.create_stream();
                stream.accept_waveform(SAMPLE_RATE, samples);
                stream.input_finished();
                while recognizer.is_ready(&stream) {
                    recognizer.decode(&stream);
                }
                recognizer
                    .get_result(&stream)
                    .map(|r| r.text)
                    .unwrap_or_default()
            }
            Backend::Moonshine(recognizer) => {
                let stream = recognizer.create_stream();
                stream.accept_waveform(SAMPLE_RATE, samples);
                recognizer.decode(&stream);
                stream.get_result().map(|r| r.text).unwrap_or_default()
            }
        };
        (text.trim().to_string(), start.elapsed())
    }
}

struct MoonshinePaths {
    dir: PathBuf,
    encoder: String,
    merged_decoder: String,
    tokens: String,
}

struct ZipformerPaths {
    dir: PathBuf,
    encoder: String,
    decoder: String,
    joiner: String,
    tokens: String,
}

/// Find a `*{stem}*` file in `dir` with an `.onnx` or `.ort` extension.
fn find_model(dir: &Path, stem: &str) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            name.contains(stem) && (name.ends_with(".onnx") || name.ends_with(".ort"))
        })
        .next()
}

/// Find a `*{stem}*.onnx` file in `dir` (zipformer layout), preferring
/// non-int8 variants.
fn find_zipformer_onnx(dir: &Path, stem: &str) -> Option<PathBuf> {
    let mut names: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name())
        .filter_map(|n| {
            let s = n.to_string_lossy();
            if s.ends_with(".onnx") && s.contains(stem) && s.contains("epoch") {
                Some(n)
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(PathBuf::from)
        .collect();
    // Prefer a non-int8 variant if both are present.
    names.sort_by_key(|p| p.to_string_lossy().contains("int8"));
    names.into_iter().next().map(|n| dir.join(n))
}

pub fn check_models_dir(models_dir: &Path) -> Result<()> {
    if !models_dir.is_dir() {
        bail!(
            "models directory not found at {:?}. Run scripts/download-models.sh first.",
            models_dir
        );
    }
    Ok(())
}
