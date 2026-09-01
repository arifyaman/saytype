use anyhow::{bail, Context, Result};
use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig};
use std::path::{Path, PathBuf};
use std::time::Instant;

const SAMPLE_RATE: i32 = 16000;

pub struct Asr {
    recognizer: OnlineRecognizer,
}

struct ModelPaths {
    encoder: String,
    decoder: String,
    joiner: String,
    tokens: String,
}

impl Asr {
    pub fn new(models_dir: &Path, num_threads: i32) -> Result<Self> {
        let paths = Self::resolve_model_paths(models_dir)?;
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
        Ok(Self { recognizer })
    }

    fn resolve_model_paths(models_dir: &Path) -> Result<ModelPaths> {
        // Model artifacts get reorganized between releases, so instead of
        // hardcoding exact filenames we scan for a directory that contains an
        // encoder/decoder/joiner ONNX pair plus tokens.txt.
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
            let Some(encoder) = find_onnx(dir, "encoder") else {
                continue;
            };
            let Some(decoder) = find_onnx(dir, "decoder") else {
                continue;
            };
            let Some(joiner) = find_onnx(dir, "joiner") else {
                continue;
            };
            let tokens = dir.join("tokens.txt");
            if tokens.is_file() {
                return Ok(ModelPaths {
                    encoder: encoder.to_string_lossy().into(),
                    decoder: decoder.to_string_lossy().into(),
                    joiner: joiner.to_string_lossy().into(),
                    tokens: tokens.to_string_lossy().into(),
                });
            }
        }

        bail!(
            "no streaming zipformer model found under {:?}. Expected a \
             sherpa-onnx-streaming-zipformer-* directory containing \
             encoder/decoder/joiner *.onnx files and tokens.txt. \
             Run scripts/download-models.sh first.",
            models_dir
        )
    }

    pub fn transcribe(&self, samples: &[f32]) -> (String, std::time::Duration) {
        let start = Instant::now();
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(SAMPLE_RATE, samples);
        stream.input_finished();
        while self.recognizer.is_ready(&stream) {
            self.recognizer.decode(&stream);
        }
        let text = self
            .recognizer
            .get_result(&stream)
            .map(|r| r.text)
            .unwrap_or_default();
        (text.trim().to_string(), start.elapsed())
    }
}

/// Find a `*{stem}*.onnx` file in `dir`, preferring non-int8 variants.
fn find_onnx(dir: &Path, stem: &str) -> Option<PathBuf> {
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
