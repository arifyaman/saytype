//! ASR backend.
//!
//! Three model families are supported behind one interface, selected from
//! `models/` and a `BackendSelection`:
//!
//! - **Nemotron Speech Streaming EN 0.6B** (mvp2 default): a
//!   `sherpa-onnx-nemotron-speech-streaming-en-*` dir with
//!   encoder/decoder/joiner `*.onnx` + `tokens.txt`. Runs as an
//!   `OnlineRecognizer` with `model_type = "nemo_transducer"` (cache-aware
//!   FastConformer + RNNT). Trained on ~530k h of audio; emits live
//!   partials and natively cased, punctuated text.
//! - **Streaming Zipformer**: a `sherpa-onnx-streaming-zipformer-*` dir
//!   with encoder/decoder/joiner `*epoch*.onnx` + `tokens.txt`. Runs as an
//!   `OnlineRecognizer`; also produces live partials. Its raw output is
//!   all-caps and unpunctuated, so it is post-processed (see below).
//! - **Moonshine v2**: a `sherpa-onnx-moonshine-*` dir with
//!   `encoder_model.*` + `decoder_model_merged.*` (`.onnx` or `.ort`) +
//!   `tokens.txt`. Runs as an `OfflineRecognizer` - batch per finalized
//!   segment. Outputs proper casing and punctuation. Cannot produce live
//!   partials (offline model).
//!
//! All model dirs can coexist in `models/`. `BackendSelection::Auto`
//! prefers Nemotron, then Zipformer, then Moonshine (the mvp2 default is
//! the Nemotron streaming backend); `Streaming` picks the best streaming
//! backend (Nemotron over Zipformer); `Zipformer`/`Moonshine` force a
//! backend and error when its model dir is missing.
//!
//! **Punctuation/casing** (streaming path only): an optional
//! `sherpa-onnx-online-punct-*` dir (`model.int8.onnx`/`model.onnx` +
//! `bpe.vocab`) is auto-detected. Zipformer output (all-caps) is always
//! lowercased and run through the `OnlinePunctuation` model when present.
//! Nemotron output is natively cased/punctuated; it is lowercased + run
//! through the punct model when present (denser punctuation), and left
//! native otherwise. Moonshine output is never post-processed.

use anyhow::{bail, Context, Result};
use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OnlinePunctuation, OnlinePunctuationConfig,
    OnlinePunctuationModelConfig, OnlineRecognizer, OnlineRecognizerConfig, OnlineStream,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

const SAMPLE_RATE: i32 = 16000;

/// The minimum input the batch `transcribe` will actually run the model on:
/// 100 ms, the same threshold the live pipeline's blip filter uses to drop
/// sub-100 ms VAD segments as noise. Nothing shorter can be a complete
/// utterance, and feeding degenerate shapes to the model is not free:
/// Moonshine's onnxruntime conv stack logs a scary "Invalid input shape"
/// error block for short inputs (verified live, even at 512 samples) instead
/// of quietly returning empty.
const MIN_TRANSCRIBE_SAMPLES: usize = SAMPLE_RATE as usize / 10;

/// Which ASR backend to load. `Auto` prefers Nemotron streaming, then
/// Zipformer streaming, then Moonshine (the mvp2 default is the Nemotron
/// streaming backend); `Streaming` picks the best streaming backend
/// (Nemotron over Zipformer); the explicit variants force a backend and
/// error when its model dir is missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSelection {
    Auto,
    Streaming,
    Zipformer,
    Moonshine,
}

/// Which backend a loaded `Asr` actually uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsrKind {
    Nemotron,
    Zipformer,
    Moonshine,
}

pub struct Asr {
    backend: Backend,
    /// Online punctuation/casing model (streaming backends only). `None`
    /// when the model dir is absent; see the module docs for how that
    /// affects each backend's output.
    punct: Option<OnlinePunctuation>,
}

enum Backend {
    Nemotron(OnlineRecognizer),
    Zipformer(OnlineRecognizer),
    Moonshine(OfflineRecognizer),
}

impl Asr {
    pub fn new(models_dir: &Path, num_threads: i32, selection: BackendSelection) -> Result<Self> {
        // mvp2 default: streaming first (it alone can produce live
        // partials); Nemotron is preferred over Zipformer; Moonshine stays
        // as the batch fallback.
        // A discovered-but-corrupt model is a hard error in every mode:
        // silently falling back would hide the broken download from the
        // user (a *missing* model is the legitimate fallback condition).
        let is_corrupt =
            |e: &anyhow::Error| e.downcast_ref::<crate::onnx::CorruptModel>().is_some();
        let backend = match selection {
            BackendSelection::Auto => match Self::create_nemotron(models_dir, num_threads) {
                Ok(b) => b,
                Err(e) if is_corrupt(&e) => return Err(e),
                Err(e) => {
                    tracing::info!("no usable Nemotron streaming model ({e}); trying Zipformer");
                    match Self::create_zipformer(models_dir, num_threads) {
                        Ok(b) => b,
                        Err(e2) if is_corrupt(&e2) => return Err(e2),
                        Err(e2) => {
                            tracing::info!(
                                "no usable streaming Zipformer model ({e2}); trying Moonshine"
                            );
                            Self::create_moonshine(models_dir, num_threads)?
                        }
                    }
                }
            },
            BackendSelection::Streaming => match Self::create_nemotron(models_dir, num_threads) {
                Ok(b) => b,
                Err(e) if is_corrupt(&e) => return Err(e),
                Err(e) => {
                    tracing::info!(
                        "no usable Nemotron streaming model ({e}); falling back to Zipformer"
                    );
                    Self::create_zipformer(models_dir, num_threads)?
                }
            },
            BackendSelection::Zipformer => Self::create_zipformer(models_dir, num_threads)?,
            BackendSelection::Moonshine => Self::create_moonshine(models_dir, num_threads)?,
        };
        // The punct model is only useful for streaming backends.
        let punct = match &backend {
            Backend::Nemotron(_) | Backend::Zipformer(_) => {
                Self::create_punct(models_dir, num_threads)
            }
            Backend::Moonshine(_) => None,
        };
        Ok(Self { backend, punct })
    }

    pub fn kind(&self) -> AsrKind {
        match &self.backend {
            Backend::Nemotron(_) => AsrKind::Nemotron,
            Backend::Zipformer(_) => AsrKind::Zipformer,
            Backend::Moonshine(_) => AsrKind::Moonshine,
        }
    }

    /// True for the backends that produce live partials.
    pub fn is_streaming(&self) -> bool {
        matches!(self.kind(), AsrKind::Nemotron | AsrKind::Zipformer)
    }

    /// A live streaming session, present only for the streaming backends.
    /// The session borrows the shared recognizer and the optional punctuation
    /// model, and owns its own `OnlineStream`. `normalize` marks backends
    /// whose raw output must always be lowercased (Zipformer: all-caps);
    /// backends with native casing (Nemotron) are only polished when the
    /// punct model is available.
    pub fn streaming_session(&self) -> Option<StreamingSession<'_>> {
        match &self.backend {
            Backend::Nemotron(recognizer) => Some(StreamingSession::new(
                recognizer,
                self.punct.as_ref(),
                false,
            )),
            Backend::Zipformer(recognizer) => {
                Some(StreamingSession::new(recognizer, self.punct.as_ref(), true))
            }
            Backend::Moonshine(_) => None,
        }
    }

    fn create_moonshine(models_dir: &Path, num_threads: i32) -> Result<Backend> {
        let paths = Self::find_moonshine(models_dir).with_context(|| {
            format!("no Moonshine model found under {:?} (a sherpa-onnx-moonshine-* dir with encoder_model.*, decoder_model_merged.*, tokens.txt)", models_dir)
        })?;
        tracing::info!("ASR backend: moonshine v2 in {:?}", paths.dir);
        // .onnx files get the corrupt-model guard (a corrupt .ort file is
        // still possible but ORT's binary format has no cheap envelope
        // check; the default install uses the Nemotron .onnx stack).
        for (path, what) in [
            (&paths.encoder, "Moonshine encoder"),
            (&paths.merged_decoder, "Moonshine merged decoder"),
        ] {
            if path.ends_with(".onnx") {
                crate::onnx::check_onnx_file(Path::new(path), what)?;
            }
        }
        let mut config = OfflineRecognizerConfig::default();
        config.model_config.moonshine.encoder = Some(paths.encoder);
        config.model_config.moonshine.merged_decoder = Some(paths.merged_decoder);
        config.model_config.tokens = Some(paths.tokens);
        config.model_config.num_threads = num_threads;
        config.model_config.provider = Some("cpu".to_string());
        let recognizer = OfflineRecognizer::create(&config)
            .context("failed to create Moonshine OfflineRecognizer (check model files)")?;
        Ok(Backend::Moonshine(recognizer))
    }

    fn create_nemotron(models_dir: &Path, num_threads: i32) -> Result<Backend> {
        let paths = Self::find_nemotron(models_dir).with_context(|| {
            format!("no Nemotron streaming model found under {:?} (a sherpa-onnx-nemotron-speech-streaming-en-* dir with encoder/decoder/joiner *.onnx + tokens.txt)", models_dir)
        })?;
        tracing::info!(
            "ASR backend: nemotron speech streaming (en 0.6b) in {:?}",
            paths.dir
        );
        for (path, what) in [
            (&paths.encoder, "Nemotron encoder"),
            (&paths.decoder, "Nemotron decoder"),
            (&paths.joiner, "Nemotron joiner"),
        ] {
            crate::onnx::check_onnx_file(Path::new(path), what)?;
        }
        let mut config = OnlineRecognizerConfig::default();
        config.model_config.transducer.encoder = Some(paths.encoder);
        config.model_config.transducer.decoder = Some(paths.decoder);
        config.model_config.transducer.joiner = Some(paths.joiner);
        config.model_config.tokens = Some(paths.tokens);
        config.model_config.num_threads = num_threads;
        config.model_config.provider = Some("cpu".to_string());
        // Cache-aware FastConformer + RNNT (NVIDIA NeMo export layout).
        config.model_config.model_type = Some("nemo_transducer".to_string());
        config.decoding_method = Some("greedy_search".to_string());
        // Endpointing stays off: VAD finalization is the commit boundary.
        config.enable_endpoint = false;

        let recognizer = OnlineRecognizer::create(&config)
            .context("failed to create Nemotron OnlineRecognizer (check model files)")?;
        Ok(Backend::Nemotron(recognizer))
    }

    fn create_zipformer(models_dir: &Path, num_threads: i32) -> Result<Backend> {
        let paths = Self::resolve_zipformer_paths(models_dir)?;
        tracing::info!("ASR backend: streaming zipformer in {:?}", paths.dir);
        for (path, what) in [
            (&paths.encoder, "Zipformer encoder"),
            (&paths.decoder, "Zipformer decoder"),
            (&paths.joiner, "Zipformer joiner"),
        ] {
            crate::onnx::check_onnx_file(Path::new(path), what)?;
        }
        let mut config = OnlineRecognizerConfig::default();
        config.model_config.transducer.encoder = Some(paths.encoder);
        config.model_config.transducer.decoder = Some(paths.decoder);
        config.model_config.transducer.joiner = Some(paths.joiner);
        config.model_config.tokens = Some(paths.tokens);
        config.model_config.num_threads = num_threads;
        config.model_config.provider = Some("cpu".to_string());
        config.decoding_method = Some("greedy_search".to_string());
        // Endpointing stays off: VAD finalization is the commit boundary.
        config.enable_endpoint = false;

        let recognizer = OnlineRecognizer::create(&config)
            .context("failed to create Zipformer OnlineRecognizer (check model files)")?;
        Ok(Backend::Zipformer(recognizer))
    }

    /// Load the optional online punctuation/casing model. Returns `None`
    /// (with a warning) when the model dir is absent - the streaming path
    /// still works, it just emits lowercased text without punctuation.
    fn create_punct(models_dir: &Path, num_threads: i32) -> Option<OnlinePunctuation> {
        let Some(paths) = Self::find_online_punct(models_dir) else {
            tracing::warn!(
                "no online punctuation model under {:?} (a sherpa-onnx-online-punct-* dir \
                 with model.int8.onnx + bpe.vocab); streaming output will be lowercased \
                 but unpunctuated. Run scripts/download-models.sh to add it.",
                models_dir
            );
            return None;
        };
        tracing::info!(
            "online punctuation model in {:?} ({})",
            paths.dir,
            if paths.model.contains("int8") {
                "int8"
            } else {
                "fp32"
            }
        );
        // The punct model is optional: a corrupt one degrades to no
        // punctuation (a warning) instead of failing the whole load.
        if let Err(e) =
            crate::onnx::check_onnx_file(Path::new(&paths.model), "online punctuation model")
        {
            tracing::warn!("online punctuation model unusable: {e}");
            return None;
        }
        let config = OnlinePunctuationConfig {
            model: OnlinePunctuationModelConfig {
                cnn_bilstm: Some(paths.model),
                bpe_vocab: Some(paths.vocab),
                num_threads,
                ..Default::default()
            },
        };
        OnlinePunctuation::create(&config)
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

    pub(crate) fn resolve_zipformer_paths(models_dir: &Path) -> Result<ZipformerPaths> {
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

    /// Find the optional online punctuation model dir: a
    /// `sherpa-onnx-online-punct-*` dir with a `bpe.vocab` and a
    /// `model.int8.onnx` (preferred) or `model.onnx`.
    fn find_online_punct(models_dir: &Path) -> Option<PunctPaths> {
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(models_dir)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_dir()
                    && p.file_name()
                        .map(|n| n.to_string_lossy().starts_with("sherpa-onnx-online-punct"))
                        .unwrap_or(false)
            })
            .collect();
        dirs.sort();

        for dir in &dirs {
            let vocab = dir.join("bpe.vocab");
            if !vocab.is_file() {
                continue;
            }
            // Prefer the int8 model (smaller, equally fast here); fall back
            // to the fp32 one.
            let model = ["model.int8.onnx", "model.onnx"]
                .iter()
                .map(|name| dir.join(name))
                .find(|p| p.is_file());
            let Some(model) = model else {
                continue;
            };
            return Some(PunctPaths {
                dir: dir.clone(),
                model: model.to_string_lossy().into(),
                vocab: vocab.to_string_lossy().into(),
            });
        }
        None
    }

    /// Find the Nemotron streaming (English) model dir: a
    /// `sherpa-onnx-nemotron-speech-streaming-en-*` dir with
    /// encoder/decoder/joiner ONNX files + `tokens.txt`.
    pub(crate) fn find_nemotron(models_dir: &Path) -> Option<NemotronPaths> {
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(models_dir)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_dir()
                    && p.file_name()
                        .map(|n| {
                            n.to_string_lossy()
                                .starts_with("sherpa-onnx-nemotron-speech-streaming-en")
                        })
                        .unwrap_or(false)
            })
            .collect();
        dirs.sort();

        for dir in &dirs {
            let Some(encoder) = find_onnx_preferring_int8(dir, "encoder") else {
                continue;
            };
            let Some(decoder) = find_onnx_preferring_int8(dir, "decoder") else {
                continue;
            };
            let Some(joiner) = find_onnx_preferring_int8(dir, "joiner") else {
                continue;
            };
            let tokens = dir.join("tokens.txt");
            if tokens.is_file() {
                return Some(NemotronPaths {
                    dir: dir.clone(),
                    encoder: encoder.to_string_lossy().into(),
                    decoder: decoder.to_string_lossy().into(),
                    joiner: joiner.to_string_lossy().into(),
                    tokens: tokens.to_string_lossy().into(),
                });
            }
        }
        None
    }

    pub fn transcribe(&self, samples: &[f32]) -> (String, std::time::Duration) {
        if samples.len() < MIN_TRANSCRIBE_SAMPLES {
            // Degenerate input (an empty or truncated WAV): no utterance to
            // decode; answer without touching the model (see
            // MIN_TRANSCRIBE_SAMPLES).
            return (String::new(), std::time::Duration::ZERO);
        }
        let start = Instant::now();
        let raw = match &self.backend {
            Backend::Nemotron(recognizer) | Backend::Zipformer(recognizer) => {
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
        // Output policy per backend (see the module docs): Zipformer is
        // always lowercased (+ punctuated when the model is present);
        // Nemotron is polished only when the punct model is present, else
        // its native cased/punctuated output is kept; Moonshine is never
        // post-processed.
        let text = match &self.backend {
            Backend::Zipformer(_) => polish(self.punct.as_ref(), &raw),
            Backend::Nemotron(_) => {
                if self.punct.is_some() {
                    polish(self.punct.as_ref(), &raw)
                } else {
                    raw.trim().to_string()
                }
            }
            Backend::Moonshine(_) => raw.trim().to_string(),
        };
        (text, start.elapsed())
    }
}

/// A live streaming ASR session on top of a shared `OnlineRecognizer`.
///
/// One session is one continuous stream of audio for the life of a
/// dictation session. Feed 16 kHz f32 chunks in order; `partial()` returns
/// the hypothesis for the utterance in progress; `commit()` finalizes the
/// current utterance at a boundary (returns its final text) and resets the
/// stream so the session keeps consuming the next utterance.
pub struct StreamingSession<'a> {
    recognizer: &'a OnlineRecognizer,
    stream: OnlineStream,
    punct: Option<&'a OnlinePunctuation>,
    /// Backends with all-caps raw output (Zipformer) must always be
    /// lowercased; backends with native casing (Nemotron) are polished only
    /// when the punct model is available.
    normalize: bool,
}

impl<'a> StreamingSession<'a> {
    pub fn new(
        recognizer: &'a OnlineRecognizer,
        punct: Option<&'a OnlinePunctuation>,
        normalize: bool,
    ) -> Self {
        Self {
            recognizer,
            stream: recognizer.create_stream(),
            punct,
            normalize,
        }
    }

    /// Append a chunk of samples and run all pending decodes.
    pub fn feed(&self, samples: &[f32]) {
        self.stream.accept_waveform(SAMPLE_RATE, samples);
        while self.recognizer.is_ready(&self.stream) {
            self.recognizer.decode(&self.stream);
        }
    }

    /// Apply this backend's output policy to raw recognizer text.
    fn finish(&self, raw: &str) -> String {
        if self.normalize || self.punct.is_some() {
            polish(self.punct, raw)
        } else {
            raw.trim().to_string()
        }
    }

    /// Current partial hypothesis for the utterance in progress, with the
    /// backend's output policy applied.
    pub fn partial(&self) -> String {
        let raw = self
            .recognizer
            .get_result(&self.stream)
            .map(|r| r.text)
            .unwrap_or_default();
        self.finish(&raw)
    }

    /// Finalize the utterance in progress: flush the decoder with end-of-
    /// input semantics, return the final text (policy applied), and reset the
    /// stream so the next utterance starts clean.
    pub fn commit(&self) -> String {
        self.stream.input_finished();
        while self.recognizer.is_ready(&self.stream) {
            self.recognizer.decode(&self.stream);
        }
        let raw = self
            .recognizer
            .get_result(&self.stream)
            .map(|r| r.text)
            .unwrap_or_default();
        self.recognizer.reset(&self.stream);
        self.finish(&raw)
    }
}

/// Normalize raw streaming ASR output for typing/display. Lowercases the text
/// and strips the sentence punctuation marks (comma, period, question mark,
/// exclamation mark, semicolon, colon) that the online-punct model would
/// regenerate, so feeding it pre-punctuated text cannot double-mark. When the
/// online punctuation model is available it restores casing + punctuation;
/// without it the text is just lowercased/stripped (more readable than
/// all-caps). Intra-word punctuation (hyphens, apostrophes) is kept, since it
/// is not the sentence punctuation the model regenerates, so contractions and
/// hyphenated words survive intact.
fn polish(punct: Option<&OnlinePunctuation>, raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut text = String::with_capacity(trimmed.len());
    for ch in trimmed.to_lowercase().chars() {
        if matches!(ch, ',' | '.' | '?' | '!' | ';' | ':') {
            continue;
        }
        text.push(ch);
    }
    match punct.and_then(|p| p.add_punctuation(&text)) {
        Some(p) if !p.trim().is_empty() => p,
        _ => text,
    }
}

struct MoonshinePaths {
    dir: PathBuf,
    encoder: String,
    merged_decoder: String,
    tokens: String,
}

/// `pub(crate)` only so the real-model tests outside this module (daemon)
/// can reuse the canonical dir detection instead of re-implementing the
/// naming rules.
#[derive(Debug)]
pub(crate) struct ZipformerPaths {
    pub(crate) dir: PathBuf,
    encoder: String,
    decoder: String,
    joiner: String,
    tokens: String,
}

struct PunctPaths {
    dir: PathBuf,
    model: String,
    vocab: String,
}

/// `pub(crate)` only so the real-model tests outside this module (daemon)
/// can reuse the canonical dir detection instead of re-implementing the
/// naming rules.
#[derive(Debug)]
pub(crate) struct NemotronPaths {
    pub(crate) dir: PathBuf,
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
        .find(|p| {
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            name.contains(stem) && (name.ends_with(".onnx") || name.ends_with(".ort"))
        })
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

/// Find a `*{stem}*.onnx` file in `dir` (Nemotron layout: no "epoch" in the
/// names), preferring the int8 variants (the official release form; an fp32
/// Nemotron encoder would be ~2.4 GB).
fn find_onnx_preferring_int8(dir: &Path, stem: &str) -> Option<PathBuf> {
    let mut names: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name())
        .filter_map(|n| {
            let s = n.to_string_lossy();
            if s.ends_with(".onnx") && s.contains(stem) {
                Some(n)
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(PathBuf::from)
        .collect();
    names.sort_by_key(|p| !p.to_string_lossy().contains("int8"));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{model_lock, repo_models_dir};
    use std::fs;

    /// Create an empty `models/`-like directory (removed on drop).
    fn models_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    /// Create a subdirectory of a models dir (removed with the parent on drop).
    fn sub(models: &Path, name: &str) -> PathBuf {
        let p = models.join(name);
        fs::create_dir_all(&p).expect("create sub dir");
        p
    }

    /// Create an empty file with the given name in `dir`.
    fn file(dir: &Path, name: &str) {
        fs::write(dir.join(name), b"").expect("write file");
    }

    fn file_name(p: impl AsRef<Path>) -> String {
        p.as_ref()
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    #[test]
    fn polish_without_punct_model_normalizes() {
        // Empty and whitespace-only input.
        assert_eq!(polish(None, ""), "");
        assert_eq!(polish(None, "   \t\n  "), "");
        // Lowercased, surrounding whitespace trimmed, punctuation stripped.
        assert_eq!(polish(None, "  Hello, World!  "), "hello world");
        assert_eq!(polish(None, "STOP; GO: NOW."), "stop go now");
        // Punctuation-only input collapses to nothing.
        assert_eq!(polish(None, "?!...;:"), "");
        // Inner whitespace runs are preserved as-is (single spaces here).
        assert_eq!(polish(None, "a, b"), "a b");
    }

    /// The strip step removes only the sentence punctuation the online-punct
    /// model would regenerate (.,!?;:), so feeding pre-punctuated text cannot
    /// double-mark it. Intra-word punctuation is intentionally preserved: the
    /// Nemotron backend is natively cased + punctuated, so contractions
    /// ("don't") and hyphenated words ("well-known") must reach the model and
    /// the typed text intact, not be shredded by an over-eager strip. (A
    /// hyphen/apostrophe is not sentence punctuation, so the model would
    /// neither add nor expect to re-mark it.)
    #[test]
    fn polish_keeps_intra_word_punctuation_strips_only_sentence_marks() {
        // Contractions keep their apostrophe; hyphenated words keep their hyphens.
        assert_eq!(polish(None, "DON'T STOP"), "don't stop");
        assert_eq!(polish(None, "well-known words"), "well-known words");
        // Sentence marks are still stripped (alongside lowercasing) while the
        // hyphen and apostrophe survive.
        assert_eq!(
            polish(None, "Well-known, don't? Stop!"),
            "well-known don't stop"
        );
        // All-caps hyphenated phrase with a trailing mark.
        assert_eq!(polish(None, "STATE-OF-THE-ART?"), "state-of-the-art");
    }

    #[test]
    fn find_model_matches_stem_and_model_extension() {
        let models = models_dir();
        let d = sub(models.path(), "m");
        file(&d, "encoder_model.onnx");
        file(&d, "decoder_model.ort");
        file(&d, "encoder_model.txt"); // wrong extension: ignored

        assert_eq!(
            find_model(&d, "encoder_model").as_ref().map(file_name),
            Some("encoder_model.onnx".to_string())
        );
        // Stem must be contained in the name.
        assert!(find_model(&d, "decoder").is_some());
        assert!(find_model(&d, "joiner").is_none());

        // Non-existent and empty dirs yield None, not an error.
        assert!(find_model(&models.path().join("no-such-dir"), "encoder_model").is_none());
        assert!(find_model(models.path(), "encoder_model").is_none());
    }

    #[test]
    fn find_zipformer_onnx_requires_epoch_and_prefers_non_int8() {
        let models = models_dir();
        let d = sub(models.path(), "z");
        file(&d, "encoder-epoch-99.int8.onnx");
        // Only int8 present: it is used.
        assert_eq!(
            find_zipformer_onnx(&d, "encoder").as_ref().map(file_name),
            Some("encoder-epoch-99.int8.onnx".to_string())
        );
        // Both present: the non-int8 one wins.
        file(&d, "encoder-epoch-99.onnx");
        assert_eq!(
            find_zipformer_onnx(&d, "encoder").as_ref().map(file_name),
            Some("encoder-epoch-99.onnx".to_string())
        );

        // Files without "epoch" in the name do not match the zipformer layout.
        let d2 = sub(models.path(), "z2");
        file(&d2, "encoder.onnx");
        assert!(find_zipformer_onnx(&d2, "encoder").is_none());
    }

    #[test]
    fn find_onnx_preferring_int8_prefers_int8() {
        let models = models_dir();
        let d = sub(models.path(), "n");
        file(&d, "encoder.onnx");
        assert_eq!(
            find_onnx_preferring_int8(&d, "encoder")
                .as_ref()
                .map(file_name),
            Some("encoder.onnx".to_string())
        );
        file(&d, "encoder.int8.onnx");
        assert_eq!(
            find_onnx_preferring_int8(&d, "encoder")
                .as_ref()
                .map(file_name),
            Some("encoder.int8.onnx".to_string())
        );

        // No match (incl. wrong extension) -> None.
        let d2 = sub(models.path(), "n2");
        file(&d2, "encoder.txt");
        assert!(find_onnx_preferring_int8(&d2, "encoder").is_none());
        assert!(find_onnx_preferring_int8(&d2, "joiner").is_none());
    }

    #[test]
    fn find_moonshine_skips_incomplete_dirs_and_checks_all_parts() {
        let models = models_dir();
        // Sorted first, but missing tokens.txt: must be skipped.
        let broken = sub(models.path(), "sherpa-onnx-moonshine-16k-v1-broken");
        file(&broken, "encoder_model.onnx");
        file(&broken, "decoder_model_merged.onnx");
        assert!(Asr::find_moonshine(models.path()).is_none());

        // Complete dir (sorted after the broken one) is found.
        let ok = sub(models.path(), "sherpa-onnx-moonshine-16k-v2");
        file(&ok, "encoder_model.ort");
        file(&ok, "decoder_model_merged.ort");
        file(&ok, "tokens.txt");
        let p = Asr::find_moonshine(models.path()).expect("complete dir found");
        assert!(p.dir.ends_with("sherpa-onnx-moonshine-16k-v2"));
        assert_eq!(file_name(p.encoder), "encoder_model.ort");
        assert_eq!(file_name(p.merged_decoder), "decoder_model_merged.ort");
        assert!(p.tokens.ends_with("tokens.txt"));

        // A stray top-level file (e.g. silero_vad.onnx) is never a candidate.
        file(models.path(), "silero_vad.onnx");
        assert!(Asr::find_moonshine(models.path()).is_some());
    }

    #[test]
    fn find_online_punct_prefers_int8_and_requires_vocab() {
        let models = models_dir();
        let d = sub(models.path(), "sherpa-onnx-online-punct-en");
        file(&d, "bpe.vocab");
        file(&d, "model.onnx");
        let p = Asr::find_online_punct(models.path()).expect("fp32 model found");
        assert_eq!(file_name(p.model), "model.onnx");
        // int8 appears and takes priority.
        file(&d, "model.int8.onnx");
        let p = Asr::find_online_punct(models.path()).expect("int8 model found");
        assert_eq!(file_name(p.model), "model.int8.onnx");
        assert_eq!(file_name(p.vocab), "bpe.vocab");

        // Missing vocab disqualifies a dir; wrong dir prefix is ignored.
        let d2 = sub(models.path(), "sherpa-onnx-online-punct-en2");
        file(&d2, "model.int8.onnx");
        let d3 = sub(models.path(), "punct-model");
        file(&d3, "bpe.vocab");
        file(&d3, "model.int8.onnx");
        let p = Asr::find_online_punct(models.path()).expect("first good dir still wins");
        assert!(p.dir.ends_with("sherpa-onnx-online-punct-en"));

        let models2 = models_dir();
        let only = sub(models2.path(), "sherpa-onnx-online-punct-en");
        file(&only, "model.int8.onnx");
        assert!(Asr::find_online_punct(models2.path()).is_none());
    }

    #[test]
    fn find_nemotron_requires_triple_and_tokens() {
        let models = models_dir();
        // Sorted first but missing the joiner: skipped.
        let broken = sub(
            models.path(),
            "sherpa-onnx-nemotron-speech-streaming-en-0.6b-broken",
        );
        file(&broken, "encoder.int8.onnx");
        file(&broken, "decoder.int8.onnx");
        file(&broken, "tokens.txt");
        assert!(Asr::find_nemotron(models.path()).is_none());

        // Complete dir is found and the int8 variants are picked.
        let ok = sub(
            models.path(),
            "sherpa-onnx-nemotron-speech-streaming-en-0.6b",
        );
        file(&ok, "encoder.onnx");
        file(&ok, "encoder.int8.onnx");
        file(&ok, "decoder.int8.onnx");
        file(&ok, "joiner.int8.onnx");
        file(&ok, "tokens.txt");
        let p = Asr::find_nemotron(models.path()).expect("complete dir found");
        assert!(p
            .dir
            .ends_with("sherpa-onnx-nemotron-speech-streaming-en-0.6b"));
        assert_eq!(file_name(p.encoder), "encoder.int8.onnx");
        assert_eq!(file_name(p.decoder), "decoder.int8.onnx");
        assert_eq!(file_name(p.joiner), "joiner.int8.onnx");

        // A non-matching dir prefix is never a candidate.
        let models2 = models_dir();
        let other = sub(models2.path(), "my-nemotron-copy");
        file(&other, "encoder.int8.onnx");
        file(&other, "decoder.int8.onnx");
        file(&other, "joiner.int8.onnx");
        file(&other, "tokens.txt");
        assert!(Asr::find_nemotron(models2.path()).is_none());
    }

    #[test]
    fn resolve_zipformer_paths_ok_and_error() {
        let models = models_dir();
        // No matching dir at all: Err with a helpful message.
        let err = Asr::resolve_zipformer_paths(models.path())
            .expect_err("error expected")
            .to_string();
        assert!(
            err.contains("no ASR model found"),
            "unexpected error: {err}"
        );

        // Complete dir resolves with the epoch-tagged files.
        let ok = sub(
            models.path(),
            "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        );
        file(&ok, "encoder-epoch-99-avg-1.onnx");
        file(&ok, "decoder-epoch-99-avg-1.onnx");
        file(&ok, "joiner-epoch-99-avg-1.onnx");
        file(&ok, "tokens.txt");
        let p = Asr::resolve_zipformer_paths(models.path()).expect("complete dir resolves");
        assert!(p
            .dir
            .ends_with("sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"));
        assert_eq!(file_name(p.encoder), "encoder-epoch-99-avg-1.onnx");
        assert_eq!(file_name(p.decoder), "decoder-epoch-99-avg-1.onnx");
        assert_eq!(file_name(p.joiner), "joiner-epoch-99-avg-1.onnx");
        assert!(p.tokens.ends_with("tokens.txt"));
    }

    #[test]
    fn asr_new_without_models_errors_for_every_selection() {
        let models = models_dir();
        // Auto walks the whole fallback chain (nemotron -> zipformer ->
        // moonshine) and only then reports failure.
        for selection in [
            BackendSelection::Auto,
            BackendSelection::Streaming,
            BackendSelection::Zipformer,
            BackendSelection::Moonshine,
        ] {
            let err = Asr::new(models.path(), 1, selection)
                .err()
                .expect("error expected")
                .to_string();
            assert!(
                !err.is_empty(),
                "{selection:?} should error on an empty models dir"
            );
        }
    }

    #[test]
    fn partial_model_dirs_fall_through_the_selection_chain() {
        let models = models_dir();
        // Realistic half-downloads: a Nemotron dir missing its joiner and a
        // Zipformer dir whose encoder lacks the "epoch" tag. Neither is a
        // candidate, so no backend may be *selected* (and then fail to
        // load) - the chain must fall through to the detection errors.
        let broken_n = sub(
            models.path(),
            "sherpa-onnx-nemotron-speech-streaming-en-0.6b",
        );
        file(&broken_n, "encoder.int8.onnx");
        file(&broken_n, "decoder.int8.onnx");
        file(&broken_n, "tokens.txt");

        let broken_z = sub(
            models.path(),
            "sherpa-onnx-streaming-zipformer-bilingual-zh-en",
        );
        file(&broken_z, "encoder.onnx"); // no "epoch" in the name
        file(&broken_z, "decoder-epoch-99.onnx");
        file(&broken_z, "joiner-epoch-99.onnx");
        file(&broken_z, "tokens.txt");

        // Auto walks nemotron -> zipformer -> moonshine; the outermost error
        // is the moonshine one, proving the chain fell through completely.
        let err = Asr::new(models.path(), 1, BackendSelection::Auto)
            .err()
            .expect("error expected")
            .to_string();
        assert!(
            err.contains("no Moonshine model found"),
            "unexpected error: {err}"
        );

        // Streaming stops after the zipformer stage (no moonshine fallback).
        let err = Asr::new(models.path(), 1, BackendSelection::Streaming)
            .err()
            .expect("error expected")
            .to_string();
        assert!(
            err.contains("no ASR model found"),
            "unexpected error: {err}"
        );

        // Forced backends report their own detection failure, not a load error.
        let err = Asr::new(models.path(), 1, BackendSelection::Zipformer)
            .err()
            .expect("error expected")
            .to_string();
        assert!(
            err.contains("no ASR model found"),
            "unexpected error: {err}"
        );
        let err = Asr::new(models.path(), 1, BackendSelection::Moonshine)
            .err()
            .expect("error expected")
            .to_string();
        assert!(
            err.contains("no Moonshine model found"),
            "unexpected error: {err}"
        );
    }

    /// `polish` with a real online-punctuation model (skipped when the
    /// model dir is absent). Pins the model's deterministic output policy
    /// and - the regression this test exists for - that pre-punctuated
    /// input cannot double-mark: the strip step runs before the model, so
    /// "HELLO, WORLD!" polishes to exactly what "hello world" does.
    #[test]
    fn polish_with_real_punct_model_restores_casing() {
        let _lock = model_lock();
        let Some(models) = repo_models_dir() else {
            return;
        };
        let Some(p) = Asr::create_punct(&models, 1) else {
            return;
        };
        // The model capitalizes sentence starts (its capitalization is a
        // learned behavior, not a regex: it may also cap some other words).
        assert_eq!(polish(Some(&p), "hello world"), "Hello World");
        assert_eq!(
            polish(Some(&p), "the quick brown fox"),
            "The quick brown fox"
        );
        // Deterministic across calls.
        assert_eq!(polish(Some(&p), "hello world"), "Hello World");
        // Pre-punctuated input strips to the same shape first, so the model
        // never sees the marks and cannot double them.
        assert_eq!(
            polish(Some(&p), "HELLO, WORLD!"),
            polish(Some(&p), "hello world")
        );
    }

    /// A VAD segment of pure noise (a cough, a click) can make the ASR
    /// backend emit a fragment that is *entirely* sentence punctuation -
    /// which the strip step collapses to an empty string. The remaining
    /// `polish` path then hands that empty text to the online-punct model
    /// (`add_punctuation("")`); the regression this test exists for is that
    /// the model must not invent text (a stray period, a phantom word) out
    /// of an empty input, which would type nothing-but-junk into the target
    /// app. Skipped when the repo's `models/` dir or the punct model is
    /// absent.
    #[test]
    fn polish_of_punct_only_input_never_invents_text() {
        let _lock = model_lock();
        let Some(models) = repo_models_dir() else {
            return;
        };
        let Some(p) = Asr::create_punct(&models, 1) else {
            return;
        };
        // Trim is non-empty but every char is a stripped sentence mark, so
        // the text handed to the model is empty; the result must be empty.
        assert_eq!(polish(Some(&p), "?!...;:"), "");
        assert_eq!(polish(Some(&p), "??"), "");
        // The trim-only short-circuit (no model call) also stays empty.
        assert_eq!(polish(Some(&p), ""), "");
        assert_eq!(polish(Some(&p), "   \t  "), "");
        // A model that *does* add marks still returns them for real text,
        // so the empty result above is not a blanket no-op of the model.
        assert!(!polish(Some(&p), "hello world").is_empty());
    }

    /// Full `StreamingSession` round trip on a real model + real speech
    /// (skipped when the repo's `models/` dir is absent): feed a 16 kHz
    /// utterance in 512-sample chunks like the live pipeline, watch live
    /// partials appear, commit at an utterance boundary, and verify the
    /// reset leaves a clean stream that serves a second utterance.
    #[test]
    fn streaming_session_roundtrip_with_real_model() {
        let _lock = model_lock();
        let Some(models) = repo_models_dir() else {
            return;
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
        let asr = match Asr::new(&models, 2, BackendSelection::Streaming) {
            Ok(a) => a,
            Err(_) => return,
        };
        assert!(asr.is_streaming());
        let session = asr
            .streaming_session()
            .expect("streaming backend has a session");
        let wave =
            sherpa_onnx::Wave::read(wav.to_str().expect("utf-8 path")).expect("read test wav");
        let samples = wave.samples().to_vec();
        assert!(samples.len() > 16000, "test wav should be at least 1 s");

        // Feed in 512-sample chunks (32 ms, the VAD window size) and track
        // that live partials appear before the commit.
        let mut saw_partial = false;
        for chunk in samples.chunks(512) {
            session.feed(chunk);
            if !session.partial().is_empty() {
                saw_partial = true;
            }
        }
        assert!(saw_partial, "no live partial was produced while feeding");

        let final_text = session.commit();
        assert!(
            final_text.to_lowercase().contains("yellow lamps"),
            "unexpected final: {final_text:?}"
        );

        // Commit resets the stream: committing again with no new audio
        // yields no text.
        assert_eq!(session.commit(), "", "stream must be clean after commit");

        // The same session serves the next utterance.
        for chunk in samples.chunks(512) {
            session.feed(chunk);
        }
        let final_text2 = session.commit();
        assert!(
            final_text2.to_lowercase().contains("yellow lamps"),
            "unexpected second final: {final_text2:?}"
        );
    }

    /// `--transcribe` may be handed a truncated or empty WAV. Input shorter
    /// than `MIN_TRANSCRIBE_SAMPLES` (100 ms, the live pipeline's blip
    /// filter threshold) cannot be a complete utterance: it must return
    /// empty text WITHOUT running the model at all - besides being a wasted
    /// decode, Moonshine's onnxruntime conv stack logs a scary "Invalid
    /// input shape" error block for short inputs (verified live, even at
    /// 512 samples). Exactly 100 ms of silence still reaches the model and
    /// must complete cleanly with empty text (skipped when a backend's
    /// model dir is absent).
    #[test]
    fn transcribe_empty_and_short_audio_returns_empty_without_crashing() {
        let _lock = model_lock();
        let Some(models) = repo_models_dir() else {
            return;
        };
        // Auto loads the preferred streaming backend (Nemotron where
        // installed, else Zipformer - the same `OnlineRecognizer` FFI
        // surface); Moonshine is the separate `OfflineRecognizer` surface.
        for selection in [BackendSelection::Auto, BackendSelection::Moonshine] {
            let Ok(asr) = Asr::new(&models, 1, selection) else {
                continue; // backend not installed on this machine
            };
            // Empty input and the boundary just below the threshold: guarded,
            // no model round trip.
            for n in [0usize, MIN_TRANSCRIBE_SAMPLES - 1] {
                let (text, elapsed) = asr.transcribe(&vec![0.0f32; n]);
                assert!(
                    text.is_empty(),
                    "{selection:?}: {n} samples must give empty text, got {text:?}"
                );
                assert_eq!(
                    elapsed,
                    std::time::Duration::ZERO,
                    "{selection:?}: {n} samples must not run the model"
                );
            }
            // Exactly the threshold of pure silence: reaches the model (no
            // guard) and must still complete with empty text.
            let (text, _) = asr.transcribe(&vec![0.0f32; MIN_TRANSCRIBE_SAMPLES]);
            assert!(
                text.is_empty(),
                "{selection:?}: 100 ms of silence must give empty text, got {text:?}"
            );
        }
    }

    #[test]
    fn check_models_dir_reports_missing_dir() {
        let models = models_dir();
        let missing = models.path().join("no-such-dir");
        let err = check_models_dir(&missing).unwrap_err().to_string();
        assert!(err.contains("no-such-dir"), "unexpected error: {err}");
        assert!(check_models_dir(models.path()).is_ok());
    }

    // --- corrupt model files: clean error, never a process abort ---
    //
    // A corrupt model (interrupted download, disk full) used to make the
    // sherpa-onnx C++ loader throw a C++ exception across the FFI
    // boundary, which aborts the whole process (SIGABRT, uncatchable in
    // Rust). The onnx envelope check must reject it with a clean Err.

    /// Write a garbage Nemotron triple into a temp models dir.
    fn corrupt_nemotron_dir(models: &Path) -> PathBuf {
        let d = models.join("sherpa-onnx-nemotron-speech-streaming-en-corrupt");
        std::fs::create_dir_all(&d).unwrap();
        for name in ["encoder", "decoder", "joiner"] {
            std::fs::write(d.join(format!("{name}.int8.onnx")), vec![0xAB; 4096]).unwrap();
        }
        std::fs::write(d.join("tokens.txt"), "a\nb\n").unwrap();
        d
    }

    #[test]
    fn asr_new_with_corrupt_nemotron_model_errors_cleanly() {
        let models = models_dir();
        corrupt_nemotron_dir(models.path());
        let err = Asr::new(models.path(), 1, BackendSelection::Streaming)
            .err()
            .expect("error expected")
            .to_string();
        assert!(
            err.contains("not a valid ONNX model"),
            "unexpected error: {err}"
        );
        assert!(
            err.contains("download-models.sh"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn asr_new_with_truncated_real_nemotron_model_errors_cleanly() {
        // An interrupted download of the real model: a file that starts
        // like a valid ONNX model but ends mid-way. Gated on the real
        // model being installed.
        let Some(models) = repo_models_dir() else {
            return;
        };
        let real =
            models.join("sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25");
        let enc = real.join("encoder.int8.onnx");
        if !enc.is_file() {
            return;
        }
        let bytes = std::fs::read(&enc).unwrap();
        let tmp = models_dir();
        let d = corrupt_nemotron_dir(tmp.path());
        std::fs::write(d.join("encoder.int8.onnx"), &bytes[..bytes.len() / 10]).unwrap();
        let err = Asr::new(tmp.path(), 1, BackendSelection::Streaming)
            .err()
            .expect("error expected")
            .to_string();
        assert!(
            err.contains("not a valid ONNX model"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn asr_new_with_corrupt_moonshine_onnx_model_errors_cleanly() {
        let models = models_dir();
        let d = models.path().join("sherpa-onnx-moonshine-corrupt");
        std::fs::create_dir_all(&d).unwrap();
        // .onnx moonshine files get the envelope guard.
        std::fs::write(d.join("encoder_model.onnx"), vec![0xAB; 4096]).unwrap();
        std::fs::write(d.join("decoder_model_merged.onnx"), vec![0xAB; 4096]).unwrap();
        std::fs::write(d.join("tokens.txt"), "a\nb\n").unwrap();
        let err = Asr::new(models.path(), 1, BackendSelection::Moonshine)
            .err()
            .expect("error expected")
            .to_string();
        assert!(
            err.contains("not a valid ONNX model"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn asr_new_with_corrupt_punct_model_degrades_to_no_punct() {
        // The punct model is optional: a corrupt one must not fail the
        // whole load - the streaming backend still comes up (with
        // unpunctuated output). Gated on the real Nemotron model; the
        // temp dir symlinks it in so the real models/ is untouched.
        let _lock = model_lock();
        let Some(models) = repo_models_dir() else {
            return;
        };
        let real =
            models.join("sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25");
        if !real.is_dir() {
            return;
        }
        let tmp = models_dir();
        let d = tmp.path();
        std::os::unix::fs::symlink(&real, d.join(real.file_name().unwrap())).unwrap();
        let punct = d.join("sherpa-onnx-online-punct-corrupt");
        std::fs::create_dir_all(&punct).unwrap();
        std::fs::write(punct.join("model.int8.onnx"), vec![0xAB; 4096]).unwrap();
        std::fs::write(punct.join("bpe.vocab"), "a\n").unwrap();
        let asr = Asr::new(d, 1, BackendSelection::Streaming).expect("load");
        assert_eq!(asr.kind(), AsrKind::Nemotron);
    }
}
