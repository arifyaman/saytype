mod asr;
mod audio;
mod daemon;
mod injector;
mod vad;

use std::path::PathBuf;

use anyhow::Context;

fn models_dir() -> PathBuf {
    if let Ok(env) = std::env::var("SAYTYPE_MODELS_DIR") {
        return PathBuf::from(env);
    }
    if let Ok(cwd) = std::env::current_dir() {
        let p = cwd.join("models");
        if p.is_dir() {
            return p;
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let p = parent.join("models");
            if p.is_dir() {
                return p;
            }
        }
    }
    PathBuf::from("models")
}

fn init_logging() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime")
}

fn run_daemon() {
    let models = models_dir();
    let rt = runtime();
    rt.block_on(async move {
        init_logging();
        if let Err(e) = daemon::run(&models).await {
            tracing::error!("daemon failed: {e:?}");
            std::process::exit(1);
        }
    });
}

/// Offline ASR check: `saytype --transcribe <file.wav>` (16 kHz mono).
fn run_transcribe(args: &[String]) {
    if args.is_empty() {
        eprintln!("usage: saytype --transcribe <file.wav>");
        std::process::exit(2);
    }
    let path = args[0].clone();
    let models = models_dir();
    let rt = runtime();
    let res = rt.block_on(async move {
        init_logging();
        let out = tokio::task::spawn_blocking(move || {
            let asr = asr::Asr::new(&models, 4)?;
            let wave = sherpa_onnx::Wave::read(&path)
                .with_context(|| format!("cannot read WAV {path:?}"))?;
            anyhow::ensure!(
                wave.sample_rate() == 16000,
                "expected a 16 kHz WAV, got {} Hz",
                wave.sample_rate()
            );
            let samples = wave.samples().to_vec();
            let (text, elapsed) = asr.transcribe(&samples);
            Ok::<_, anyhow::Error>((text, elapsed, samples.len()))
        })
        .await
        .expect("transcribe task panicked")?;
        let (text, elapsed, n) = out;
        println!("samples: {n} ({:.2}s)", n as f32 / 16000.0);
        println!("latency: {elapsed:?}");
        println!("text: {text}");
        Ok::<(), anyhow::Error>(())
    });
    if let Err(e) = res {
        eprintln!("transcribe failed: {e:?}");
        std::process::exit(1);
    }
}

/// Offline VAD check: `saytype --vad-test <file.wav>` (16 kHz mono).
fn run_vad_test(args: &[String]) {
    if args.is_empty() {
        eprintln!("usage: saytype --vad-test <file.wav>");
        std::process::exit(2);
    }
    let path = args[0].clone();
    let models = models_dir();
    let rt = runtime();
    let res = rt.block_on(async move {
        init_logging();
        let cfg = vad::VadConfig::from_models_dir(&models);
        let (n_seg, total, n) = tokio::task::spawn_blocking(move || {
            vad::check_model_file(std::path::Path::new(&cfg.model_path))?;
            let vad = vad::Vad::new(&cfg)?;
            let wave = sherpa_onnx::Wave::read(&path)
                .with_context(|| format!("cannot read WAV {path:?}"))?;
            anyhow::ensure!(
                wave.sample_rate() == 16000,
                "expected a 16 kHz WAV, got {} Hz",
                wave.sample_rate()
            );
            let samples = wave.samples();
            let mut n_seg = 0usize;
            let mut total = 0usize;
            for chunk in samples.chunks(512) {
                vad.feed(chunk);
                while let Some((seg, _)) = vad.take_segment() {
                    n_seg += 1;
                    total += seg.len();
                    println!("segment {n_seg}: {:.2}s", seg.len() as f32 / 16000.0);
                }
            }
            vad.flush();
            while let Some((seg, _)) = vad.take_segment() {
                n_seg += 1;
                total += seg.len();
                println!(
                    "segment {n_seg}: {:.2}s (flushed)",
                    seg.len() as f32 / 16000.0
                );
            }
            Ok::<_, anyhow::Error>((n_seg, total, samples.len()))
        })
        .await
        .expect("vad test task panicked")?;
        println!(
            "total: {n_seg} segments, {:.2}s of speech in {:.2}s of audio",
            total as f32 / 16000.0,
            n as f32 / 16000.0
        );
        Ok::<(), anyhow::Error>(())
    });
    if let Err(e) = res {
        eprintln!("vad test failed: {e:?}");
        std::process::exit(1);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--transcribe") => run_transcribe(&args[2..]),
        Some("--vad-test") => run_vad_test(&args[2..]),
        Some("--help") | Some("-h") => {
            println!(
                "saytype - background speech-to-text dictation daemon\n\n\
                  usage:\n  \
                  saytype                        run the D-Bus daemon (systemd user service)\n  \
                  saytype --transcribe <wav>     offline ASR check (16 kHz mono WAV)\n  \
                  saytype --vad-test <wav>       offline VAD check (16 kHz mono WAV)"
            );
        }
        Some(other) => {
            eprintln!("unknown argument: {other}. See `saytype --help`.");
            std::process::exit(2);
        }
        None => run_daemon(),
    }
}
