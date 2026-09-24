mod asr;
mod audio;
mod daemon;
mod injector;
mod transcript;
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

fn run_daemon(args: &[String]) {
    let (rest, selection) = parse_asr_selection(args);
    let typing_mode = parse_typing_mode(&rest);
    let rest: Vec<String> = rest
        .into_iter()
        .filter(|a| a != "--live-typing" && a != "--no-live-typing")
        .collect();
    if !rest.is_empty() {
        eprintln!("daemon takes no positional arguments (got {:?})", rest[0]);
        std::process::exit(2);
    }
    let models = models_dir();
    let rt = runtime();
    rt.block_on(async move {
        init_logging();
        if let Err(e) = daemon::run(&models, selection, typing_mode).await {
            tracing::error!("daemon failed: {e:?}");
            std::process::exit(1);
        }
    });
}

/// Parse the typing-mode flags out of a daemon subcommand's remaining args
/// (after `--asr` has already been stripped by `parse_asr_selection`),
/// returning the mode and (via `retain`) removing the flags so leftover
/// positional-argument validation still works. Default: `Deferred` -
/// nothing is typed into the target app until the session stops, so
/// mid-dictation erase/undo never touches whatever real app has focus
/// while you are still speaking.
fn parse_typing_mode(args: &[String]) -> daemon::TypingMode {
    if args.iter().any(|a| a == "--live-typing") {
        daemon::TypingMode::Live
    } else if args.iter().any(|a| a == "--no-live-typing") {
        daemon::TypingMode::FinalOnly
    } else {
        daemon::TypingMode::Deferred
    }
}

/// Parse `--asr <auto|streaming|zipformer|moonshine>` out of a subcommand's
/// args, returning the remaining positional args and the selection.
fn parse_asr_selection(args: &[String]) -> (Vec<String>, asr::BackendSelection) {
    let mut rest = Vec::new();
    let mut selection = asr::BackendSelection::Auto;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--asr" {
            i += 1;
            selection = match args.get(i).map(String::as_str) {
                Some("streaming") => asr::BackendSelection::Streaming,
                Some("zipformer") => asr::BackendSelection::Zipformer,
                Some("moonshine") => asr::BackendSelection::Moonshine,
                Some("auto") => asr::BackendSelection::Auto,
                _ => {
                    eprintln!("--asr expects auto, streaming, zipformer, or moonshine");
                    std::process::exit(2);
                }
            };
        } else {
            rest.push(args[i].clone());
        }
        i += 1;
    }
    (rest, selection)
}

/// Offline ASR check: `saytype --transcribe [--asr <sel>] <file.wav>` (16 kHz mono).
fn run_transcribe(args: &[String]) {
    let (rest, selection) = parse_asr_selection(args);
    if rest.is_empty() {
        eprintln!("usage: saytype --transcribe [--asr auto|streaming|zipformer|moonshine] <file.wav>");
        std::process::exit(2);
    }
    let path = rest[0].clone();
    let models = models_dir();
    let rt = runtime();
    let res = rt.block_on(async move {
        init_logging();
        let out = tokio::task::spawn_blocking(move || {
            let asr = asr::Asr::new(&models, 4, selection)?;
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

/// Commit every finalized VAD segment through the streaming session,
/// printing the final text and how it differs from the last partial.
fn drain_commits(
    vad: &vad::Vad,
    session: &asr::StreamingSession,
    last_partial: &mut String,
    n_commits: &mut usize,
) {
    // Same blip filter the daemon uses (~100 ms).
    while let Some((seg, _)) = vad.take_segment() {
        if seg.len() < audio::SAMPLE_RATE as usize / 10 {
            continue;
        }
        let text = session.commit();
        if text.is_empty() {
            continue;
        }
        *n_commits += 1;
        let note = if *last_partial == text {
            "final == last partial".to_string()
        } else {
            format!("last partial was {:?}", last_partial)
        };
        println!("[final] {text}   ({note})");
        last_partial.clear();
    }
}

/// Offline injector check: `saytype --paste-test "<text>"`. Exercises the
/// exact real subprocess chain `injector::paste_text` uses (clipboard set
/// via `xclip`/`wl-copy`, then `ydotool key ctrl+v`) in the real tokio
/// runtime, timed - a direct regression check for a real bug where `xclip`
/// hung under the daemon's actual process context (not reproducible
/// testing `xclip` standalone from an interactive shell) until
/// `run_with_stdin` stopped waiting for it to fully exit.
fn run_paste_test(args: &[String]) {
    if args.is_empty() {
        eprintln!("usage: saytype --paste-test \"<text>\"");
        std::process::exit(2);
    }
    let text = args.join(" ");
    let rt = runtime();
    rt.block_on(async move {
        init_logging();
        let start = std::time::Instant::now();
        match injector::paste_text(&text).await {
            Ok(()) => println!("paste_text OK in {:?}", start.elapsed()),
            Err(e) => {
                eprintln!("paste_text FAILED in {:?}: {e}", start.elapsed());
                std::process::exit(1);
            }
        }
    });
}

/// Offline streaming check: `saytype --stream-test [--asr streaming] <file.wav>`
/// (16 kHz mono). Replays the file headlessly through the same pieces the
/// live pipeline uses - a VAD for commit boundaries and one continuous
/// streaming session - printing each changed partial and each committed
/// final, plus decode-cost stats. This is the regression harness for the
/// streaming path (what `--transcribe` is for the batch path).
fn run_stream_test(args: &[String]) {
    let (rest, selection) = parse_asr_selection(args);
    if rest.is_empty() {
        eprintln!("usage: saytype --stream-test [--asr streaming] <file.wav>");
        std::process::exit(2);
    }
    let path = rest[0].clone();
    let models = models_dir();
    let rt = runtime();
    let res = rt.block_on(async move {
        init_logging();
        tokio::task::spawn_blocking(move || {
            let model = asr::Asr::new(&models, 4, selection)?;
            let Some(session) = model.streaming_session() else {
                anyhow::bail!("streaming backend required (pass --asr streaming)");
            };
            let wave = sherpa_onnx::Wave::read(&path)
                .with_context(|| format!("cannot read WAV {path:?}"))?;
            anyhow::ensure!(
                wave.sample_rate() == 16000,
                "expected a 16 kHz WAV, got {} Hz",
                wave.sample_rate()
            );
            let samples = wave.samples().to_vec();
            let rate = 16000usize;

            let vad = vad::Vad::new(&vad::VadConfig::from_models_dir(&models))?;

            let mut last_partial = String::new();
            let mut n_partials = 0usize;
            let mut n_commits = 0usize;
            let mut decode = std::time::Duration::ZERO;

            // 512 samples = 32 ms, the VAD's window.
            for (idx, chunk) in samples.chunks(512).enumerate() {
                vad.feed(chunk);
                let t = std::time::Instant::now();
                session.feed(chunk);
                decode += t.elapsed();

                drain_commits(&vad, &session, &mut last_partial, &mut n_commits);

                let partial = session.partial();
                if partial != last_partial {
                    n_partials += 1;
                    let t_sec = (idx + 1) as f32 * 512.0 / rate as f32;
                    println!("[partial {t_sec:7.2}s] {partial}");
                    last_partial = partial;
                }
            }
            // Session end: flush trailing speech, like the live pipeline.
            vad.flush();
            drain_commits(&vad, &session, &mut last_partial, &mut n_commits);

            let audio_secs = samples.len() as f32 / rate as f32;
            let rtf = decode.as_secs_f32() / audio_secs.max(f32::EPSILON);
            println!("---");
            println!(
                "audio {:.2}s | commits {n_commits} | partials {n_partials} | decode {decode:?} (RTF {rtf:.3}) | backend {:?}",
                audio_secs,
                model.kind()
            );
            Ok::<(), anyhow::Error>(())
        })
        .await
        .expect("stream test task panicked")?;
        Ok::<(), anyhow::Error>(())
    });
    if let Err(e) = res {
        eprintln!("stream test failed: {e:?}");
        std::process::exit(1);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--transcribe") => run_transcribe(&args[2..]),
        Some("--vad-test") => run_vad_test(&args[2..]),
        Some("--stream-test") => run_stream_test(&args[2..]),
        Some("--paste-test") => run_paste_test(&args[2..]),
        Some("--help") | Some("-h") => {
            println!(
                "saytype - background speech-to-text dictation daemon\n\n\
                   usage:\n  \
                   saytype [--asr <sel>] [--live-typing | --no-live-typing]\n  \
                   \t                run the D-Bus daemon (systemd user service)\n  \
                   saytype --transcribe [--asr <sel>] <wav>\n  \
                   \t                  offline ASR check (16 kHz mono WAV)\n  \
                   saytype --vad-test <wav>         offline VAD check (16 kHz mono WAV)\n  \
                    saytype --stream-test [--asr streaming] <wav>\n  \
                    \t                  offline streaming replay: partials + committed finals\n  \
                    saytype --paste-test <text>\n  \
                    \t                  offline injector check: clipboard set + ydotool paste\n\n\
                    <sel> = auto | streaming | zipformer | moonshine\n  \
                    \t      auto = best available (nemotron > zipformer > moonshine)\n  \
                    \t      streaming = best streaming backend (nemotron > zipformer)\n\n\
                    typing mode (default: deferred - nothing is typed into the target\n  \
                    app until the session stops; erase/undo only affect the HUD while\n  \
                    recording, so they never touch a real app's live buffer):\n  \
                    --live-typing      type partials live as they stabilize (mvp2\n  \
                    \t\t\t   behavior); erase/undo edit the live-typed buffer too\n  \
                    --no-live-typing   type only committed finals, immediately, one\n  \
                    \t\t\t   chunk per utterance (MVP1 behavior); erase/undo edit\n  \
                    \t\t\t   the live-typed buffer too"
            );
        }
        // Daemon with a flag: `saytype --asr <sel>`.
        Some(flag) if flag.starts_with("--") => run_daemon(&args[1..]),
        Some(other) => {
            eprintln!("unknown argument: {other}. See `saytype --help`.");
            std::process::exit(2);
        }
        None => run_daemon(&args[1..]),
    }
}

#[cfg(test)]
mod typing_mode_tests {
    use super::*;

    #[test]
    fn defaults_to_deferred() {
        let args: Vec<String> = vec![];
        assert_eq!(parse_typing_mode(&args), daemon::TypingMode::Deferred);
    }

    #[test]
    fn live_typing_flag_selects_live() {
        let args: Vec<String> = vec!["--live-typing".to_string()];
        assert_eq!(parse_typing_mode(&args), daemon::TypingMode::Live);
    }

    #[test]
    fn no_live_typing_flag_selects_final_only() {
        let args: Vec<String> = vec!["--no-live-typing".to_string()];
        assert_eq!(parse_typing_mode(&args), daemon::TypingMode::FinalOnly);
    }
}
