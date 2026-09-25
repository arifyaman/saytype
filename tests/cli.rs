//! Binary-level contract tests for the offline check subcommands
//! (`--transcribe`, `--vad-test`, `--stream-test`), run against the real
//! built binary (cargo builds it as a dependency of this test target).
//!
//! They pin the fail-fast input-validation order: an unreadable or
//! wrong-rate WAV is rejected immediately, before any model load (which can
//! take several seconds). With an empty models dir the two orderings are
//! observable: if a model load ran first, the error would be a model error
//! ("no ASR model found" / "VAD model not found") instead of the WAV error.
//!
//! They also pin the user-facing CLI contract: every misuse path exits 2
//! with a diagnostic (missing argument, unknown argument, daemon positional,
//! invalid `--asr` value), repeated `--asr` flags are last-wins rather than a
//! usage error (an invalid value on any occurrence still exits 2),
//! `--help`/`-h` exits 0 and lists every subcommand, and the only offline
//! check with a subprocess-free success path (`--paste-test ""`) exits 0
//! without spawning any tool.

use std::process::Command;

/// Write a minimal mono S16LE WAV at the requested sample rate. The content
/// is silence; only the header matters for input validation.
fn write_wav(path: &std::path::Path, sample_rate: u32) {
    let n_samples = 1600usize; // 100 ms
    let data_len = (n_samples * 2) as u32;
    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&u32::to_le_bytes(36 + data_len));
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&1u16.to_le_bytes()); // mono
    b.extend_from_slice(&sample_rate.to_le_bytes());
    b.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    b.extend_from_slice(&2u16.to_le_bytes()); // block align
    b.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    b.resize(b.len() + n_samples * 2, 0);
    std::fs::write(path, b).expect("write WAV");
}

/// Run one offline-check subcommand against `arg` with the models dir
/// pointed at `models`; return (success, stderr).
fn run(sub: &str, arg: &std::path::Path, models: &std::path::Path) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_saytype"))
        .arg(sub)
        .arg(arg)
        .env("SAYTYPE_MODELS_DIR", models)
        .output()
        .expect("run saytype");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn wrong_rate_wav_is_rejected_before_any_model_load() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let bad = tmp.path().join("bad.wav");
    write_wav(&bad, 24000);
    let models = tempfile::tempdir().expect("tempdir");
    for sub in ["--transcribe", "--vad-test", "--stream-test"] {
        let (ok, stderr) = run(sub, &bad, models.path());
        assert!(!ok, "{sub} must fail on a 24 kHz WAV");
        assert!(
            stderr.contains("expected a 16 kHz WAV, got 24000 Hz"),
            "{sub}: expected the sample-rate error, got: {stderr}"
        );
        assert!(
            !stderr.contains("no ASR model found") && !stderr.contains("VAD model not found"),
            "{sub}: a model error means the WAV check no longer runs first: {stderr}"
        );
    }
}

#[test]
fn missing_wav_is_reported_cleanly() {
    let models = tempfile::tempdir().expect("tempdir");
    let (ok, stderr) = run(
        "--transcribe",
        &models.path().join("nope.wav"),
        models.path(),
    );
    assert!(!ok, "a missing WAV must fail");
    assert!(stderr.contains("cannot read WAV"), "got: {stderr}");
}

/// Run the real binary with arbitrary args and return (exit code, stdout,
/// stderr). No env mutation: every case below exits before models/, audio,
/// or D-Bus are touched, so the tests are headless-safe and parallel-safe.
fn run_args(args: &[&str]) -> (Option<i32>, String, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_saytype"));
    for a in args {
        cmd.arg(a);
    }
    let out = cmd.output().expect("run saytype");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn missing_positional_arg_prints_usage_and_exits_2() {
    for sub in [
        "--transcribe",
        "--vad-test",
        "--stream-test",
        "--paste-test",
    ] {
        let (code, _stdout, stderr) = run_args(&[sub]);
        assert_eq!(code, Some(2), "{sub} without its argument must exit 2");
        assert!(
            stderr.contains("usage:"),
            "{sub}: expected a usage line on stderr, got: {stderr}"
        );
    }
}

#[test]
fn unknown_argument_exits_2_with_pointer_to_help() {
    let (code, _stdout, stderr) = run_args(&["frobnicate"]);
    assert_eq!(code, Some(2), "an unknown argument must exit 2");
    assert!(stderr.contains("unknown argument"), "got: {stderr}");
    assert!(stderr.contains("--help"), "got: {stderr}");
}

#[test]
fn daemon_rejects_positional_arguments() {
    let (code, _stdout, stderr) = run_args(&["--live-typing", "stray.wav"]);
    assert_eq!(code, Some(2), "a daemon positional must exit 2");
    assert!(
        stderr.contains("daemon takes no positional arguments"),
        "got: {stderr}"
    );
}

#[test]
fn daemon_rejects_contradictory_typing_mode_flags() {
    // `--live-typing` and `--no-live-typing` are mutually exclusive (the
    // `--help` usage shows them as an either/or). Passing both is a typo
    // that used to silently run in Live mode; it must now be a usage error
    // (exit 2) before any model/D-Bus work, so it is headless-safe.
    let (code, _stdout, stderr) = run_args(&["--live-typing", "--no-live-typing"]);
    assert_eq!(code, Some(2), "contradictory typing-mode flags must exit 2");
    assert!(stderr.contains("mutually exclusive"), "got: {stderr}");
    // The diagnostic runs before the daemon loads: no models/D-Bus error.
    assert!(
        !stderr.contains("daemon failed"),
        "the flag check must run before the daemon starts: {stderr}"
    );
}

#[test]
fn invalid_asr_value_exits_2_before_any_wav_or_model_work() {
    let (code, _stdout, stderr) =
        run_args(&["--transcribe", "--asr", "bogus", "does-not-matter.wav"]);
    assert_eq!(code, Some(2), "an invalid --asr value must exit 2");
    assert!(
        stderr.contains("--asr expects auto, streaming, zipformer, or moonshine"),
        "got: {stderr}"
    );
    assert!(
        !stderr.contains("cannot read WAV"),
        "the value check must run before the WAV check: {stderr}"
    );
}

#[test]
fn asr_flag_missing_or_empty_value_exits_2() {
    // A dangling `--asr` (no following value) and an explicitly empty value
    // are both malformed: they exit 2 with the value diagnostic, on the same
    // path as an invalid value, before any WAV/model work.
    for args in [
        &["--transcribe", "--asr"][..],
        &["--transcribe", "--asr", ""][..],
    ] {
        let (code, _stdout, stderr) = run_args(args);
        assert_eq!(
            code,
            Some(2),
            "a missing/empty --asr value must exit 2: {stderr}"
        );
        assert!(
            stderr.contains("--asr expects auto, streaming, zipformer, or moonshine"),
            "expected the value diagnostic, got: {stderr}"
        );
        assert!(
            !stderr.contains("cannot read WAV"),
            "the --asr check must run before the WAV check: {stderr}"
        );
    }
}

#[test]
fn repeated_asr_flags_are_last_wins_not_a_usage_error() {
    // Repeated `--asr` flags are accepted (the last valid value wins; which
    // value wins is pinned by the unit test in src/main.rs), so the command
    // proceeds past flag parsing to the WAV check. An invalid value on any
    // occurrence still exits 2, even when an earlier occurrence was valid.
    let (code, _stdout, stderr) = run_args(&[
        "--transcribe",
        "--asr",
        "auto",
        "--asr",
        "moonshine",
        "does-not-exist.wav",
    ]);
    assert_eq!(
        code,
        Some(1),
        "repeated valid --asr flags must not be a usage error: {stderr}"
    );
    assert!(
        stderr.contains("cannot read WAV"),
        "the command must proceed to the WAV check, got: {stderr}"
    );

    let (code, _stdout, stderr) = run_args(&[
        "--transcribe",
        "--asr",
        "moonshine",
        "--asr",
        "bogus",
        "does-not-exist.wav",
    ]);
    assert_eq!(
        code,
        Some(2),
        "an invalid later --asr value must still exit 2: {stderr}"
    );
    assert!(
        stderr.contains("--asr expects auto, streaming, zipformer, or moonshine"),
        "got: {stderr}"
    );
}

#[test]
fn paste_test_with_empty_text_exits_0_without_any_subprocess() {
    // paste_text("") early-returns before spawning any clipboard/paste tool,
    // so this is the one offline check whose success path needs neither a
    // display, a model, nor ydotool - safe to assert headlessly.
    let (code, stdout, stderr) = run_args(&["--paste-test", ""]);
    assert_eq!(
        code,
        Some(0),
        "empty paste-test text must succeed, stderr: {stderr}"
    );
    assert!(stdout.contains("paste_text OK"), "got: {stdout} {stderr}");
}

#[test]
fn help_exits_0_and_lists_all_subcommands() {
    for flag in ["--help", "-h"] {
        let (code, stdout, _stderr) = run_args(&[flag]);
        assert_eq!(code, Some(0), "{flag} must exit 0");
        for sub in [
            "--transcribe",
            "--vad-test",
            "--stream-test",
            "--paste-test",
        ] {
            assert!(stdout.contains(sub), "{flag} output must mention {sub}");
        }
        assert!(
            stdout.contains("typing mode"),
            "help must document the typing modes, got: {stdout}"
        );
    }
}
