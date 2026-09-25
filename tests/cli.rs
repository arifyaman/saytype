//! Binary-level contract tests for the offline check subcommands
//! (`--transcribe`, `--vad-test`, `--stream-test`), run against the real
//! built binary (cargo builds it as a dependency of this test target).
//!
//! They pin the fail-fast input-validation order: an unreadable or
//! wrong-rate WAV is rejected immediately, before any model load (which can
//! take several seconds). With an empty models dir the two orderings are
//! observable: if a model load ran first, the error would be a model error
//! ("no ASR model found" / "VAD model not found") instead of the WAV error.

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
