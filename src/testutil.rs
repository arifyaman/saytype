//! Shared helpers for the unit tests that drive real (faked) subprocesses
//! or real models.
//!
//! Lives in its own `cfg(test)` module so `injector.rs`, `daemon.rs`, `asr.rs`
//! and `vad.rs` share the same locks: every fake tool is a `#!/bin/sh` script (concurrent
//! execs of the same shared interpreter can race in the kernel's exec
//! write-count bookkeeping - observed ETXTBSY from `spawn` under the full
//! parallel suite - so no two tool-spawning tests may overlap their spawns,
//! crate-wide), and the PATH-sandboxing tests mutate process-global env
//! vars (PATH / WAYLAND_DISPLAY), so they must not overlap each other
//! either.

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

/// Serializes every test that spawns a fake `#!/bin/sh` tool, no matter
/// which module it lives in.
pub static TOOL_SPAWN_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Serializes the tests that temporarily rewrite process-global env
/// (PATH / WAYLAND_DISPLAY). See `EnvPatch`: acquiring it together with
/// `TOOL_SPAWN_LOCK` (in that fixed order) keeps an env-mutating test from
/// racing every other tool-spawning test as well.
pub static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Serializes the real-model tests across modules (asr, vad): a Nemotron
/// encoder alone is ~623 MB and each loaded model keeps its ONNX
/// allocations alive for the whole test, so loading several of them in
/// parallel (the harness runs up to one test per CPU) is wasteful and,
/// on a loaded box, slow.
pub static MODEL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Lock the real-model serialization mutex. Recovers from poisoning so
/// one test panicking cannot strand the others.
pub fn model_lock() -> std::sync::MutexGuard<'static, ()> {
    MODEL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// The repo's `models/` directory, or None when it is absent (e.g. a
/// fresh clone before `scripts/download-models.sh`) - the real-model
/// tests skip silently in that case.
pub fn repo_models_dir() -> Option<std::path::PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
    p.is_dir().then_some(p)
}

/// Write an executable fake tool to `dir` and return its absolute path
/// (commands are looked up by exact path, so no PATH or env mutation is
/// needed - safe under parallel tests).
pub fn make_fake_tool(dir: &Path, name: &str, body: &str) -> String {
    let path = dir.join(name);
    std::fs::write(&path, body).unwrap();
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms).unwrap();
    path.to_str().unwrap().to_string()
}

/// Temporarily put `dir` first on PATH (process-global) and set or clear
/// WAYLAND_DISPLAY; both are restored on drop. Prepending (not replacing)
/// keeps the fake tools' own `cat` resolvable, while any bare name the
/// tested code spawns still resolves to the fake in `dir` first, regardless
/// of what the machine has installed.
///
/// `new` also holds `ENV_LOCK` and (behind it) `TOOL_SPAWN_LOCK` for the
/// guard's whole lifetime - always in that fixed order, so an
/// `EnvPatch`-based test can neither race another env-mutating test nor
/// any other test's fake-tool spawns (both would otherwise be invisible
/// flake sources: a leaked sandboxed PATH, or ETXTBSY on concurrent
/// `#!/bin/sh` execs).
pub struct EnvPatch {
    _env: tokio::sync::MutexGuard<'static, ()>,
    _spawn: tokio::sync::MutexGuard<'static, ()>,
    old_path: String,
    old_wayland: Option<std::ffi::OsString>,
}

impl EnvPatch {
    pub async fn new(dir: &Path, wayland: bool) -> Self {
        let env = ENV_LOCK.lock().await;
        let spawn = TOOL_SPAWN_LOCK.lock().await;
        let old_path = std::env::var("PATH").unwrap_or_default();
        let old_wayland = std::env::var_os("WAYLAND_DISPLAY");
        std::env::set_var(
            "PATH",
            format!("{}:{}", dir.display(), old_path),
        );
        if wayland {
            std::env::set_var("WAYLAND_DISPLAY", "wayland-0");
        } else {
            std::env::remove_var("WAYLAND_DISPLAY");
        }
        Self {
            _env: env,
            _spawn: spawn,
            old_path,
            old_wayland,
        }
    }
}

impl Drop for EnvPatch {
    fn drop(&mut self) {
        std::env::set_var("PATH", self.old_path.clone());
        match &self.old_wayland {
            Some(v) => std::env::set_var("WAYLAND_DISPLAY", v),
            None => std::env::remove_var("WAYLAND_DISPLAY"),
        }
    }
}
