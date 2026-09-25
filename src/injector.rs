use std::io;
use std::time::Duration;

/// Hard cap on any single subprocess-based injector operation (ydotool,
/// xclip/wl-copy). Without this, a hung subprocess blocks the entire
/// session-stop path: everything here runs strictly serialized through one
/// injector task, which `Engine::stop()` waits on (bounded, but only at
/// 30s) before it can emit `StateChanged("Idle")` - so a hang anywhere in
/// here previously meant the HUD stayed visibly stuck on screen for up to
/// 30 seconds (observed live). Every call in this module is wrapped in
/// this timeout so a hang degrades to one logged error in a few seconds
/// instead.
const SUBPROCESS_TIMEOUT: Duration = Duration::from_secs(5);

async fn with_timeout<T>(
    what: &str,
    fut: impl std::future::Future<Output = io::Result<T>>,
) -> io::Result<T> {
    match tokio::time::timeout(SUBPROCESS_TIMEOUT, fut).await {
        Ok(res) => res,
        Err(_) => Err(io::Error::new(
            io::ErrorKind::TimedOut,
            format!("{what} did not finish within {SUBPROCESS_TIMEOUT:?}"),
        )),
    }
}

/// Type `text` into whatever window has keyboard focus, using ydotool.
///
/// Each call is a single `ydotool type` invocation. Serialization (no overlapping
/// typing) is guaranteed by the caller feeding this through a single-consumer task.
///
/// The text is always passed after `--` (end-of-options): ydotool parses its
/// CLI with boost::program_options, so a transcript chunk that starts with `-`
/// (a hyphenated list item, "-20 degrees", ...) would otherwise be rejected
/// as "unrecognised option" and silently lost - ydotool even exits 0 in that
/// case, so without `--` the word simply never arrives with no error logged.
pub async fn type_text(text: &str) -> io::Result<()> {
    with_timeout("ydotool type", async {
        let output = tokio::process::Command::new("ydotool")
            .arg("type")
            .arg("--")
            .arg(text)
            .output()
            .await
            .map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!("failed to spawn ydotool (is it installed and on PATH?): {e}"),
                )
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(io::Error::other(format!(
                "ydotool exited with {}: {}",
                output.status,
                stderr.trim()
            )));
        }
        Ok(())
    })
    .await
}

/// Insert `text` into whatever window has keyboard focus as a single paste
/// (clipboard set + Ctrl+V), instead of `type_text`'s synthesized
/// keystroke-per-character typing. Used for the one-shot injection at the
/// end of a `Deferred`-mode session: a paste lands instantly, rather than
/// visibly "typing itself out" character by character (which `type_text`
/// does even called once with a whole sentence, since ydotool simulates
/// real keystrokes) and without firing the target app's per-keystroke
/// handlers for a block of text the user never watched arrive.
pub async fn paste_text(text: &str) -> io::Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    let start = std::time::Instant::now();
    copy_to_clipboard(text).await?;
    tracing::debug!("clipboard set in {:?}", start.elapsed());
    let paste_start = std::time::Instant::now();
    with_timeout("ydotool key ctrl+v", async {
        let output = tokio::process::Command::new("ydotool")
            .args(["key", "ctrl+v"])
            .output()
            .await
            .map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!("failed to spawn ydotool (is it installed and on PATH?): {e}"),
                )
            })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(io::Error::other(format!(
                "ydotool exited with {}: {}",
                output.status,
                stderr.trim()
            )));
        }
        Ok(())
    })
    .await?;
    tracing::debug!("paste keystroke sent in {:?}", paste_start.elapsed());
    Ok(())
}

/// Set the system clipboard to `text`. Tries the tool matching the current
/// session first (`wl-copy` under Wayland, `xclip` under X11 - detected via
/// `WAYLAND_DISPLAY`), then falls back to the other if the first is
/// missing or fails, so a misdetected session type still has a chance to
/// work.
async fn copy_to_clipboard(text: &str) -> io::Result<()> {
    let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
    try_clipboard_tools(text, &clipboard_attempts(wayland)).await
}

/// The clipboard tools to try, in order, for one session type: the tool
/// matching the session first (wl-copy under Wayland, xclip under X11),
/// the other as the fallback. xclip always needs `-selection clipboard`
/// (its default selection is PRIMARY, which would not paste on Ctrl+V).
fn clipboard_attempts(wayland: bool) -> [(&'static str, &'static [&'static str]); 2] {
    if wayland {
        [("wl-copy", &[]), ("xclip", &["-selection", "clipboard"])]
    } else {
        [("xclip", &["-selection", "clipboard"]), ("wl-copy", &[])]
    }
}

/// Pipe `text` into each `(command, args)` in order; the first tool that
/// succeeds wins, and when every attempt fails the last error is returned
/// (the empty-attempts list reports "no tool available").
async fn try_clipboard_tools(text: &str, attempts: &[(&str, &[&str])]) -> io::Result<()> {
    let mut last_err = None;
    for (cmd, args) in attempts {
        match with_timeout(cmd, run_with_stdin(cmd, args, text)).await {
            Ok(()) => return Ok(()),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| io::Error::other("no clipboard tool available (wl-copy/xclip)")))
}

async fn run_with_stdin(cmd: &str, args: &[&str], input: &str) -> io::Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut child = tokio::process::Command::new(cmd)
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        // A broken pipe here means the tool exited before reading all of
        // stdin (a fast failure: bad args, no display, ...) - its exit
        // status and stderr below carry the real reason, so the write error
        // must not mask them. Any other write error is propagated.
        if let Err(e) = stdin.write_all(input.as_bytes()).await {
            if e.kind() != io::ErrorKind::BrokenPipe {
                return Err(e);
            }
        }
        // Dropping `stdin` here (end of scope) closes the pipe, which is
        // what tells xclip it has read the whole input.
    }
    // xclip (unlike wl-copy) intentionally forks into the background to
    // keep serving the CLIPBOARD selection after this process would
    // otherwise exit - X11 selections need a live owner process, and
    // GNOME's own clipboard handling takes over from there so pasted
    // content survives. Waiting for it to fully exit can therefore hang
    // indefinitely: verified live, it hit our outer subprocess timeout
    // every time in this daemon's actual process context (did not
    // reproduce testing xclip standalone from an interactive shell - the
    // detach behavior is sensitive to the parent's session/process
    // group). Give it a short grace period to report a real, fast failure
    // (bad args, no display); if it is still running after that, treat it
    // as the expected background-server case, not a hang.
    const DETACH_GRACE: Duration = Duration::from_millis(400);
    match tokio::time::timeout(DETACH_GRACE, child.wait()).await {
        // Exited within the grace period: either a fast success (wl-copy
        // always does this) or a fast, real failure - both meaningful.
        Ok(Ok(status)) if !status.success() => {
            let mut stderr_buf = String::new();
            if let Some(mut stderr) = child.stderr.take() {
                let _ = stderr.read_to_string(&mut stderr_buf).await;
            }
            Err(io::Error::other(format!(
                "{cmd} exited with {status}: {}",
                stderr_buf.trim()
            )))
        }
        Ok(Ok(_)) => Ok(()),
        Ok(Err(e)) => Err(e),
        // Still running past the grace period: the expected xclip
        // backgrounding case, not a hang. Leave it running (tokio does
        // not kill-on-drop by default) - it will keep serving the
        // clipboard until something else takes ownership.
        Err(_) => Ok(()),
    }
}

/// Delete `n` characters from the end of the focused input, using ydotool.
/// One ydotool invocation repeats the BackSpace key `n` times with no
/// inter-key delay (verified ~0.1 ms per key at the uinput level).
pub async fn backspaces(n: u32) -> io::Result<()> {
    if n == 0 {
        return Ok(());
    }
    with_timeout("ydotool key Backspace", async {
        let output = tokio::process::Command::new("ydotool")
            .args(["key", "--repeat", &n.to_string(), "--delay", "0", "--repeat-delay", "0", "Backspace"])
            .output()
            .await
            .map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!("failed to spawn ydotool (is it installed and on PATH?): {e}"),
                )
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(io::Error::other(format!(
                "ydotool exited with {}: {}",
                output.status,
                stderr.trim()
            )));
        }
        Ok(())
    })
    .await
}

/// Compute how to transform the currently typed text into `new` using the
/// minimal number of edits, given that `typed` is a prefix-anchored edit
/// target (we can only backspace from the end and append).
///
/// Returns `(backspaces, to_type)`: press BackSpace `backspaces` times, then
/// type `to_type`. When `typed` is a prefix of `new` this is a pure
/// extension (0 backspaces). When they diverge, it rewinds to the longest
/// common prefix and retypes the rest.
pub fn diff<'a>(typed: &str, new: &'a str) -> (usize, &'a str) {
    let typed_chars: Vec<char> = typed.chars().collect();
    let common = typed_chars
        .iter()
        .zip(new.chars())
        .take_while(|(t, n)| *t == n)
        .count();
    let backspaces = typed_chars.len() - common;
    // Byte offset of the common prefix in `new`.
    let byte_offset: usize = new.chars().take(common).map(|c| c.len_utf8()).sum();
    (backspaces, &new[byte_offset..])
}

/// MVP punctuation stand-in: capitalize the first letter, keep the rest as-is.
pub fn capitalize_first(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(c) => {
            let mut out = String::with_capacity(text.len());
            out.extend(c.to_uppercase());
            out.extend(chars);
            out
        }
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{make_fake_tool, EnvPatch};

    #[test]
    fn capitalizes_first_char() {
        assert_eq!(capitalize_first("hello world"), "Hello world");
        assert_eq!(capitalize_first("HELLO"), "HELLO");
        assert_eq!(capitalize_first(""), "");
        assert_eq!(capitalize_first("123 abc"), "123 abc");
    }

    #[test]
    fn diff_pure_extension() {
        assert_eq!(diff("", "hello"), (0, "hello"));
        assert_eq!(diff("Hello", "Hello world"), (0, " world"));
        assert_eq!(diff("Hello", "Hello"), (0, ""));
    }

    #[test]
    fn diff_tail_correction() {
        // Diverge at the tail: rewind to common prefix "The", retype "re".
        assert_eq!(diff("Their", "There"), (2, "re"));
        assert_eq!(diff("con", "connect"), (0, "nect"));
        // Full rewind when first char changes.
        assert_eq!(diff("the", "The"), (3, "The"));
    }

    #[test]
    fn diff_shrinking_target() {
        // Target shorter than typed: backspace the rest, type nothing.
        assert_eq!(diff("Hello", "Hell"), (1, ""));
        assert_eq!(diff("abc", ""), (3, ""));
    }

    #[test]
    fn diff_multibyte() {
        // Common prefix must be counted in chars, not bytes.
        assert_eq!(diff("café", "cafés"), (0, "s"));
        // Replace the multibyte é with e: one backspace, type "e".
        assert_eq!(diff("café", "cafe"), (1, "e"));
    }

    #[test]
    fn diff_empty_both_sides() {
        assert_eq!(diff("", ""), (0, ""));
    }

    #[test]
    fn diff_multibyte_boundaries() {
        // `new` is a char-prefix of `typed` ending just before a multibyte
        // char: the backspace count is in chars, and the retype offset must
        // land on a char boundary in `new` (it does: it is a sum of char
        // widths, never a byte mid-sequence).
        assert_eq!(diff("café", "caf"), (1, ""));
        // Diverge at a multibyte char: rewind to the common prefix, retype
        // the (ASCII) replacement and everything after it.
        assert_eq!(diff("naïve", "naive"), (3, "ive"));
        // Pure extension past a multibyte common prefix.
        assert_eq!(diff("café", "café au lait"), (0, " au lait"));
        // Full replacement of a single multibyte char.
        assert_eq!(diff("é", "e"), (1, "e"));
    }

    #[test]
    fn capitalize_first_multichar_uppercase() {
        // ß uppercases to two chars (SS), not one: the extend-based build
        // must absorb the whole grapheme without truncating the rest.
        assert_eq!(capitalize_first("ßtraße"), "SStraße");
    }

    #[test]
    fn capitalize_first_leading_non_letters() {
        // Whitespace-only and leading punctuation: "capitalize" is a no-op
        // on them, and nothing is dropped or reordered.
        assert_eq!(capitalize_first("   "), "   ");
        assert_eq!(capitalize_first(", hello"), ", hello");
    }

    #[tokio::test]
    async fn empty_paste_is_a_noop() {
        // A deferred session in which the user erased everything ends with
        // an empty transcript: the one-shot stop paste must succeed without
        // touching the clipboard or the focused app (no subprocess spawned).
        assert!(paste_text("").await.is_ok());
    }

    #[tokio::test]
    async fn zero_backspaces_is_a_noop() {
        // Zero backspaces must not spawn ydotool at all.
        assert!(backspaces(0).await.is_ok());
    }

    #[tokio::test]
    async fn clipboard_tool_receives_stdin_and_success_wins() {
        let _guard = crate::testutil::TOOL_SPAWN_LOCK.lock().await;
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("got.txt");
        let tool = make_fake_tool(tmp.path(), "tool", &format!("#!/bin/sh\ncat > {}\n", out.display()));
        try_clipboard_tools("hello clipboard", &[(tool.as_str(), &[])]).await.unwrap();
        assert_eq!(std::fs::read_to_string(&out).unwrap(), "hello clipboard");
    }

    #[tokio::test]
    async fn clipboard_tool_failure_reports_stderr() {
        let _guard = crate::testutil::TOOL_SPAWN_LOCK.lock().await;
        let tmp = tempfile::tempdir().unwrap();
        let tool = make_fake_tool(tmp.path(), "tool", "#!/bin/sh\necho boom >&2\nexit 1\n");
        let err = try_clipboard_tools("x", &[(tool.as_str(), &[])]).await.unwrap_err();
        assert!(err.to_string().contains("boom"), "{err}");
    }

    #[tokio::test]
    async fn clipboard_missing_tool_is_an_error() {
        let _guard = crate::testutil::TOOL_SPAWN_LOCK.lock().await;
        // A nonexistent command fails at spawn time, before any timeout.
        let err = try_clipboard_tools("x", &[("/nonexistent/clipboard-tool-xyz", &[])]).await.unwrap_err();
        assert!(matches!(err.kind(), io::ErrorKind::NotFound), "{err}");
    }

    #[tokio::test]
    async fn clipboard_backgrounding_is_treated_as_success() {
        let _guard = crate::testutil::TOOL_SPAWN_LOCK.lock().await;
        // xclip forks into the background to keep serving the selection: a
        // tool still running after the detach grace period must count as
        // success, and we must not wait for it to actually exit (2 s here).
        let tmp = tempfile::tempdir().unwrap();
        let tool = make_fake_tool(tmp.path(), "tool", "#!/bin/sh\nsleep 2\n");
        let start = std::time::Instant::now();
        try_clipboard_tools("x", &[(tool.as_str(), &[])]).await.unwrap();
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_secs(1), "took {elapsed:?}");
    }

    #[tokio::test]
    async fn clipboard_falls_back_to_second_tool() {
        let _guard = crate::testutil::TOOL_SPAWN_LOCK.lock().await;
        let tmp = tempfile::tempdir().unwrap();
        let bad = make_fake_tool(tmp.path(), "bad", "#!/bin/sh\nexit 1\n");
        let good = make_fake_tool(tmp.path(), "good", "#!/bin/sh\ncat >/dev/null\n");
        try_clipboard_tools("x", &[(bad.as_str(), &[]), (good.as_str(), &[])]).await.unwrap();
    }

    #[tokio::test]
    async fn clipboard_all_tools_fail_returns_last_error() {
        let _guard = crate::testutil::TOOL_SPAWN_LOCK.lock().await;
        let tmp = tempfile::tempdir().unwrap();
        let first = make_fake_tool(tmp.path(), "first", "#!/bin/sh\necho first-fail >&2\nexit 1\n");
        let second = make_fake_tool(tmp.path(), "second", "#!/bin/sh\necho second-fail >&2\nexit 1\n");
        let err = try_clipboard_tools(
            "x",
            &[(first.as_str(), &[]), (second.as_str(), &[])],
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("second-fail"), "{err}");
    }

    #[test]
    fn clipboard_attempts_orders_the_session_tool_first() {
        let wayland = clipboard_attempts(true);
        assert_eq!(wayland[0].0, "wl-copy");
        assert!(wayland[0].1.is_empty());
        assert_eq!(wayland[1].0, "xclip");
        assert_eq!(wayland[1].1, ["-selection", "clipboard"]);
        let x11 = clipboard_attempts(false);
        assert_eq!(x11[0].0, "xclip");
        assert_eq!(x11[0].1, ["-selection", "clipboard"]);
        assert_eq!(x11[1].0, "wl-copy");
        assert!(x11[1].1.is_empty());
    }

    /// A transcript chunk that starts with `-` must reach ydotool as a
    /// positional argument (after the `--` end-of-options marker), not be
    /// parsed as a ydotool CLI option - otherwise the word is silently lost
    /// (ydotool's boost::program_options rejects it as "unrecognised
    /// option" and still exits 0).
    #[tokio::test]
    async fn type_text_passes_leading_dash_text_after_end_of_options() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        make_fake_tool(d, "ydotool", &format!("#!/bin/sh\nprintf '%s\\n' \"$@\" > {}\n", d.join("ydotool.log").display()));
        let _env = EnvPatch::new(d, true).await;
        type_text("- two hyphenated items").await.unwrap();
        assert_eq!(
            std::fs::read_to_string(d.join("ydotool.log")).unwrap(),
            "type\n--\n- two hyphenated items\n"
        );
    }

    /// The Deferred-mode one-shot paste, end to end, on a Wayland session:
    /// wl-copy (the session tool) receives the whole text, the xclip
    /// fallback is never tried, and then - and only then - exactly
    /// `ydotool key ctrl+v` is sent.
    #[tokio::test]
    async fn paste_text_copies_via_wl_copy_then_pastes_under_wayland() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        make_fake_tool(d, "wl-copy", &format!("#!/bin/sh\ncat > {}\n", d.join("wl-copy.log").display()));
        make_fake_tool(
            d,
            "xclip",
            &format!("#!/bin/sh\nprintf '%s ' \"$@\" > {}\ncat > {}\n", d.join("xclip.args").display(), d.join("xclip.log").display()),
        );
        make_fake_tool(d, "ydotool", &format!("#!/bin/sh\nprintf '%s\\n' \"$@\" > {}\n", d.join("ydotool.log").display()));
        let _env = EnvPatch::new(d, true).await;
        paste_text("hello world").await.unwrap();
        assert_eq!(std::fs::read_to_string(d.join("wl-copy.log")).unwrap(), "hello world");
        assert!(!d.join("xclip.log").exists());
        assert_eq!(std::fs::read_to_string(d.join("ydotool.log")).unwrap(), "key\nctrl+v\n");
    }

    /// The same pipeline on an X11 session: xclip is tried first and must
    /// be given `-selection clipboard` (xclip's default selection is
    /// PRIMARY, which Ctrl+V would not paste); wl-copy is only the
    /// fallback.
    #[tokio::test]
    async fn paste_text_prefers_xclip_under_x11_and_passes_selection_arg() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        make_fake_tool(d, "wl-copy", &format!("#!/bin/sh\ncat > {}\n", d.join("wl-copy.log").display()));
        make_fake_tool(
            d,
            "xclip",
            &format!("#!/bin/sh\nprintf '%s ' \"$@\" > {}\ncat > {}\n", d.join("xclip.args").display(), d.join("xclip.log").display()),
        );
        make_fake_tool(d, "ydotool", &format!("#!/bin/sh\nprintf '%s\\n' \"$@\" > {}\n", d.join("ydotool.log").display()));
        let _env = EnvPatch::new(d, false).await;
        paste_text("hi there").await.unwrap();
        assert_eq!(std::fs::read_to_string(d.join("xclip.log")).unwrap(), "hi there");
        assert_eq!(std::fs::read_to_string(d.join("xclip.args")).unwrap(), "-selection clipboard ");
        assert!(!d.join("wl-copy.log").exists());
        assert_eq!(std::fs::read_to_string(d.join("ydotool.log")).unwrap(), "key\nctrl+v\n");
    }

    /// Safety property: if the clipboard cannot be set (both tools fail),
    /// the paste keystroke must NOT be sent - a bare Ctrl+V would paste
    /// whatever stale text the clipboard already held.
    #[tokio::test]
    async fn paste_text_does_not_send_paste_keystroke_when_clipboard_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        make_fake_tool(d, "wl-copy", "#!/bin/sh\necho clip-fail >&2\nexit 1\n");
        make_fake_tool(d, "xclip", "#!/bin/sh\necho xclip-fail >&2\nexit 1\n");
        make_fake_tool(d, "ydotool", &format!("#!/bin/sh\nprintf '%s\\n' \"$@\" > {}\n", d.join("ydotool.log").display()));
        let _env = EnvPatch::new(d, true).await;
        // Both attempts fail: the error is the last one tried (xclip).
        let err = paste_text("x").await.unwrap_err();
        assert!(err.to_string().contains("xclip-fail"), "{err}");
        assert!(!d.join("ydotool.log").exists());
    }
}
