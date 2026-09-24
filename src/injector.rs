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
pub async fn type_text(text: &str) -> io::Result<()> {
    with_timeout("ydotool type", async {
        let output = tokio::process::Command::new("ydotool")
            .arg("type")
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
    let attempts: [(&str, &[&str]); 2] = if wayland {
        [("wl-copy", &[]), ("xclip", &["-selection", "clipboard"])]
    } else {
        [("xclip", &["-selection", "clipboard"]), ("wl-copy", &[])]
    };
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
        stdin.write_all(input.as_bytes()).await?;
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
}
