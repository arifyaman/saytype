use std::io;

/// Type `text` into whatever window has keyboard focus, using ydotool.
///
/// Each call is a single `ydotool type` invocation. Serialization (no overlapping
/// typing) is guaranteed by the caller feeding this through a single-consumer task.
pub async fn type_text(text: &str) -> io::Result<()> {
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
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("ydotool exited with {}: {}", output.status, stderr.trim()),
        ));
    }
    Ok(())
}

/// Delete `n` characters from the end of the focused input, using ydotool.
/// One ydotool invocation repeats the BackSpace key `n` times with no
/// inter-key delay (verified ~0.1 ms per key at the uinput level).
pub async fn backspaces(n: u32) -> io::Result<()> {
    if n == 0 {
        return Ok(());
    }
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
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("ydotool exited with {}: {}", output.status, stderr.trim()),
        ));
    }
    Ok(())
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
