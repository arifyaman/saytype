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
}
