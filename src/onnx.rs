//! Plausibility checking for ONNX model files, before they are handed to
//! the sherpa-onnx C++ library.
//!
//! Why this exists: a corrupt model file (interrupted download, disk full,
//! bit rot) makes the C++ model loader throw a C++ exception across the
//! FFI boundary. Rust cannot catch foreign exceptions, so the whole
//! process aborts (SIGABRT) - the daemon dies instead of reporting an
//! error. All sherpa-onnx load paths (VAD, streaming ASR, batch ASR,
//! online punctuation) were verified to abort this way on truncated model
//! files. The check below rejects the realistic corruption cases with a
//! clean, actionable error before any FFI load runs.

use anyhow::Context;
use std::io::{self, Read};

/// Marker error type so callers can distinguish "model file found but
/// corrupt" (a hard error - the user must re-download) from "model not
/// found" (a legitimate fallback condition in the auto/streaming backend
/// chain). Check the chain with `err.downcast_ref::<CorruptModel>()`.
#[derive(Debug)]
pub struct CorruptModel;

impl std::fmt::Display for CorruptModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("corrupt model file")
    }
}

impl std::error::Error for CorruptModel {}

/// Stream a file and check that it has a plausible ONNX envelope.
///
/// ONNX files are Protocol Buffers. This validates the envelope only:
///
/// - the file must start with the `ir_version` field (field 1, wire type
///   0, tag byte `0x08`) - the first field of every real ONNX file, which
///   rejects empty files and non-ONNX garbage in one test;
/// - the top-level fields must then walk to the end of the file without a
///   length-delimited field running past the end (truncation) and without
///   reserved wire types (3, 4, 6, 7 - group start/end, which ONNX never
///   uses).
///
/// That catches the realistic corruption cases - an interrupted download
/// (truncated file) or a file that is not an ONNX model at all (wrong
/// bytes) - which would otherwise abort the process. A file that is
/// structurally valid protobuf but semantically broken ONNX is *not*
/// caught; that is far rarer and would need a full ONNX parser.
///
/// `Ok(false)` means "structurally not an ONNX file" (garbage, truncated
/// at any point, or malformed - the interrupted-download case); `Err`
/// means the file could not be read at all (I/O failure).
pub fn is_plausible_onnx<R: Read>(r: R) -> io::Result<bool> {
    match walk(r) {
        Ok(v) => Ok(v),
        // Truncation anywhere (an interrupted download) is "not an ONNX
        // file", not an I/O failure.
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
        // A varint over our 5-byte cap is malformed protobuf (no real ONNX
        // top-level field does this): also "not an ONNX file", so the
        // caller gets the re-download message rather than a raw I/O error.
        Err(e) if e.kind() == io::ErrorKind::InvalidData => Ok(false),
        Err(e) => Err(e),
    }
}

/// The actual envelope walk: `Ok(true)` when the top-level protobuf fields
/// walk to a clean end of file, `Ok(false)` when the structure is simply
/// wrong, `Err(UnexpectedEof)` when the file ends in the middle of a field
/// (truncation), and `Err(InvalidData)` when a varint exceeds the 5-byte
/// cap (malformed - `is_plausible_onnx` folds that into `Ok(false)`).
fn walk<R: Read>(mut r: R) -> io::Result<bool> {
    let first = read_byte(&mut r)?;
    if first != Some(0x08) {
        return Ok(false);
    }
    // First field: tag 0x08 = field 1 (ir_version), wire type 0 (varint).
    // The file ending inside the very first field is truncation.
    if read_varint(&mut r)?.is_none() {
        return Err(truncated());
    }
    loop {
        match read_varint(&mut r)? {
            // Clean end of file exactly at a field boundary.
            None => return Ok(true),
            Some(tag) => match tag & 0x07 {
                0 => {
                    // Varint value: an early end is truncation.
                    if read_varint(&mut r)?.is_none() {
                        return Err(truncated());
                    }
                }
                1 => skip_fixed(&mut r, 8)?,
                2 => {
                    // Length-delimited: read the length, then skip it. A
                    // declared length past EOF is the truncated-download
                    // case.
                    let Some(len) = read_varint(&mut r)? else {
                        return Err(truncated());
                    };
                    skip_fixed(&mut r, len as u64)?;
                }
                5 => skip_fixed(&mut r, 4)?,
                // Wire types 3, 4, 6, 7 are reserved/invalid in modern
                // protobuf: no real ONNX file uses them.
                _ => return Ok(false),
            },
        }
    }
}

fn truncated() -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "onnx file truncated")
}

/// One byte, or `None` at a clean end of file.
fn read_byte(r: &mut impl Read) -> io::Result<Option<u8>> {
    let mut b = [0u8; 1];
    match r.read(&mut b)? {
        0 => Ok(None),
        _ => Ok(Some(b[0])),
    }
}

/// A base-128 varint of at most 5 bytes (a 6th byte is malformed for our
/// purposes). `Ok(None)` at a clean end of file before any byte;
/// `Err(UnexpectedEof)` if the file ends mid-varint; `Err(InvalidData)`
/// if the varint runs past 5 bytes.
fn read_varint(r: &mut impl Read) -> io::Result<Option<usize>> {
    let mut result: usize = 0;
    for shift in (0..35).step_by(7) {
        match read_byte(r)? {
            None => {
                return if shift == 0 {
                    Ok(None)
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "varint truncated",
                    ))
                };
            }
            Some(b) => {
                result |= ((b & 0x7F) as usize) << shift;
                if b & 0x80 == 0 {
                    return Ok(Some(result));
                }
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "varint longer than 5 bytes",
    ))
}

/// Skip exactly `n` bytes; an early end of file is an error (truncation).
fn skip_fixed(r: &mut impl Read, n: u64) -> io::Result<()> {
    let copied = io::copy(&mut r.take(n), &mut io::sink())?;
    if copied != n {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "field payload truncated",
        ));
    }
    Ok(())
}

/// Check a model file on disk. Returns a clean, actionable error when the
/// file is missing, unreadable, or not a plausible ONNX file.
///
/// `what` labels the file in the message (e.g. "VAD model",
/// "Nemotron encoder") so the user knows which download to re-run.
pub fn check_onnx_file(path: &std::path::Path, what: &str) -> anyhow::Result<()> {
    let file =
        std::fs::File::open(path).with_context(|| format!("opening {what} {}", path.display()))?;
    if !is_plausible_onnx(file).with_context(|| format!("reading {what} {}", path.display()))? {
        return Err(anyhow::Error::new(CorruptModel).context(format!(
            "{what} {} is not a valid ONNX model (corrupt or incomplete - an \
             interrupted download?); delete it and re-run scripts/download-models.sh",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::repo_models_dir;

    /// A minimal structurally-valid ONNX-like protobuf: ir_version=3,
    /// then a length-delimited field (graph) and a varint field.
    fn valid_envelope() -> Vec<u8> {
        vec![
            0x08, // field 1 (ir_version), wire 0
            0x03, // value 3
            0x42, // field 8 (graph), wire 2
            0x02, // length 2
            b'g', b'r', 0x18, // field 3 (model_version), wire 0
            0x05, // value 5
        ]
    }

    #[test]
    fn valid_envelope_accepted() {
        assert!(is_plausible_onnx(&valid_envelope()[..]).unwrap());
    }

    #[test]
    fn empty_file_rejected() {
        assert!(!is_plausible_onnx(&b""[..]).unwrap());
    }

    #[test]
    fn non_onnx_garbage_rejected_on_first_byte() {
        // Random bytes fail the first-byte test (0xAB is not the 0x08 tag).
        assert!(!is_plausible_onnx(&vec![0xAB; 100][..]).unwrap());
        // A length-delimited file whose first field is not field 1.
        assert!(!is_plausible_onnx(&[0x12, 0x02, 0x78, 0x79][..]).unwrap());
    }

    #[test]
    fn truncated_payload_rejected() {
        let mut v = valid_envelope();
        // Cut inside the 2-byte graph payload.
        v.truncate(v.len() - 1);
        assert!(!is_plausible_onnx(&v[..]).unwrap());
    }

    #[test]
    fn truncated_length_varint_rejected() {
        let mut v = valid_envelope();
        // Keep the graph tag but drop its length byte.
        v.truncate(3);
        assert!(!is_plausible_onnx(&v[..]).unwrap());
    }

    #[test]
    fn dangling_first_field_rejected() {
        // Tag 0x08 with its varint value missing.
        assert!(!is_plausible_onnx(&[0x08][..]).unwrap());
        // Continuation bit set, file ends: truncated varint.
        assert!(!is_plausible_onnx(&[0x08, 0x80][..]).unwrap());
    }

    #[test]
    fn reserved_wire_type_rejected() {
        // Field 1 again with wire type 3 (group start): reserved.
        let v = [0x08, 0x03, 0x0B, 0x01];
        assert!(!is_plausible_onnx(&v[..]).unwrap());
    }

    #[test]
    fn fixed_width_fields_accepted_and_truncation_rejected() {
        // The two remaining wire types the envelope walker accepts are the
        // fixed-width ones: wire type 1 (fixed 64-bit, skip 8 bytes) and wire
        // type 5 (fixed 32-bit, skip 4 bytes). A full field of the right
        // width walks cleanly; a file ending inside the fixed payload is the
        // interrupted-download truncation case (Ok(false), not an I/O error).
        // Field 2, wire type 1 -> tag 0x11, then exactly 8 payload bytes.
        let fixed64 = [0x08, 0x03, 0x11, 0u8, 0, 0, 0, 0, 0, 0, 0];
        assert!(is_plausible_onnx(&fixed64[..]).unwrap());
        // Same tag but only 3 of the 8 payload bytes present: truncated.
        assert!(!is_plausible_onnx(&fixed64[..6][..]).unwrap());
        // Field 2, wire type 5 -> tag 0x15, then exactly 4 payload bytes.
        let fixed32 = [0x08, 0x03, 0x15, 0u8, 0, 0, 0];
        assert!(is_plausible_onnx(&fixed32[..]).unwrap());
        // Same tag but only 2 of the 4 payload bytes present: truncated.
        assert!(!is_plausible_onnx(&fixed32[..5][..]).unwrap());
    }

    #[test]
    fn multi_byte_varint_tag_and_value_accepted() {
        // A top-level field number >= 16 encodes its tag as a multi-byte
        // varint (field 16, wire type 0 -> tag bytes 0x80 0x01); real
        // protobuf messages use these, so the envelope must accept them.
        let tag = [0x08, 0x03, 0x80, 0x01, 0x01];
        assert!(is_plausible_onnx(&tag[..]).unwrap());
        // A multi-byte varint *value* (ir_version = 300 = 0xAC 0x02) too.
        let value = [0x08, 0xAC, 0x02, 0x18, 0x05];
        assert!(is_plausible_onnx(&value[..]).unwrap());
        // A multi-byte tag on a length-delimited field (field 16, wire 2).
        let tag_len = [0x08, 0x03, 0x82, 0x01, 0x02, b'x', b'y'];
        assert!(is_plausible_onnx(&tag_len[..]).unwrap());
    }

    #[test]
    fn over_long_varint_is_not_onnx_not_io_error() {
        // A varint whose first 5 bytes all carry the continuation bit never
        // terminates within the 5-byte cap: malformed protobuf, classified
        // as "not an ONNX file" (Ok(false)) rather than an I/O error, so
        // check_onnx_file yields the re-download hint.
        let malformed = [0x08, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01];
        assert!(!is_plausible_onnx(&malformed[..]).unwrap());
    }

    #[test]
    fn real_models_pass_and_truncated_reals_fail() {
        let Some(models) = repo_models_dir() else {
            return; // models/ not installed
        };
        let silero = models.join("silero_vad.onnx");
        if !silero.is_file() {
            return;
        }
        let bytes = std::fs::read(&silero).unwrap();
        assert!(
            is_plausible_onnx(&bytes[..]).unwrap(),
            "real silero rejected"
        );
        // An interrupted download of the real model must be caught at
        // several cut points (middle of a payload, near the start, 3/4).
        for frac in [1usize, 3, 7, 9] {
            let cut = bytes.len() * frac / 10;
            assert!(
                !is_plausible_onnx(&bytes[..cut]).unwrap(),
                "truncated silero at {cut} bytes accepted"
            );
        }
        // Every other .onnx model in the tree must pass too.
        let mut found = 0;
        for entry in std::fs::read_dir(&models).unwrap().flatten() {
            if !entry.path().is_dir() {
                continue;
            }
            for sub in std::fs::read_dir(entry.path()).unwrap().flatten() {
                let p = sub.path();
                if p.extension().is_some_and(|e| e == "onnx") {
                    assert!(
                        is_plausible_onnx(&std::fs::read(&p).unwrap()[..]).unwrap(),
                        "{:?} rejected",
                        p
                    );
                    found += 1;
                }
            }
        }
        assert!(found > 0, "no .onnx models found in {}", models.display());
    }

    #[test]
    fn check_onnx_file_reports_missing_and_corrupt() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("nope.onnx");
        let err = check_onnx_file(&missing, "VAD model")
            .unwrap_err()
            .to_string();
        assert!(err.contains("nope.onnx"), "{err}");

        let garbage = tmp.path().join("garbage.onnx");
        std::fs::write(&garbage, vec![0u8; 64]).unwrap();
        let err = check_onnx_file(&garbage, "VAD model")
            .unwrap_err()
            .to_string();
        assert!(err.contains("not a valid ONNX model"), "{err}");
        assert!(err.contains("download-models.sh"), "{err}");
    }

    #[test]
    fn check_onnx_file_accepts_valid() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("good.onnx");
        std::fs::write(&p, valid_envelope()).unwrap();
        check_onnx_file(&p, "VAD model").unwrap();
    }
}
