//! The visible dictation transcript.
//!
//! The single source of truth for what text the target app should contain
//! and what the HUD should show, with mid-dictation word erase/undo applied:
//!
//! - `erase_last` (left mouse press) removes the last visible word: from
//!   the live partial if an utterance is in progress, else from the tail of
//!   the committed text.
//! - `undo_last` (right mouse press) restores the most recently erased word
//!   (LIFO: multiple erases are undone one by one in reverse order), putting
//!   it back at its **original position** in the text.
//! - An erase is restorable only until the next **new word** is transcribed:
//!   a partial that grows beyond any earlier partial of the utterance, or
//!   a final containing words no partial showed. From then on the word is
//!   gone for good (not restorable), matching "the user kept talking".
//!
//! Erased words are tracked two ways, because a word can be erased while it
//! is still only a live hypothesis or after it has committed:
//!
//! - committed erasures hide a slot of the committed token sequence
//!   (restoring un-hides it, so the word returns to where it was);
//! - partial erasures record the token **position** in the current
//!   utterance, so decoder revisions (a token changing or dropping at that
//!   position) keep the mapping correct. At the final, surviving positions
//!   convert into restorable committed-slot erasures.
//!
//! Committed tokens are stored raw (as the ASR emitted them); casing is
//! applied at render time - the first visible word of each segment is
//! capitalized - so erasing/undoing the first word of a segment re-derives
//! the casing of whatever word takes its place.
//!
//! The state machine is pure: no audio, no ASR, no ydotool. The injector
//! task feeds it and applies the resulting buffer edits (see
//! `buffer_target`), and the daemon relays `display()` to the HUD.

use std::collections::BTreeSet;

use crate::injector::capitalize_first;

/// One entry of the LIFO undo stack.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Undo {
    /// A committed slot hidden by a pending erasure; restoring un-hides it.
    Committed(usize),
    /// A token position erased in the current partial; restoring
    /// re-includes the position.
    Partial(usize),
}

/// One dictation session's visible transcript.
#[derive(Debug, Default)]
pub struct Transcript {
    /// The full token sequence of the committed segments, in order.
    /// `None` marks a permanently erased position: a gap the visible text
    /// never fills in. Tokens are raw; casing is applied at render time.
    committed: Vec<Option<String>>,
    /// The start index in `committed` of each committed segment; the last
    /// segment runs to the end of the vec.
    segment_starts: Vec<usize>,
    /// Committed slots currently hidden by a pending (restorable)
    /// erasure. Disjoint with the `None` slots: a slot is either
    /// visible, hidden-pending, or gone.
    hidden: BTreeSet<usize>,
    /// LIFO of erased words that are still restorable; the last element is
    /// what the next `undo_last` restores.
    undo: Vec<Undo>,
    /// The latest partial's tokens (the utterance in progress); empty
    /// between finals.
    partial_tokens: Vec<String>,
    /// The most tokens any partial of this utterance has had so far. A
    /// partial (or final) beyond it contains new words and makes pending
    /// erasures permanent.
    partial_max_tokens: usize,
    /// Token positions permanently erased in the current utterance (never
    /// shown again, not restorable). Disjoint with the `Partial` entries of
    /// `undo`: a position is in exactly one of the two.
    erased: BTreeSet<usize>,
}

impl Transcript {
    pub fn new() -> Self {
        Self::default()
    }

    fn tokens(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_string).collect()
    }

    /// Positions currently excluded from the visible partial: permanently
    /// erased plus pending (restorable) erasures.
    fn excluded_positions(&self) -> BTreeSet<usize> {
        let mut s = self.erased.clone();
        for u in &self.undo {
            if let Undo::Partial(i) = u {
                s.insert(*i);
            }
        }
        s
    }

    /// The current live partial with all erased positions excluded
    /// (whitespace-joined, raw - no casing transform).
    pub fn visible_partial(&self) -> String {
        let excluded = self.excluded_positions();
        self.partial_tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| !excluded.contains(i))
            .map(|(_, t)| t.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// One committed segment's visible tokens, in order (raw).
    fn segment_tokens(&self, start: usize, end: usize) -> Vec<&str> {
        (start..end)
            .filter(|&i| !self.hidden.contains(&i))
            .filter_map(|i| self.committed[i].as_deref())
            .collect()
    }

    /// One committed segment's visible text: its visible tokens joined by a
    /// single space, the first word capitalized; empty when the segment has
    /// no visible tokens.
    fn segment_text(&self, start: usize, end: usize) -> String {
        let toks = self.segment_tokens(start, end);
        if toks.is_empty() {
            return String::new();
        }
        capitalize_first(&toks.join(" "))
    }

    /// The committed text: each segment's visible text, one space between
    /// non-empty segments.
    fn committed_text(&self) -> String {
        let mut segs: Vec<String> = Vec::new();
        let mut prev = 0usize;
        for &end in self.segment_starts.iter().chain(std::iter::once(&self.committed.len())) {
            let t = self.segment_text(prev, end);
            if !t.is_empty() {
                segs.push(t);
            }
            prev = end;
        }
        segs.join(" ")
    }

    /// New words were transcribed: pending erasures become permanent.
    /// Partial positions join the `erased` set; hidden committed slots
    /// become gaps (the token is dropped from the sequence).
    fn flush_undos(&mut self) {
        for u in self.undo.drain(..) {
            match u {
                Undo::Partial(i) => {
                    self.erased.insert(i);
                }
                Undo::Committed(i) => {
                    self.hidden.remove(&i);
                    self.committed[i] = None;
                }
            }
        }
    }

    /// Feed a live partial (streaming path). Returns the visible partial.
    ///
    /// A partial with more tokens than any earlier partial of this
    /// utterance contains new words: pending erasures become permanent
    /// first, so the new words appear without the erased ones.
    pub fn feed_partial(&mut self, text: &str) -> String {
        let tokens = Self::tokens(text);
        if tokens.len() > self.partial_max_tokens {
            self.flush_undos();
            self.partial_max_tokens = tokens.len();
        }
        self.partial_tokens = tokens;
        self.visible_partial()
    }

    /// The utterance in progress committed with this final text
    /// (streaming path). Its tokens are appended to the committed sequence:
    /// permanently erased positions as gaps, pending partial erasures as
    /// hidden slots (same LIFO order, converted to slot indexes), the rest
    /// visible.
    pub fn feed_final(&mut self, text: &str) {
        let tokens = Self::tokens(text);
        if tokens.len() > self.partial_max_tokens {
            // The final contains words no partial ever showed: new words,
            // pending erasures are permanent.
            self.flush_undos();
            self.partial_max_tokens = tokens.len();
        }
        let base = self.committed.len();
        // Pending partial erasures whose position the final still contains
        // become restorable committed-slot erasures; positions the final
        // dropped (the decoder revised the word away) are discarded.
        // Committed erasures of earlier segments pass through unchanged.
        let mut converted: Vec<Undo> = Vec::new();
        for u in self.undo.drain(..) {
            match u {
                Undo::Partial(i) if i < tokens.len() => {
                    let slot = base + i;
                    self.hidden.insert(slot);
                    converted.push(Undo::Committed(slot));
                }
                Undo::Committed(i) => converted.push(Undo::Committed(i)),
                _ => {}
            }
        }
        // Append the segment: permanently erased positions become gaps.
        for (i, tok) in tokens.into_iter().enumerate() {
            self.committed
                .push(if self.erased.contains(&i) { None } else { Some(tok) });
        }
        if base < self.committed.len() {
            self.segment_starts.push(base);
        }

        // Start a fresh utterance.
        self.partial_tokens.clear();
        self.partial_max_tokens = 0;
        self.erased.clear();
        self.undo = converted;
    }

    /// A new utterance's final (batch path: no partials in flight). Any
    /// pending erasures are permanent - a whole new utterance was
    /// transcribed - then the text is committed.
    pub fn feed_batch_final(&mut self, text: &str) {
        self.flush_undos();
        self.partial_tokens.clear();
        self.partial_max_tokens = 0;
        self.erased.clear();
        let tokens = Self::tokens(text);
        if tokens.is_empty() {
            return;
        }
        let base = self.committed.len();
        for tok in tokens {
            self.committed.push(Some(tok));
        }
        self.segment_starts.push(base);
    }

    /// Erase the last visible word (left mouse press). The last word of the
    /// live partial if one is visible, else the last committed token.
    /// Returns true if a word was erased.
    pub fn erase_last(&mut self) -> bool {
        let excluded = self.excluded_positions();
        // The last visible partial token, scanning from the tail.
        let last = (0..self.partial_tokens.len())
            .rev()
            .find(|i| !excluded.contains(i));
        if let Some(i) = last {
            self.undo.push(Undo::Partial(i));
            return true;
        }
        // The last visible committed slot, scanning from the tail.
        let last = (0..self.committed.len())
            .rev()
            .find(|&i| self.committed[i].is_some() && !self.hidden.contains(&i));
        match last {
            Some(i) => {
                self.hidden.insert(i);
                self.undo.push(Undo::Committed(i));
                true
            }
            None => false,
        }
    }

    /// Restore the last erased word (right mouse press), LIFO. Returns true
    /// if an erase was undone. Restoring a committed slot puts the word
    /// back at its original position; restoring a partial position whose
    /// token the decoder since dropped is a no-op on the visible text but
    /// still consumes the undo entry.
    pub fn undo_last(&mut self) -> bool {
        match self.undo.pop() {
            Some(Undo::Committed(i)) => {
                self.hidden.remove(&i);
                true
            }
            Some(Undo::Partial(_)) => true,
            None => false,
        }
    }

    /// The full visible text for the HUD: committed + live partial (raw,
    /// erasures applied).
    pub fn display(&self) -> String {
        let p = self.visible_partial();
        let c = self.committed_text();
        if p.is_empty() {
            c
        } else if c.is_empty() {
            p
        } else {
            format!("{c} {p}")
        }
    }

    /// The text the target buffer should hold: the committed text plus,
    /// while an utterance is in progress, the transformed `stable` prefix
    /// of the visible partial. `stable` is the live-typing stability prefix
    /// (a prefix of the visible partial); an empty `stable` types nothing
    /// of the partial yet.
    pub fn buffer_target(&self, stable: &str) -> String {
        let mut out = self.committed_text();
        if !self.visible_partial().is_empty() && !stable.is_empty() {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&capitalize_first(stable));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_transcript_is_inert() {
        let mut t = Transcript::new();
        assert_eq!(t.display(), "");
        assert_eq!(t.visible_partial(), "");
        assert_eq!(t.buffer_target(""), "");
        assert!(!t.erase_last());
        assert!(!t.undo_last());
    }

    #[test]
    fn partial_display_is_raw_target_is_transformed() {
        let mut t = Transcript::new();
        assert_eq!(t.feed_partial("the quick brown"), "the quick brown");
        assert_eq!(t.display(), "the quick brown");
        // First segment: capitalized, no leading space.
        assert_eq!(t.buffer_target("the quick"), "The quick");
        // Nothing stable yet: only the committed text (none).
        assert_eq!(t.buffer_target(""), "");
    }

    #[test]
    fn finals_transform_and_later_segments_get_a_space() {
        let mut t = Transcript::new();
        t.feed_final("hello world");
        assert_eq!(t.display(), "Hello world");
        t.feed_partial("next sentence");
        assert_eq!(t.display(), "Hello world next sentence");
        assert_eq!(t.buffer_target("next"), "Hello world Next");
        t.feed_final("next sentence here");
        assert_eq!(t.display(), "Hello world Next sentence here");
    }

    #[test]
    fn erase_last_partial_word_and_undo() {
        let mut t = Transcript::new();
        t.feed_partial("the quick brown");
        assert!(t.erase_last());
        assert_eq!(t.display(), "the quick");
        assert_eq!(t.buffer_target("the quick"), "The quick");
        assert!(t.undo_last());
        assert_eq!(t.display(), "the quick brown");
        assert!(!t.undo_last());
    }

    #[test]
    fn erase_multiple_words_undoes_in_reverse_order() {
        let mut t = Transcript::new();
        t.feed_partial("one two three four");
        assert!(t.erase_last());
        assert!(t.erase_last());
        assert_eq!(t.display(), "one two");
        assert!(t.undo_last());
        assert_eq!(t.display(), "one two three");
        assert!(t.undo_last());
        assert_eq!(t.display(), "one two three four");
        assert!(!t.undo_last());
    }

    #[test]
    fn erase_from_committed_when_no_partial() {
        let mut t = Transcript::new();
        t.feed_final("hello world");
        assert!(t.erase_last());
        assert_eq!(t.display(), "Hello");
        assert!(t.erase_last());
        assert_eq!(t.display(), "");
        assert!(t.undo_last());
        assert_eq!(t.display(), "Hello");
        assert!(t.undo_last());
        assert_eq!(t.display(), "Hello world");
    }

    #[test]
    fn erase_spans_segments_and_restores_spaces() {
        let mut t = Transcript::new();
        t.feed_final("first segment");
        t.feed_final("second segment");
        assert_eq!(t.display(), "First segment Second segment");
        assert!(t.erase_last());
        assert_eq!(t.display(), "First segment Second");
        assert!(t.erase_last());
        assert_eq!(t.display(), "First segment");
        assert!(t.erase_last());
        assert_eq!(t.display(), "First");
        assert!(t.erase_last());
        assert_eq!(t.display(), "");
        assert!(t.undo_last());
        assert_eq!(t.display(), "First");
        assert!(t.undo_last());
        assert_eq!(t.display(), "First segment");
        assert!(t.undo_last());
        assert_eq!(t.display(), "First segment Second");
        assert!(t.undo_last());
        assert_eq!(t.display(), "First segment Second segment");
        assert!(!t.undo_last());
    }

    #[test]
    fn new_word_makes_partial_erasure_permanent() {
        let mut t = Transcript::new();
        t.feed_partial("the quick brown");
        assert!(t.erase_last());
        assert_eq!(t.display(), "the quick");
        // A new word ("fox") is transcribed: "brown" is gone for good,
        // even though the decoder still emits it.
        t.feed_partial("the quick brown fox");
        assert_eq!(t.display(), "the quick fox");
        assert!(!t.undo_last());
    }

    #[test]
    fn same_length_revision_keeps_erasure_restorable() {
        let mut t = Transcript::new();
        t.feed_partial("the quick brwn");
        assert!(t.erase_last());
        // No growth: a revision, not a new word. The revised token at the
        // same position is what a restore brings back.
        t.feed_partial("the quick brown");
        assert_eq!(t.display(), "the quick");
        assert!(t.undo_last());
        assert_eq!(t.display(), "the quick brown");
    }

    #[test]
    fn new_utterance_makes_committed_erasure_permanent() {
        let mut t = Transcript::new();
        t.feed_final("hello world");
        assert!(t.erase_last());
        // The user starts a new utterance: "world" is gone for good.
        t.feed_partial("again");
        assert_eq!(t.display(), "Hello again");
        assert!(!t.undo_last());
    }

    #[test]
    fn partial_erasure_survives_commit_as_restorable() {
        let mut t = Transcript::new();
        t.feed_partial("hello world");
        assert!(t.erase_last());
        // The final commits with the same words: no new word, the erasure
        // becomes a restorable committed erasure.
        t.feed_final("hello world");
        assert_eq!(t.display(), "Hello");
        assert!(t.undo_last());
        assert_eq!(t.display(), "Hello world");
    }

    #[test]
    fn final_longer_than_all_partials_flushes_erasure() {
        let mut t = Transcript::new();
        t.feed_partial("hello world");
        assert!(t.erase_last());
        // The final contains a word no partial ever showed: new word, the
        // erased word is permanent and stays out of the final too.
        t.feed_final("hello world again");
        assert_eq!(t.display(), "Hello again");
        assert!(!t.undo_last());
    }

    #[test]
    fn streaming_final_with_new_word_makes_committed_erasure_permanent() {
        let mut t = Transcript::new();
        t.feed_final("hello world");
        assert!(t.erase_last()); // "world" pending (committed slot)
        // A new utterance finalizes with words no partial showed: the
        // pending committed erasure is permanent, leaving a gap that a
        // later segment's casing ignores.
        t.feed_final("new text here");
        assert_eq!(t.display(), "Hello New text here");
        assert!(!t.undo_last());
        // The gap stays a gap: the next segment still renders correctly.
        t.feed_final("next");
        assert_eq!(t.display(), "Hello New text here Next");
    }

    #[test]
    fn final_shorter_than_partial_drops_dangling_erasure() {
        let mut t = Transcript::new();
        t.feed_partial("hello world goodbye");
        assert!(t.erase_last());
        // The decoder revised "goodbye" away in the final: the erasure had
        // nothing left to erase.
        t.feed_final("hello world");
        assert_eq!(t.display(), "Hello world");
        assert!(!t.undo_last());
    }

    #[test]
    fn batch_path_erase_undo_and_new_final_flushes() {
        let mut t = Transcript::new();
        t.feed_batch_final("one two three");
        assert!(t.erase_last());
        assert!(t.erase_last());
        assert_eq!(t.display(), "One");
        assert!(t.undo_last());
        assert_eq!(t.display(), "One two");
        // A new utterance: the still-pending "three" erasure is permanent
        // ("two" was restored and stays).
        t.feed_batch_final("four five");
        assert_eq!(t.display(), "One two Four five");
        assert!(!t.undo_last());
    }

    #[test]
    fn shrinking_partial_then_growth_tracks_positions() {
        let mut t = Transcript::new();
        t.feed_partial("the cat sat");
        assert!(t.erase_last());
        t.feed_partial("the cat"); // the decoder drops "sat"
        assert_eq!(t.display(), "the cat");
        assert!(t.erase_last());
        assert_eq!(t.display(), "the");
        // A genuinely new word arrives: both pending erasures are
        // permanent, tracked by position (the decoder re-emits "sat").
        t.feed_partial("the cat sat down");
        assert_eq!(t.display(), "the down");
        assert!(!t.undo_last());
    }

    #[test]
    fn fully_erased_partial_falls_back_to_committed() {
        let mut t = Transcript::new();
        t.feed_final("done");
        t.feed_partial("new words");
        assert!(t.erase_last());
        assert!(t.erase_last());
        // The partial is fully erased; the next erase hits committed.
        assert!(t.erase_last());
        assert_eq!(t.display(), "");
        // Undo order: the committed word, then the partial words LIFO.
        assert!(t.undo_last());
        assert_eq!(t.display(), "Done");
        assert!(t.undo_last());
        assert_eq!(t.display(), "Done new");
        assert!(t.undo_last());
        assert_eq!(t.display(), "Done new words");
    }

    #[test]
    fn buffer_target_combines_committed_and_stable_prefix() {
        let mut t = Transcript::new();
        t.feed_final("hello");
        t.feed_partial("world is big");
        assert_eq!(t.buffer_target("world is"), "Hello World is");
        assert_eq!(t.buffer_target(""), "Hello");
        t.feed_final("world is big");
        assert_eq!(t.buffer_target(""), "Hello World is big");
    }

    #[test]
    fn erased_first_word_recapitalizes_the_next() {
        let mut t = Transcript::new();
        t.feed_partial("a");
        assert!(t.erase_last());
        // "a" is permanent; "b" becomes the first visible word.
        t.feed_partial("a b");
        assert_eq!(t.display(), "b");
        assert_eq!(t.buffer_target("b"), "B");
    }

    #[test]
    fn punctuation_attaches_to_tokens() {
        let mut t = Transcript::new();
        t.feed_final("hello, world");
        assert!(t.erase_last());
        assert_eq!(t.display(), "Hello,");
        assert!(t.undo_last());
        assert_eq!(t.display(), "Hello, world");
    }

    #[test]
    fn multibyte_tokens_roundtrip() {
        let mut t = Transcript::new();
        t.feed_final("café naïve");
        assert!(t.erase_last());
        assert_eq!(t.display(), "Café");
        assert!(t.undo_last());
        assert_eq!(t.display(), "Café naïve");
    }

    #[test]
    fn fresh_utterance_starts_clean() {
        let mut t = Transcript::new();
        t.feed_partial("old words here");
        t.feed_final("old words here");
        assert_eq!(t.feed_partial("fresh"), "fresh");
        assert_eq!(t.display(), "Old words here fresh");
    }

    #[test]
    fn full_streaming_session_with_mid_utterance_erase() {
        let mut t = Transcript::new();
        t.feed_partial("the quick");
        t.feed_partial("the quick brown");
        assert!(t.erase_last());
        // Growth: "brown" is permanent for the rest of the session.
        t.feed_partial("the quick brown fox");
        assert_eq!(t.display(), "the quick fox");
        t.feed_final("the quick brown fox");
        assert_eq!(t.display(), "The quick fox");
        assert_eq!(t.buffer_target(""), "The quick fox");
        // The next utterance builds on the erased committed text.
        t.feed_partial("jumps over");
        assert_eq!(t.display(), "The quick fox jumps over");
        assert_eq!(t.buffer_target("jumps"), "The quick fox Jumps");
    }

    #[test]
    fn restored_word_returns_to_its_original_position() {
        let mut t = Transcript::new();
        t.feed_partial("a b c d e");
        assert!(t.erase_last()); // e (position 4)
        // The decoder collapses the tail: "d" and "e" drop out of the
        // partial. No new word, so the erasure stays restorable.
        t.feed_partial("a b");
        assert!(t.erase_last()); // b (position 1)
        // The final re-expands to all five words: both erasures survive as
        // committed-slot erasures.
        t.feed_final("a b c d e");
        assert_eq!(t.display(), "A c d");
        // Undo restores "b" to its original position (between "a" and
        // "c"), not to the end; "e" is still erased.
        assert!(t.undo_last());
        assert_eq!(t.display(), "A b c d");
        assert!(t.undo_last());
        assert_eq!(t.display(), "A b c d e");
        assert!(!t.undo_last());
        // The typed buffer follows the same ordering.
        assert_eq!(t.buffer_target(""), "A b c d e");
    }

    #[test]
    fn fully_erased_segment_leaves_a_gap_that_undoes_cleanly() {
        let mut t = Transcript::new();
        t.feed_final("one two");
        t.feed_final("three four");
        // Erase both words of the second segment.
        assert!(t.erase_last()); // four
        assert!(t.erase_last()); // three
        assert_eq!(t.display(), "One two");
        // Undo restores them in place, before any later segment.
        assert!(t.undo_last());
        assert_eq!(t.display(), "One two Three");
        assert!(t.undo_last());
        assert_eq!(t.display(), "One two Three four");
        // A new utterance makes the still-pending erasure permanent only
        // for words that were not restored: erase "four" again, then speak.
        assert!(t.erase_last());
        t.feed_batch_final("five");
        assert_eq!(t.display(), "One two Three Five");
        assert!(!t.undo_last());
    }

    #[test]
    fn erasing_and_restoring_a_segments_first_word_keeps_casing() {
        let mut t = Transcript::new();
        t.feed_final("hello world");
        assert!(t.erase_last()); // world
        assert!(t.erase_last()); // hello
        assert_eq!(t.display(), "");
        assert!(t.undo_last());
        assert_eq!(t.display(), "Hello");
        assert!(t.undo_last());
        assert_eq!(t.display(), "Hello world");
    }
}
