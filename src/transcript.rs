//! The visible dictation transcript.
//!
//! The single source of truth for what text the target app should contain
//! and what the HUD should show, with mid-dictation word erase/undo applied:
//!
//! - `erase_last` (left mouse press) removes the last visible word: from
//!   the live partial if an utterance is in progress, else from the tail of
//!   the committed text.
//! - `undo_last` (right mouse press) restores the most recently erased word
//!   (LIFO: multiple erases are undone one by one in reverse order).
//! - An erase is restorable only until the next **new word** is transcribed:
//!   a partial that grows beyond any earlier partial of the utterance, or
//!   a final containing words no partial showed. From then on the word is
//!   gone for good (not restorable), matching "the user kept talking".
//!
//! Erased words are tracked two ways, because a word can be erased while it
//! is still only a live hypothesis or after it has committed:
//!
//! - committed erasures pop a token off the committed text (restoring
//!   re-appends it);
//! - partial erasures record the token **position** in the current
//!   utterance, so decoder revisions (a token changing or dropping at that
//!   position) keep the mapping correct. At the final, surviving positions
//!   convert into restorable committed erasures.
//!
//! The state machine is pure: no audio, no ASR, no ydotool. The injector
//! task feeds it and applies the resulting buffer edits (see
//! `buffer_target`), and the daemon relays `display()` to the HUD.

use std::collections::BTreeSet;

use crate::injector::capitalize_first;

/// One entry of the LIFO undo stack.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Undo {
    /// A token popped off the committed text; restoring re-appends it.
    Committed(String),
    /// A token position erased in the current partial; restoring
    /// re-includes the position.
    Partial(usize),
}

/// One dictation session's visible transcript.
#[derive(Debug, Default)]
pub struct Transcript {
    /// Exact text the target buffer holds from committed segments, after
    /// all erasures. Each segment's first visible word is capitalized and a
    /// single space separates segments (and words), so the string is also
    /// the token layout of the buffer.
    committed: String,
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

    /// New words were transcribed: pending partial erasures become
    /// permanent, pending committed erasures are discarded (their tokens
    /// are already out of the committed text).
    fn flush_undos(&mut self) {
        for u in self.undo.drain(..) {
            if let Undo::Partial(i) = u {
                self.erased.insert(i);
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
    /// (streaming path). Its visible tokens are appended to the committed
    /// text (transformed). Pending partial erasures at positions the final
    /// still contains become restorable committed-token erasures (same LIFO
    /// order); positions the final dropped (the decoder revised the word
    /// away) are discarded.
    pub fn feed_final(&mut self, text: &str) {
        let tokens = Self::tokens(text);
        if tokens.len() > self.partial_max_tokens {
            // The final contains words no partial ever showed: new words,
            // pending erasures are permanent.
            self.flush_undos();
            self.partial_max_tokens = tokens.len();
        }
        let mut converted: Vec<Undo> = Vec::new();
        let mut convert_at: BTreeSet<usize> = BTreeSet::new();
        for u in self.undo.drain(..) {
            match u {
                Undo::Partial(i) if i < tokens.len() => {
                    convert_at.insert(i);
                    converted.push(Undo::Committed(tokens[i].clone()));
                }
                Undo::Partial(_) => {}
                Undo::Committed(c) => converted.push(Undo::Committed(c)),
            }
        }
        let mut visible = tokens;
        let mut drop = self.erased.clone();
        drop.extend(convert_at);
        for &i in drop.iter().rev() {
            if i < visible.len() {
                visible.remove(i);
            }
        }
        self.append_segment(&visible);

        // Start a fresh utterance.
        self.partial_tokens.clear();
        self.partial_max_tokens = 0;
        self.erased.clear();
        self.undo = converted;
    }

    /// A new utterance's final (batch path: no partials in flight). Any
    /// pending erasures are permanent - a whole new utterance was
    /// transcribed - then the text is appended transformed.
    pub fn feed_batch_final(&mut self, text: &str) {
        self.undo.clear();
        self.partial_tokens.clear();
        self.partial_max_tokens = 0;
        self.erased.clear();
        self.append_segment(&Self::tokens(text));
    }

    /// Append one committed segment's visible tokens, transformed: first
    /// letter capitalized, a single space in front when the committed text
    /// is non-empty.
    fn append_segment(&mut self, tokens: &[String]) {
        if tokens.is_empty() {
            return;
        }
        let mut seg = capitalize_first(&tokens.join(" "));
        if !self.committed.is_empty() {
            seg.insert(0, ' ');
        }
        self.committed.push_str(&seg);
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
        match last {
            Some(i) => {
                self.undo.push(Undo::Partial(i));
                true
            }
            None if !self.committed.is_empty() => {
                // Pop the last whitespace token, together with its
                // separating space (or the whole string when it is the
                // only token).
                let sp = self.committed.rfind(' ');
                let start = sp.map(|p| p + 1).unwrap_or(0);
                let tok = self.committed[start..].to_string();
                self.committed.truncate(sp.unwrap_or(0));
                self.undo.push(Undo::Committed(tok));
                true
            }
            None => false,
        }
    }

    /// Restore the last erased word (right mouse press), LIFO. Returns true
    /// if an erase was undone. Restoring a partial position whose token the
    /// decoder since dropped is a no-op on the visible text but still
    /// consumes the undo entry.
    pub fn undo_last(&mut self) -> bool {
        match self.undo.pop() {
            Some(Undo::Committed(tok)) => {
                if self.committed.is_empty() {
                    self.committed.push_str(&tok);
                } else {
                    self.committed.push(' ');
                    self.committed.push_str(&tok);
                }
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
        if p.is_empty() {
            self.committed.clone()
        } else if self.committed.is_empty() {
            p
        } else {
            format!("{} {}", self.committed, p)
        }
    }

    /// The text the target buffer should hold: the committed text plus,
    /// while an utterance is in progress, the transformed `stable` prefix
    /// of the visible partial. `stable` is the live-typing stability prefix
    /// (a prefix of the visible partial); an empty `stable` types nothing
    /// of the partial yet.
    pub fn buffer_target(&self, stable: &str) -> String {
        let mut out = self.committed.clone();
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
        t.feed_final("Hello world");
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
}
