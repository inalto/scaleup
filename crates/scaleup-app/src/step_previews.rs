//! Which worker previews belong to the history step on screen.
//!
//! Going to a history step sends `UiMsg::ShowStep`. The worker answers every such request, in the
//! order sent, with exactly one of: the step's "before" then its "after"; an error naming one of
//! its two files (gone, or "before" undecodable); the "before" then an error naming the output
//! (the "after" undecodable). Requests can be overtaken: the user goes to another step, picks
//! another image or starts a run before the answer arrives. This type keeps every request still
//! unanswered, oldest first, knows which one (if any) is still wanted ("live"), and tells the
//! window what to do with each incoming preview or error: show it, hold it, or drop it.
//!
//! Events of other requests come between whole answers (the worker does one message at a time),
//! so an event that fits the oldest unanswered request is its answer; anything else is not a
//! step's and passes through. One case it cannot tell apart: a plain preview request for the very
//! file a step reads, sent just before that step's request; that preview is then taken as the
//! step's own (same file, same picture).
//!
//! `T` is the decoded image; the type does nothing with it but keep the live step's "before"
//! until its "after" arrives, so both reach the image area in the same frame.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use crate::worker::Which;

/// What to do with a preview.
#[derive(Debug, PartialEq)]
pub enum PreviewFate<T> {
    /// Not a step's preview: show it as usual.
    Show(T),
    /// The live step's "before": kept until its "after".
    Hold,
    /// The live step's answer is complete: show `before` (when it came) and `after` together.
    Pair { before: Option<T>, after: T },
    /// A preview of a step that is no longer wanted.
    Drop,
}

/// What to do with a worker error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorFate {
    /// Not about a step: report it as usual.
    NotAStep,
    /// The live step cannot be shown: say why, where its picture would be.
    Live,
    /// About a step that is no longer wanted: nothing to report.
    Drop,
}

#[derive(Debug)]
struct Request {
    id: u64,
    input: PathBuf,
    output: PathBuf,
}

/// The unanswered `ShowStep` requests and the one still wanted.
#[derive(Debug)]
pub struct StepPreviews<T> {
    pending: VecDeque<Request>,
    live: Option<u64>,
    next: u64,
    held: Option<T>,
}

impl<T> Default for StepPreviews<T> {
    fn default() -> Self {
        Self {
            pending: VecDeque::new(),
            live: None,
            next: 0,
            held: None,
        }
    }
}

impl<T> StepPreviews<T> {
    /// A `ShowStep` for these files was sent; it is now the one wanted, any earlier one is not.
    pub fn ask(&mut self, input: &Path, output: &Path) {
        self.held = None;
        self.live = Some(self.next);
        self.pending.push_back(Request {
            id: self.next,
            input: input.to_path_buf(),
            output: output.to_path_buf(),
        });
        self.next += 1;
    }

    /// The picture is being replaced by something else (another image, a run, an empty history):
    /// no request is wanted any more. Their answers are still expected, and will be dropped.
    pub fn leave(&mut self) {
        self.live = None;
        self.held = None;
    }

    /// True while the wanted step's answer has not arrived.
    pub fn waiting(&self) -> bool {
        self.live.is_some()
    }

    fn front_is_live(&self) -> bool {
        self.pending
            .front()
            .is_some_and(|r| Some(r.id) == self.live)
    }

    /// Classifies a preview of `path`.
    pub fn preview(&mut self, which: Which, path: &Path, image: T) -> PreviewFate<T> {
        let Some(front) = self.pending.front() else {
            return PreviewFate::Show(image);
        };
        let live = self.front_is_live();
        match which {
            Which::Before if path == front.input => {
                if live {
                    self.held = Some(image);
                    PreviewFate::Hold
                } else {
                    PreviewFate::Drop
                }
            }
            Which::After if path == front.output => {
                self.pending.pop_front();
                if live {
                    self.live = None;
                    PreviewFate::Pair {
                        before: self.held.take(),
                        after: image,
                    }
                } else {
                    PreviewFate::Drop
                }
            }
            _ => PreviewFate::Show(image),
        }
    }

    /// Classifies a worker error. The worker names the file first (`<path> no longer exists`,
    /// `<path>: <why>`), which is how an error is matched to the oldest unanswered request.
    pub fn error(&mut self, message: &str) -> ErrorFate {
        let Some(front) = self.pending.front() else {
            return ErrorFate::NotAStep;
        };
        let names = |p: &Path| message.starts_with(&p.display().to_string());
        if !(names(&front.input) || names(&front.output)) {
            return ErrorFate::NotAStep;
        }
        let live = self.front_is_live();
        self.pending.pop_front();
        if live {
            self.leave();
            ErrorFate::Live
        } else {
            ErrorFate::Drop
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    /// Step `n` reads `in<n>.png` and wrote `out<n>.png`.
    fn ask(s: &mut StepPreviews<&'static str>, n: u8) {
        s.ask(&p(&format!("in{n}.png")), &p(&format!("out{n}.png")));
    }

    fn before(
        s: &mut StepPreviews<&'static str>,
        n: u8,
        img: &'static str,
    ) -> PreviewFate<&'static str> {
        s.preview(Which::Before, &p(&format!("in{n}.png")), img)
    }

    fn after(
        s: &mut StepPreviews<&'static str>,
        n: u8,
        img: &'static str,
    ) -> PreviewFate<&'static str> {
        s.preview(Which::After, &p(&format!("out{n}.png")), img)
    }

    #[test]
    fn a_step_answered_shows_both_previews_together() {
        let mut s = StepPreviews::default();
        assert_eq!(before(&mut s, 1, "b1"), PreviewFate::Show("b1"));
        ask(&mut s, 1);
        assert!(s.waiting());
        assert_eq!(before(&mut s, 1, "b1"), PreviewFate::Hold);
        assert_eq!(
            after(&mut s, 1, "a1"),
            PreviewFate::Pair {
                before: Some("b1"),
                after: "a1"
            }
        );
        assert!(!s.waiting());
        // Nothing pending: previews of a run pass through.
        assert_eq!(after(&mut s, 1, "again"), PreviewFate::Show("again"));
    }

    /// (a) Step 1 then step 2 before any preview: only step 2 reaches the view.
    #[test]
    fn a_second_step_overtakes_the_first() {
        let mut s = StepPreviews::default();
        ask(&mut s, 1);
        ask(&mut s, 2);
        assert_eq!(before(&mut s, 1, "b1"), PreviewFate::Drop);
        assert_eq!(after(&mut s, 1, "a1"), PreviewFate::Drop);
        assert!(s.waiting());
        assert_eq!(before(&mut s, 2, "b2"), PreviewFate::Hold);
        assert_eq!(
            after(&mut s, 2, "a2"),
            PreviewFate::Pair {
                before: Some("b2"),
                after: "a2"
            }
        );
    }

    /// (b) A step, then another image picked: the step's answer is dropped, the image's own
    /// "before" (which comes after it) shows.
    #[test]
    fn picking_another_image_drops_the_steps_answer() {
        let mut s = StepPreviews::default();
        ask(&mut s, 1);
        s.leave();
        assert!(!s.waiting());
        assert_eq!(before(&mut s, 1, "b1"), PreviewFate::Drop);
        assert_eq!(after(&mut s, 1, "a1"), PreviewFate::Drop);
        let other = s.preview(Which::Before, &p("other.png"), "x");
        assert_eq!(other, PreviewFate::Show("x"));
        // Picking the step's own input again: its preview shows once the answer is through.
        ask(&mut s, 1);
        s.leave();
        assert_eq!(before(&mut s, 1, "b1"), PreviewFate::Drop);
        assert_eq!(after(&mut s, 1, "a1"), PreviewFate::Drop);
        assert_eq!(before(&mut s, 1, "mine"), PreviewFate::Show("mine"));
    }

    /// (c) A step, then Start: the run's previews come after the step's answer, which is dropped;
    /// the step's errors make no report either.
    #[test]
    fn a_run_started_drops_the_steps_answer_and_its_error() {
        let mut s = StepPreviews::default();
        ask(&mut s, 1);
        s.leave();
        assert_eq!(before(&mut s, 1, "b1"), PreviewFate::Drop);
        assert_eq!(after(&mut s, 1, "a1"), PreviewFate::Drop);
        let run = s.preview(Which::After, &p("result.png"), "r");
        assert_eq!(run, PreviewFate::Show("r"));
        ask(&mut s, 2);
        s.leave();
        assert_eq!(s.error("out2.png no longer exists"), ErrorFate::Drop);
        assert_eq!(
            s.error("the model could not be loaded"),
            ErrorFate::NotAStep
        );
    }

    /// (d) The output is gone: one error, about the live step.
    #[test]
    fn a_missing_result_is_the_live_steps_error() {
        let mut s = StepPreviews::<&str>::default();
        ask(&mut s, 1);
        assert_eq!(s.error("out1.png no longer exists"), ErrorFate::Live);
        assert!(!s.waiting());
        // Nothing pending any more.
        assert_eq!(s.error("out1.png no longer exists"), ErrorFate::NotAStep);
    }

    /// (e) The input is gone, or the "after" cannot be decoded after a "before" was held: the
    /// held "before" goes, and a later step is not confused by it.
    #[test]
    fn a_missing_source_or_an_undecodable_result_ends_the_request() {
        let mut s = StepPreviews::default();
        ask(&mut s, 1);
        assert_eq!(s.error("in1.png no longer exists"), ErrorFate::Live);
        ask(&mut s, 2);
        assert_eq!(before(&mut s, 2, "b2"), PreviewFate::Hold);
        assert_eq!(s.error("out2.png: unexpected end of file"), ErrorFate::Live);
        ask(&mut s, 3);
        assert_eq!(
            after(&mut s, 3, "a3"),
            PreviewFate::Pair {
                before: None,
                after: "a3"
            }
        );
        // The same, abandoned in between: dropped, and the next one is whole.
        ask(&mut s, 1);
        ask(&mut s, 2);
        assert_eq!(before(&mut s, 1, "b1"), PreviewFate::Drop);
        assert_eq!(s.error("out1.png: bad data"), ErrorFate::Drop);
        assert_eq!(before(&mut s, 2, "b2"), PreviewFate::Hold);
        assert_eq!(
            after(&mut s, 2, "a2"),
            PreviewFate::Pair {
                before: Some("b2"),
                after: "a2"
            }
        );
    }

    /// (f) Chained steps: step 2 reads what step 1 wrote. Each file is taken for what the request
    /// in front expects it to be.
    #[test]
    fn one_steps_output_is_the_next_steps_input() {
        let mut s = StepPreviews::default();
        s.ask(&p("a.png"), &p("b.png"));
        s.ask(&p("b.png"), &p("c.png"));
        assert_eq!(
            s.preview(Which::Before, &p("a.png"), "a"),
            PreviewFate::Drop
        );
        assert_eq!(
            s.preview(Which::After, &p("b.png"), "b as after"),
            PreviewFate::Drop
        );
        assert_eq!(
            s.preview(Which::Before, &p("b.png"), "b as before"),
            PreviewFate::Hold
        );
        assert_eq!(
            s.preview(Which::After, &p("c.png"), "c"),
            PreviewFate::Pair {
                before: Some("b as before"),
                after: "c"
            }
        );
        // Going back to the first: b.png is now its "after".
        s.ask(&p("a.png"), &p("b.png"));
        assert_eq!(
            s.preview(Which::Before, &p("a.png"), "a"),
            PreviewFate::Hold
        );
        assert_eq!(
            s.preview(Which::After, &p("b.png"), "b"),
            PreviewFate::Pair {
                before: Some("a"),
                after: "b"
            }
        );
    }

    /// (g) An error about something else while a step is pending is reported as usual, and the
    /// step's answer still lands.
    #[test]
    fn an_unrelated_error_passes_through() {
        let mut s = StepPreviews::default();
        ask(&mut s, 1);
        assert_eq!(
            s.error("Cannot use the GPU: lost device"),
            ErrorFate::NotAStep
        );
        assert_eq!(s.error("other.png: not an image"), ErrorFate::NotAStep);
        assert!(s.waiting());
        assert_eq!(before(&mut s, 1, "b1"), PreviewFate::Hold);
        assert!(matches!(after(&mut s, 1, "a1"), PreviewFate::Pair { .. }));
    }

    /// (h) Remove while a step is pending: the next step is asked for (it overtakes), or, with
    /// nothing left, the request is left.
    #[test]
    fn remove_while_pending() {
        let mut s = StepPreviews::default();
        ask(&mut s, 2);
        // Removed: the cursor moves to step 3.
        ask(&mut s, 3);
        assert_eq!(before(&mut s, 2, "b2"), PreviewFate::Drop);
        assert_eq!(after(&mut s, 2, "a2"), PreviewFate::Drop);
        assert_eq!(before(&mut s, 3, "b3"), PreviewFate::Hold);
        // Removed too, the history is empty: the selected image is asked for instead.
        s.leave();
        assert_eq!(after(&mut s, 3, "a3"), PreviewFate::Drop);
        let own = s.preview(Which::Before, &p("in3.png"), "own");
        assert_eq!(own, PreviewFate::Show("own"));
        assert!(!s.waiting());
    }
}
