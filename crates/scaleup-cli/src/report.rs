//! Human (upscayl-bin style) and JSON-lines output on stderr.

use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

use scaleup_core::{Phase, Progress};
use serde_json::json;

pub struct Reporter {
    json: bool,
    verbose: bool,
    sink: Mutex<Box<dyn Write + Send>>,
}

impl Reporter {
    pub fn new(json: bool, verbose: bool) -> Self {
        Self::with_sink(json, verbose, Box::new(std::io::stderr()))
    }

    pub fn with_sink(json: bool, verbose: bool, sink: Box<dyn Write + Send>) -> Self {
        Self {
            json,
            verbose,
            sink: Mutex::new(sink),
        }
    }

    fn line(&self, text: &str) {
        let mut sink = self.sink.lock().unwrap_or_else(|e| e.into_inner());
        let _ = writeln!(sink, "{text}");
    }

    fn event(&self, value: serde_json::Value) {
        self.line(&value.to_string());
    }

    pub fn progress(&self, percent: f32) {
        if self.json {
            self.event(json!({"event": "progress", "percent": (f64::from(percent) * 100.0).round() / 100.0}));
        } else {
            self.line(&format!("{percent:.2}%"));
        }
    }

    pub fn phase(&self, phase: Phase) {
        if self.json {
            if self.verbose || phase == Phase::Resizing {
                self.event(json!({"event": "phase", "phase": format!("{phase:?}").to_lowercase()}));
            }
        } else if self.verbose || phase == Phase::Resizing {
            self.line(&format!("{phase:?}"));
        }
    }

    pub fn done(&self, path: &Path) {
        if self.json {
            self.event(json!({"event": "done", "path": path.display().to_string()}));
        } else {
            self.line(" Upscayled Successfully!");
        }
    }

    pub fn error(&self, msg: &str) {
        if self.json {
            self.event(json!({"event": "error", "message": msg}));
        } else {
            self.line(&format!(" Error: {msg}"));
        }
    }

    pub fn warn(&self, msg: &str) {
        if self.json {
            self.event(json!({"event": "warning", "message": msg}));
        } else {
            self.line(&format!("warning: {msg}"));
        }
    }

    /// Verbose-only diagnostics.
    pub fn info(&self, msg: &str) {
        if !self.verbose {
            return;
        }
        if self.json {
            self.event(json!({"event": "info", "message": msg}));
        } else {
            self.line(msg);
        }
    }

    pub fn file_start(&self, index: usize, total: usize, path: &Path) {
        if self.json {
            self.event(json!({"event": "file", "index": index, "total": total,
                "path": path.display().to_string()}));
        } else {
            self.line(&format!("[{index}/{total}] {}", path.display()));
        }
    }

    pub fn summary(&self, done: usize, failed: usize, skipped: usize) {
        if self.json {
            self.event(
                json!({"event": "summary", "done": done, "failed": failed, "skipped": skipped}),
            );
        } else {
            self.line(&format!("{done} done, {failed} failed, {skipped} skipped"));
        }
    }

    /// Forwards core progress: tiles become percentages, phases are printed per verbosity.
    pub fn core_progress(&self, p: Progress) {
        match p {
            Progress::Phase(phase) => self.phase(phase),
            tiles => {
                if let Some(pct) = percent_from(&tiles, 1.0) {
                    self.progress(pct);
                }
            }
        }
    }
}

/// `Tiles{done,total}` → percent scaled by `phase_weight`; phases have no percentage.
pub fn percent_from(p: &Progress, phase_weight: f32) -> Option<f32> {
    match *p {
        Progress::Tiles { done, total } if total > 0 => {
            Some(done as f32 / total as f32 * 100.0 * phase_weight)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[derive(Clone, Default)]
    struct Buf(Arc<Mutex<Vec<u8>>>);
    impl Write for Buf {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl Buf {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }

    fn reporter(json: bool, verbose: bool) -> (Reporter, Buf) {
        let buf = Buf::default();
        (
            Reporter::with_sink(json, verbose, Box::new(buf.clone())),
            buf,
        )
    }

    #[test]
    fn json_lines_are_valid_json() {
        let (r, buf) = reporter(true, true);
        r.progress(12.345);
        r.phase(Phase::Saving);
        r.done(Path::new("a b.png"));
        r.error("bad \"quote\"");
        r.file_start(1, 2, Path::new("x.png"));
        r.summary(1, 0, 0);
        let text = buf.text();
        let events: Vec<_> = text
            .lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
            .collect();
        assert_eq!(events.len(), 6);
        assert_eq!(events[0]["event"], "progress");
        assert_eq!(events[0]["percent"], 12.35);
        assert_eq!(events[2]["path"], "a b.png");
        assert_eq!(events[3]["message"], "bad \"quote\"");
    }

    #[test]
    fn human_progress_format() {
        let (r, buf) = reporter(false, false);
        r.progress(12.3412);
        r.phase(Phase::Saving); // quiet
        r.phase(Phase::Resizing);
        r.done(Path::new("x"));
        assert_eq!(buf.text(), "12.34%\nResizing\n Upscayled Successfully!\n");
    }

    #[test]
    fn percent_from_tiles() {
        assert_eq!(
            percent_from(&Progress::Tiles { done: 1, total: 4 }, 1.0),
            Some(25.0)
        );
        assert_eq!(percent_from(&Progress::Phase(Phase::Saving), 1.0), None);
    }
}
