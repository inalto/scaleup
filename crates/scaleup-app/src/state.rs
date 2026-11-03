//! Everything the UI remembers: persisted [`Settings`] and the transient [`AppState`].

use std::path::{Path, PathBuf};

use scaleup_core::UpscaleOptions;
use scaleup_core::io::is_supported_input;
use serde::{Deserialize, Serialize};

use crate::ui::toast::Toast;
use crate::worker::WorkerEvent;

/// Every setting visible in the sidebar; survives restarts.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Settings {
    /// Catalogue id, or a path to a model file.
    pub model_id: String,
    pub scale: u32,
    pub use_width: bool,
    pub width: u32,
    pub format: String,
    pub quality: u8,
    /// 0 = automatic.
    pub tile: u32,
    /// `auto`, `cpu` or a GPU index.
    pub gpu: String,
    pub tta: bool,
    pub double: bool,
    pub overwrite: bool,
    pub output_dir: Option<PathBuf>,
    pub show_nc_models: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model_id: "realesr-general-x4v3".into(),
            scale: 4,
            use_width: false,
            width: 1920,
            format: "png".into(),
            quality: 90,
            tile: 0,
            gpu: "auto".into(),
            tta: false,
            double: false,
            overwrite: false,
            output_dir: None,
            show_nc_models: false,
        }
    }
}

impl Settings {
    /// The core options these settings stand for. A custom width overrides the scale.
    pub fn to_options(&self) -> UpscaleOptions {
        UpscaleOptions {
            tile_size: self.tile,
            tta: self.tta,
            output_scale: (!self.use_width).then_some(self.scale),
            output_width: self.use_width.then_some(self.width),
            ..UpscaleOptions::default()
        }
    }

    /// The configured output folder, else the folder `input` is in.
    pub fn output_dir_for(&self, input: &Path) -> PathBuf {
        self.output_dir
            .clone()
            .unwrap_or_else(|| input.parent().map(Path::to_path_buf).unwrap_or_default())
    }
}

/// What the next run will process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    Single(PathBuf),
    Batch(PathBuf),
}

#[derive(Clone, Debug, PartialEq, Default)]
pub enum Status {
    #[default]
    Idle,
    Downloading {
        done: u64,
        total: u64,
    },
    Running {
        file: usize,
        files: usize,
        percent: f32,
        phase: String,
    },
    Done(PathBuf),
    Error(String),
    Stopped {
        done: usize,
        total: usize,
    },
}

#[derive(Clone, Debug, Default)]
pub struct AppState {
    pub settings: Settings,
    pub mode: Option<Mode>,
    pub status: Status,
    pub last_result: Option<PathBuf>,
    pub device_name: String,
    pub log: Vec<String>,
    /// GPU choices the worker found (probing them is too slow for the UI thread).
    pub devices: Vec<scaleup_core::DeviceInfo>,
    /// Message of the most recent failed file, for the end-of-run status.
    pub last_file_error: Option<String>,
}

/// What a selected or dropped path means: a folder is a batch, a supported image a single run.
pub fn mode_from_path(p: &Path) -> Option<Mode> {
    if p.is_dir() {
        Some(Mode::Batch(p.to_path_buf()))
    } else if p.is_file() && is_supported_input(p) {
        Some(Mode::Single(p.to_path_buf()))
    } else {
        None
    }
}

/// Folds one worker event into the state; the returned toast, if any, is for the user to see.
pub fn apply_event(state: &mut AppState, ev: WorkerEvent) -> Option<Toast> {
    match ev {
        WorkerEvent::Device(name) => state.device_name = name,
        WorkerEvent::Devices(devices) => state.devices = devices,
        WorkerEvent::Download { done, total } => state.status = Status::Downloading { done, total },
        WorkerEvent::Progress {
            file,
            files,
            percent,
            phase,
        } => {
            state.status = Status::Running {
                file,
                files,
                percent,
                phase,
            }
        }
        WorkerEvent::FileDone(path) => state.last_result = Some(path),
        WorkerEvent::FileFailed { input, error } => {
            let text = format!("{}: {error}", input.display());
            state.log.push(text.clone());
            state.last_file_error = Some(text.clone());
            return Some(Toast::error(text));
        }
        WorkerEvent::AllDone { done, failed } => {
            let error = state.last_file_error.take();
            match (&state.last_result, done, error) {
                (Some(path), 1.., _) => state.status = Status::Done(path.clone()),
                (_, _, Some(error)) => state.status = Status::Error(error),
                _ => state.status = Status::Idle,
            }
            return Some(if failed > 0 {
                Toast::error(format!("{done} done, {failed} failed"))
            } else {
                Toast::success(format!(
                    "Done: {done} file{}",
                    if done == 1 { "" } else { "s" }
                ))
            });
        }
        WorkerEvent::Stopped { done, total } => {
            state.last_file_error = None;
            state.status = Status::Stopped { done, total };
        }
        WorkerEvent::Error(message) => {
            state.last_file_error = None;
            state.log.push(message.clone());
            state.status = Status::Error(message.clone());
            return Some(Toast::error(message));
        }
        WorkerEvent::Log(line) => state.log.push(line),
    }
    None
}

impl Status {
    pub fn is_running(&self) -> bool {
        matches!(self, Status::Downloading { .. } | Status::Running { .. })
    }

    /// One line for the status bar.
    pub fn label(&self) -> String {
        const MB: f64 = 1024.0 * 1024.0;
        match self {
            Status::Idle => "Ready".into(),
            Status::Downloading { done, total: 0 } => {
                format!("Downloading model: {:.1} MB", *done as f64 / MB)
            }
            Status::Downloading { done, total } => format!(
                "Downloading model: {:.1} / {:.1} MB",
                *done as f64 / MB,
                *total as f64 / MB
            ),
            Status::Running {
                phase, files: 1, ..
            } => phase.clone(),
            Status::Running {
                file, files, phase, ..
            } => format!("{phase} (file {} of {files})", file + 1),
            Status::Done(path) => format!("Saved {}", path.display()),
            Status::Error(message) => message.clone(),
            Status::Stopped { done, total } => format!("Stopped after {done} of {total}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::toast::ToastKind;

    #[test]
    fn mode_from_path_file_dir_and_unsupported() {
        let d = tempfile::tempdir().unwrap();
        let png = d.path().join("a.PNG");
        let txt = d.path().join("a.txt");
        std::fs::write(&png, b"x").unwrap();
        std::fs::write(&txt, b"x").unwrap();
        assert_eq!(mode_from_path(&png), Some(Mode::Single(png.clone())));
        assert_eq!(
            mode_from_path(d.path()),
            Some(Mode::Batch(d.path().to_path_buf()))
        );
        assert_eq!(mode_from_path(&txt), None);
        assert_eq!(mode_from_path(&d.path().join("missing.png")), None);
    }

    #[test]
    fn progress_updates_status() {
        let mut st = AppState::default();
        apply_event(
            &mut st,
            WorkerEvent::Progress {
                file: 1,
                files: 3,
                percent: 50.0,
                phase: "Upscaling".into(),
            },
        );
        assert_eq!(
            st.status,
            Status::Running {
                file: 1,
                files: 3,
                percent: 50.0,
                phase: "Upscaling".into()
            }
        );
        assert!(st.status.is_running());
        assert_eq!(st.status.label(), "Upscaling (file 2 of 3)");
    }

    #[test]
    fn error_sets_error_and_keeps_last_result() {
        let mut st = AppState {
            last_result: Some(PathBuf::from("old.png")),
            ..AppState::default()
        };
        let toast = apply_event(&mut st, WorkerEvent::Error("boom".into())).unwrap();
        assert_eq!(st.status, Status::Error("boom".into()));
        assert!(!st.status.is_running());
        assert_eq!(st.last_result, Some(PathBuf::from("old.png")));
        assert_eq!(toast.kind, ToastKind::Error);
    }

    #[test]
    fn stopped_sets_counts() {
        let mut st = AppState::default();
        apply_event(&mut st, WorkerEvent::Stopped { done: 1, total: 3 });
        assert_eq!(st.status, Status::Stopped { done: 1, total: 3 });
        assert_eq!(st.status.label(), "Stopped after 1 of 3");
    }

    #[test]
    fn download_shows_bytes() {
        let mut st = AppState::default();
        apply_event(
            &mut st,
            WorkerEvent::Download {
                done: 1_048_576,
                total: 4_194_304,
            },
        );
        assert!(st.status.is_running());
        assert_eq!(st.status.label(), "Downloading model: 1.0 / 4.0 MB");
    }

    #[test]
    fn all_done_after_file_done_is_done_with_a_green_toast() {
        let mut st = AppState::default();
        apply_event(&mut st, WorkerEvent::FileDone(PathBuf::from("o.png")));
        let toast = apply_event(&mut st, WorkerEvent::AllDone { done: 1, failed: 0 }).unwrap();
        assert_eq!(st.status, Status::Done(PathBuf::from("o.png")));
        assert_eq!(toast.kind, ToastKind::Success);
    }

    #[test]
    fn failed_file_keeps_the_run_going_and_is_reported_at_the_end() {
        let mut st = AppState {
            status: Status::Running {
                file: 0,
                files: 1,
                percent: 0.0,
                phase: String::new(),
            },
            last_result: Some(PathBuf::from("older-run.png")),
            ..AppState::default()
        };
        apply_event(
            &mut st,
            WorkerEvent::FileFailed {
                input: PathBuf::from("a.png"),
                error: "bad".into(),
            },
        );
        assert!(st.status.is_running());
        apply_event(&mut st, WorkerEvent::AllDone { done: 0, failed: 1 });
        assert!(matches!(&st.status, Status::Error(e) if e.contains("bad")));
    }

    #[test]
    fn settings_default_values() {
        let s = Settings::default();
        assert_eq!(s.model_id, "realesr-general-x4v3");
        assert_eq!(s.scale, 4);
        assert!(!s.use_width);
        assert_eq!(s.width, 1920);
        assert_eq!(s.format, "png");
        assert_eq!(s.quality, 90);
        assert_eq!(s.tile, 0);
        assert_eq!(s.gpu, "auto");
        assert!(!s.tta && !s.double && !s.overwrite && !s.show_nc_models);
        assert_eq!(s.output_dir, None);
    }

    #[test]
    fn settings_json_roundtrip() {
        let s = Settings {
            model_id: "x".into(),
            scale: 2,
            use_width: true,
            width: 800,
            format: "webp".into(),
            quality: 40,
            tile: 256,
            gpu: "cpu".into(),
            tta: true,
            double: true,
            overwrite: true,
            output_dir: Some(PathBuf::from("out")),
            show_nc_models: true,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<Settings>(&json).unwrap(), s);
    }

    #[test]
    fn settings_from_older_json_fill_missing_fields() {
        let s: Settings = serde_json::from_str(r#"{"scale":2}"#).unwrap();
        assert_eq!(s.scale, 2);
        assert_eq!(s.format, "png");
    }

    #[test]
    fn to_options_width_overrides_scale() {
        let mut st = Settings {
            tile: 128,
            tta: true,
            ..Settings::default()
        };
        let o = st.to_options();
        assert_eq!(o.output_scale, Some(4));
        assert_eq!(o.output_width, None);
        assert_eq!(o.tile_size, 128);
        assert!(o.tta);
        st.use_width = true;
        let o = st.to_options();
        assert_eq!(o.output_scale, None);
        assert_eq!(o.output_width, Some(1920));
    }

    #[test]
    fn output_dir_defaults_to_input_parent() {
        let mut st = Settings::default();
        assert_eq!(
            st.output_dir_for(Path::new("a/b/c.png")),
            PathBuf::from("a/b")
        );
        st.output_dir = Some(PathBuf::from("out"));
        assert_eq!(
            st.output_dir_for(Path::new("a/b/c.png")),
            PathBuf::from("out")
        );
    }
}
