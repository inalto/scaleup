//! Everything the UI remembers: persisted [`Settings`] and the transient [`AppState`].

use std::path::{Path, PathBuf};

use scaleup_core::io::is_supported_input;
use scaleup_core::resize::target_size;
use scaleup_core::{ExtendOptions, Fill, MarginSpec, Margins, ScaleUpError, UpscaleOptions};
use serde::{Deserialize, Serialize};

use crate::ui::toast::Toast;
use crate::worker::{Which, WorkerEvent};

/// How the four side values of the "Extend canvas" section are read.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ExtendMode {
    #[default]
    Pixels,
    /// Of the side each margin extends.
    Percent,
    /// Grow, centred, to `extend_aspect`.
    Aspect,
}

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
    /// Add canvas around the image before upscaling.
    pub extend_enabled: bool,
    pub extend_mode: ExtendMode,
    /// Side values in pixels or percent, as `extend_mode` says.
    pub extend_left: f32,
    pub extend_top: f32,
    pub extend_right: f32,
    pub extend_bottom: f32,
    /// Every side takes `extend_left`.
    pub extend_linked: bool,
    /// Target ratio for [`ExtendMode::Aspect`], width:height.
    pub extend_aspect: (u32, u32),
    /// `ai`, `mirror`, `edge`, `transparent` or `#rrggbb`, as [`Fill::parse`] reads it.
    pub extend_fill: String,
    /// Inpainting model of the AI fill (catalogue id).
    pub extend_model: String,
    /// The filter toggles of the OpenModelDB window (its search text is not kept).
    pub hub: HubFilters,
}

/// Filters of the "Browse OpenModelDB" window.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
#[serde(default)]
pub struct HubFilters {
    /// Any of these scales; every scale when empty.
    pub scales: Vec<u32>,
    /// Any of these content tags; every model when empty.
    pub content: Vec<String>,
    /// Only licenses that clearly allow commercial use. On by default, so NonCommercial and
    /// Unknown licenses are never pre-selected.
    pub commercial_only: bool,
    /// Only models that are installed or run here without conversion.
    pub runnable_only: bool,
    pub arch: Option<String>,
}

impl Default for HubFilters {
    fn default() -> Self {
        Self {
            scales: Vec::new(),
            content: Vec::new(),
            commercial_only: true,
            runnable_only: true,
            arch: None,
        }
    }
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
            extend_enabled: false,
            extend_mode: ExtendMode::Pixels,
            extend_left: 64.0,
            extend_top: 64.0,
            extend_right: 64.0,
            extend_bottom: 64.0,
            extend_linked: true,
            extend_aspect: (16, 9),
            extend_fill: "ai".into(),
            extend_model: "lama-512".into(),
            hub: HubFilters::default(),
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

    /// Left, top, right, bottom as the user set them; all the left value when linked.
    pub fn extend_sides(&self) -> [f32; 4] {
        if self.extend_linked {
            [self.extend_left; 4]
        } else {
            [
                self.extend_left,
                self.extend_top,
                self.extend_right,
                self.extend_bottom,
            ]
        }
    }

    /// The canvas extension these settings ask for: `None` when it is off, an error when the fill
    /// cannot be read.
    pub fn to_extend_options(&self) -> Result<Option<ExtendOptions>, String> {
        if !self.extend_enabled {
            return Ok(None);
        }
        let fill = Fill::parse(&self.extend_fill)?;
        let [left, top, right, bottom] = self.extend_sides();
        let px = |v: f32| v.max(0.0).round() as u32;
        let margins = match self.extend_mode {
            ExtendMode::Pixels => MarginSpec::Pixels(Margins {
                left: px(left),
                top: px(top),
                right: px(right),
                bottom: px(bottom),
            }),
            ExtendMode::Percent => MarginSpec::Percent {
                left,
                top,
                right,
                bottom,
            },
            ExtendMode::Aspect => MarginSpec::Aspect {
                w: self.extend_aspect.0,
                h: self.extend_aspect.1,
            },
        };
        Ok(Some(ExtendOptions { margins, fill }))
    }

    /// The "resulting size" line of the Extend section, `None` when extension is off. With the
    /// source size: `1200×800 ➡ 1600×800 ➡ 6400×3200`; without it (a folder, where every file has
    /// its own size) the rule, `+64 px each side ➡ 4×`. `Err` holds why the settings cannot be used,
    /// for this image when its size is known.
    pub fn extend_summary(&self, source: Option<[u32; 2]>) -> Option<Result<String, String>> {
        let ext = match self.to_extend_options() {
            Ok(None) => return None,
            Ok(Some(ext)) => ext,
            Err(e) => return Some(Err(e)),
        };
        let reason = |e: ScaleUpError| match e {
            ScaleUpError::InvalidOption(m) => m,
            other => other.to_string(),
        };
        let factor = if self.double {
            self.scale * self.scale
        } else {
            self.scale
        };
        let Some([w, h]) = source else {
            let upscale = if self.use_width {
                format!("{} px wide", self.width)
            } else {
                format!("{factor}×")
            };
            return Some(Ok(format!("{} ➡ {upscale}", margin_rule(&ext.margins))));
        };
        Some(
            ext.margins
                .resolve(w, h)
                .and_then(|m| {
                    let (cw, ch) = (w + m.left + m.right, h + m.top + m.bottom);
                    let (ow, oh) = if self.use_width {
                        target_size(cw, ch, 1, None, Some(self.width))?
                    } else {
                        (cw * factor, ch * factor)
                    };
                    Ok(format!("{w}×{h} ➡ {cw}×{ch} ➡ {ow}×{oh}"))
                })
                .map_err(reason),
        )
    }

    /// The configured output folder, else the folder `input` is in.
    pub fn output_dir_for(&self, input: &Path) -> PathBuf {
        self.output_dir
            .clone()
            .unwrap_or_else(|| input.parent().map(Path::to_path_buf).unwrap_or_default())
    }
}

/// `+64 px each side`, `+1/2/3/4 px (L/T/R/B)`, `+10% each side` or `grow to 16:9`.
fn margin_rule(spec: &MarginSpec) -> String {
    fn sides<T: PartialEq + std::fmt::Display>(v: [T; 4], unit: &str) -> String {
        if v.iter().all(|x| *x == v[0]) {
            format!("+{}{unit} each side", v[0])
        } else {
            let [l, t, r, b] = v;
            format!("+{l}/{t}/{r}/{b}{unit} (L/T/R/B)")
        }
    }
    match *spec {
        MarginSpec::Pixels(m) => sides([m.left, m.top, m.right, m.bottom], " px"),
        MarginSpec::Percent {
            left,
            top,
            right,
            bottom,
        } => sides([left, top, right, bottom], "%"),
        MarginSpec::Aspect { w, h } => format!("grow to {w}:{h}"),
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
    /// Stop was pressed and the worker has not confirmed yet.
    pub stopping: bool,
    /// Size of the selected single image, once its preview is decoded; `None` for a folder.
    pub source_size: Option<[u32; 2]>,
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

/// Lines kept in the log panel.
const MAX_LOG_LINES: usize = 1000;

/// Folds one worker event into the state; the returned toast, if any, is for the user to see.
pub fn apply_event(state: &mut AppState, ev: WorkerEvent) -> Option<Toast> {
    let toast = reduce(state, ev);
    if state.log.len() > MAX_LOG_LINES {
        state.log.drain(..state.log.len() - MAX_LOG_LINES);
    }
    toast
}

fn reduce(state: &mut AppState, ev: WorkerEvent) -> Option<Toast> {
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
            state.stopping = false;
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
            state.stopping = false;
            state.last_file_error = None;
            state.status = Status::Stopped { done, total };
        }
        WorkerEvent::Error(message) => {
            state.stopping = false;
            state.last_file_error = None;
            state.log.push(message.clone());
            state.status = Status::Error(message.clone());
            return Some(Toast::error(message));
        }
        WorkerEvent::Log(line) => state.log.push(line),
        WorkerEvent::ModelDownloaded(id) => {
            state.status = Status::Idle;
            state.log.push(format!("downloaded {id}"));
            return Some(Toast::success(format!("Downloaded {id}")));
        }
        WorkerEvent::HubInstalled(m) => {
            state.stopping = false;
            state.status = Status::Idle;
            state.log.push(format!(
                "installed {} from OpenModelDB: {}",
                m.id,
                m.file.display()
            ));
            return Some(if m.verified {
                Toast::success(format!("Installed {}", m.name))
            } else {
                state.log.push(format!(
                    "{}: OpenModelDB lists no sha256, not verified",
                    m.id
                ));
                Toast::error(format!(
                    "Installed {}, but it could not be verified: OpenModelDB lists no sha256 for it",
                    m.name
                ))
            });
        }
        // Nothing was running: the status stays as it was.
        WorkerEvent::HubCatalogFailed(message) => {
            state.log.push(message.clone());
            return Some(Toast::error(message));
        }
        WorkerEvent::HubRemoved(id) => {
            if state.settings.model_id == id {
                state.settings.model_id = Settings::default().model_id;
            }
            state.log.push(format!("removed {id}"));
            return Some(Toast::success(format!("Removed {id}")));
        }
        // Pixels and the catalogue belong to the views, not the state.
        WorkerEvent::Preview { .. } | WorkerEvent::HubCatalog(_) | WorkerEvent::HubThumb { .. } => {
        }
    }
    None
}

impl AppState {
    /// Records that Stop was pressed; the run ends when the worker says so.
    pub fn request_stop(&mut self) {
        self.stopping = self.status.is_running();
    }

    /// Remembers the size of a decoded preview: the "before" of a single image is the source.
    pub fn note_preview(&mut self, which: Which, size: [u32; 2]) {
        if which == Which::Before && matches!(self.mode, Some(Mode::Single(_))) {
            self.source_size = Some(size);
        }
    }

    /// The status bar text.
    pub fn status_line(&self) -> String {
        if self.stopping {
            "Stopping…".into()
        } else {
            self.status.label()
        }
    }
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
            Status::Done(path) => format!(
                "Saved {}",
                path.file_name()
                    .unwrap_or(path.as_os_str())
                    .to_string_lossy()
            ),
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
    fn stop_request_shows_until_the_worker_confirms() {
        let mut st = AppState {
            status: Status::Running {
                file: 0,
                files: 3,
                percent: 10.0,
                phase: "Saving".into(),
            },
            ..AppState::default()
        };
        st.request_stop();
        assert_eq!(st.status_line(), "Stopping…");
        apply_event(
            &mut st,
            WorkerEvent::Progress {
                file: 0,
                files: 3,
                percent: 10.0,
                phase: "Saving".into(),
            },
        );
        assert_eq!(st.status_line(), "Stopping…");
        apply_event(&mut st, WorkerEvent::Stopped { done: 0, total: 3 });
        assert_eq!(st.status_line(), "Stopped after 0 of 3");
    }

    #[test]
    fn done_label_names_only_the_file() {
        let st = Status::Done(PathBuf::from("a/b/c.png"));
        assert_eq!(st.label(), "Saved c.png");
    }

    #[test]
    fn model_downloaded_returns_to_idle_with_a_toast() {
        let mut st = AppState {
            status: Status::Downloading { done: 1, total: 2 },
            ..AppState::default()
        };
        let toast = apply_event(&mut st, WorkerEvent::ModelDownloaded("m".into())).unwrap();
        assert_eq!(st.status, Status::Idle);
        assert_eq!(toast.kind, ToastKind::Success);
    }

    fn installed(verified: bool) -> scaleup_core::hub::InstalledModel {
        scaleup_core::hub::InstalledModel {
            id: "4x-Hub".into(),
            name: "Hub model".into(),
            arch: "span".into(),
            scale: 4,
            license: Some("MIT".into()),
            format: "onnx".into(),
            file: PathBuf::from("hub/4x-Hub/m.onnx"),
            sha256: None,
            verified,
            source_url: String::new(),
            installed_at: String::new(),
        }
    }

    #[test]
    fn hub_installed_returns_to_idle_and_warns_when_unverified() {
        let mut st = AppState {
            status: Status::Downloading { done: 1, total: 2 },
            stopping: true,
            ..AppState::default()
        };
        let toast = apply_event(&mut st, WorkerEvent::HubInstalled(installed(true))).unwrap();
        assert_eq!(st.status, Status::Idle);
        assert!(!st.stopping);
        assert_eq!(toast.kind, ToastKind::Success);
        let toast = apply_event(&mut st, WorkerEvent::HubInstalled(installed(false))).unwrap();
        assert_eq!(toast.kind, ToastKind::Error);
        assert!(
            toast.text.contains("could not be verified"),
            "{}",
            toast.text
        );
    }

    #[test]
    fn hub_removed_unselects_the_removed_model() {
        let mut st = AppState::default();
        st.settings.model_id = "4x-Hub".into();
        let toast = apply_event(&mut st, WorkerEvent::HubRemoved("4x-Hub".into())).unwrap();
        assert_eq!(toast.kind, ToastKind::Success);
        assert_eq!(st.settings.model_id, Settings::default().model_id);
        st.settings.model_id = "other".into();
        apply_event(&mut st, WorkerEvent::HubRemoved("4x-Hub".into()));
        assert_eq!(st.settings.model_id, "other");
    }

    #[test]
    fn hub_catalogue_failure_toasts_without_touching_the_status() {
        let mut st = AppState {
            status: Status::Done(PathBuf::from("o.png")),
            ..AppState::default()
        };
        let toast = apply_event(&mut st, WorkerEvent::HubCatalogFailed("offline".into())).unwrap();
        assert_eq!(toast.kind, ToastKind::Error);
        assert_eq!(st.status, Status::Done(PathBuf::from("o.png")));
        assert_eq!(st.log, ["offline"]);
    }

    #[test]
    fn hub_filters_default_to_commercial_and_runnable() {
        let f = HubFilters::default();
        assert!(f.commercial_only && f.runnable_only);
        assert!(f.scales.is_empty() && f.content.is_empty() && f.arch.is_none());
        let s: Settings = serde_json::from_str(r#"{"hub":{"scales":[4]}}"#).unwrap();
        assert_eq!(s.hub.scales, [4]);
        assert!(s.hub.commercial_only, "missing toggles keep their default");
    }

    #[test]
    fn log_is_capped() {
        let mut st = AppState::default();
        for i in 0..MAX_LOG_LINES + 5 {
            apply_event(&mut st, WorkerEvent::Log(i.to_string()));
        }
        assert_eq!(st.log.len(), MAX_LOG_LINES);
        assert_eq!(st.log[0], "5");
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
            extend_enabled: true,
            extend_mode: ExtendMode::Percent,
            extend_left: 1.0,
            extend_top: 2.0,
            extend_right: 3.0,
            extend_bottom: 4.5,
            extend_linked: false,
            extend_aspect: (9, 16),
            extend_fill: "#102030".into(),
            extend_model: "migan-512-places2".into(),
            hub: HubFilters {
                scales: vec![2, 4],
                content: vec!["anime".into()],
                commercial_only: false,
                runnable_only: false,
                arch: Some("span".into()),
            },
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
    fn extend_settings_default_and_load_from_older_json() {
        let s: Settings = serde_json::from_str(r#"{"scale":2}"#).unwrap();
        assert!(!s.extend_enabled);
        assert_eq!(s.extend_mode, ExtendMode::Pixels);
        assert_eq!(
            [s.extend_left, s.extend_top, s.extend_right, s.extend_bottom],
            [64.0; 4]
        );
        assert!(s.extend_linked);
        assert_eq!(s.extend_aspect, (16, 9));
        assert_eq!(s.extend_fill, "ai");
        assert_eq!(s.extend_model, "lama-512");
        assert_eq!(s.to_extend_options(), Ok(None));
    }

    fn extending(mode: ExtendMode, sides: [f32; 4], fill: &str) -> Settings {
        let [l, t, r, b] = sides;
        Settings {
            extend_enabled: true,
            extend_mode: mode,
            extend_left: l,
            extend_top: t,
            extend_right: r,
            extend_bottom: b,
            extend_linked: false,
            extend_aspect: (4, 3),
            extend_fill: fill.into(),
            ..Settings::default()
        }
    }

    #[test]
    fn extend_settings_roundtrip_to_core_options() {
        let s = extending(ExtendMode::Pixels, [1.0, 2.4, 3.6, 0.0], "mirror");
        let json = serde_json::to_string(&s).unwrap();
        let back: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
        assert_eq!(
            back.to_extend_options(),
            Ok(Some(ExtendOptions {
                margins: MarginSpec::Pixels(Margins {
                    left: 1,
                    top: 2,
                    right: 4,
                    bottom: 0
                }),
                fill: Fill::Mirror,
            }))
        );
        let s = extending(ExtendMode::Percent, [10.0, 0.0, 12.5, 5.0], "#FF8000");
        assert_eq!(
            s.to_extend_options(),
            Ok(Some(ExtendOptions {
                margins: MarginSpec::Percent {
                    left: 10.0,
                    top: 0.0,
                    right: 12.5,
                    bottom: 5.0
                },
                fill: Fill::Color([255, 128, 0]),
            }))
        );
        let s = extending(ExtendMode::Aspect, [0.0; 4], "ai");
        assert_eq!(
            s.to_extend_options(),
            Ok(Some(ExtendOptions {
                margins: MarginSpec::Aspect { w: 4, h: 3 },
                fill: Fill::Ai,
            }))
        );
        assert!(
            extending(ExtendMode::Pixels, [1.0; 4], "blur")
                .to_extend_options()
                .is_err()
        );
        // Linked sides all follow the left value.
        let mut s = extending(ExtendMode::Pixels, [8.0, 1.0, 2.0, 3.0], "edge");
        s.extend_linked = true;
        let Ok(Some(o)) = s.to_extend_options() else {
            panic!()
        };
        assert_eq!(
            o.margins,
            MarginSpec::Pixels(Margins {
                left: 8,
                top: 8,
                right: 8,
                bottom: 8
            })
        );
    }

    #[test]
    fn extend_summary_shows_sizes_rules_and_errors() {
        let mut s = extending(ExtendMode::Pixels, [200.0, 0.0, 200.0, 0.0], "ai");
        s.extend_linked = false;
        assert_eq!(
            s.extend_summary(Some([1200, 800])),
            Some(Ok("1200×800 ➡ 1600×800 ➡ 6400×3200".into()))
        );
        assert_eq!(
            s.extend_summary(None),
            Some(Ok("+200/0/200/0 px (L/T/R/B) ➡ 4×".into()))
        );
        s.extend_linked = true;
        s.extend_left = 64.0;
        assert_eq!(
            s.extend_summary(None),
            Some(Ok("+64 px each side ➡ 4×".into()))
        );
        s.double = true;
        assert_eq!(
            s.extend_summary(None),
            Some(Ok("+64 px each side ➡ 16×".into()))
        );
        s.use_width = true;
        assert_eq!(
            s.extend_summary(Some([100, 100])),
            Some(Ok("100×100 ➡ 228×228 ➡ 1920×1920".into()))
        );
        assert_eq!(
            s.extend_summary(None),
            Some(Ok("+64 px each side ➡ 1920 px wide".into()))
        );
        let mut s = extending(ExtendMode::Percent, [12.5; 4], "ai");
        assert_eq!(
            s.extend_summary(None),
            Some(Ok("+12.5% each side ➡ 4×".into()))
        );
        s.extend_mode = ExtendMode::Aspect;
        assert_eq!(s.extend_summary(None), Some(Ok("grow to 4:3 ➡ 4×".into())));
        // Over the limits for this image: the core's reason, without the error-kind prefix.
        let s = extending(ExtendMode::Pixels, [500.0; 4], "ai");
        let Some(Err(e)) = s.extend_summary(Some([100, 100])) else {
            panic!()
        };
        assert!(e.starts_with("left margin 500 px exceeds"), "{e}");
        assert!(
            extending(ExtendMode::Pixels, [1.0; 4], "blur")
                .extend_summary(None)
                .is_some_and(|r| r.is_err())
        );
        assert_eq!(Settings::default().extend_summary(Some([10, 10])), None);
    }

    #[test]
    fn source_size_comes_from_the_single_image_preview() {
        let mut st = AppState {
            mode: Some(Mode::Single(PathBuf::from("a.png"))),
            ..AppState::default()
        };
        st.note_preview(crate::worker::Which::Before, [30, 20]);
        assert_eq!(st.source_size, Some([30, 20]));
        st.note_preview(crate::worker::Which::After, [120, 80]);
        assert_eq!(st.source_size, Some([30, 20]));
        // A batch shows each file's preview: none of them is "the" source size.
        st.mode = Some(Mode::Batch(PathBuf::from("dir")));
        st.source_size = None;
        st.note_preview(crate::worker::Which::Before, [30, 20]);
        assert_eq!(st.source_size, None);
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
