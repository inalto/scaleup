//! Everything the UI remembers: persisted [`Settings`] and the transient [`AppState`].

use std::path::{Path, PathBuf};

use scaleup_core::io::is_supported_input;
use scaleup_core::resize::target_size;
use scaleup_core::{
    ExtendOptions, Fill, MarginSpec, Margins, Registry, ScaleUpError, UpscaleOptions,
};
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
    /// Upscale factor; `None` only extends the canvas (no model is loaded). Older settings files
    /// hold a plain number, which reads as `Some`.
    pub scale: Option<u32>,
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
            scale: Some(4),
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
            output_scale: if self.use_width { None } else { self.scale },
            output_width: self.use_width.then_some(self.width),
            ..UpscaleOptions::default()
        }
    }

    /// Whether a run loads the upscaling model. A custom width is a resize of its own, so it
    /// upscales even when the scale is "None", as it overrides any other scale.
    pub fn upscales(&self) -> bool {
        self.use_width || self.scale.is_some()
    }

    /// The upscale factor of the whole run (the scale squared with "Double upscale"), `None` when
    /// a custom width decides the size or nothing is upscaled.
    fn factor(&self) -> Option<u32> {
        let s = self.scale.filter(|_| !self.use_width)?;
        Some(if self.double { s * s } else { s })
    }

    /// Why Start cannot be pressed with these settings, `None` when it can. `source` is the size of
    /// the selected single image, as for [`Settings::extend_summary`].
    pub fn start_blocker(&self, source: Option<[u32; 2]>) -> Option<String> {
        if !self.upscales() && !self.extend_enabled {
            return Some("Turn on Extend canvas or choose a scale".into());
        }
        match self.extend_summary(source) {
            Some(Err(e)) => Some(format!("Extend canvas: {e}")),
            _ => None,
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
        Ok(Some(ExtendOptions {
            margins: self.margin_spec(),
            fill,
        }))
    }

    /// The margins the "Extend canvas" fields ask for, whether or not extension is on and the fill
    /// can be read.
    pub fn margin_spec(&self) -> MarginSpec {
        let [left, top, right, bottom] = self.extend_sides();
        let px = |v: f32| v.max(0.0).round() as u32;
        match self.extend_mode {
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
        }
    }

    /// The "resulting size" line of the Extend section, `None` when extension is off. With the
    /// source size: `1200×800 to 1600×800 to 6400×3200`; without it (a folder, where every file has
    /// its own size) the rule, `+64 px each side, then 4×`. With the scale "None" the canvas is the
    /// final size: `1200×800 to 1600×800`, or the rule alone. `Err` holds why the settings cannot be
    /// used, for this image when its size is known.
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
        let Some([w, h]) = source else {
            let rule = margin_rule(&ext.margins);
            return Some(Ok(match (self.use_width, self.factor()) {
                (true, _) => format!("{rule}, then {} px wide", self.width),
                (false, Some(f)) => format!("{rule}, then {f}×"),
                (false, None) => rule,
            }));
        };
        Some(
            ext.margins
                .resolve(w, h)
                .and_then(|m| {
                    let (cw, ch) = (w + m.left + m.right, h + m.top + m.bottom);
                    let canvas = format!("{w}×{h} to {cw}×{ch}");
                    let (ow, oh) = match (self.use_width, self.factor()) {
                        (true, _) => target_size(cw, ch, 1, None, Some(self.width))?,
                        (false, Some(f)) => (cw * f, ch * f),
                        (false, None) => return Ok(canvas),
                    };
                    Ok(format!("{canvas} to {ow}×{oh}"))
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

/// `+128 px`, `+1/2/3/4 px`, `+10%` or `to 16:9`: the short form of [`margin_rule`] for labels.
fn margin_short(spec: &MarginSpec) -> String {
    fn sides<T: PartialEq + std::fmt::Display>(v: [T; 4], unit: &str) -> String {
        if v.iter().all(|x| *x == v[0]) {
            format!("+{}{unit}", v[0])
        } else {
            let [l, t, r, b] = v;
            format!("+{l}/{t}/{r}/{b}{unit}")
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
        MarginSpec::Aspect { w, h } => format!("to {w}:{h}"),
    }
}

/// `fill` as [`Settings::extend_fill`] stores it, the text [`Fill::parse`] reads back.
pub fn fill_setting(fill: &Fill) -> String {
    match fill {
        Fill::Ai => "ai".into(),
        Fill::Mirror => "mirror".into(),
        Fill::Edge => "edge".into(),
        Fill::Transparent => "transparent".into(),
        Fill::Color([r, g, b]) => format!("#{r:02x}{g:02x}{b:02x}"),
    }
}

/// The catalogue name of `id`, else the file stem of a model file, else `id` itself.
fn model_name(id: &str, registry: &Registry) -> String {
    match registry.get(id) {
        Some(e) => e.name.clone(),
        None => Path::new(id)
            .file_stem()
            .map_or_else(|| id.to_string(), |s| s.to_string_lossy().into_owned()),
    }
}

/// What fills the new canvas, for people: `Mirror`, `Colour #ffffff`, or the inpainting model's
/// name for the AI fill (`LaMa (512)`). `?` when the fill cannot be read.
pub fn fill_title(s: &Settings, registry: &Registry) -> String {
    match Fill::parse(&s.extend_fill) {
        Ok(Fill::Ai) => model_name(&s.extend_model, registry),
        Ok(Fill::Mirror) => "Mirror".into(),
        Ok(Fill::Edge) => "Edge pixels".into(),
        Ok(Fill::Transparent) => "Transparent".into(),
        Ok(fill @ Fill::Color(_)) => format!("Colour {}", fill_setting(&fill)),
        Err(_) => "?".into(),
    }
}

/// The history label of a run made with `s`: `Extend +128 px · LaMa (512) · 4×`,
/// `Extend +10% · Mirror` (extend only), `Upscale 4× · Real-ESRGAN General v3 (fast)`. Names come
/// from `registry`. Only ASCII, `×` and `·`: the app's fonts have no other symbols.
pub fn step_label(s: &Settings, registry: &Registry) -> String {
    let size = match (s.use_width, s.factor()) {
        (true, _) => Some(format!("{} px wide", s.width)),
        (false, Some(f)) => Some(format!("{f}×")),
        (false, None) => None,
    };
    match s.to_extend_options() {
        Ok(Some(ext)) => {
            let mut parts = vec![
                format!("Extend {}", margin_short(&ext.margins)),
                fill_title(s, registry),
            ];
            parts.extend(size);
            parts.join(" · ")
        }
        _ => format!(
            "Upscale {} · {}",
            size.unwrap_or_default(),
            model_name(&s.model_id, registry)
        ),
    }
}

/// One fill of a *Try several* queue.
#[allow(
    dead_code,
    reason = "wired by the history strip and the Try several list (plan 12, Tasks 6 and 7)"
)]
#[derive(Clone, Debug, PartialEq)]
pub enum FillChoice {
    /// A fill that needs no model, as [`Fill::parse`] reads it. `Fill::Ai` here means the AI fill
    /// with the inpainting model of the settings.
    Plain(Fill),
    /// The AI fill with this inpainting model (catalogue id).
    Ai(String),
}

impl FillChoice {
    /// `base` with this fill; everything else, the extension included, stays as it is.
    pub fn apply(&self, base: &Settings) -> Settings {
        let mut s = base.clone();
        match self {
            FillChoice::Plain(fill) => s.extend_fill = fill_setting(fill),
            FillChoice::Ai(model) => {
                s.extend_fill = "ai".into();
                s.extend_model = model.clone();
            }
        }
        s
    }
}

/// Most steps the history keeps; the oldest goes first (D3).
pub const HISTORY_LIMIT: usize = 30;

/// One finished single-image run, as the history shows it again. The files stay on disk whatever
/// happens to the step.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub input: PathBuf,
    pub output: PathBuf,
    /// The settings of the run, for "Restore settings".
    pub settings: Settings,
    /// Size of the input with its EXIF orientation applied, as the run saw it.
    pub source_size: [u32; 2],
    pub result_size: [u32; 2],
    /// Canvas the run added around the input, in input pixels; zero when it did not extend.
    pub margins: Margins,
    /// What the run did, from [`step_label`].
    pub label: String,
}

/// The results of this session, oldest first, and the one being looked at. Moving the cursor never
/// drops steps: a new result goes after the last one wherever the cursor is.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct History {
    steps: Vec<Step>,
    at: Option<usize>,
}

impl History {
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// Index of the current step; `None` only when there are no steps.
    pub fn at(&self) -> Option<usize> {
        self.at
    }

    pub fn current(&self) -> Option<&Step> {
        self.steps.get(self.at?)
    }

    /// Adds `step` after the last one and makes it current. Past [`HISTORY_LIMIT`] the oldest step
    /// is dropped.
    pub fn push(&mut self, step: Step) {
        self.steps.push(step);
        if self.steps.len() > HISTORY_LIMIT {
            self.steps.remove(0);
        }
        self.at = Some(self.steps.len() - 1);
    }

    pub fn can_back(&self) -> bool {
        self.at.is_some_and(|i| i > 0)
    }

    pub fn can_forward(&self) -> bool {
        self.at.is_some_and(|i| i + 1 < self.steps.len())
    }

    /// The step before the current one becomes current; `None`, and no change, at the first.
    pub fn back(&mut self) -> Option<&Step> {
        let i = self.at.filter(|_| self.can_back())? - 1;
        self.go(i)
    }

    /// The step after the current one becomes current; `None`, and no change, at the last.
    pub fn forward(&mut self) -> Option<&Step> {
        let i = self.at.filter(|_| self.can_forward())? + 1;
        self.go(i)
    }

    /// Makes step `index` current; `None`, and no change, when there is no such step.
    pub fn go(&mut self, index: usize) -> Option<&Step> {
        if index >= self.steps.len() {
            return None;
        }
        self.at = Some(index);
        self.steps.get(index)
    }

    /// Takes step `index` out of the history (its files stay on disk). The cursor stays on the
    /// step it was on; when that is the one removed, it moves to the next step, else the previous.
    pub fn remove(&mut self, index: usize) -> Option<Step> {
        if index >= self.steps.len() {
            return None;
        }
        let step = self.steps.remove(index);
        self.at = match self.at {
            _ if self.steps.is_empty() => None,
            Some(at) if at > index => Some(at - 1),
            Some(at) => Some(at.min(self.steps.len() - 1)),
            None => None,
        };
        Some(step)
    }

    /// "Use as input": the file to select, the output of step `index`. Extending, looking, then
    /// upscaling is chained this way.
    pub fn use_as_input(&self, index: usize) -> Option<&Path> {
        self.steps.get(index).map(|s| s.output.as_path())
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
    /// Finished single-image runs of this session.
    pub history: History,
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
        WorkerEvent::FileDone { output, step } => {
            state.last_result = Some(output);
            if let Some(step) = step {
                state.history.push(*step);
            }
        }
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

    /// Remembers the size of a decoded preview: the "before" of a single image is the source. A
    /// history step's "before" is its input, which going to the step made the selected image, so
    /// the size always describes the image the next run reads.
    pub fn note_preview(&mut self, which: Which, size: [u32; 2]) {
        if which == Which::Before && matches!(self.mode, Some(Mode::Single(_))) {
            self.source_size = Some(size);
        }
    }

    /// "Restore settings": the settings of step `index`, with this window's own preferences (the
    /// OpenModelDB filters, the non-commercial group) kept as they are now.
    pub fn restore_settings(&self, index: usize) -> Option<Settings> {
        let step = self.history.steps().get(index)?;
        Some(Settings {
            hub: self.settings.hub.clone(),
            show_nc_models: self.settings.show_nc_models,
            ..step.settings.clone()
        })
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
        apply_event(
            &mut st,
            WorkerEvent::FileDone {
                output: PathBuf::from("o.png"),
                step: None,
            },
        );
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
        assert_eq!(s.scale, Some(4));
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
            scale: Some(2),
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
        assert_eq!(s.scale, Some(2));
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
            Some(Ok("1200×800 to 1600×800 to 6400×3200".into()))
        );
        assert_eq!(
            s.extend_summary(None),
            Some(Ok("+200/0/200/0 px (L/T/R/B), then 4×".into()))
        );
        s.extend_linked = true;
        s.extend_left = 64.0;
        assert_eq!(
            s.extend_summary(None),
            Some(Ok("+64 px each side, then 4×".into()))
        );
        s.double = true;
        assert_eq!(
            s.extend_summary(None),
            Some(Ok("+64 px each side, then 16×".into()))
        );
        s.use_width = true;
        assert_eq!(
            s.extend_summary(Some([100, 100])),
            Some(Ok("100×100 to 228×228 to 1920×1920".into()))
        );
        assert_eq!(
            s.extend_summary(None),
            Some(Ok("+64 px each side, then 1920 px wide".into()))
        );
        let mut s = extending(ExtendMode::Percent, [12.5; 4], "ai");
        assert_eq!(
            s.extend_summary(None),
            Some(Ok("+12.5% each side, then 4×".into()))
        );
        s.extend_mode = ExtendMode::Aspect;
        assert_eq!(
            s.extend_summary(None),
            Some(Ok("grow to 4:3, then 4×".into()))
        );
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
    fn scale_none_loads_saves_and_older_numbers_still_load() {
        let old: Settings = serde_json::from_str(r#"{"scale":3}"#).unwrap();
        assert_eq!(old.scale, Some(3));
        let missing: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(missing.scale, Some(4));
        let none = Settings {
            scale: None,
            ..Settings::default()
        };
        let json = serde_json::to_string(&none).unwrap();
        assert!(json.contains(r#""scale":null"#), "{json}");
        assert_eq!(serde_json::from_str::<Settings>(&json).unwrap(), none);
    }

    #[test]
    fn scale_none_upscales_only_with_a_custom_width() {
        let mut s = Settings {
            scale: None,
            ..Settings::default()
        };
        assert!(!s.upscales());
        assert_eq!(s.to_options().output_scale, None);
        assert_eq!(s.to_options().output_width, None);
        // A custom width overrides the scale, "None" included: the model runs, then the resize.
        s.use_width = true;
        assert!(s.upscales());
        assert_eq!(s.to_options().output_width, Some(1920));
        assert!(Settings::default().upscales());
    }

    #[test]
    fn extend_only_summary_ends_at_the_canvas() {
        let mut s = extending(ExtendMode::Pixels, [64.0; 4], "mirror");
        s.scale = None;
        s.double = true;
        assert_eq!(
            s.extend_summary(Some([100, 50])),
            Some(Ok("100×50 to 228×178".into()))
        );
        assert_eq!(s.extend_summary(None), Some(Ok("+64 px each side".into())));
        s.use_width = true;
        assert_eq!(
            s.extend_summary(Some([100, 100])),
            Some(Ok("100×100 to 228×228 to 1920×1920".into()))
        );
    }

    #[test]
    fn start_is_blocked_with_nothing_to_do_or_a_bad_extension() {
        let mut s = Settings::default();
        assert_eq!(s.start_blocker(Some([10, 10])), None);
        s.scale = None;
        assert_eq!(
            s.start_blocker(None).as_deref(),
            Some("Turn on Extend canvas or choose a scale")
        );
        s.extend_enabled = true;
        assert_eq!(s.start_blocker(Some([100, 100])), None);
        s.extend_left = 500.0;
        let why = s.start_blocker(Some([100, 100])).unwrap();
        assert!(
            why.starts_with("Extend canvas: left margin 500 px"),
            "{why}"
        );
        s.extend_enabled = false;
        s.use_width = true;
        assert_eq!(s.start_blocker(None), None);
    }

    fn catalogue() -> (tempfile::TempDir, Registry) {
        let d = tempfile::tempdir().unwrap();
        let reg = Registry::with_dir(d.path().to_path_buf());
        (d, reg)
    }

    #[test]
    fn step_labels_say_what_the_run_did() {
        let (_d, reg) = catalogue();
        assert_eq!(
            step_label(&Settings::default(), &reg),
            "Upscale 4× · Real-ESRGAN General v3 (fast)"
        );
        let mut s = extending(ExtendMode::Pixels, [128.0; 4], "ai");
        s.extend_linked = true;
        assert_eq!(step_label(&s, &reg), "Extend +128 px · LaMa (512) · 4×");
        s.double = true;
        assert_eq!(step_label(&s, &reg), "Extend +128 px · LaMa (512) · 16×");
        let mut s = extending(ExtendMode::Percent, [10.0; 4], "mirror");
        s.scale = None;
        assert_eq!(step_label(&s, &reg), "Extend +10% · Mirror");
        s.extend_mode = ExtendMode::Pixels;
        s.extend_fill = "#FF8000".into();
        s.extend_left = 1.0;
        assert_eq!(
            step_label(&s, &reg),
            "Extend +1/10/10/10 px · Colour #ff8000"
        );
        s.extend_mode = ExtendMode::Aspect;
        s.use_width = true;
        assert_eq!(
            step_label(&s, &reg),
            "Extend to 4:3 · Colour #ff8000 · 1920 px wide"
        );
        let custom = Settings {
            model_id: "C:/models/my model.onnx".into(),
            scale: Some(2),
            ..Settings::default()
        };
        assert_eq!(step_label(&custom, &reg), "Upscale 2× · my model");
        // Only characters the app's fonts have.
        for label in [step_label(&s, &reg), step_label(&custom, &reg)] {
            assert!(
                label.chars().all(|c| c.is_ascii() || "×·…".contains(c)),
                "{label}"
            );
        }
    }

    #[test]
    fn fill_settings_read_back_as_the_same_fill() {
        for fill in [
            Fill::Ai,
            Fill::Mirror,
            Fill::Edge,
            Fill::Transparent,
            Fill::Color([1, 0xab, 0xff]),
        ] {
            assert_eq!(Fill::parse(&fill_setting(&fill)), Ok(fill));
        }
    }

    #[test]
    fn fill_titles_name_the_fill_or_its_model() {
        let (_d, reg) = catalogue();
        let mut s = Settings::default();
        assert_eq!(fill_title(&s, &reg), "LaMa (512)");
        s.extend_model = "migan-512-places2".into();
        assert_eq!(fill_title(&s, &reg), "MI-GAN Places2 (512, fast)");
        s.extend_fill = "edge".into();
        assert_eq!(fill_title(&s, &reg), "Edge pixels");
        s.extend_fill = "plaid".into();
        assert_eq!(fill_title(&s, &reg), "?");
    }

    #[test]
    fn fill_choices_change_only_the_fill() {
        let base = extending(ExtendMode::Pixels, [8.0; 4], "ai");
        let s = FillChoice::Plain(Fill::Mirror).apply(&base);
        assert_eq!(s.extend_fill, "mirror");
        assert_eq!(
            Settings {
                extend_fill: "ai".into(),
                ..s.clone()
            },
            base
        );
        let s = FillChoice::Ai("migan-512-places2".into()).apply(&base);
        assert_eq!(
            (s.extend_fill.as_str(), s.extend_model.as_str()),
            ("ai", "migan-512-places2")
        );
        let s = FillChoice::Plain(Fill::Ai).apply(&base);
        assert_eq!(s, base, "the AI fill with the settings' own model");
    }

    fn step(n: usize) -> Step {
        Step {
            input: PathBuf::from(format!("in{n}.png")),
            output: PathBuf::from(format!("out{n}.png")),
            settings: Settings::default(),
            source_size: [10, 10],
            result_size: [40, 40],
            margins: Margins::default(),
            label: format!("step {n}"),
        }
    }

    fn labels(h: &History) -> Vec<&str> {
        h.steps().iter().map(|s| s.label.as_str()).collect()
    }

    #[test]
    fn history_push_back_forward_and_go() {
        let mut h = History::default();
        assert_eq!((h.current(), h.at()), (None, None));
        assert!(!h.can_back() && !h.can_forward());
        assert_eq!(h.back(), None);
        assert_eq!(h.forward(), None);
        for n in 0..3 {
            h.push(step(n));
        }
        assert_eq!(h.at(), Some(2));
        assert!(h.can_back() && !h.can_forward());
        assert_eq!(h.forward(), None);
        assert_eq!(h.back().map(|s| s.label.as_str()), Some("step 1"));
        assert_eq!(h.back().map(|s| s.label.as_str()), Some("step 0"));
        assert_eq!(h.back(), None);
        assert_eq!(h.at(), Some(0));
        assert_eq!(h.forward().map(|s| s.label.as_str()), Some("step 1"));
        assert!(h.can_back() && h.can_forward());
        // A new result goes after the last step, wherever the cursor is, and becomes current.
        h.push(step(3));
        assert_eq!(labels(&h), ["step 0", "step 1", "step 2", "step 3"]);
        assert_eq!(h.at(), Some(3));
        assert_eq!(h.go(1).map(|s| s.label.as_str()), Some("step 1"));
        assert_eq!(h.go(9), None);
        assert_eq!(h.current().map(|s| s.label.as_str()), Some("step 1"));
        assert_eq!(h.use_as_input(2), Some(Path::new("out2.png")));
        assert_eq!(h.use_as_input(9), None);
    }

    #[test]
    fn history_keeps_the_last_thirty_steps() {
        let mut h = History::default();
        for n in 0..HISTORY_LIMIT + 2 {
            h.push(step(n));
        }
        assert_eq!(h.steps().len(), HISTORY_LIMIT);
        assert_eq!(h.steps()[0].label, "step 2");
        assert_eq!(h.at(), Some(HISTORY_LIMIT - 1));
        assert_eq!(
            h.current().map(|s| s.label.clone()),
            Some(format!("step {}", HISTORY_LIMIT + 1))
        );
    }

    #[test]
    fn history_push_at_the_cap_with_the_cursor_back_goes_to_the_new_step() {
        let mut h = History::default();
        for n in 0..HISTORY_LIMIT {
            h.push(step(n));
        }
        h.go(5);
        h.push(step(99));
        assert_eq!(h.steps().len(), HISTORY_LIMIT);
        assert_eq!(h.steps()[0].label, "step 1");
        assert_eq!(h.at(), Some(HISTORY_LIMIT - 1));
        assert_eq!(h.current().map(|s| s.label.as_str()), Some("step 99"));
    }

    #[test]
    fn history_remove_keeps_the_cursor_on_its_step() {
        let mut h = History::default();
        for n in 0..4 {
            h.push(step(n));
        }
        h.go(2);
        // Before the cursor: the cursor follows its step.
        assert_eq!(h.remove(0).map(|s| s.label), Some("step 0".into()));
        assert_eq!(h.current().map(|s| s.label.as_str()), Some("step 2"));
        // After the cursor: nothing moves.
        h.remove(2);
        assert_eq!(h.current().map(|s| s.label.as_str()), Some("step 2"));
        assert_eq!(labels(&h), ["step 1", "step 2"]);
        // The current one, last: the previous step.
        h.remove(1);
        assert_eq!(h.current().map(|s| s.label.as_str()), Some("step 1"));
        assert_eq!(h.remove(5), None);
        h.push(step(7));
        h.go(0);
        // The current one, not last: the next step.
        h.remove(0);
        assert_eq!(h.current().map(|s| s.label.as_str()), Some("step 7"));
        h.remove(0);
        assert_eq!((h.current(), h.at()), (None, None));
        assert!(!h.can_back() && !h.can_forward());
    }

    #[test]
    fn removing_a_step_leaves_its_files() {
        let d = tempfile::tempdir().unwrap();
        let out = d.path().join("out.png");
        std::fs::write(&out, b"x").unwrap();
        let mut h = History::default();
        h.push(Step {
            output: out.clone(),
            ..step(0)
        });
        h.remove(0);
        assert!(out.is_file());
    }

    #[test]
    fn file_done_with_a_step_adds_it_to_the_history() {
        let mut st = AppState::default();
        apply_event(
            &mut st,
            WorkerEvent::FileDone {
                output: PathBuf::from("out0.png"),
                step: Some(Box::new(step(0))),
            },
        );
        assert_eq!(labels(&st.history), ["step 0"]);
        assert_eq!(st.last_result, Some(PathBuf::from("out0.png")));
        // A folder file carries no step.
        apply_event(
            &mut st,
            WorkerEvent::FileDone {
                output: PathBuf::from("b.png"),
                step: None,
            },
        );
        assert_eq!(st.history.steps().len(), 1);
        // Failures and stops add nothing.
        apply_event(
            &mut st,
            WorkerEvent::FileFailed {
                input: PathBuf::from("a.png"),
                error: "bad".into(),
            },
        );
        apply_event(&mut st, WorkerEvent::Stopped { done: 0, total: 1 });
        apply_event(&mut st, WorkerEvent::Error("boom".into()));
        assert_eq!(st.history.steps().len(), 1);
    }

    #[test]
    fn restore_settings_brings_back_the_run_and_keeps_window_preferences() {
        let mut st = AppState::default();
        let mut made = extending(ExtendMode::Percent, [5.0; 4], "edge");
        made.scale = None;
        made.show_nc_models = true;
        st.history.push(Step {
            settings: made.clone(),
            ..step(0)
        });
        st.settings.hub.commercial_only = false;
        let restored = st.restore_settings(0).unwrap();
        assert_eq!(restored.extend_fill, "edge");
        assert_eq!(restored.scale, None);
        assert!(!restored.hub.commercial_only && !restored.show_nc_models);
        assert_eq!(st.restore_settings(1), None);
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
