//! The one background thread: owns the [`Upscaler`] and the loaded model. The UI talks to it over
//! two channels and never blocks.
//!
//! While a run is in progress the thread is busy, so it reads its inbox from the progress callbacks
//! and between files: that is where `Stop` takes effect.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;

use eframe::egui;
use scaleup_core::hub::{
    Capabilities, Catalog, Http, HttpResponse, InstalledModel, ModelCard, Python, UreqHttp,
    find_python, find_python_from, import_converted, import_file, import_needs_conversion, install,
    install_converted, uninstall,
};
use scaleup_core::io::is_supported_input;
use scaleup_core::{
    CancelToken, DeviceInfo, DevicePreference, FileJob, Fill, Inpainter, Margins, Model,
    OutputFormat, Phase, Progress, Registry, Result, ScaleUpError, Upscaler, list_devices,
    load_image,
};

use crate::state::{FillChoice, Mode, Settings, Step, fill_title, step_label};
use crate::thumbs;

/// Where the OpenModelDB work of the worker gets its catalogue, files and Python.
pub struct HubConfig {
    /// A catalogue file that replaces the network fetch and the cache (`SCALEUP_HUB_CATALOG`,
    /// as for the command line).
    pub catalog_file: Option<PathBuf>,
    pub http: Arc<dyn Http + Send + Sync>,
    /// The Python to convert with, instead of `SCALEUP_PYTHON` or `PATH`.
    pub python: Option<OsString>,
}

impl HubConfig {
    /// The real network, `SCALEUP_HUB_CATALOG` when it is set, Python found the usual way.
    pub fn from_env() -> Self {
        Self {
            catalog_file: std::env::var_os("SCALEUP_HUB_CATALOG").map(PathBuf::from),
            http: Arc::new(UreqHttp),
            python: None,
        }
    }
}

pub enum UiMsg {
    Start {
        mode: Mode,
        settings: Box<Settings>,
    },
    /// *Try several*: run each fill in turn on the same single image with the same extension.
    /// Each finished fill is a history step; a failed one is reported and the queue goes on; Stop
    /// ends the queue.
    #[allow(
        dead_code,
        reason = "wired by the history strip and the Try several list (plan 12, Tasks 6 and 7)"
    )]
    StartFills {
        mode: Mode,
        settings: Box<Settings>,
        fills: Vec<FillChoice>,
    },
    /// Show a history step: its input as "before", its output as "after" with its margins.
    ShowStep {
        input: PathBuf,
        output: PathBuf,
        margins: Margins,
    },
    Stop,
    ChangeDevice(String),
    /// Makes the worker panic, as a GPU driver or validation failure would.
    #[cfg(test)]
    TestPanic,
    /// Fetch a catalogue model without running anything.
    Download(String),
    /// Decode this file as the new "before" image.
    Preview(PathBuf),
    /// Load the OpenModelDB catalogue: from the cache when it is under 24 h old, unless `force`.
    HubRefresh {
        force: bool,
    },
    /// Download, verify and install an OpenModelDB model.
    HubInstall(String),
    /// Delete an installed OpenModelDB model.
    HubRemove(String),
    /// Install a file the user downloaded by hand, after checking it against the catalogue.
    HubImport {
        id: String,
        file: PathBuf,
    },
    /// Download the PyTorch file of a card and install it converted to ONNX.
    HubConvert(String),
    /// Fetch the before/after thumbnails of a card.
    HubThumbs(String),
    Quit,
}

impl UiMsg {
    /// The message that shows `step` again.
    pub fn show_step(step: &Step) -> Self {
        UiMsg::ShowStep {
            input: step.input.clone(),
            output: step.output.clone(),
            margins: step.margins,
        }
    }
}

/// Which side of the comparison a preview belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    Before,
    After,
}

#[derive(Debug, PartialEq)]
pub enum WorkerEvent {
    Device(String),
    Devices(Vec<DeviceInfo>),
    Download {
        done: u64,
        total: u64,
    },
    Progress {
        file: usize,
        files: usize,
        percent: f32,
        phase: String,
    },
    /// A file was written; in a folder run also one that was already there and kept.
    FileDone {
        output: PathBuf,
        /// What the history keeps of it: set for a single image (a fill of *Try several*
        /// included), `None` in a folder run.
        step: Option<Box<Step>>,
    },
    AllDone {
        done: usize,
        failed: usize,
    },
    Stopped {
        done: usize,
        total: usize,
    },
    /// One file could not be processed; the run goes on. Always followed by `AllDone`.
    FileFailed {
        input: PathBuf,
        error: String,
    },
    /// The run ended or could not start, or an idle-time request failed. Nothing is running now.
    Error(String),
    Log(String),
    ModelDownloaded(String),
    Preview {
        which: Which,
        /// The file decoded: the window matches a history step's previews and thumbnail by it.
        path: PathBuf,
        image: egui::ColorImage,
        /// Size of the file, which `image` may be a downscaled copy of.
        size: [u32; 2],
        /// Canvas this result added around its source image (single image or batch file), so
        /// the view can place that file's "before" inside it. Zero for a "before" and when
        /// nothing was extended.
        margins: Margins,
    },
    HubCatalog(Arc<Catalog>),
    /// The catalogue could not be loaded. Separate from `Error`, so other failures do not touch the
    /// window's loading state.
    HubCatalogFailed(String),
    /// A model was installed (downloaded, imported or converted); the worker's registry lists it.
    HubInstalled(InstalledModel),
    /// A model was removed; the worker's registry no longer lists it.
    HubRemoved(String),
    /// The before and after thumbnails of card `id`; a missing one is `None`.
    HubThumb {
        id: String,
        img: [Option<egui::ColorImage>; 2],
    },
}

pub struct WorkerHandle {
    pub tx: Sender<UiMsg>,
    pub rx: Receiver<WorkerEvent>,
    join: Option<JoinHandle<()>>,
    wake: Arc<OnceLock<egui::Context>>,
}

impl WorkerHandle {
    /// From now on every event repaints `ctx`, so idle windows notice results without polling.
    pub fn wake_on_events(&self, ctx: egui::Context) {
        let _ = self.wake.set(ctx);
    }
}

/// The sending half of the event channel.
struct Events {
    tx: Sender<WorkerEvent>,
    wake: Arc<OnceLock<egui::Context>>,
}

impl Events {
    fn send(&self, ev: WorkerEvent) {
        let _ = self.tx.send(ev);
        if let Some(ctx) = self.wake.get() {
            ctx.request_repaint();
        }
    }
}

impl Drop for WorkerHandle {
    /// Asks the worker to quit and gives it a moment, so a file being saved is finished rather than
    /// left behind as a temporary file; a slow tile must not hold up closing the window for long.
    fn drop(&mut self) {
        let _ = self.tx.send(UiMsg::Quit);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        if let Some(join) = self.join.take() {
            while !join.is_finished() && std::time::Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
    }
}

pub fn spawn(registry: Registry, initial_gpu: &str) -> WorkerHandle {
    spawn_with(registry, initial_gpu, HubConfig::from_env())
}

pub fn spawn_with(registry: Registry, initial_gpu: &str, hub: HubConfig) -> WorkerHandle {
    let (ui_tx, ui_rx) = channel();
    let (ev_tx, ev_rx) = channel();
    let gpu = initial_gpu.to_string();
    let wake = Arc::new(OnceLock::new());
    let events = Events {
        tx: ev_tx,
        wake: wake.clone(),
    };
    let join = std::thread::Builder::new()
        .name("scaleup-worker".into())
        .spawn(move || {
            let mut worker = Worker {
                registry,
                events,
                inbox: Inbox {
                    rx: ui_rx,
                    deferred: VecDeque::new(),
                    quit: false,
                },
                gpu,
                upscaler: None,
                model: None,
                inpainter: None,
                hub,
                catalog: None,
            };
            worker.run();
        })
        .expect("spawning the worker thread");
    WorkerHandle {
        tx: ui_tx,
        rx: ev_rx,
        join: Some(join),
        wake,
    }
}

/// `model_id` as it appears in output names: the file stem for a custom model file.
fn model_label(model_id: &str) -> String {
    let path = Path::new(model_id);
    if (path.extension().is_some() || path.components().count() > 1)
        && let Some(stem) = path.file_stem()
    {
        return stem.to_string_lossy().into_owned();
    }
    model_id.to_string()
}

/// The files a run processes. Batch mode takes the supported images directly inside the folder,
/// sorted, leaving out earlier results of this app.
pub(crate) fn plan_jobs(mode: &Mode, s: &Settings) -> Vec<FileJob> {
    let inputs = match mode {
        Mode::Single(p) => vec![p.clone()],
        Mode::Batch(dir) => {
            let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file() && is_supported_input(p))
                .filter(|p| {
                    !p.file_stem()
                        .is_some_and(|n| n.to_string_lossy().contains("_scaleup_"))
                })
                .collect();
            files.sort();
            files
        }
    };
    let format = OutputFormat::parse(&s.format).unwrap_or(OutputFormat::Png);
    let opts = s.to_options();
    // Extend only: no model, so none in the name either.
    let label = s.upscales().then(|| model_label(&s.model_id));
    // `start` refuses settings whose fill cannot be read before it plans anything.
    let extend = s.to_extend_options().ok().flatten();
    inputs
        .into_iter()
        .map(|input| FileJob {
            output: Upscaler::output_name_for(
                &input,
                &s.output_dir_for(&input),
                label.as_deref(),
                extend.as_ref().map(|e| &e.fill),
                Some(&s.extend_model),
                &opts,
                format,
            ),
            input,
            format,
            quality: s.quality,
            overwrite: s.overwrite,
            double: s.double,
            extend: extend.clone(),
        })
        .collect()
}

/// Size of `path` as [`load_image`] returns it: with EXIF orientation applied, so a quarter turn
/// swaps the sides. Reads the header only, the way `load_image` opens the file.
fn oriented_dimensions(path: &Path) -> Option<(u32, u32)> {
    use image::ImageDecoder;
    use image::metadata::Orientation;
    let mut decoder = image::ImageReader::open(path)
        .ok()?
        .with_guessed_format()
        .ok()?
        .into_decoder()
        .ok()?;
    let (w, h) = decoder.dimensions();
    let turned = matches!(
        decoder.orientation().ok()?,
        Orientation::Rotate90
            | Orientation::Rotate270
            | Orientation::Rotate90FlipH
            | Orientation::Rotate270FlipH
    );
    Some(if turned { (h, w) } else { (w, h) })
}

/// The canvas `job` adds around its input, for the compare view: the margins resolved against the
/// oriented image, as `run_file` resolves them. Zero when it does not extend or the input cannot
/// be measured.
fn job_margins(job: &FileJob) -> Margins {
    job.extend
        .as_ref()
        .and_then(|ext| {
            let (w, h) = oriented_dimensions(&job.input)?;
            ext.margins.resolve(w, h).ok()
        })
        .unwrap_or_default()
}

/// Long side limit of preview textures.
const MAX_PREVIEW_SIDE: u32 = 4096;

/// `img`, downscaled if needed so its long side is at most `max`: GPUs refuse larger textures.
pub(crate) fn cap_long_side(img: image::RgbaImage, max: u32) -> image::RgbaImage {
    let (w, h) = img.dimensions();
    let long = w.max(h);
    if long <= max {
        return img;
    }
    let scaled =
        |side: u32| ((side as u64 * max as u64 + long as u64 / 2) / long as u64).max(1) as u32;
    image::imageops::resize(
        &img,
        scaled(w),
        scaled(h),
        image::imageops::FilterType::Triangle,
    )
}

/// Decodes `path` into a preview event.
fn preview_event(path: &Path, which: Which, margins: Margins) -> Result<WorkerEvent> {
    let img = load_image(path)?;
    let size = [img.width(), img.height()];
    let img = cap_long_side(img, MAX_PREVIEW_SIDE);
    let dims = [img.width() as usize, img.height() as usize];
    Ok(WorkerEvent::Preview {
        which,
        path: path.to_path_buf(),
        image: egui::ColorImage::from_rgba_unmultiplied(dims, img.as_raw()),
        size,
        margins,
    })
}

fn phase_name(phase: Phase) -> &'static str {
    match phase {
        Phase::Loading => "Loading image",
        Phase::Extending => "Extending",
        Phase::Upscaling => "Upscaling",
        Phase::Resizing => "Resizing",
        Phase::Saving => "Saving",
    }
}

/// Where a file is in its run, for its progress events: file `file` of `files`, and in a fill
/// queue the `Fill 2 of 3: ` put before each phase.
#[derive(Clone, Copy)]
struct Place<'a> {
    file: usize,
    files: usize,
    prefix: &'a str,
}

impl Place<'_> {
    fn progress(&self, percent: f32, phase: &str) -> WorkerEvent {
        WorkerEvent::Progress {
            file: self.file,
            files: self.files,
            percent,
            phase: format!("{}{phase}", self.prefix),
        }
    }
}

/// How one file of a run ended. Its events are already sent.
enum Outcome {
    Done,
    /// This file failed; the run goes on. The caller reports it with `FileFailed`.
    Failed(String),
    Stopped,
    /// The upscaler or the upscaling model could not be loaded: every later file, and every
    /// later fill of a queue, would fail the same way. Not reported yet.
    Fatal(String),
    /// The inpainting model of this file's AI fill could not be loaded. Ends a normal run (every
    /// file uses the same model); in a fill queue only this fill fails. Not reported yet.
    NoInpainter(String),
}

/// `path` when no file is there, else the first free `<stem>-2.<ext>`, `<stem>-3.<ext>`, ... next
/// to it. A single-image run writes there rather than reuse a file from an earlier run, whose
/// settings may differ in what the name does not say (margins, aspect).
pub(crate) fn next_free_name(path: &Path) -> PathBuf {
    if !path.exists() {
        return path.to_path_buf();
    }
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let ext = path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    (2..)
        .map(|n| path.with_file_name(format!("{stem}-{n}{ext}")))
        .find(|p| !p.exists())
        .expect("some number is free")
}

/// The history step of a finished single-image `job`. `Err` when the input or the result cannot
/// be measured: a step with no size would mislead the view.
fn step_for(
    job: &FileJob,
    settings: &Settings,
    registry: &Registry,
) -> std::result::Result<Step, String> {
    let size = |p: &Path| {
        oriented_dimensions(p)
            .map(|(w, h)| [w, h])
            .ok_or_else(|| format!("cannot read the size of {}", p.display()))
    };
    Ok(Step {
        input: job.input.clone(),
        output: job.output.clone(),
        settings: settings.clone(),
        source_size: size(&job.input)?,
        result_size: size(&job.output)?,
        margins: job_margins(job),
        label: step_label(settings, registry),
    })
}

/// The worker's end of the UI channel; messages that cannot be acted on mid-run wait in `deferred`.
struct Inbox {
    rx: Receiver<UiMsg>,
    deferred: VecDeque<UiMsg>,
    quit: bool,
}

impl Inbox {
    /// Applies what can happen during a run: `Stop` and `Quit` cancel it.
    fn poll(&mut self, cancel: &CancelToken) {
        loop {
            match self.rx.try_recv() {
                Ok(UiMsg::Stop) => cancel.cancel(),
                Ok(UiMsg::Quit) | Err(TryRecvError::Disconnected) => {
                    self.quit = true;
                    cancel.cancel();
                    return;
                }
                Ok(other) => self.deferred.push_back(other),
                Err(TryRecvError::Empty) => return,
            }
        }
    }

    fn next(&mut self) -> Option<UiMsg> {
        if self.quit {
            return None;
        }
        self.deferred.pop_front().or_else(|| self.rx.recv().ok())
    }
}

struct Worker {
    registry: Registry,
    events: Events,
    inbox: Inbox,
    gpu: String,
    upscaler: Option<Upscaler>,
    /// The loaded model and the `model_id` it came from.
    model: Option<(String, Box<dyn Model>)>,
    /// The loaded inpainting model of the AI fill and its id.
    inpainter: Option<(String, Box<dyn Inpainter>)>,
    hub: HubConfig,
    /// The OpenModelDB catalogue, once loaded.
    catalog: Option<Arc<Catalog>>,
}

/// Reads that fail once `cancel` is set, so a download stops at the next chunk.
struct CancelRead {
    inner: Box<dyn Read + Send>,
    cancel: CancelToken,
}

impl Read for CancelRead {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.cancel.is_cancelled() {
            // Not `Interrupted`: readers retry that.
            return Err(std::io::Error::other("cancelled"));
        }
        self.inner.read(buf)
    }
}

/// `inner` with bodies that stop when `cancel` is set.
struct CancelHttp<'a> {
    inner: &'a dyn Http,
    cancel: CancelToken,
}

impl Http for CancelHttp<'_> {
    fn get(&self, url: &str, cookies: &[(String, String)]) -> Result<HttpResponse> {
        if self.cancel.is_cancelled() {
            return Err(ScaleUpError::Cancelled);
        }
        let mut response = self.inner.get(url, cookies)?;
        response.body = Box::new(CancelRead {
            inner: response.body,
            cancel: self.cancel.clone(),
        });
        Ok(response)
    }
}

/// How a card is to be installed.
enum HubJob {
    Download,
    Convert,
    Import(PathBuf),
}

impl Worker {
    fn emit(&self, ev: WorkerEvent) {
        self.events.send(ev);
    }

    fn run(&mut self) {
        self.guarded(|w| {
            w.emit(WorkerEvent::Devices(list_devices()));
            if let Err(e) = w.ensure_upscaler() {
                w.emit(WorkerEvent::Error(e.to_string()));
            }
        });
        while let Some(msg) = self.inbox.next() {
            if matches!(msg, UiMsg::Quit) {
                return;
            }
            self.guarded(|w| w.handle(msg));
        }
    }

    /// Runs `work`, turning a panic (wgpu raises GPU failures that way) into an `Error` event. The UI
    /// would otherwise wait forever. Device and model are dropped: they may be what failed.
    fn guarded(&mut self, work: impl FnOnce(&mut Self)) {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(self)));
        if let Err(payload) = outcome {
            let detail = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown failure");
            self.model = None;
            self.inpainter = None;
            self.upscaler = None;
            self.emit(WorkerEvent::Error(format!(
                "processing stopped unexpectedly: {detail}"
            )));
        }
    }

    fn handle(&mut self, msg: UiMsg) {
        match msg {
            UiMsg::Start { mode, settings } => self.start(&mode, &settings),
            UiMsg::StartFills {
                mode,
                settings,
                fills,
            } => self.start_fills(&mode, &settings, &fills),
            UiMsg::ShowStep {
                input,
                output,
                margins,
            } => self.show_step(&input, &output, margins),
            // Nothing is running, so there is nothing to stop.
            UiMsg::Stop => {}
            UiMsg::ChangeDevice(gpu) => self.change_device(gpu),
            UiMsg::Preview(path) => match preview_event(&path, Which::Before, Margins::default()) {
                Ok(ev) => self.emit(ev),
                Err(e) => self.emit(WorkerEvent::Error(format!("{}: {e}", path.display()))),
            },
            UiMsg::Download(id) => self.download(&id),
            UiMsg::HubRefresh { force } => match self.catalog(force) {
                Ok(cat) => self.emit(WorkerEvent::HubCatalog(cat)),
                Err(e) => self.emit(WorkerEvent::HubCatalogFailed(format!(
                    "OpenModelDB catalogue: {e}"
                ))),
            },
            UiMsg::HubInstall(id) => self.hub_install(&id, HubJob::Download),
            UiMsg::HubConvert(id) => self.hub_install(&id, HubJob::Convert),
            UiMsg::HubImport { id, file } => self.hub_install(&id, HubJob::Import(file)),
            UiMsg::HubRemove(id) => match uninstall(self.registry.models_dir(), &id) {
                Ok(()) => {
                    self.reload_registry();
                    self.emit(WorkerEvent::HubRemoved(id));
                }
                Err(e) => self.emit(WorkerEvent::Error(e.to_string())),
            },
            UiMsg::HubThumbs(id) => self.hub_thumbs(id),
            #[cfg(test)]
            UiMsg::TestPanic => panic!("test panic"),
            UiMsg::Quit => {}
        }
    }

    fn download(&mut self, id: &str) {
        let cancel = CancelToken::new();
        let (events, inbox) = (&self.events, &mut self.inbox);
        let result = self.registry.ensure_cancellable(
            id,
            &mut |done, total| {
                events.send(WorkerEvent::Download { done, total });
                inbox.poll(&cancel);
            },
            &cancel,
        );
        match result {
            Ok(_) => self.emit(WorkerEvent::ModelDownloaded(id.to_string())),
            Err(ScaleUpError::Cancelled) => {
                self.emit(WorkerEvent::Stopped { done: 0, total: 1 });
            }
            Err(e) => self.emit(WorkerEvent::Error(e.to_string())),
        }
    }

    /// The OpenModelDB catalogue, loaded once (or again when `force`).
    fn catalog(&mut self, force: bool) -> Result<Arc<Catalog>> {
        if !force && let Some(cat) = &self.catalog {
            return Ok(cat.clone());
        }
        let cat = match &self.hub.catalog_file {
            Some(file) => Catalog::parse(
                &std::fs::read_to_string(file)
                    .map_err(|e| ScaleUpError::Model(format!("{}: {e}", file.display())))?,
            )?,
            None => Catalog::fetch(self.registry.models_dir(), force)?,
        };
        let cat = Arc::new(cat);
        self.catalog = Some(cat.clone());
        Ok(cat)
    }

    fn card(&mut self, id: &str) -> Result<ModelCard> {
        self.catalog(false)?.get(id).cloned().ok_or_else(|| {
            ScaleUpError::Model(format!("unknown model '{id}' in the OpenModelDB catalogue"))
        })
    }

    fn python(&self) -> Result<Python> {
        match &self.hub.python {
            Some(exe) => find_python_from(Some(exe.clone())),
            None => find_python(),
        }
    }

    /// The registry reads `installed.json` when it is built: build it again so the new list is
    /// used, and drop the upscaler, which holds a copy.
    fn reload_registry(&mut self) {
        self.registry = Registry::with_dir(self.registry.models_dir().to_path_buf());
        self.upscaler = None;
        self.model = None;
        self.inpainter = None;
    }

    fn hub_install(&mut self, id: &str, job: HubJob) {
        let card = match self.card(id) {
            Ok(card) => card,
            Err(e) => return self.emit(WorkerEvent::Error(e.to_string())),
        };
        let converts = match &job {
            HubJob::Convert => true,
            HubJob::Import(file) => import_needs_conversion(&card, file, &Capabilities::current()),
            HubJob::Download => false,
        };
        // Before downloading anything: without Python the file would be of no use.
        let python = if converts {
            match self.python() {
                Ok(p) => Some(p),
                Err(e) => return self.emit(WorkerEvent::Error(e.to_string())),
            }
        } else {
            None
        };
        let models_dir = self.registry.models_dir().to_path_buf();
        let cancel = CancelToken::new();
        let http = CancelHttp {
            inner: self.hub.http.as_ref(),
            cancel: cancel.clone(),
        };
        let events = &self.events;
        // Both the download progress and the conversion's cancel check read the inbox.
        let inbox = RefCell::new(&mut self.inbox);
        let mut progress = |done: u64, total: Option<u64>| {
            events.send(WorkerEvent::Download {
                done,
                total: total.unwrap_or(0),
            });
            inbox.borrow_mut().poll(&cancel);
        };
        let mut cancelled = || {
            inbox.borrow_mut().poll(&cancel);
            cancel.is_cancelled()
        };
        let converting = || {
            events.send(WorkerEvent::Progress {
                file: 0,
                files: 1,
                percent: 0.0,
                phase: "Converting to ONNX".into(),
            })
        };
        let result = match (&job, &python) {
            (HubJob::Import(file), Some(python)) => {
                converting();
                import_converted(&card, file, &models_dir, python, &mut cancelled)
            }
            (HubJob::Import(file), None) => import_file(&card, file, &models_dir),
            (_, Some(python)) => install_converted(
                &card,
                &models_dir,
                python,
                &http,
                &mut progress,
                &mut || converting(),
                &mut cancelled,
            ),
            (_, None) => install(
                &card,
                &models_dir,
                &Capabilities::current(),
                &http,
                &mut progress,
            ),
        };
        match result {
            Ok(model) => {
                self.reload_registry();
                self.emit(WorkerEvent::HubInstalled(model));
            }
            Err(_) if cancel.is_cancelled() => {
                self.emit(WorkerEvent::Stopped { done: 0, total: 1 });
            }
            Err(e) => self.emit(WorkerEvent::Error(e.to_string())),
        }
    }

    /// Always answers, so the window stops waiting; a thumbnail that failed is `None`.
    fn hub_thumbs(&mut self, id: String) {
        let urls = self
            .catalog
            .as_ref()
            .and_then(|c| c.get(&id))
            .map(thumbs::pair_urls)
            .unwrap_or_default();
        let cache = self.registry.models_dir().join("hub").join("thumbs");
        let img = urls.map(|url| {
            let img = thumbs::load(self.hub.http.as_ref(), url.as_deref()?, &cache)?;
            let size = [img.width() as usize, img.height() as usize];
            Some(egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw()))
        });
        self.emit(WorkerEvent::HubThumb { id, img });
    }

    /// A preview that cannot be decoded must not disturb the run: it only reaches the log.
    fn emit_preview(&self, path: &Path, which: Which, margins: Margins) {
        match preview_event(path, which, margins) {
            Ok(ev) => self.emit(ev),
            Err(e) => self.emit(WorkerEvent::Log(format!(
                "no preview for {}: {e}",
                path.display()
            ))),
        }
    }

    fn change_device(&mut self, gpu: String) {
        if gpu != self.gpu {
            self.gpu = gpu;
            self.upscaler = None;
            self.model = None;
            self.inpainter = None;
        }
        if let Err(e) = self.ensure_upscaler() {
            self.emit(WorkerEvent::Error(e.to_string()));
        }
    }

    fn ensure_upscaler(&mut self) -> Result<()> {
        if self.upscaler.is_none() {
            let pref = DevicePreference::parse(&self.gpu).map_err(ScaleUpError::InvalidOption)?;
            let upscaler = Upscaler::new(pref, self.registry.clone())?;
            self.emit(WorkerEvent::Device(upscaler.device().name.clone()));
            self.upscaler = Some(upscaler);
        }
        Ok(())
    }

    /// Makes `model_id` the loaded model, downloading it first if needed.
    fn ensure_model(&mut self, model_id: &str, cancel: &CancelToken) -> Result<()> {
        if self.model.as_ref().is_some_and(|(id, _)| id == model_id) {
            return Ok(());
        }
        self.model = None;
        self.ensure_upscaler()?;
        let upscaler = self.upscaler.as_ref().expect("just ensured");
        if upscaler.registry().get(model_id).is_some() {
            let (events, inbox) = (&self.events, &mut self.inbox);
            upscaler.registry().ensure_cancellable(
                model_id,
                &mut |done, total| {
                    events.send(WorkerEvent::Download { done, total });
                    inbox.poll(cancel);
                },
                cancel,
            )?;
        }
        let model = upscaler.load(model_id, &mut |_, _| {})?;
        let device = upscaler.active_device().name;
        self.emit(WorkerEvent::Log(format!(
            "model {model_id} ({}x) on {device}, automatic tile size {}",
            model.scale(),
            upscaler.effective_tile(0, model.preferred_overlap())
        )));
        self.emit(WorkerEvent::Device(device));
        self.model = Some((model_id.to_string(), model));
        Ok(())
    }

    /// Makes `id` the loaded inpainting model, downloading it first if needed. Like the upscale
    /// model, the download reads the inbox so Stop aborts it; loading itself cannot be interrupted
    /// and takes many seconds for LaMa, so the status says what is happening.
    fn ensure_inpainter(&mut self, id: &str, place: Place, cancel: &CancelToken) -> Result<()> {
        if self
            .inpainter
            .as_ref()
            .is_some_and(|(loaded, _)| loaded == id)
        {
            return Ok(());
        }
        self.inpainter = None;
        self.ensure_upscaler()?;
        let upscaler = self.upscaler.as_ref().expect("just ensured");
        let loading = || place.progress(0.0, "Loading extension model");
        self.emit(loading());
        // A download in between replaces that status with its own; its last chunk brings it back.
        let (events, inbox) = (&self.events, &mut self.inbox);
        let inpainter = upscaler.load_inpainter(
            id,
            &mut |done, total| {
                events.send(WorkerEvent::Download { done, total });
                if total > 0 && done >= total {
                    events.send(loading());
                }
                inbox.poll(cancel);
            },
            cancel,
        )?;
        self.emit(WorkerEvent::Log(format!(
            "inpainting model {id}, {} px window",
            inpainter.window()
        )));
        self.inpainter = Some((id.to_string(), inpainter));
        Ok(())
    }

    fn start(&mut self, mode: &Mode, settings: &Settings) {
        if settings.gpu != self.gpu {
            self.change_device(settings.gpu.clone());
        }
        if let Some(why) = settings.start_blocker(None) {
            self.emit(WorkerEvent::Error(why));
            return;
        }
        let jobs = plan_jobs(mode, settings);
        if jobs.is_empty() {
            self.emit(WorkerEvent::Error("no supported images found".into()));
            return;
        }
        let cancel = CancelToken::new();
        let files = jobs.len();
        let single = matches!(mode, Mode::Single(_));
        let (mut done, mut failed) = (0, 0);
        for (index, job) in jobs.iter().enumerate() {
            self.inbox.poll(&cancel);
            if cancel.is_cancelled() {
                self.emit(WorkerEvent::Stopped { done, total: files });
                return;
            }
            let place = Place {
                file: index,
                files,
                prefix: "",
            };
            match self.run_one(place, job, settings, single, &cancel) {
                Outcome::Done => done += 1,
                Outcome::Failed(error) => {
                    self.emit(WorkerEvent::FileFailed {
                        input: job.input.clone(),
                        error,
                    });
                    failed += 1;
                }
                Outcome::Stopped => {
                    self.emit(WorkerEvent::Stopped { done, total: files });
                    return;
                }
                // A missing inpainting model is reported, never replaced by a plain fill.
                Outcome::Fatal(e) | Outcome::NoInpainter(e) => {
                    self.emit(WorkerEvent::Error(e));
                    return;
                }
            }
        }
        self.emit(WorkerEvent::AllDone { done, failed });
    }

    /// *Try several*: each fill in turn on one image. A fill that fails, its inpainting model
    /// included, is reported as a failed file named after the fill, and the next one runs. A
    /// failure every fill would repeat (the upscaling model) ends the queue with one error. Two
    /// identical fills both run: the second writes the next free name.
    fn start_fills(&mut self, mode: &Mode, settings: &Settings, fills: &[FillChoice]) {
        let Mode::Single(input) = mode else {
            self.emit(WorkerEvent::Error(
                "Try several works on a single image, not a folder".into(),
            ));
            return;
        };
        if !settings.extend_enabled {
            self.emit(WorkerEvent::Error(
                "Turn on Extend canvas to try fills".into(),
            ));
            return;
        }
        if fills.is_empty() {
            self.emit(WorkerEvent::Error("Choose at least one fill to try".into()));
            return;
        }
        if settings.gpu != self.gpu {
            self.change_device(settings.gpu.clone());
        }
        let cancel = CancelToken::new();
        let total = fills.len();
        let (mut done, mut failed) = (0, 0);
        for (n, choice) in fills.iter().enumerate() {
            self.inbox.poll(&cancel);
            if cancel.is_cancelled() {
                self.emit(WorkerEvent::Stopped { done, total });
                return;
            }
            let s = choice.apply(settings);
            let name = fill_title(&s, &self.registry);
            let fail = |error: String| WorkerEvent::FileFailed {
                input: input.clone(),
                error: format!("{name}: {error}"),
            };
            if let Some(why) = s.start_blocker(None) {
                self.emit(fail(why));
                failed += 1;
                continue;
            }
            let job = plan_jobs(mode, &s).remove(0);
            let prefix = format!("Fill {} of {total}: ", n + 1);
            let place = Place {
                file: 0,
                files: 1,
                prefix: &prefix,
            };
            match self.run_one(place, &job, &s, true, &cancel) {
                Outcome::Done => done += 1,
                Outcome::Failed(e) | Outcome::NoInpainter(e) => {
                    self.emit(fail(e));
                    failed += 1;
                }
                Outcome::Stopped => {
                    self.emit(WorkerEvent::Stopped { done, total });
                    return;
                }
                Outcome::Fatal(e) => {
                    self.emit(WorkerEvent::Error(e));
                    return;
                }
            }
        }
        self.emit(WorkerEvent::AllDone { done, failed });
    }

    /// One file of a run: loads the models it needs, runs it and reports the result (a failure
    /// is left to the caller). `single` marks a single-image run, whose result becomes a history
    /// step: it never reuses an existing file without Overwrite, but writes the next free name,
    /// so the step describes a file this run wrote. A folder run skips files already done.
    fn run_one(
        &mut self,
        place: Place,
        job: &FileJob,
        settings: &Settings,
        single: bool,
        cancel: &CancelToken,
    ) -> Outcome {
        let free;
        let job = if single && !job.overwrite {
            free = FileJob {
                output: next_free_name(&job.output),
                ..job.clone()
            };
            &free
        } else {
            job
        };
        if job.output.exists() && !job.overwrite {
            self.emit(WorkerEvent::Log(format!(
                "{} exists, not overwriting",
                job.output.display()
            )));
            return match self.finished(job, settings, single) {
                Ok(()) => Outcome::Done,
                Err(e) => Outcome::Failed(e),
            };
        }
        // Extend only still runs through the upscaler; it loads no model then.
        let mut loaded = self.ensure_upscaler();
        if loaded.is_ok() && settings.upscales() {
            self.emit(place.progress(0.0, "Loading model"));
            loaded = self.ensure_model(&settings.model_id, cancel);
        }
        match loaded {
            Ok(()) => {}
            Err(ScaleUpError::Cancelled) => return Outcome::Stopped,
            Err(e) => return Outcome::Fatal(e.to_string()),
        }
        if job.extend.as_ref().is_some_and(|e| e.fill == Fill::Ai) {
            match self.ensure_inpainter(&settings.extend_model, place, cancel) {
                Ok(()) => {}
                Err(ScaleUpError::Cancelled) => return Outcome::Stopped,
                Err(e) => return Outcome::NoInpainter(e.to_string()),
            }
        }
        if !single {
            self.emit_preview(&job.input, Which::Before, Margins::default());
        }
        let started = std::time::Instant::now();
        let opts = settings.to_options();
        match self.run_job(place, job, settings.upscales(), &opts, cancel) {
            Ok(path) => {
                self.emit(WorkerEvent::Log(format!(
                    "{} in {:.1} s",
                    path.display(),
                    started.elapsed().as_secs_f32()
                )));
                match self.finished(job, settings, single) {
                    Ok(()) => Outcome::Done,
                    Err(e) => Outcome::Failed(e),
                }
            }
            Err(ScaleUpError::Cancelled) => Outcome::Stopped,
            Err(e) => Outcome::Failed(e.to_string()),
        }
    }

    /// Reports that `job.output` is there: `FileDone`, with a history step for a single image,
    /// then its "after" preview. `Err`, with nothing sent, when the step cannot be made.
    fn finished(
        &self,
        job: &FileJob,
        settings: &Settings,
        single: bool,
    ) -> std::result::Result<(), String> {
        let step = if single {
            Some(Box::new(step_for(job, settings, &self.registry)?))
        } else {
            None
        };
        self.emit(WorkerEvent::FileDone {
            output: job.output.clone(),
            step,
        });
        self.emit_preview(&job.output, Which::After, job_margins(job));
        Ok(())
    }

    /// Both previews of a history step. A file that is gone, or cannot be read, is an error that
    /// names it.
    fn show_step(&self, input: &Path, output: &Path, margins: Margins) {
        if let Some(gone) = [input, output].into_iter().find(|p| !p.is_file()) {
            self.emit(WorkerEvent::Error(format!(
                "{} no longer exists",
                gone.display()
            )));
            return;
        }
        for (path, which, margins) in [
            (input, Which::Before, Margins::default()),
            (output, Which::After, margins),
        ] {
            match preview_event(path, which, margins) {
                Ok(ev) => self.emit(ev),
                Err(e) => {
                    self.emit(WorkerEvent::Error(format!("{}: {e}", path.display())));
                    return;
                }
            }
        }
    }

    /// Runs `job` with the loaded models: the upscaling model only when `upscale`.
    fn run_job(
        &mut self,
        place: Place,
        job: &FileJob,
        upscale: bool,
        opts: &scaleup_core::UpscaleOptions,
        cancel: &CancelToken,
    ) -> Result<PathBuf> {
        let upscaler = self.upscaler.as_ref().expect("the upscaler is ready");
        let model = if upscale {
            let (_, model) = self.model.as_ref().expect("model is loaded");
            Some(model.as_ref())
        } else {
            None
        };
        // Loaded by `run_one` when this job's fill needs it.
        let inpainter = self.inpainter.as_ref().map(|(_, i)| i.as_ref());
        let (events, inbox) = (&self.events, &mut self.inbox);
        let mut phase = phase_name(Phase::Loading);
        let mut percent = 0.0;
        upscaler.run_file(
            model,
            inpainter,
            job,
            opts,
            &mut |p| {
                match p {
                    // Extension and upscaling each count their own tiles from zero.
                    Progress::Phase(ph) => {
                        phase = phase_name(ph);
                        if matches!(ph, Phase::Extending | Phase::Upscaling) {
                            percent = 0.0;
                        }
                    }
                    Progress::Tiles { done, total } => {
                        percent = done as f32 * 100.0 / total.max(1) as f32;
                    }
                }
                events.send(place.progress(percent, phase));
                inbox.poll(cancel);
            },
            cancel,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scaleup_core::ModelEntry;
    use std::io::{Read, Write};
    use std::time::Duration;

    fn fixture() -> String {
        format!(
            "{}/../../tests/fixtures/nearest2x.onnx",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    fn settings(out: &Path) -> Settings {
        Settings {
            model_id: fixture(),
            scale: Some(2),
            gpu: "cpu".into(),
            tile: 32,
            output_dir: Some(out.to_path_buf()),
            ..Settings::default()
        }
    }

    fn write_png(path: &Path, size: u32) {
        image::RgbaImage::from_fn(size, size, |x, y| {
            image::Rgba([(x % 251) as u8, (y % 251) as u8, 7, 255])
        })
        .save(path)
        .unwrap();
    }

    fn cpu_worker(dir: &Path) -> WorkerHandle {
        spawn(Registry::with_dir(dir.join("models")), "cpu")
    }

    /// Events up to and including the first `AllDone`, `Stopped` or `Error`.
    fn until_end(h: &WorkerHandle) -> Vec<WorkerEvent> {
        let mut out = Vec::new();
        loop {
            let ev =
                h.rx.recv_timeout(Duration::from_secs(120))
                    .expect("worker went quiet");
            let end = matches!(
                ev,
                WorkerEvent::AllDone { .. } | WorkerEvent::Stopped { .. } | WorkerEvent::Error(_)
            );
            out.push(ev);
            if end {
                return out;
            }
        }
    }

    #[test]
    fn download_failure_from_dialog_reports_error() {
        let d = tempfile::tempdir().unwrap();
        let mut entry = scaleup_core::registry::parse_manifest(
            r#"{"models":[{"id":"remote","name":"R","arch":"x","scale":2,"license":"MIT",
                "commercial":true,"size_bytes":1,"url":"","sha256":"00"}]}"#,
        )
        .unwrap();
        entry[0].url = failing_server();
        let h = spawn(
            Registry::from_entries(entry, d.path().join("models")),
            "cpu",
        );
        h.tx.send(UiMsg::Download("remote".into())).unwrap();
        let evs = until_end(&h);
        assert!(matches!(evs.last(), Some(WorkerEvent::Error(_))), "{evs:?}");
    }

    #[test]
    fn panic_in_worker_reports_error_and_recovers() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 8);
        let h = cpu_worker(d.path());
        h.tx.send(UiMsg::TestPanic).unwrap();
        let evs = until_end(&h);
        assert!(
            matches!(evs.last(), Some(WorkerEvent::Error(e)) if e.contains("stopped unexpectedly")),
            "{evs:?}"
        );
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input),
            settings: Box::new(settings(d.path())),
        })
        .unwrap();
        let evs = until_end(&h);
        assert!(
            matches!(
                evs.last(),
                Some(WorkerEvent::AllDone { done: 1, failed: 0 })
            ),
            "{evs:?}"
        );
    }

    #[test]
    fn preview_downscale_caps_4096() {
        let out = cap_long_side(image::RgbaImage::new(8000, 6000), 4096);
        assert_eq!(out.dimensions(), (4096, 3072));
        let out = cap_long_side(image::RgbaImage::new(100, 5000), 4096);
        assert_eq!(out.dimensions(), (82, 4096));
        let out = cap_long_side(image::RgbaImage::new(20, 20), 4096);
        assert_eq!(out.dimensions(), (20, 20));
    }

    #[test]
    fn preview_msg_emits_before_image_with_file_size() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 9);
        let h = cpu_worker(d.path());
        h.tx.send(UiMsg::Preview(input.clone())).unwrap();
        loop {
            match h.rx.recv_timeout(Duration::from_secs(120)).unwrap() {
                WorkerEvent::Preview {
                    which,
                    path,
                    image,
                    size,
                    margins,
                } => {
                    assert_eq!(path, input);
                    assert_eq!(margins, Margins::default());
                    assert_eq!(which, Which::Before);
                    assert_eq!(size, [9, 9]);
                    assert_eq!(image.size, [9, 9]);
                    return;
                }
                WorkerEvent::Error(e) => panic!("{e}"),
                _ => {}
            }
        }
    }

    #[test]
    fn preview_of_undecodable_file_is_an_error() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        std::fs::write(&input, b"nope").unwrap();
        let h = cpu_worker(d.path());
        h.tx.send(UiMsg::Preview(input)).unwrap();
        let evs = until_end(&h);
        assert!(matches!(evs.last(), Some(WorkerEvent::Error(_))));
    }

    #[test]
    fn finished_file_sends_after_preview_before_alldone() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 10);
        let h = cpu_worker(d.path());
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input),
            settings: Box::new(settings(d.path())),
        })
        .unwrap();
        let evs = until_end(&h);
        let has_after = evs.iter().any(|e| {
            matches!(
                e,
                WorkerEvent::Preview {
                    which: Which::After,
                    size: [20, 20],
                    ..
                }
            )
        });
        assert!(has_after, "no after preview");
    }

    #[test]
    fn plan_jobs_batch_skips_non_images() {
        let d = tempfile::tempdir().unwrap();
        for n in ["b.png", "a.jpg", "notes.txt", "c.webp"] {
            std::fs::write(d.path().join(n), b"x").unwrap();
        }
        std::fs::create_dir(d.path().join("sub.png")).unwrap();
        let out = d.path().join("out");
        let jobs = plan_jobs(&Mode::Batch(d.path().to_path_buf()), &settings(&out));
        let names: Vec<_> = jobs
            .iter()
            .map(|j| j.input.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, ["a.jpg", "b.png", "c.webp"]);
        assert!(jobs[0].output.starts_with(&out));
    }

    #[test]
    fn plan_jobs_batch_skips_previous_outputs_in_the_same_folder() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("a.png"), b"x").unwrap();
        std::fs::write(d.path().join("a_scaleup_4x_model.png"), b"x").unwrap();
        let mut s = settings(d.path());
        s.output_dir = None;
        let jobs = plan_jobs(&Mode::Batch(d.path().to_path_buf()), &s);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].output.parent().unwrap(), d.path());
    }

    #[test]
    fn plan_jobs_names_custom_model_files_by_stem() {
        let d = tempfile::tempdir().unwrap();
        let jobs = plan_jobs(&Mode::Single(d.path().join("a.png")), &settings(d.path()));
        assert_eq!(
            jobs[0].output.file_name().unwrap().to_str().unwrap(),
            "a_scaleup_2x_nearest2x.png"
        );
    }

    #[test]
    fn start_single_emits_progress_then_filedone_then_alldone() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 40);
        let h = cpu_worker(d.path());
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input),
            settings: Box::new(settings(d.path())),
        })
        .unwrap();
        let evs = until_end(&h);
        let pos = |f: fn(&WorkerEvent) -> bool| evs.iter().position(f);
        let progress = pos(|e| matches!(e, WorkerEvent::Progress { .. })).expect("progress");
        let file_done = pos(|e| matches!(e, WorkerEvent::FileDone { .. })).expect("file done");
        let all_done =
            pos(|e| matches!(e, WorkerEvent::AllDone { done: 1, failed: 0 })).expect("all done");
        assert!(progress < file_done && file_done < all_done, "{evs:?}");
        let Some(WorkerEvent::FileDone { output: p, .. }) = evs.get(file_done) else {
            unreachable!()
        };
        assert!(p.is_file());
        assert_eq!(image::image_dimensions(p).unwrap(), (80, 80));
    }

    #[test]
    fn stop_mid_batch_emits_stopped() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("in");
        std::fs::create_dir(&input).unwrap();
        for n in ["1.png", "2.png", "3.png"] {
            write_png(&input.join(n), 256);
        }
        let out = d.path().join("out");
        let h = cpu_worker(d.path());
        h.tx.send(UiMsg::Start {
            mode: Mode::Batch(input),
            settings: Box::new(settings(&out)),
        })
        .unwrap();
        loop {
            match h.rx.recv_timeout(Duration::from_secs(120)).unwrap() {
                WorkerEvent::FileDone { .. } => break,
                WorkerEvent::Error(e) => panic!("{e}"),
                _ => {}
            }
        }
        h.tx.send(UiMsg::Stop).unwrap();
        let evs = until_end(&h);
        let Some(WorkerEvent::Stopped { done, total }) = evs.last() else {
            panic!("{evs:?}")
        };
        assert_eq!(*total, 3);
        assert!((1..3).contains(done), "{done}");
        // Files that were not finished leave nothing behind.
        assert_eq!(std::fs::read_dir(&out).unwrap().count(), *done);
    }

    #[test]
    fn a_folder_rerun_skips_existing_outputs_without_progress() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("in");
        std::fs::create_dir(&dir).unwrap();
        let input = dir.join("a.png");
        write_png(&input, 8);
        let s = settings(&d.path().join("out"));
        let existing = plan_jobs(&Mode::Batch(dir.clone()), &s).remove(0).output;
        std::fs::create_dir_all(existing.parent().unwrap()).unwrap();
        std::fs::write(&existing, b"keep me").unwrap();
        let h = cpu_worker(d.path());
        h.tx.send(UiMsg::Start {
            mode: Mode::Batch(dir),
            settings: Box::new(s),
        })
        .unwrap();
        let evs = until_end(&h);
        assert!(
            !evs.iter()
                .any(|e| matches!(e, WorkerEvent::Progress { .. })),
            "{evs:?}"
        );
        assert!(evs.iter().any(|e| matches!(
            e,
            WorkerEvent::FileDone { output, step: None } if *output == existing
        )));
        assert_eq!(std::fs::read(&existing).unwrap(), b"keep me");
    }

    #[test]
    fn next_free_name_numbers_before_the_extension() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("a_scaleup_ext_mirror.png");
        assert_eq!(next_free_name(&p), p);
        std::fs::write(&p, b"x").unwrap();
        let second = d.path().join("a_scaleup_ext_mirror-2.png");
        assert_eq!(next_free_name(&p), second);
        std::fs::write(&second, b"x").unwrap();
        assert_eq!(
            next_free_name(&p),
            d.path().join("a_scaleup_ext_mirror-3.png")
        );
        let bare = d.path().join("noext");
        std::fs::write(&bare, b"x").unwrap();
        assert_eq!(next_free_name(&bare), d.path().join("noext-2"));
    }

    /// Runs a single image with `s` and returns the events up to the end.
    fn run_single(h: &WorkerHandle, input: &Path, s: Settings) -> Vec<WorkerEvent> {
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input.to_path_buf()),
            settings: Box::new(s),
        })
        .unwrap();
        until_end(h)
    }

    #[test]
    fn a_single_rerun_without_overwrite_writes_a_new_file_its_step_describes() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 10);
        let h = cpu_worker(d.path());
        let only = |px| Settings {
            scale: None,
            ..extend_settings(d.path(), "mirror", px)
        };
        let evs = run_single(&h, &input, only(4.0));
        let first = steps(&evs)[0].clone();
        // The name says nothing of the margins: the same name, now taken.
        let evs = run_single(&h, &input, only(8.0));
        let [second] = steps(&evs)[..] else {
            panic!("{evs:?}")
        };
        assert_eq!(
            first.output.file_name().unwrap(),
            "a_scaleup_ext_mirror.png"
        );
        assert_eq!(
            second.output.file_name().unwrap(),
            "a_scaleup_ext_mirror-2.png"
        );
        assert_eq!(first.result_size, [18, 18]);
        assert_eq!((second.result_size, second.margins.left), ([26, 26], 8));
        assert_eq!(image::image_dimensions(&first.output).unwrap(), (18, 18));
        assert_eq!(image::image_dimensions(&second.output).unwrap(), (26, 26));
    }

    #[test]
    fn a_single_rerun_with_overwrite_replaces_the_file() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 10);
        let h = cpu_worker(d.path());
        let only = |px| Settings {
            scale: None,
            overwrite: true,
            ..extend_settings(d.path(), "mirror", px)
        };
        let first = steps(&run_single(&h, &input, only(4.0)))[0].clone();
        let evs = run_single(&h, &input, only(8.0));
        let [second] = steps(&evs)[..] else {
            panic!("{evs:?}")
        };
        assert_eq!(second.output, first.output);
        assert_eq!(second.result_size, [26, 26]);
        assert_eq!(image::image_dimensions(&first.output).unwrap(), (26, 26));
    }

    #[test]
    fn custom_width_with_scale_none_upscales_to_the_width() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 10);
        let h = cpu_worker(d.path());
        let s = Settings {
            scale: None,
            use_width: true,
            width: 30,
            ..settings(d.path())
        };
        let evs = run_single(&h, &input, s);
        let [step] = steps(&evs)[..] else {
            panic!("{evs:?}")
        };
        assert_eq!(step.result_size, [30, 30]);
        assert_eq!(step.label, "Upscale 30 px wide · nearest2x");
        assert_eq!(
            step.output.file_name().unwrap(),
            "a_scaleup_30px_nearest2x.png"
        );
    }

    #[test]
    fn a_step_of_a_rotated_input_has_the_oriented_size_and_margins() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("portrait.jpg");
        write_rotated_jpeg(&input, 20, 10);
        let h = cpu_worker(d.path());
        let s = Settings {
            extend_mode: crate::state::ExtendMode::Percent,
            scale: None,
            ..extend_settings(d.path(), "mirror", 50.0)
        };
        let evs = run_single(&h, &input, s);
        let [step] = steps(&evs)[..] else {
            panic!("{evs:?}")
        };
        assert_eq!(step.source_size, [10, 20]);
        assert_eq!(
            step.margins,
            Margins {
                left: 5,
                top: 10,
                right: 5,
                bottom: 10
            }
        );
        assert_eq!(step.result_size, [20, 40]);
    }

    #[test]
    fn a_result_whose_size_cannot_be_read_makes_no_step() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 4);
        let output = d.path().join("out.png");
        std::fs::write(&output, b"not an image").unwrap();
        let mut job = plan_jobs(&Mode::Single(input.clone()), &settings(d.path())).remove(0);
        let reg = Registry::with_dir(d.path().join("models"));
        job.output = output;
        let e = step_for(&job, &settings(d.path()), &reg).unwrap_err();
        assert!(e.contains("out.png"), "{e}");
        job.output = input;
        assert!(step_for(&job, &settings(d.path()), &reg).is_ok());
    }

    /// Serves HTTP 500 to every request on a loopback port.
    fn failing_server() -> String {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            for mut c in l.incoming().flatten() {
                let mut buf = [0u8; 1024];
                let _ = c.read(&mut buf);
                let _ = c.write_all(
                    b"HTTP/1.1 500 Oops\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
            }
        });
        format!("http://{addr}/m.onnx")
    }

    #[test]
    fn download_failure_emits_error_and_recovers() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 8);
        let entry = ModelEntry {
            id: "remote".into(),
            name: "Remote".into(),
            arch: "x".into(),
            scale: 2,
            license: "MIT".into(),
            commercial: true,
            size_bytes: 10,
            url: failing_server(),
            sha256: "00".into(),
            tags: vec![],
            notes: String::new(),
            format: "onnx".into(),
            source: "catalogue".into(),
            kind: Default::default(),
            good_for: String::new(),
            layout: Default::default(),
        };
        let reg = Registry::from_entries(vec![entry], d.path().join("models"));
        let h = spawn(reg, "cpu");
        let mut s = settings(d.path());
        s.model_id = "remote".into();
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input.clone()),
            settings: Box::new(s.clone()),
        })
        .unwrap();
        let evs = until_end(&h);
        assert!(matches!(evs.last(), Some(WorkerEvent::Error(_))), "{evs:?}");

        s.model_id = fixture();
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input),
            settings: Box::new(s),
        })
        .unwrap();
        let evs = until_end(&h);
        assert!(
            matches!(
                evs.last(),
                Some(WorkerEvent::AllDone { done: 1, failed: 0 })
            ),
            "{evs:?}"
        );
    }

    #[test]
    fn failed_file_in_batch_is_counted_and_batch_continues() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("in");
        std::fs::create_dir(&input).unwrap();
        std::fs::write(input.join("1.png"), b"not an image").unwrap();
        write_png(&input.join("2.png"), 8);
        let h = cpu_worker(d.path());
        h.tx.send(UiMsg::Start {
            mode: Mode::Batch(input),
            settings: Box::new(settings(&d.path().join("out"))),
        })
        .unwrap();
        let mut last = None;
        while !matches!(last, Some(WorkerEvent::AllDone { .. })) {
            last = Some(h.rx.recv_timeout(Duration::from_secs(120)).unwrap());
        }
        assert_eq!(last, Some(WorkerEvent::AllDone { done: 1, failed: 1 }));
    }

    /// Serves fixed bodies by URL; anything else fails. Records every URL asked for.
    #[derive(Default)]
    struct MapHttp {
        bodies: std::collections::HashMap<String, Vec<u8>>,
        seen: std::sync::Mutex<Vec<String>>,
    }

    impl Http for MapHttp {
        fn get(&self, url: &str, _: &[(String, String)]) -> Result<HttpResponse> {
            self.seen.lock().unwrap().push(url.to_owned());
            let body = self
                .bodies
                .get(url)
                .cloned()
                .ok_or_else(|| ScaleUpError::Download(format!("{url}: 404")))?;
            Ok(HttpResponse {
                content_type: Some("application/octet-stream".into()),
                content_disposition: None,
                content_length: Some(body.len() as u64),
                cookies: Vec::new(),
                body: Box::new(std::io::Cursor::new(body)),
            })
        }
    }

    const DIRECT_URL: &str = "https://github.com/t/t/releases/download/v1/test-direct.onnx";
    const THUMB_URL: &str = "https://openmodeldb.info/thumbs/lr.png";
    const NEAREST_SHA: &str = "5ab9a633273dd61a59a49a289c33e15da8f3f4d8e0bb81d068b562e54964a0ff";

    /// A catalogue of synthetic cards around the nearest-neighbour fixture model.
    fn write_catalogue(dir: &Path) -> PathBuf {
        let card = |name: &str, arch: &str, resource: &str, extra: &str| {
            format!(
                r#""{name}": {{"name": "{name}", "author": "tests", "license": "MIT", "tags": [],
                "description": "", "date": "2026-01-01", "architecture": "{arch}", "scale": 2,
                "inputChannels": 3, "outputChannels": 3, "resources": [{resource}]{extra}}}"#
            )
        };
        let res = |platform: &str, kind: &str, url: &str| {
            format!(
                r#"{{"platform": "{platform}", "type": "{kind}", "sha256": "{NEAREST_SHA}", "urls": ["{url}"]}}"#
            )
        };
        let json = format!(
            "{{{}, {}, {}}}",
            card(
                "test-direct",
                "compact",
                &res("onnx", "onnx", DIRECT_URL),
                r#", "thumbnail": {"type": "paired", "LR": "/thumbs/lr.png", "SR": "http://plain.example/sr.png"}"#
            ),
            card(
                "test-manual",
                "compact",
                &res("onnx", "onnx", "https://mega.nz/file/x"),
                ""
            ),
            card(
                "test-convert",
                "dat",
                &res(
                    "pytorch",
                    "pth",
                    "https://github.com/t/t/releases/download/v1/c.pth"
                ),
                ""
            ),
        );
        let file = dir.join("catalogue.json");
        std::fs::write(&file, json).unwrap();
        file
    }

    fn hub_worker(dir: &Path, http: Arc<MapHttp>, python: Option<&str>) -> WorkerHandle {
        spawn_with(
            Registry::with_dir(dir.join("models")),
            "cpu",
            HubConfig {
                catalog_file: Some(write_catalogue(dir)),
                http,
                python: python.map(OsString::from),
            },
        )
    }

    /// Events up to and including the first hub answer, `Stopped` or `Error`.
    fn until_hub(h: &WorkerHandle) -> Vec<WorkerEvent> {
        let mut out = Vec::new();
        loop {
            let ev =
                h.rx.recv_timeout(Duration::from_secs(120))
                    .expect("worker went quiet");
            let end = matches!(
                ev,
                WorkerEvent::HubCatalog(_)
                    | WorkerEvent::HubCatalogFailed(_)
                    | WorkerEvent::HubInstalled(_)
                    | WorkerEvent::HubRemoved(_)
                    | WorkerEvent::HubThumb { .. }
                    | WorkerEvent::Stopped { .. }
                    | WorkerEvent::Error(_)
            );
            out.push(ev);
            if end {
                return out;
            }
        }
    }

    fn nearest_bytes() -> Vec<u8> {
        std::fs::read(fixture()).unwrap()
    }

    #[test]
    fn hub_refresh_sends_the_catalogue() {
        let d = tempfile::tempdir().unwrap();
        let sample = format!(
            "{}/../../tests/fixtures/hub/models.sample.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let h = spawn_with(
            Registry::with_dir(d.path().join("models")),
            "cpu",
            HubConfig {
                catalog_file: Some(PathBuf::from(sample)),
                http: Arc::new(MapHttp::default()),
                python: None,
            },
        );
        h.tx.send(UiMsg::HubRefresh { force: false }).unwrap();
        let evs = until_hub(&h);
        let Some(WorkerEvent::HubCatalog(cat)) = evs.last() else {
            panic!("{evs:?}")
        };
        assert_eq!(cat.models.len(), 40);
    }

    #[test]
    fn hub_install_downloads_verifies_and_the_model_runs() {
        let d = tempfile::tempdir().unwrap();
        let http = Arc::new(MapHttp {
            bodies: [(DIRECT_URL.to_owned(), nearest_bytes())].into(),
            ..MapHttp::default()
        });
        let h = hub_worker(d.path(), http, None);
        h.tx.send(UiMsg::HubInstall("test-direct".into())).unwrap();
        let evs = until_hub(&h);
        assert!(
            evs.iter()
                .any(|e| matches!(e, WorkerEvent::Download { .. })),
            "{evs:?}"
        );
        let Some(WorkerEvent::HubInstalled(m)) = evs.last() else {
            panic!("{evs:?}")
        };
        assert!(m.verified);
        assert!(
            m.file
                .starts_with(d.path().join("models").join("hub").join("test-direct"))
        );

        // The worker lists it at once: a run with its id needs no restart.
        let input = d.path().join("a.png");
        write_png(&input, 8);
        let mut s = settings(d.path());
        s.model_id = "test-direct".into();
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input),
            settings: Box::new(s),
        })
        .unwrap();
        let evs = until_end(&h);
        assert_eq!(
            evs.last(),
            Some(&WorkerEvent::AllDone { done: 1, failed: 0 }),
            "{evs:?}"
        );
    }

    #[test]
    fn hub_import_then_remove() {
        let d = tempfile::tempdir().unwrap();
        let h = hub_worker(d.path(), Arc::new(MapHttp::default()), None);
        let file = d.path().join("downloaded.onnx");
        std::fs::copy(fixture(), &file).unwrap();
        h.tx.send(UiMsg::HubImport {
            id: "test-manual".into(),
            file: file.clone(),
        })
        .unwrap();
        let evs = until_hub(&h);
        let Some(WorkerEvent::HubInstalled(m)) = evs.last() else {
            panic!("{evs:?}")
        };
        assert!(m.verified && m.file.is_file() && !file.exists());

        h.tx.send(UiMsg::HubRemove("test-manual".into())).unwrap();
        let evs = until_hub(&h);
        assert_eq!(
            evs.last(),
            Some(&WorkerEvent::HubRemoved("test-manual".into()))
        );
        assert!(
            !d.path()
                .join("models")
                .join("hub")
                .join("test-manual")
                .exists()
        );
        h.tx.send(UiMsg::HubRemove("test-manual".into())).unwrap();
        let evs = until_hub(&h);
        assert!(matches!(evs.last(), Some(WorkerEvent::Error(_))), "{evs:?}");
    }

    #[test]
    fn hub_import_of_a_wrong_file_is_an_error() {
        let d = tempfile::tempdir().unwrap();
        let h = hub_worker(d.path(), Arc::new(MapHttp::default()), None);
        let file = d.path().join("other.onnx");
        std::fs::write(&file, b"not the model").unwrap();
        h.tx.send(UiMsg::HubImport {
            id: "test-manual".into(),
            file: file.clone(),
        })
        .unwrap();
        let evs = until_hub(&h);
        assert!(
            matches!(evs.last(), Some(WorkerEvent::Error(e)) if e.contains("sha256")),
            "{evs:?}"
        );
        assert!(file.exists());
    }

    #[test]
    fn hub_convert_without_python_names_the_pip_line_and_downloads_nothing() {
        let d = tempfile::tempdir().unwrap();
        let http = Arc::new(MapHttp::default());
        let h = hub_worker(d.path(), http.clone(), Some("scaleup-no-such-python-xyz"));
        h.tx.send(UiMsg::HubConvert("test-convert".into())).unwrap();
        let evs = until_hub(&h);
        let Some(WorkerEvent::Error(e)) = evs.last() else {
            panic!("{evs:?}")
        };
        assert!(
            e.lines()
                .any(|l| l.trim() == scaleup_core::hub::PIP_INSTALL),
            "{e}"
        );
        assert!(http.seen.lock().unwrap().is_empty());
    }

    const CONVERT_URL: &str = "https://github.com/t/t/releases/download/v1/c.pth";

    /// A stand-in for Python: passes the `-c` probes; as the converter it writes
    /// `<output>.part`, sleeps about 5 s and fails.
    fn slow_python(dir: &Path) -> PathBuf {
        #[cfg(windows)]
        let (name, text) = (
            "python.cmd",
            "@echo off\r\nif \"%~1\"==\"-c\" (\r\necho 3\r\nexit /b 0\r\n)\r\necho partial> \"%~3.part\"\r\nping -n 6 127.0.0.1 >nul\r\nexit /b 1\r\n",
        );
        #[cfg(not(windows))]
        let (name, text) = (
            "python",
            "#!/bin/sh\nif [ \"$1\" = \"-c\" ]; then echo 3; exit 0; fi\necho partial > \"$3.part\"\nsleep 5\nexit 1\n",
        );
        let path = dir.join(name);
        std::fs::write(&path, text).unwrap();
        #[cfg(not(windows))]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }

    #[test]
    fn hub_refresh_failure_is_its_own_event() {
        let d = tempfile::tempdir().unwrap();
        let h = spawn_with(
            Registry::with_dir(d.path().join("models")),
            "cpu",
            HubConfig {
                catalog_file: Some(d.path().join("missing.json")),
                http: Arc::new(MapHttp::default()),
                python: None,
            },
        );
        h.tx.send(UiMsg::HubRefresh { force: false }).unwrap();
        let evs = until_hub(&h);
        assert!(
            matches!(evs.last(), Some(WorkerEvent::HubCatalogFailed(e)) if e.contains("missing.json")),
            "{evs:?}"
        );
    }

    #[test]
    fn stop_during_a_conversion_kills_python_and_leaves_nothing() {
        let d = tempfile::tempdir().unwrap();
        let python = slow_python(d.path());
        let http = Arc::new(MapHttp {
            bodies: [(CONVERT_URL.to_owned(), nearest_bytes())].into(),
            ..MapHttp::default()
        });
        let h = hub_worker(d.path(), http, python.to_str());
        h.tx.send(UiMsg::HubConvert("test-convert".into())).unwrap();
        loop {
            match h.rx.recv_timeout(Duration::from_secs(60)).unwrap() {
                WorkerEvent::Progress { phase, .. } if phase.contains("Converting") => break,
                WorkerEvent::Error(e) => panic!("{e}"),
                _ => {}
            }
        }
        std::thread::sleep(Duration::from_millis(500));
        let stopped_at = std::time::Instant::now();
        h.tx.send(UiMsg::Stop).unwrap();
        let evs = until_hub(&h);
        assert_eq!(
            evs.last(),
            Some(&WorkerEvent::Stopped { done: 0, total: 1 }),
            "{evs:?}"
        );
        assert!(
            stopped_at.elapsed() < Duration::from_secs(3),
            "{:?}",
            stopped_at.elapsed()
        );
        let models = d.path().join("models");
        let folder = models.join("hub").join("test-convert");
        let left: Vec<_> = std::fs::read_dir(&folder).unwrap().flatten().collect();
        assert!(left.is_empty(), "{left:?}");
        assert!(
            scaleup_core::hub::Installed::load(&models)
                .models
                .is_empty()
        );
    }

    #[test]
    fn hub_import_of_a_pytorch_file_converts_with_python() {
        let d = tempfile::tempdir().unwrap();
        let h = hub_worker(
            d.path(),
            Arc::new(MapHttp::default()),
            Some("scaleup-no-such-python-xyz"),
        );
        let file = d.path().join("c.pth");
        std::fs::copy(fixture(), &file).unwrap();
        h.tx.send(UiMsg::HubImport {
            id: "test-convert".into(),
            file: file.clone(),
        })
        .unwrap();
        let evs = until_hub(&h);
        let Some(WorkerEvent::Error(e)) = evs.last() else {
            panic!("{evs:?}")
        };
        assert!(
            e.lines()
                .any(|l| l.trim() == scaleup_core::hub::PIP_INSTALL),
            "{e}"
        );
        assert!(file.exists());
    }

    #[test]
    fn hub_thumbs_are_fetched_cached_and_missing_ones_are_none() {
        let d = tempfile::tempdir().unwrap();
        let http = Arc::new(MapHttp {
            bodies: [(THUMB_URL.to_owned(), crate::thumbs::tests::png(512, 256))].into(),
            ..MapHttp::default()
        });
        let h = hub_worker(d.path(), http.clone(), None);
        h.tx.send(UiMsg::HubRefresh { force: false }).unwrap();
        until_hub(&h);
        h.tx.send(UiMsg::HubThumbs("test-direct".into())).unwrap();
        let evs = until_hub(&h);
        let Some(WorkerEvent::HubThumb { id, img: [lr, sr] }) = evs.last() else {
            panic!("{evs:?}")
        };
        assert_eq!(id, "test-direct");
        assert_eq!(lr.as_ref().map(|i| i.size), Some([256, 128]));
        assert!(sr.is_none(), "plain http is never fetched");
        let cache = d.path().join("models").join("hub").join("thumbs");
        assert!(cache.join(crate::thumbs::cache_key(THUMB_URL)).is_file());
        assert_eq!(*http.seen.lock().unwrap(), [THUMB_URL]);
    }

    fn extend_settings(out: &Path, fill: &str, px: f32) -> Settings {
        Settings {
            extend_enabled: true,
            extend_left: px,
            extend_fill: fill.into(),
            extend_model: "test-fill".into(),
            ..settings(out)
        }
    }

    /// A registry with one inpainting entry, `test-fill`, served from `url`; when `installed`, the
    /// mean-filling fixture is already in place as its downloaded file.
    fn inpaint_fixture() -> PathBuf {
        PathBuf::from(format!(
            "{}/../../tests/fixtures/inpaint_mean_lama.onnx",
            env!("CARGO_MANIFEST_DIR")
        ))
    }

    fn inpaint_entry(url: &str, sha256: String) -> ModelEntry {
        ModelEntry {
            id: "test-fill".into(),
            name: "Test fill".into(),
            arch: "lama".into(),
            scale: 1,
            license: "MIT".into(),
            commercial: true,
            size_bytes: 100_000,
            url: url.into(),
            sha256,
            tags: vec![],
            notes: String::new(),
            format: "onnx".into(),
            source: "catalogue".into(),
            kind: scaleup_core::ModelKind::Inpaint,
            good_for: String::new(),
            layout: Default::default(),
        }
    }

    fn inpaint_registry(dir: &Path, url: &str, installed: bool) -> Registry {
        let models = dir.join("models");
        if installed {
            std::fs::create_dir_all(&models).unwrap();
            std::fs::copy(inpaint_fixture(), models.join("test-fill.onnx")).unwrap();
        }
        Registry::from_entries(vec![inpaint_entry(url, "00".repeat(32))], models)
    }

    #[test]
    fn plan_jobs_extend_names_and_carries_the_extension() {
        let d = tempfile::tempdir().unwrap();
        let s = extend_settings(d.path(), "mirror", 4.0);
        let jobs = plan_jobs(&Mode::Single(d.path().join("a.png")), &s);
        assert_eq!(
            jobs[0].output.file_name().unwrap().to_str().unwrap(),
            "a_scaleup_ext_mirror_2x_nearest2x.png"
        );
        assert_eq!(jobs[0].extend, s.to_extend_options().unwrap());
        assert!(jobs[0].extend.is_some());
    }

    #[test]
    fn plan_jobs_names_an_ai_fill_by_its_inpainting_model() {
        let d = tempfile::tempdir().unwrap();
        let mut s = extend_settings(d.path(), "ai", 4.0);
        let name = |s: &Settings| {
            plan_jobs(&Mode::Single(d.path().join("a.png")), s)[0]
                .output
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        };
        assert_eq!(name(&s), "a_scaleup_ext_ai-test-fill_2x_nearest2x.png");
        s.extend_model = "migan-512-places2".into();
        assert_eq!(
            name(&s),
            "a_scaleup_ext_ai-migan-512-places2_2x_nearest2x.png"
        );
    }

    /// The after preview of a run, with its size and margins.
    fn after_preview(evs: &[WorkerEvent]) -> Option<([u32; 2], Margins)> {
        evs.iter().find_map(|e| match e {
            WorkerEvent::Preview {
                which: Which::After,
                size,
                margins,
                ..
            } => Some((*size, *margins)),
            _ => None,
        })
    }

    #[test]
    fn single_run_extends_then_upscales_and_the_preview_carries_the_margins() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 10);
        let h = cpu_worker(d.path());
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input),
            settings: Box::new(extend_settings(d.path(), "mirror", 4.0)),
        })
        .unwrap();
        let evs = until_end(&h);
        assert_eq!(
            evs.last(),
            Some(&WorkerEvent::AllDone { done: 1, failed: 0 }),
            "{evs:?}"
        );
        let four = Margins {
            left: 4,
            top: 4,
            right: 4,
            bottom: 4,
        };
        assert_eq!(after_preview(&evs), Some(([36, 36], four)));
        assert!(
            evs.iter()
                .any(|e| matches!(e, WorkerEvent::Progress { phase, .. } if phase == "Extending"))
        );
    }

    /// A `w`×`h` JPEG whose EXIF orientation is 6 (rotate 90° clockwise to view): it loads as
    /// `h`×`w`.
    fn write_rotated_jpeg(path: &Path, w: u32, h: u32) {
        let img =
            image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x * 20) as u8, (y * 9) as u8, 90]));
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 95)
            .encode_image(&img)
            .unwrap();
        // APP1 "Exif": a little-endian TIFF header and one IFD entry, Orientation (0x0112) = 6.
        let tiff: [u8; 26] = [
            b'I', b'I', 0x2a, 0, 8, 0, 0, 0, // header, IFD at offset 8
            1, 0, // one entry
            0x12, 0x01, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, // Orientation, SHORT, 1, value 6
            0, 0, 0, 0, // no next IFD
        ];
        let mut app1 = vec![0xff, 0xe1, 0, (2 + 6 + tiff.len()) as u8];
        app1.extend_from_slice(b"Exif\0\0");
        app1.extend_from_slice(&tiff);
        let mut out = jpeg[..2].to_vec();
        out.extend_from_slice(&app1);
        out.extend_from_slice(&jpeg[2..]);
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn preview_margins_follow_the_exif_orientation_core_applies() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("portrait.jpg");
        write_rotated_jpeg(&input, 20, 10);
        assert_eq!(load_image(&input).unwrap().dimensions(), (10, 20));
        let s = Settings {
            extend_mode: crate::state::ExtendMode::Percent,
            ..extend_settings(d.path(), "mirror", 50.0)
        };
        let h = cpu_worker(d.path());
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input),
            settings: Box::new(s),
        })
        .unwrap();
        let evs = until_end(&h);
        assert_eq!(
            evs.last(),
            Some(&WorkerEvent::AllDone { done: 1, failed: 0 }),
            "{evs:?}"
        );
        // 50% of the oriented 10×20: 5 left and right, 10 top and bottom; the raw 20×10 would give
        // the opposite. The result is (oriented size + margins) × 2.
        let (size, m) = after_preview(&evs).unwrap();
        assert_eq!(
            m,
            Margins {
                left: 5,
                top: 10,
                right: 5,
                bottom: 10
            }
        );
        assert_eq!(size, [(10 + 5 + 5) * 2, (20 + 10 + 10) * 2]);
    }

    #[test]
    fn batch_previews_carry_each_files_margins() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("in");
        std::fs::create_dir(&dir).unwrap();
        write_png(&dir.join("a.png"), 10);
        let h = cpu_worker(d.path());
        h.tx.send(UiMsg::Start {
            mode: Mode::Batch(dir),
            settings: Box::new(extend_settings(&d.path().join("out"), "mirror", 4.0)),
        })
        .unwrap();
        let evs = until_end(&h);
        assert_eq!(
            evs.last(),
            Some(&WorkerEvent::AllDone { done: 1, failed: 0 }),
            "{evs:?}"
        );
        assert!(evs.iter().any(|e| matches!(
            e,
            WorkerEvent::Preview { which: Which::Before, size: [10, 10], margins, .. }
                if *margins == Margins::default()
        )));
        let four = Margins {
            left: 4,
            top: 4,
            right: 4,
            bottom: 4,
        };
        assert_eq!(after_preview(&evs), Some(([36, 36], four)));
    }

    #[test]
    fn ai_fill_loads_the_inpainter_and_says_so() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 16);
        let h = spawn(inpaint_registry(d.path(), "", true), "cpu");
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input),
            settings: Box::new(extend_settings(d.path(), "ai", 4.0)),
        })
        .unwrap();
        let evs = until_end(&h);
        assert_eq!(
            evs.last(),
            Some(&WorkerEvent::AllDone { done: 1, failed: 0 }),
            "{evs:?}"
        );
        assert!(evs.iter().any(|e| matches!(
            e,
            WorkerEvent::Progress { phase, .. } if phase == "Loading extension model"
        )));
        assert_eq!(after_preview(&evs).map(|(size, _)| size), Some([48, 48]));
    }

    #[test]
    fn ai_fill_without_its_model_is_an_error_not_a_plain_fill() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 16);
        let h = spawn(inpaint_registry(d.path(), &failing_server(), false), "cpu");
        let s = extend_settings(d.path(), "ai", 4.0);
        let output = plan_jobs(&Mode::Single(input.clone()), &s).remove(0).output;
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input),
            settings: Box::new(s),
        })
        .unwrap();
        let evs = until_end(&h);
        assert!(matches!(evs.last(), Some(WorkerEvent::Error(_))), "{evs:?}");
        assert!(!output.exists());
    }

    #[test]
    fn a_downloaded_inpainter_reports_the_download_then_the_load() {
        use sha2::{Digest, Sha256};
        let body = std::fs::read(inpaint_fixture()).unwrap();
        let sha: String = Sha256::digest(&body)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            if let Ok((mut c, _)) = l.accept() {
                let mut buf = [0u8; 1024];
                let _ = c.read(&mut buf);
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = c.write_all(head.as_bytes());
                let _ = c.write_all(&body);
            }
        });
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 16);
        let models = d.path().join("models");
        let entry = inpaint_entry(&format!("http://{addr}/fill.onnx"), sha);
        let h = spawn(Registry::from_entries(vec![entry], models.clone()), "cpu");
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input),
            settings: Box::new(extend_settings(d.path(), "ai", 4.0)),
        })
        .unwrap();
        let evs = until_end(&h);
        assert_eq!(
            evs.last(),
            Some(&WorkerEvent::AllDone { done: 1, failed: 0 }),
            "{evs:?}"
        );
        assert!(models.join("test-fill.onnx").is_file());
        let loading = |e: &WorkerEvent| matches!(e, WorkerEvent::Progress { phase, .. } if phase == "Loading extension model");
        let last_download = evs
            .iter()
            .rposition(|e| matches!(e, WorkerEvent::Download { .. }))
            .expect("the download reported its progress");
        // The status says the model is loading once its download is over.
        assert!(evs[last_download..].iter().any(loading), "{evs:?}");
    }

    /// Answers one request with a long body that arrives a byte every 50 ms.
    fn dripping_server() -> String {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            if let Ok((mut c, _)) = l.accept() {
                let mut buf = [0u8; 1024];
                let _ = c.read(&mut buf);
                let _ = c.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100000\r\n\r\n");
                for _ in 0..400 {
                    if c.write_all(b"x").and_then(|()| c.flush()).is_err() {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        });
        format!("http://{addr}/slow.onnx")
    }

    #[test]
    fn stop_during_the_inpainter_download_stops_the_run() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 16);
        let h = spawn(inpaint_registry(d.path(), &dripping_server(), false), "cpu");
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input),
            settings: Box::new(extend_settings(d.path(), "ai", 4.0)),
        })
        .unwrap();
        loop {
            match h.rx.recv_timeout(Duration::from_secs(60)).unwrap() {
                WorkerEvent::Download { .. } => break,
                WorkerEvent::Error(e) => panic!("{e}"),
                _ => {}
            }
        }
        let stopped_at = std::time::Instant::now();
        h.tx.send(UiMsg::Stop).unwrap();
        let evs = until_end(&h);
        assert_eq!(
            evs.last(),
            Some(&WorkerEvent::Stopped { done: 0, total: 1 }),
            "{evs:?}"
        );
        assert!(stopped_at.elapsed() < Duration::from_secs(3));
        let left: Vec<_> = std::fs::read_dir(d.path().join("models"))
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .collect();
        assert!(left.is_empty(), "{left:?}");
    }

    /// The history steps a run sent, in order.
    fn steps(evs: &[WorkerEvent]) -> Vec<&Step> {
        evs.iter()
            .filter_map(|e| match e {
                WorkerEvent::FileDone { step, .. } => step.as_deref(),
                _ => None,
            })
            .collect()
    }

    fn fixture_path(name: &str) -> String {
        format!("{}/../../tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
    }

    /// A registry whose upscaling model can only be fetched from a server that always fails, so a
    /// run that touches it fails.
    fn unreachable_model_registry(dir: &Path) -> Registry {
        let entry = ModelEntry {
            id: "remote".into(),
            name: "Remote".into(),
            arch: "x".into(),
            scale: 2,
            license: "MIT".into(),
            commercial: true,
            size_bytes: 10,
            url: failing_server(),
            sha256: "00".into(),
            tags: vec![],
            notes: String::new(),
            format: "onnx".into(),
            source: "catalogue".into(),
            kind: Default::default(),
            good_for: String::new(),
            layout: Default::default(),
        };
        Registry::from_entries(vec![entry], dir.join("models"))
    }

    #[test]
    fn extend_only_never_loads_the_upscaling_model() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 10);
        let h = spawn(unreachable_model_registry(d.path()), "cpu");
        let s = Settings {
            model_id: "remote".into(),
            scale: None,
            ..extend_settings(d.path(), "mirror", 4.0)
        };
        let output = plan_jobs(&Mode::Single(input.clone()), &s).remove(0).output;
        assert_eq!(
            output.file_name().unwrap().to_str().unwrap(),
            "a_scaleup_ext_mirror.png"
        );
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input),
            settings: Box::new(s),
        })
        .unwrap();
        let evs = until_end(&h);
        assert_eq!(
            evs.last(),
            Some(&WorkerEvent::AllDone { done: 1, failed: 0 }),
            "{evs:?}"
        );
        let loads_model = |e: &WorkerEvent| match e {
            WorkerEvent::Download { .. } => true,
            WorkerEvent::Progress { phase, .. } => phase == "Loading model",
            _ => false,
        };
        assert!(!evs.iter().any(loads_model), "{evs:?}");
        // The canvas is the final size.
        assert_eq!(image::image_dimensions(&output).unwrap(), (18, 18));
    }

    #[test]
    fn nothing_to_do_is_refused_before_anything_runs() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 8);
        let h = cpu_worker(d.path());
        let s = Settings {
            scale: None,
            ..settings(d.path())
        };
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input),
            settings: Box::new(s),
        })
        .unwrap();
        let evs = until_end(&h);
        assert!(
            matches!(evs.last(), Some(WorkerEvent::Error(e)) if e.contains("choose a scale")),
            "{evs:?}"
        );
    }

    #[test]
    fn a_single_run_sends_a_step_and_a_folder_run_does_not() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 10);
        let h = cpu_worker(d.path());
        let s = extend_settings(d.path(), "mirror", 4.0);
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input.clone()),
            settings: Box::new(s.clone()),
        })
        .unwrap();
        let evs = until_end(&h);
        let [step] = steps(&evs)[..] else {
            panic!("{evs:?}")
        };
        let four = Margins {
            left: 4,
            top: 4,
            right: 4,
            bottom: 4,
        };
        assert_eq!(step.input, input);
        assert_eq!(step.output, plan_jobs(&Mode::Single(input), &s)[0].output);
        assert_eq!(step.settings, s);
        assert_eq!(
            (step.source_size, step.result_size, step.margins),
            ([10, 10], [36, 36], four)
        );
        assert_eq!(step.label, "Extend +4 px · Mirror · 2×");

        let dir = d.path().join("in");
        std::fs::create_dir(&dir).unwrap();
        write_png(&dir.join("b.png"), 10);
        h.tx.send(UiMsg::Start {
            mode: Mode::Batch(dir),
            settings: Box::new(extend_settings(&d.path().join("out"), "mirror", 4.0)),
        })
        .unwrap();
        let evs = until_end(&h);
        assert!(
            evs.iter()
                .any(|e| matches!(e, WorkerEvent::FileDone { step: None, .. })),
            "{evs:?}"
        );
        assert!(steps(&evs).is_empty());
    }

    #[test]
    fn a_failed_run_sends_no_step() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("1.png");
        std::fs::write(&input, b"not an image").unwrap();
        let h = cpu_worker(d.path());
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input),
            settings: Box::new(settings(d.path())),
        })
        .unwrap();
        let evs = until_end(&h);
        assert!(steps(&evs).is_empty(), "{evs:?}");
        assert!(
            !evs.iter()
                .any(|e| matches!(e, WorkerEvent::FileDone { .. }))
        );
    }

    #[test]
    fn try_several_runs_each_fill_and_goes_on_past_a_failed_one() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 16);
        // The upscaling model is unreachable: extend-only fills must not need it.
        let h = spawn(unreachable_model_registry(d.path()), "cpu");
        let s = Settings {
            model_id: "remote".into(),
            scale: None,
            ..extend_settings(d.path(), "ai", 4.0)
        };
        let fills = vec![
            FillChoice::Ai(fixture_path("inpaint_mean_lama.onnx")),
            FillChoice::Plain(Fill::Mirror),
            FillChoice::Ai("no-such-inpainter".into()),
            FillChoice::Ai(fixture_path("inpaint_mean_migan.onnx")),
        ];
        h.tx.send(UiMsg::StartFills {
            mode: Mode::Single(input.clone()),
            settings: Box::new(s),
            fills,
        })
        .unwrap();
        let evs = until_end(&h);
        assert_eq!(
            evs.last(),
            Some(&WorkerEvent::AllDone { done: 3, failed: 1 }),
            "{evs:?}"
        );
        let names: Vec<String> = steps(&evs)
            .iter()
            .map(|s| s.output.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            [
                "a_scaleup_ext_ai-inpaint_mean_lama.png",
                "a_scaleup_ext_mirror.png",
                "a_scaleup_ext_ai-inpaint_mean_migan.png",
            ]
        );
        for step in steps(&evs) {
            assert!(step.output.is_file());
            assert_eq!(step.result_size, [24, 24]);
            assert_eq!(step.input, input);
        }
        assert!(
            evs.iter().any(|e| matches!(
                e,
                WorkerEvent::FileFailed { input: i, error } if *i == input && error.contains("no-such-inpainter")
            )),
            "{evs:?}"
        );
        // Each fill says where it is in the queue, with its phase.
        for n in [1, 2, 4] {
            let prefix = format!("Fill {n} of 4: ");
            assert!(
                evs.iter().any(|e| matches!(
                    e,
                    WorkerEvent::Progress { phase, files: 1, .. } if phase.starts_with(&prefix) && phase.len() > prefix.len()
                )),
                "{prefix} {evs:?}"
            );
        }
    }

    #[test]
    fn try_several_refuses_a_folder_and_an_empty_list() {
        let d = tempfile::tempdir().unwrap();
        let h = cpu_worker(d.path());
        let s = extend_settings(d.path(), "mirror", 4.0);
        h.tx.send(UiMsg::StartFills {
            mode: Mode::Batch(d.path().to_path_buf()),
            settings: Box::new(s.clone()),
            fills: vec![FillChoice::Plain(Fill::Edge)],
        })
        .unwrap();
        let evs = until_end(&h);
        assert!(
            matches!(evs.last(), Some(WorkerEvent::Error(e)) if e.contains("single image")),
            "{evs:?}"
        );
        h.tx.send(UiMsg::StartFills {
            mode: Mode::Single(d.path().join("a.png")),
            settings: Box::new(s),
            fills: vec![],
        })
        .unwrap();
        let evs = until_end(&h);
        assert!(matches!(evs.last(), Some(WorkerEvent::Error(_))), "{evs:?}");
    }

    #[test]
    fn try_several_runs_a_repeated_fill_to_a_new_name() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 10);
        let h = cpu_worker(d.path());
        let s = extend_settings(d.path(), "mirror", 2.0);
        h.tx.send(UiMsg::StartFills {
            mode: Mode::Single(input),
            settings: Box::new(s),
            fills: vec![
                FillChoice::Plain(Fill::Mirror),
                FillChoice::Plain(Fill::Edge),
                FillChoice::Plain(Fill::Mirror),
            ],
        })
        .unwrap();
        let evs = until_end(&h);
        assert_eq!(
            evs.last(),
            Some(&WorkerEvent::AllDone { done: 3, failed: 0 }),
            "{evs:?}"
        );
        let names: Vec<String> = steps(&evs)
            .iter()
            .map(|s| s.output.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            [
                "a_scaleup_ext_mirror_2x_nearest2x.png",
                "a_scaleup_ext_edge_2x_nearest2x.png",
                "a_scaleup_ext_mirror_2x_nearest2x-2.png",
            ]
        );
    }

    #[test]
    fn a_fill_that_fails_while_running_is_named() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        std::fs::write(&input, b"not an image").unwrap();
        let h = cpu_worker(d.path());
        let s = Settings {
            scale: None,
            ..extend_settings(d.path(), "mirror", 2.0)
        };
        h.tx.send(UiMsg::StartFills {
            mode: Mode::Single(input.clone()),
            settings: Box::new(s),
            fills: vec![
                FillChoice::Plain(Fill::Mirror),
                FillChoice::Plain(Fill::Edge),
            ],
        })
        .unwrap();
        let evs = until_end(&h);
        assert_eq!(
            evs.last(),
            Some(&WorkerEvent::AllDone { done: 0, failed: 2 }),
            "{evs:?}"
        );
        let errors: Vec<&str> = evs
            .iter()
            .filter_map(|e| match e {
                WorkerEvent::FileFailed { input: i, error } if *i == input => Some(error.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(errors.len(), 2, "{evs:?}");
        assert!(errors[0].starts_with("Mirror: "), "{}", errors[0]);
        assert!(errors[1].starts_with("Edge pixels: "), "{}", errors[1]);
    }

    #[test]
    fn an_upscaling_model_that_cannot_load_ends_the_queue_once() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 8);
        let h = spawn(unreachable_model_registry(d.path()), "cpu");
        let s = Settings {
            model_id: "remote".into(),
            ..extend_settings(d.path(), "mirror", 2.0)
        };
        h.tx.send(UiMsg::StartFills {
            mode: Mode::Single(input),
            settings: Box::new(s),
            fills: vec![
                FillChoice::Plain(Fill::Mirror),
                FillChoice::Plain(Fill::Edge),
            ],
        })
        .unwrap();
        let evs = until_end(&h);
        assert!(matches!(evs.last(), Some(WorkerEvent::Error(_))), "{evs:?}");
        assert!(
            !evs.iter()
                .any(|e| matches!(e, WorkerEvent::FileFailed { .. })),
            "{evs:?}"
        );
        // The second fill never started.
        assert!(
            !evs.iter().any(
                |e| matches!(e, WorkerEvent::Progress { phase, .. } if phase.starts_with("Fill 2"))
            ),
            "{evs:?}"
        );
    }

    #[test]
    fn stop_ends_the_fill_queue() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 16);
        // The second fill's inpainting model downloads a byte every 50 ms: the queue is held
        // open there until Stop.
        let h = spawn(inpaint_registry(d.path(), &dripping_server(), false), "cpu");
        let s = Settings {
            scale: None,
            ..extend_settings(d.path(), "mirror", 4.0)
        };
        let fills = vec![
            FillChoice::Plain(Fill::Mirror),
            FillChoice::Ai("test-fill".into()),
            FillChoice::Plain(Fill::Edge),
        ];
        h.tx.send(UiMsg::StartFills {
            mode: Mode::Single(input),
            settings: Box::new(s),
            fills,
        })
        .unwrap();
        let mut evs = Vec::new();
        loop {
            let ev = h.rx.recv_timeout(Duration::from_secs(120)).unwrap();
            if let WorkerEvent::Error(e) = &ev {
                panic!("{e}");
            }
            let downloading = matches!(ev, WorkerEvent::Download { .. });
            evs.push(ev);
            if downloading {
                break;
            }
        }
        h.tx.send(UiMsg::Stop).unwrap();
        evs.extend(until_end(&h));
        assert_eq!(
            evs.last(),
            Some(&WorkerEvent::Stopped { done: 1, total: 3 }),
            "{evs:?}"
        );
        // One step per finished fill; the interrupted one and the one after it add none.
        let made = steps(&evs);
        assert_eq!(made.len(), 1, "{evs:?}");
        assert_eq!(
            made[0].output.file_name().unwrap(),
            "a_scaleup_ext_mirror.png"
        );
    }

    #[test]
    fn show_step_sends_both_previews_with_the_steps_margins() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        let output = d.path().join("b.png");
        write_png(&input, 10);
        write_png(&output, 30);
        let margins = Margins {
            left: 5,
            top: 0,
            right: 5,
            bottom: 0,
        };
        let step = Step {
            input: input.clone(),
            output: output.clone(),
            settings: Settings::default(),
            source_size: [10, 10],
            result_size: [30, 30],
            margins,
            label: String::new(),
        };
        let msg = UiMsg::show_step(&step);
        assert!(matches!(
            &msg,
            UiMsg::ShowStep { input: i, output: o, margins: m } if *i == input && *o == output && *m == margins
        ));
        let h = cpu_worker(d.path());
        h.tx.send(msg).unwrap();
        let mut previews = Vec::new();
        while previews.len() < 2 {
            match h.rx.recv_timeout(Duration::from_secs(120)).unwrap() {
                WorkerEvent::Preview {
                    which,
                    size,
                    margins,
                    ..
                } => previews.push((which, size, margins)),
                WorkerEvent::Error(e) => panic!("{e}"),
                _ => {}
            }
        }
        assert_eq!(
            previews,
            [
                (Which::Before, [10, 10], Margins::default()),
                (Which::After, [30, 30], margins)
            ]
        );

        std::fs::remove_file(&output).unwrap();
        h.tx.send(UiMsg::show_step(&step)).unwrap();
        let evs = until_end(&h);
        assert!(
            matches!(evs.last(), Some(WorkerEvent::Error(e)) if e.contains("b.png") && e.contains("no longer exists")),
            "{evs:?}"
        );
    }

    #[test]
    fn failed_single_file_reports_filefailed_not_terminal_error() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("1.png");
        std::fs::write(&input, b"not an image").unwrap();
        let h = cpu_worker(d.path());
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input.clone()),
            settings: Box::new(settings(d.path())),
        })
        .unwrap();
        let evs = until_end(&h);
        assert!(
            evs.iter()
                .any(|e| matches!(e, WorkerEvent::FileFailed { input: i, .. } if *i == input)),
            "{evs:?}"
        );
        assert_eq!(
            evs.last(),
            Some(&WorkerEvent::AllDone { done: 0, failed: 1 })
        );
    }
}
