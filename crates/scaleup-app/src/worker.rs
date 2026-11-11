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
    CancelToken, DeviceInfo, DevicePreference, FileJob, Model, OutputFormat, Phase, Progress,
    Registry, Result, ScaleUpError, Upscaler, list_devices, load_image,
};

use crate::state::{Mode, Settings};
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
        settings: Settings,
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
    FileDone(PathBuf),
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
        image: egui::ColorImage,
        /// Size of the file, which `image` may be a downscaled copy of.
        size: [u32; 2],
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
    let label = model_label(&s.model_id);
    inputs
        .into_iter()
        .map(|input| FileJob {
            output: Upscaler::output_name(&input, &s.output_dir_for(&input), &label, &opts, format),
            input,
            format,
            quality: s.quality,
            overwrite: s.overwrite,
            double: s.double,
            extend: None,
        })
        .collect()
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
fn preview_event(path: &Path, which: Which) -> Result<WorkerEvent> {
    let img = load_image(path)?;
    let size = [img.width(), img.height()];
    let img = cap_long_side(img, MAX_PREVIEW_SIDE);
    let dims = [img.width() as usize, img.height() as usize];
    Ok(WorkerEvent::Preview {
        which,
        image: egui::ColorImage::from_rgba_unmultiplied(dims, img.as_raw()),
        size,
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
            self.upscaler = None;
            self.emit(WorkerEvent::Error(format!(
                "processing stopped unexpectedly: {detail}"
            )));
        }
    }

    fn handle(&mut self, msg: UiMsg) {
        match msg {
            UiMsg::Start { mode, settings } => self.start(&mode, &settings),
            // Nothing is running, so there is nothing to stop.
            UiMsg::Stop => {}
            UiMsg::ChangeDevice(gpu) => self.change_device(gpu),
            UiMsg::Preview(path) => match preview_event(&path, Which::Before) {
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
    fn emit_preview(&self, path: &Path, which: Which) {
        match preview_event(path, which) {
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

    fn start(&mut self, mode: &Mode, settings: &Settings) {
        if settings.gpu != self.gpu {
            self.change_device(settings.gpu.clone());
        }
        let jobs = plan_jobs(mode, settings);
        if jobs.is_empty() {
            self.emit(WorkerEvent::Error("no supported images found".into()));
            return;
        }
        let cancel = CancelToken::new();
        let opts = settings.to_options();
        let files = jobs.len();
        let (mut done, mut failed) = (0, 0);
        for (index, job) in jobs.iter().enumerate() {
            self.inbox.poll(&cancel);
            if cancel.is_cancelled() {
                self.emit(WorkerEvent::Stopped { done, total: files });
                return;
            }
            if job.output.exists() && !job.overwrite {
                self.emit(WorkerEvent::Log(format!(
                    "{} exists, not overwriting",
                    job.output.display()
                )));
                self.emit(WorkerEvent::FileDone(job.output.clone()));
                self.emit_preview(&job.output, Which::After);
                done += 1;
                continue;
            }
            self.emit(WorkerEvent::Progress {
                file: index,
                files,
                percent: 0.0,
                phase: "Loading model".into(),
            });
            match self.ensure_model(&settings.model_id, &cancel) {
                Ok(()) => {}
                Err(ScaleUpError::Cancelled) => {
                    self.emit(WorkerEvent::Stopped { done, total: files });
                    return;
                }
                // Every remaining file would fail the same way.
                Err(e) => {
                    self.emit(WorkerEvent::Error(e.to_string()));
                    return;
                }
            }
            if matches!(mode, Mode::Batch(_)) {
                self.emit_preview(&job.input, Which::Before);
            }
            let started = std::time::Instant::now();
            match self.run_job(index, files, job, &opts, &cancel) {
                Ok(path) => {
                    self.emit(WorkerEvent::Log(format!(
                        "{} in {:.1} s",
                        path.display(),
                        started.elapsed().as_secs_f32()
                    )));
                    self.emit(WorkerEvent::FileDone(path.clone()));
                    self.emit_preview(&path, Which::After);
                    done += 1;
                }
                Err(ScaleUpError::Cancelled) => {
                    self.emit(WorkerEvent::Stopped { done, total: files });
                    return;
                }
                Err(e) => {
                    self.emit(WorkerEvent::FileFailed {
                        input: job.input.clone(),
                        error: e.to_string(),
                    });
                    failed += 1;
                }
            }
        }
        self.emit(WorkerEvent::AllDone { done, failed });
    }

    fn run_job(
        &mut self,
        file: usize,
        files: usize,
        job: &FileJob,
        opts: &scaleup_core::UpscaleOptions,
        cancel: &CancelToken,
    ) -> Result<PathBuf> {
        let upscaler = self.upscaler.as_ref().expect("model is loaded");
        let (_, model) = self.model.as_ref().expect("model is loaded");
        let (events, inbox) = (&self.events, &mut self.inbox);
        let mut phase = phase_name(Phase::Loading);
        let mut percent = 0.0;
        upscaler.run_file(
            Some(model.as_ref()),
            None,
            job,
            opts,
            &mut |p| {
                match p {
                    Progress::Phase(ph) => phase = phase_name(ph),
                    Progress::Tiles { done, total } => {
                        percent = done as f32 * 100.0 / total.max(1) as f32;
                    }
                }
                events.send(WorkerEvent::Progress {
                    file,
                    files,
                    percent,
                    phase: phase.into(),
                });
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
            scale: 2,
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
            settings: settings(d.path()),
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
        h.tx.send(UiMsg::Preview(input)).unwrap();
        loop {
            match h.rx.recv_timeout(Duration::from_secs(120)).unwrap() {
                WorkerEvent::Preview { which, image, size } => {
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
            settings: settings(d.path()),
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
            settings: settings(d.path()),
        })
        .unwrap();
        let evs = until_end(&h);
        let pos = |f: fn(&WorkerEvent) -> bool| evs.iter().position(f);
        let progress = pos(|e| matches!(e, WorkerEvent::Progress { .. })).expect("progress");
        let file_done = pos(|e| matches!(e, WorkerEvent::FileDone(_))).expect("file done");
        let all_done =
            pos(|e| matches!(e, WorkerEvent::AllDone { done: 1, failed: 0 })).expect("all done");
        assert!(progress < file_done && file_done < all_done, "{evs:?}");
        let Some(WorkerEvent::FileDone(p)) = evs.get(file_done) else {
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
            settings: settings(&out),
        })
        .unwrap();
        loop {
            match h.rx.recv_timeout(Duration::from_secs(120)).unwrap() {
                WorkerEvent::FileDone(_) => break,
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
    fn existing_output_without_overwrite_emits_filedone_without_progress() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("a.png");
        write_png(&input, 8);
        let s = settings(d.path());
        let existing = plan_jobs(&Mode::Single(input.clone()), &s).remove(0).output;
        std::fs::write(&existing, b"keep me").unwrap();
        let h = cpu_worker(d.path());
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input),
            settings: s,
        })
        .unwrap();
        let evs = until_end(&h);
        assert!(
            !evs.iter()
                .any(|e| matches!(e, WorkerEvent::Progress { .. })),
            "{evs:?}"
        );
        assert!(evs.contains(&WorkerEvent::FileDone(existing.clone())));
        assert_eq!(std::fs::read(&existing).unwrap(), b"keep me");
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
        };
        let reg = Registry::from_entries(vec![entry], d.path().join("models"));
        let h = spawn(reg, "cpu");
        let mut s = settings(d.path());
        s.model_id = "remote".into();
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input.clone()),
            settings: s.clone(),
        })
        .unwrap();
        let evs = until_end(&h);
        assert!(matches!(evs.last(), Some(WorkerEvent::Error(_))), "{evs:?}");

        s.model_id = fixture();
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input),
            settings: s,
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
            settings: settings(&d.path().join("out")),
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
            settings: s,
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

    #[test]
    fn failed_single_file_reports_filefailed_not_terminal_error() {
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("1.png");
        std::fs::write(&input, b"not an image").unwrap();
        let h = cpu_worker(d.path());
        h.tx.send(UiMsg::Start {
            mode: Mode::Single(input.clone()),
            settings: settings(d.path()),
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
