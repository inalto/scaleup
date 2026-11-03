//! The one background thread: owns the [`Upscaler`] and the loaded model. The UI talks to it over
//! two channels and never blocks.
//!
//! While a run is in progress the thread is busy, so it reads its inbox from the progress callbacks
//! and between files: that is where `Stop` takes effect.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;

use eframe::egui;
use scaleup_core::io::is_supported_input;
use scaleup_core::{
    CancelToken, DeviceInfo, DevicePreference, FileJob, Model, OutputFormat, Phase, Progress,
    Registry, Result, ScaleUpError, Upscaler, list_devices, load_image,
};

use crate::state::{Mode, Settings};

pub enum UiMsg {
    Start {
        mode: Mode,
        settings: Settings,
    },
    Stop,
    ChangeDevice(String),
    /// Fetch a catalogue model without running anything.
    Download(String),
    /// Decode this file as the new "before" image.
    Preview(PathBuf),
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
}

impl Worker {
    fn emit(&self, ev: WorkerEvent) {
        self.events.send(ev);
    }

    fn run(&mut self) {
        self.emit(WorkerEvent::Devices(list_devices()));
        if let Err(e) = self.ensure_upscaler() {
            self.emit(WorkerEvent::Error(e.to_string()));
        }
        while let Some(msg) = self.inbox.next() {
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
                UiMsg::Quit => return,
            }
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
        self.emit(WorkerEvent::Log(format!(
            "model {model_id} ({}x) on {}, automatic tile size {}",
            model.scale(),
            upscaler.device().name,
            upscaler.effective_tile(0)
        )));
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
            model.as_ref(),
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
