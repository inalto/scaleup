//! The eframe application shell: routes input to state, state to the worker, events back to state.

use std::path::{Path, PathBuf};

use eframe::egui;
use scaleup_core::Registry;

use crate::paste;
use crate::state::Mode;
use crate::state::{AppState, Settings, Status, Step, apply_event, mode_from_path};
use crate::ui::compare::{CompareView, StageInput};
use crate::ui::history::{self, HistoryAction, HistoryStrip};
use crate::ui::hub_dialog::{HubAction, HubDialog};
use crate::ui::models_dialog::{DialogAction, ModelsDialog};
use crate::ui::toast::{Toast, Toasts};
use crate::ui::{canvas_view, sidebar, statusbar, theme};
use crate::worker::{self, UiMsg, Which, WorkerEvent, WorkerHandle};

const SETTINGS_KEY: &str = "settings";
const SIDEBAR_WIDTH: f32 = 320.0;
/// Toasts sit this far above the window's bottom edge when no history strip shows.
const TOAST_GAP: f32 = 52.0;

/// A history step asked of the worker, until its "after" preview arrives. Its "before" is held
/// back until then, so the image area goes from the old pair to the new pair in one frame (a
/// "before" alone would flash the edit stage).
struct PendingStep {
    input: PathBuf,
    output: PathBuf,
    before: Option<(egui::ColorImage, [u32; 2])>,
}

pub struct App {
    pub state: AppState,
    registry: Registry,
    worker: WorkerHandle,
    toasts: Toasts,
    log_open: bool,
    compare: CompareView,
    models_dialog: ModelsDialog,
    hub_dialog: HubDialog,
    strip: HistoryStrip,
    /// The step whose previews are on their way.
    showing: Option<PendingStep>,
    /// Why the current step cannot be shown (a file of it is gone), for the empty image area.
    step_error: Option<String>,
    /// The outputs of the steps the worker was asked to show, for the tests to check.
    #[cfg(test)]
    asked: Vec<PathBuf>,
    /// Frames the image area asked to draw again after changing the settings (tests).
    #[cfg(test)]
    discards: usize,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, initial: Option<PathBuf>) -> Self {
        let settings = cc
            .storage
            .and_then(|s| s.get_string(SETTINGS_KEY))
            .and_then(|json| serde_json::from_str::<Settings>(&json).ok())
            .unwrap_or_default();
        theme::apply(&cc.egui_ctx);
        let registry = Registry::builtin();
        let worker = worker::spawn(registry.clone(), &settings.gpu);
        worker.wake_on_events(cc.egui_ctx.clone());
        let mut app = Self {
            state: AppState {
                settings,
                ..AppState::default()
            },
            registry,
            worker,
            toasts: Toasts::default(),
            log_open: false,
            compare: CompareView::default(),
            models_dialog: ModelsDialog::default(),
            hub_dialog: HubDialog::default(),
            strip: HistoryStrip::default(),
            showing: None,
            step_error: None,
            #[cfg(test)]
            asked: Vec::new(),
            #[cfg(test)]
            discards: 0,
        };
        if let Some(path) = initial {
            app.select(&path);
        }
        app
    }

    fn send(&self, msg: UiMsg) {
        // The worker only goes away with the app.
        let _ = self.worker.tx.send(msg);
    }

    fn select(&mut self, path: &Path) {
        if self.state.status.is_running() {
            return;
        }
        match mode_from_path(path) {
            Some(mode) => {
                self.compare.clear();
                self.showing = None;
                self.step_error = None;
                self.state.source_size = None;
                if let Mode::Single(p) = &mode {
                    self.send(UiMsg::Preview(p.clone()));
                }
                self.state.mode = Some(mode);
                self.state.status = Status::Idle;
            }
            None => self.toasts.push(Toast::error(format!(
                "{} is not a supported image or folder",
                path.display()
            ))),
        }
    }

    fn start(&mut self) {
        let Some(mode) = self.state.mode.clone() else {
            return;
        };
        if self.state.status.is_running() {
            return;
        }
        // The sidebar disables Start then; the keyboard shortcut lands here.
        if let Some(why) = self.state.settings.start_blocker(self.state.source_size) {
            self.toasts.push(Toast::error(why));
            return;
        }
        self.state.status = Status::Running {
            file: 0,
            files: 1,
            percent: 0.0,
            phase: "Starting".into(),
        };
        // Previews of a step still on their way now show as they come: the run's own follow them.
        self.showing = None;
        self.send(UiMsg::Start {
            mode,
            settings: Box::new(self.state.settings.clone()),
        });
    }

    fn drain_events(&mut self, ctx: &egui::Context) {
        while let Ok(ev) = self.worker.rx.try_recv() {
            self.handle_event(ctx, ev);
        }
    }

    fn handle_event(&mut self, ctx: &egui::Context, ev: WorkerEvent) {
        match ev {
            WorkerEvent::Preview {
                which,
                path,
                image,
                size,
                margins,
            } => {
                if which == Which::After {
                    self.strip
                        .note_result(ctx, &self.state.history, &path, &image);
                }
                if let Some(pending) = &mut self.showing {
                    match which {
                        Which::Before if path == pending.input => {
                            pending.before = Some((image, size));
                        }
                        Which::After if path == pending.output => {
                            let before = self.showing.take().and_then(|p| p.before);
                            if let Some((image, size)) = before {
                                let none = Default::default();
                                self.show_preview(ctx, Which::Before, image, size, none);
                            }
                            self.show_preview(ctx, Which::After, image, size, margins);
                        }
                        // A preview of a step that was left before it arrived.
                        _ => {}
                    }
                    return;
                }
                self.show_preview(ctx, which, image, size, margins);
                return;
            }
            // The step being shown has a file that is gone or unreadable: say so in the image
            // area too, where its picture would be. The step stays in the history.
            WorkerEvent::Error(ref message)
                if self.showing.as_ref().is_some_and(|p| {
                    [&p.input, &p.output]
                        .iter()
                        .any(|f| message.starts_with(&f.display().to_string()))
                }) =>
            {
                self.showing = None;
                self.compare.clear();
                self.step_error = Some(message.clone());
            }
            WorkerEvent::HubCatalog(cat) => {
                self.hub_dialog.set_catalog(cat);
                return;
            }
            WorkerEvent::HubThumb { id, img } => {
                self.hub_dialog.set_thumbs(ctx, id, img);
                return;
            }
            WorkerEvent::HubInstalled(ref m) => {
                self.hub_dialog.on_installed(m.clone());
                self.reload_registry();
            }
            WorkerEvent::HubRemoved(ref id) => {
                self.hub_dialog.on_removed(id);
                self.reload_registry();
            }
            WorkerEvent::HubCatalogFailed(_) => self.hub_dialog.on_catalog_failed(),
            _ => {}
        }
        if let Some(toast) = apply_event(&mut self.state, ev) {
            self.toasts.push(toast);
        }
    }

    /// Puts a decoded preview in the image area. A result shows as a result: the image area stays
    /// on it until the extension is changed.
    fn show_preview(
        &mut self,
        ctx: &egui::Context,
        which: Which,
        image: egui::ColorImage,
        size: [u32; 2],
        margins: scaleup_core::Margins,
    ) {
        self.state.note_preview(which, size);
        self.compare.set(ctx, which, image, size, margins);
        if which == Which::After {
            self.compare.result_settings(&self.state.settings);
        }
    }

    /// Does what the history strip or a history shortcut asked. Nothing while a run is going.
    fn history_action(&mut self, action: HistoryAction) {
        if self.state.status.is_running() {
            return;
        }
        let history = &mut self.state.history;
        let Some(at) = history.at() else {
            return;
        };
        match action {
            HistoryAction::Back | HistoryAction::Forward | HistoryAction::Go(_) => {
                let step = match action {
                    HistoryAction::Back => history.back(),
                    HistoryAction::Forward => history.forward(),
                    HistoryAction::Go(i) => history.go(i),
                    _ => None,
                };
                if let Some(step) = step.cloned() {
                    self.show_step(&step);
                }
            }
            HistoryAction::UseAsInput => {
                if let Some(path) = history.use_as_input(at).map(Path::to_path_buf) {
                    self.select(&path);
                }
            }
            HistoryAction::RestoreSettings => {
                let Some(settings) = self.state.restore_settings(at) else {
                    return;
                };
                let device = settings.gpu != self.state.settings.gpu;
                self.state.settings = settings;
                if device {
                    self.send(UiMsg::ChangeDevice(self.state.settings.gpu.clone()));
                }
                // The result on screen stays a result: restoring is not editing the extension.
                self.compare.result_settings(&self.state.settings);
                self.toasts.push(Toast::success(format!(
                    "Settings of step {} restored",
                    at + 1
                )));
            }
            HistoryAction::OpenFolder => {
                if let Some(step) = history.current() {
                    let output = step.output.clone();
                    self.open_folder_of(&output);
                }
            }
            HistoryAction::Remove => {
                history.remove(at);
                match history.current().cloned() {
                    Some(step) => self.show_step(&step),
                    // The last step went: back to the selected image on its own.
                    None => {
                        if let Some(Mode::Single(path)) = self.state.mode.clone() {
                            self.select(&path);
                        }
                    }
                }
                self.toasts.push(Toast::success(
                    "Removed from the history. The file is kept.",
                ));
            }
        }
    }

    /// Shows the current history step: its input becomes the selected image (so the sidebar, the
    /// size line, the edit stage and the next run all work on the picture shown as "before"), and
    /// the worker is asked for both previews with the step's margins.
    fn show_step(&mut self, step: &Step) {
        self.state.mode = Some(Mode::Single(step.input.clone()));
        self.state.source_size = Some(step.source_size);
        if matches!(self.state.status, Status::Error(_)) {
            self.state.status = Status::Idle;
        }
        self.step_error = None;
        self.showing = Some(PendingStep {
            input: step.input.clone(),
            output: step.output.clone(),
            before: None,
        });
        #[cfg(test)]
        self.asked.push(step.output.clone());
        self.send(UiMsg::show_step(step));
    }

    /// The registry reads `installed.json` when it is built: build it again after an install or
    /// a removal, so the model list shows the change.
    fn reload_registry(&mut self) {
        self.registry = Registry::with_dir(self.registry.models_dir().to_path_buf());
    }

    fn hub_action(&mut self, action: HubAction) {
        let busy = |phase: &str| Status::Running {
            file: 0,
            files: 1,
            percent: 0.0,
            phase: phase.into(),
        };
        match action {
            HubAction::Refresh { force } => self.send(UiMsg::HubRefresh { force }),
            HubAction::Install(id) => {
                self.state.status = Status::Downloading { done: 0, total: 0 };
                self.send(UiMsg::HubInstall(id));
            }
            HubAction::Convert(id) => {
                self.state.status = busy("Preparing the conversion");
                self.send(UiMsg::HubConvert(id));
            }
            HubAction::Import { id, file } => {
                self.state.status = busy("Checking the model file");
                self.send(UiMsg::HubImport { id, file });
            }
            HubAction::Remove(id) => self.send(UiMsg::HubRemove(id)),
            HubAction::Thumbs(id) => self.send(UiMsg::HubThumbs(id)),
        }
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        use egui::{Key, Modifiers};
        let (open, start, stop, paste) = ctx.input_mut(|i| {
            (
                i.consume_key(Modifiers::COMMAND, Key::O),
                i.consume_key(Modifiers::COMMAND, Key::Enter),
                i.consume_key(Modifiers::NONE, Key::Escape),
                i.events.iter().any(paste::is_paste_key_release),
            )
        });
        if open && let Some(p) = sidebar::pick_image() {
            self.select(&p);
        }
        if start {
            self.start();
        }
        if stop && self.state.status.is_running() {
            self.state.request_stop();
            self.send(UiMsg::Stop);
        }
        if paste && !self.state.status.is_running() {
            match paste::paste_from_clipboard() {
                Some(Ok(path)) => self.select(&path),
                Some(Err(e)) => self.toasts.push(Toast::error(e)),
                // Text or nothing on the clipboard: an ordinary paste, not for us.
                None => {}
            }
        }
        if self.history_keys_apply(ctx) {
            let nav = ctx.input_mut(|i| {
                let mut nav = None;
                i.events.retain(|e| match e {
                    egui::Event::Key {
                        key,
                        pressed: true,
                        modifiers,
                        ..
                    } => match history::nav_for(*modifiers, *key) {
                        Some(n) => {
                            nav = Some(n);
                            false
                        }
                        None => true,
                    },
                    _ => true,
                });
                nav
            });
            if let Some(nav) = nav {
                self.history_action(nav.into());
            }
        }
    }

    /// The history keys work when there is a history, nothing runs, and the keys do not belong to
    /// something else: a text field keeps its own undo, a focused canvas handle its arrows.
    fn history_keys_apply(&self, ctx: &egui::Context) -> bool {
        !self.state.history.steps().is_empty()
            && !self.state.status.is_running()
            && !ctx.text_edit_focused()
            && !canvas_view::handle_focused(ctx)
    }

    fn open_output_folder(&mut self) {
        let Status::Done(path) = &self.state.status else {
            return;
        };
        let path = path.clone();
        self.open_folder_of(&path);
    }

    /// Opens the folder `path` is in, in the system's file manager.
    fn open_folder_of(&mut self, path: &Path) {
        let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        if let Err(e) = open::that(&dir) {
            self.toasts
                .push(Toast::error(format!("Cannot open {}: {e}", dir.display())));
        }
    }
}

/// The two lines in the middle of the image area while there is no picture to show.
fn empty_state(ui: &mut egui::Ui, title: &str, hint: &str) {
    let width = ui.available_width().min(460.0);
    ui.vertical_centered(|ui| {
        ui.set_max_width(width);
        // Two lines of text, placed a little above the middle.
        ui.add_space((ui.available_height() * 0.42 - 24.0).max(0.0));
        ui.label(egui::RichText::new(title).size(18.0));
        ui.add_space(2.0);
        ui.weak(hint);
    });
}

/// The frame and caption shown while a file is held over the window.
fn drop_target(ui: &egui::Ui, area: egui::Rect) {
    let painter = ui.painter_at(area);
    let frame = area.shrink(8.0);
    painter.rect_filled(frame, 8.0, theme::ACCENT.gamma_multiply(0.25));
    painter.rect_stroke(
        frame,
        8.0,
        egui::Stroke::new(2.0, theme::ACCENT),
        egui::StrokeKind::Inside,
    );
    let pos = frame.center();
    let font = egui::FontId::proportional(18.0);
    let text = "Drop to open";
    let galley = painter.layout_no_wrap(text.into(), font, egui::Color32::WHITE);
    let rect = egui::Rect::from_center_size(pos, galley.size()).expand2(egui::vec2(14.0, 8.0));
    painter.rect_filled(rect, 6.0, theme::ACCENT);
    painter.galley(
        rect.shrink2(egui::vec2(14.0, 8.0)).min,
        galley,
        egui::Color32::WHITE,
    );
}

impl App {
    /// What the empty image area says: the state, then what to do about it.
    fn empty_text(&self, ctx: &egui::Context) -> (String, String) {
        let key =
            |k| ctx.format_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, k));
        if let Some(why) = &self.step_error {
            return (
                "This step cannot be shown".into(),
                format!("{why}.\nThe step stays in the history: go to another step, or remove it."),
            );
        }
        match &self.state.mode {
            Some(Mode::Batch(p)) => (
                "Folder selected".into(),
                format!("{}\nEvery image in it will be processed.", p.display()),
            ),
            Some(Mode::Single(_)) if matches!(self.state.status, Status::Error(_)) => (
                "This image could not be opened".into(),
                "Choose another image, or see the status line below for the reason.".into(),
            ),
            Some(Mode::Single(_)) => ("Loading preview…".into(), String::new()),
            None => (
                "Drop an image or a folder here".into(),
                format!(
                    "or choose one with Select Image ({}), or paste an image ({})",
                    key(egui::Key::O),
                    key(egui::Key::V)
                ),
            ),
        }
    }

    /// One frame of the whole window.
    fn frame(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.drain_events(&ctx);
        self.handle_shortcuts(&ctx);

        let dropped = ctx.input(|i| i.raw.dropped_files.first().map(|f| f.path().to_path_buf()));
        if let Some(path) = dropped {
            self.select(&path);
        }
        if self.state.status.is_running() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        let busy_now = self.state.status.is_running();
        let mut open_folder = false;
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.add_space(4.0);
            open_folder = statusbar::show(ui, &self.state, &mut self.log_open);
            ui.add_space(4.0);
        });
        if open_folder {
            self.open_output_folder();
        }

        if self.log_open {
            egui::Panel::bottom("log")
                .resizable(true)
                .default_size(140.0)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .auto_shrink(false)
                        .show(ui, |ui| {
                            if self.state.log.is_empty() {
                                ui.weak("Nothing logged yet. Messages from a run appear here.");
                            }
                            for line in &self.state.log {
                                ui.monospace(line);
                            }
                        });
                });
        }

        let side = egui::Panel::left("sidebar")
            .exact_size(SIDEBAR_WIDTH)
            .resizable(false)
            .show(ui, |ui| sidebar::show(ui, &mut self.state, &self.registry))
            .inner;
        if let Some(p) = side.picked {
            self.select(&p);
        }
        if side.device_changed {
            self.send(UiMsg::ChangeDevice(self.state.settings.gpu.clone()));
        }
        if side.open_models {
            self.models_dialog.open = true;
        }
        if side.open_hub {
            self.hub_dialog.open_with(self.registry.models_dir());
        }
        if side.start {
            self.start();
        }
        if side.stop {
            self.state.request_stop();
            self.send(UiMsg::Stop);
        }

        // The history strip: under the image area, right of the sidebar, once there is a step.
        let mut toast_gap = TOAST_GAP;
        self.strip.forget_dropped(&self.state.history);
        if !self.state.history.steps().is_empty() {
            let running = self.state.status.is_running();
            let strip = egui::Panel::bottom("history")
                .exact_size(history::STRIP_HEIGHT)
                .resizable(false)
                .show(ui, |ui| self.strip.show(ui, &self.state.history, running));
            toast_gap = ctx.content_rect().bottom() - strip.response.rect.top() + 12.0;
            if let Some(action) = strip.inner {
                self.history_action(action);
            }
        }

        let stage = egui::Frame::central_panel(ui.style()).fill(theme::stage_fill(ui.visuals()));
        egui::CentralPanel::default().frame(stage).show(ui, |ui| {
            let area = ui.max_rect();
            if self.compare.is_empty() {
                let (title, hint) = self.empty_text(&ctx);
                empty_state(ui, &title, &hint);
            } else {
                let single = matches!(self.state.mode, Some(Mode::Single(_)));
                let changed = self.compare.show(
                    ui,
                    StageInput {
                        settings: &mut self.state.settings,
                        registry: &self.registry,
                        single,
                        running: self.state.status.is_running(),
                    },
                );
                // The sidebar was drawn before the image area: draw the frame again, so its fields
                // show what a handle just did in the same frame.
                if changed {
                    #[cfg(test)]
                    {
                        self.discards += 1;
                    }
                    ctx.request_discard("the image area changed the extend settings");
                    ctx.request_repaint();
                }
            }
            // A file held over the window: say that letting go opens it.
            if !busy_now && ctx.input(|i| !i.raw.hovered_files.is_empty()) {
                drop_target(ui, area);
            }
        });

        let busy = self.state.status.is_running();
        match self.models_dialog.show(&ctx, &self.registry, busy) {
            Some(DialogAction::Download(id)) => {
                self.state.status = Status::Downloading { done: 0, total: 0 };
                self.send(UiMsg::Download(id));
            }
            Some(DialogAction::RemoveHub(id)) => self.send(UiMsg::HubRemove(id)),
            Some(DialogAction::Delete(id)) => {
                if let Err(e) = self.registry.delete_local(&id) {
                    self.toasts
                        .push(Toast::error(format!("Cannot delete {id}: {e}")));
                }
            }
            None => {}
        }
        for action in self
            .hub_dialog
            .show(&ctx, &mut self.state.settings.hub, busy)
        {
            self.hub_action(action);
        }

        self.toasts.show(&ctx, toast_gap);
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.frame(ui);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if let Ok(json) = serde_json::to_string(&self.state.settings) {
            storage.set_string(SETTINGS_KEY, json);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::canvas_edit::Handle;
    use egui_kittest::kittest::Queryable;
    use scaleup_core::Margins;

    fn app(dir: &Path) -> App {
        app_with(dir, Settings::default())
    }

    fn app_with(dir: &Path, settings: Settings) -> App {
        let registry = Registry::with_dir(dir.to_path_buf());
        std::fs::write(registry.local_path(&settings.model_id), b"x").unwrap();
        App {
            state: AppState {
                settings,
                device_name: "NVIDIA GeForce GTX 1660 SUPER (DirectML)".into(),
                ..AppState::default()
            },
            worker: worker::spawn(registry.clone(), "cpu"),
            registry,
            toasts: Toasts::default(),
            log_open: false,
            compare: CompareView::default(),
            models_dialog: ModelsDialog::default(),
            hub_dialog: HubDialog::default(),
            strip: HistoryStrip::default(),
            showing: None,
            step_error: None,
            asked: Vec::new(),
            discards: 0,
        }
    }

    fn render(
        app: App,
        size: [f32; 2],
        theme: egui::Theme,
        name: &str,
        prep: impl Fn(&mut App, &egui::Context),
    ) {
        render_then(app, size, theme, name, prep, |_| {});
    }

    /// Like [`render`], with something done to the window (a click, a held file) before the shot.
    fn render_then(
        app: App,
        size: [f32; 2],
        theme: egui::Theme,
        name: &str,
        prep: impl Fn(&mut App, &egui::Context),
        act: impl Fn(&mut egui_kittest::Harness<'_, App>),
    ) {
        let out = PathBuf::from(std::env::var_os("SCALEUP_RENDER_DIR").unwrap());
        let mut harness = egui_kittest::Harness::builder()
            .with_size(size)
            .wgpu()
            .build_ui_state(|ui, app: &mut App| app.frame(ui), app);
        theme::apply(&harness.ctx);
        harness.ctx.set_theme(theme);
        let ctx = harness.ctx.clone();
        prep(harness.state_mut(), &ctx);
        for _ in 0..6 {
            harness.step();
        }
        act(&mut harness);
        for _ in 0..6 {
            harness.step();
        }
        let file = out.join(format!("{name}.png"));
        harness.render().expect("render").save(&file).unwrap();
        eprintln!("wrote {}", file.display());
    }

    fn with_images(app: &mut App, ctx: &egui::Context, after: bool) {
        let photo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/images/photo_like_256x192.jpg");
        let img = image::open(&photo).unwrap().into_rgba8();
        let color = |i: &image::RgbaImage| {
            egui::ColorImage::from_rgba_unmultiplied(
                [i.width() as usize, i.height() as usize],
                i.as_raw(),
            )
        };
        app.state.mode = Some(Mode::Single(PathBuf::from(
            r"C:\Users\alain\Pictures\Holidays 2026\harbour at night.jpg",
        )));
        app.state.source_size = Some([256, 192]);
        let none = Margins::default();
        app.compare
            .set(ctx, Which::Before, color(&img), [256, 192], none);
        if after {
            let filter = image::imageops::FilterType::Lanczos3;
            let big = image::imageops::resize(&img, 1024, 768, filter);
            app.compare
                .set(ctx, Which::After, color(&big), [1024, 768], none);
        }
    }

    /// Every control of the window has a name a screen reader can say: the harness checks it on
    /// each run. No GPU needed.
    #[test]
    fn every_control_has_an_accessible_name() {
        let dir = tempfile::tempdir().unwrap();
        let settings = Settings {
            extend_enabled: true,
            extend_linked: false,
            ..Settings::default()
        };
        // The Models window over the sidebar, then the edit stage of an image with Extend on.
        for edit in [false, true] {
            let mut app = app_with(dir.path(), settings.clone());
            app.models_dialog.open = !edit;
            let mut harness = egui_kittest::Harness::builder()
                .with_size([1200.0, 800.0])
                .build_ui_state(|ui, app: &mut App| app.frame(ui), app);
            if edit {
                let ctx = harness.ctx.clone();
                with_images(harness.state_mut(), &ctx, false);
            }
            for _ in 0..4 {
                harness.step();
            }
            harness.check_accessibility();
            if edit {
                assert!(harness.state().compare.edit_layout().is_some());
                for handle in Handle::ALL {
                    harness.get_by_label(handle.name());
                }
            }
        }
        // The history strip, with the middle of three steps current.
        let steps = three_steps(dir.path());
        let mut harness = history_harness(dir.path(), settings, &steps, 1);
        harness.step();
        harness.check_accessibility();
        for name in [
            "Back",
            "Forward",
            "Use as input",
            "Restore settings",
            "Open folder",
            "Remove",
        ] {
            harness.get_by_label(name);
        }
        for (i, s) in steps.iter().enumerate() {
            harness.get_by_label(&history::step_name(i, s));
        }
    }

    /// Extend on, mirror fill, 32 px on every side (not linked).
    fn edit_settings() -> Settings {
        Settings {
            extend_enabled: true,
            extend_linked: false,
            extend_left: 32.0,
            extend_top: 32.0,
            extend_right: 32.0,
            extend_bottom: 32.0,
            extend_fill: "mirror".into(),
            ..Settings::default()
        }
    }

    /// An app on the edit stage of the fixture photo (256×192) with 32 px on every side.
    fn edit_harness(dir: &Path) -> egui_kittest::Harness<'static, App> {
        edit_harness_with(dir, edit_settings())
    }

    /// An app on the edit stage of the fixture photo with `settings`.
    fn edit_harness_with(dir: &Path, settings: Settings) -> egui_kittest::Harness<'static, App> {
        let app = app_with(dir, settings);
        let mut harness = egui_kittest::Harness::builder()
            .with_size([1200.0, 800.0])
            .build_ui_state(|ui, app: &mut App| app.frame(ui), app);
        let ctx = harness.ctx.clone();
        with_images(harness.state_mut(), &ctx, false);
        for _ in 0..3 {
            harness.step();
        }
        harness
    }

    fn sides(s: &Settings) -> [f32; 4] {
        [s.extend_left, s.extend_top, s.extend_right, s.extend_bottom]
    }

    #[test]
    fn dragging_the_right_handle_adds_canvas_on_the_right() {
        let dir = tempfile::tempdir().unwrap();
        let mut harness = edit_harness(dir.path());
        let layout = harness.state().compare.edit_layout().unwrap();
        let at = Handle::Right.position(layout.canvas);
        let to = at + egui::vec2(40.0, 0.0);
        harness.hover_at(at);
        harness.drag_at(at);
        harness.hover_at(to);
        harness.step();
        let grown = 32.0 + (40.0 / layout.scale).round();
        // Mid-drag the sidebar already shows the new value: it was drawn again in the same frame.
        let s = &harness.state().state.settings;
        assert_eq!(sides(s), [32.0, 32.0, grown, 32.0]);
        let field = harness.get_by_label("Right").value().unwrap_or_default();
        assert!(field.contains(&format!("{grown}")), "{field}");
        // With the scale frozen, the edge is under the pointer.
        let mid = harness.state().compare.edit_layout().unwrap();
        assert!((mid.canvas.max.x - to.x).abs() <= layout.scale / 2.0 + 0.01);
        harness.drop_at(to);
        harness.step();
        let s = &harness.state().state.settings;
        assert_eq!(sides(s), [32.0, 32.0, grown, 32.0]);
        assert!(!s.extend_linked);
        assert_eq!(s.extend_mode, crate::state::ExtendMode::Pixels);
        // Released: the wider canvas is fitted again, so it is drawn smaller.
        let after = harness.state().compare.edit_layout().unwrap();
        assert!(after.scale < layout.scale);
    }

    #[test]
    fn a_focused_handle_moves_with_the_arrow_keys() {
        let dir = tempfile::tempdir().unwrap();
        let mut harness = edit_harness(dir.path());
        harness.get_by_label("Right edge").focus();
        harness.step();
        harness.key_press(egui::Key::ArrowRight);
        harness.step();
        assert_eq!(
            sides(&harness.state().state.settings),
            [32.0, 32.0, 33.0, 32.0]
        );
        harness.key_press_modifiers(egui::Modifiers::SHIFT, egui::Key::ArrowLeft);
        harness.step();
        assert_eq!(
            sides(&harness.state().state.settings),
            [32.0, 32.0, 23.0, 32.0]
        );
        // Up and down are across the right edge: nothing moves, and focus stays.
        harness.key_press(egui::Key::ArrowUp);
        harness.step();
        assert_eq!(
            sides(&harness.state().state.settings),
            [32.0, 32.0, 23.0, 32.0]
        );
        assert!(harness.get_by_label("Right edge").is_focused());
    }

    #[test]
    fn the_chips_switch_between_result_and_edit() {
        let dir = tempfile::tempdir().unwrap();
        // Extend off, an image loaded: a quiet offer to extend, which opens the edit stage.
        let mut harness = egui_kittest::Harness::builder()
            .with_size([1200.0, 800.0])
            .build_ui_state(|ui, app: &mut App| app.frame(ui), app(dir.path()));
        let ctx = harness.ctx.clone();
        with_images(harness.state_mut(), &ctx, true);
        harness.step();
        assert!(harness.query_by_label("Adjust canvas").is_none());
        harness.get_by_label("Extend canvas…").click();
        harness.step();
        harness.step();
        assert!(harness.state().state.settings.extend_enabled);
        assert!(harness.state().compare.edit_layout().is_some());
        assert!(harness.query_by_label("Extend canvas…").is_none());
        // A new result shows as a result, with the way back to Edit.
        let photo = egui::ColorImage::filled([4, 3], egui::Color32::GRAY);
        let none = scaleup_core::Margins::default();
        harness
            .state_mut()
            .compare
            .set(&ctx, Which::After, photo, [4, 3], none);
        harness.step();
        assert!(harness.state().compare.edit_layout().is_none());
        harness.get_by_label("Adjust canvas").click();
        harness.step();
        harness.step();
        assert!(harness.state().compare.edit_layout().is_some());
    }

    #[test]
    fn no_offer_to_extend_a_folder() {
        let dir = tempfile::tempdir().unwrap();
        let mut harness = egui_kittest::Harness::builder()
            .with_size([1200.0, 800.0])
            .build_ui_state(|ui, app: &mut App| app.frame(ui), app(dir.path()));
        harness.state_mut().state.mode = Some(Mode::Batch(dir.path().to_path_buf()));
        harness.step();
        assert!(harness.query_by_label("Extend canvas…").is_none());
    }

    /// Shows a result of the edit settings for the fixture photo, as a finished run does.
    fn give_result(harness: &mut egui_kittest::Harness<'static, App>) {
        let ctx = harness.ctx.clone();
        let image = egui::ColorImage::filled([320, 256], egui::Color32::GRAY);
        let m = Margins {
            left: 32,
            top: 32,
            right: 32,
            bottom: 32,
        };
        harness
            .state_mut()
            .show_preview(&ctx, Which::After, image, [320, 256], m);
        harness.step();
    }

    fn on_edit(harness: &egui_kittest::Harness<'static, App>) -> bool {
        harness.state().compare.edit_layout().is_some()
    }

    #[test]
    fn a_screen_reader_steps_a_handle_and_hears_its_margin() {
        let dir = tempfile::tempdir().unwrap();
        let mut harness = edit_harness(dir.path());
        let value = |h: &egui_kittest::Harness<'static, App>, name: &str| {
            h.get_by_label(name).value().unwrap_or_default()
        };
        assert_eq!(value(&harness, "Right edge"), "32 px");
        assert_eq!(value(&harness, "Top-left corner"), "left 32 px, top 32 px");
        harness.get_by_label("Right edge").increment_accesskit();
        harness.step();
        assert_eq!(
            sides(&harness.state().state.settings),
            [32.0, 32.0, 33.0, 32.0]
        );
        assert_eq!(value(&harness, "Right edge"), "33 px");
        harness
            .get_by_label("Top-left corner")
            .decrement_accesskit();
        harness.step();
        assert_eq!(
            sides(&harness.state().state.settings),
            [31.0, 31.0, 33.0, 32.0]
        );
    }

    #[test]
    fn changing_the_extension_after_a_result_goes_back_to_edit_and_stays() {
        let dir = tempfile::tempdir().unwrap();
        let mut harness = edit_harness(dir.path());
        give_result(&mut harness);
        assert!(!on_edit(&harness));
        // Not a change of the output: "Same on every side" over four equal values.
        harness.state_mut().state.settings.extend_linked = true;
        harness.step();
        assert!(!on_edit(&harness));
        harness.state_mut().state.settings.extend_left = 40.0;
        harness.step();
        assert!(on_edit(&harness));
        // Back to what the result was made with: still editing until asked otherwise.
        harness.state_mut().state.settings.extend_left = 32.0;
        harness.step();
        assert!(on_edit(&harness));
        // "Show result" goes back to the result without a run; the next change returns.
        harness.get_by_label("Show result").click();
        harness.step();
        harness.step();
        assert!(!on_edit(&harness));
        assert!(harness.query_by_label("Show result").is_none());
        harness.state_mut().state.settings.extend_fill = "edge".into();
        harness.step();
        assert!(on_edit(&harness));
    }

    #[test]
    fn selecting_another_image_forgets_the_edit() {
        let dir = tempfile::tempdir().unwrap();
        let mut harness = edit_harness(dir.path());
        give_result(&mut harness);
        harness.state_mut().state.settings.extend_left = 40.0;
        harness.step();
        assert!(harness.state().compare.editing());
        let photo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/images/photo_like_256x192.jpg");
        harness.state_mut().select(&photo);
        assert!(!harness.state().compare.editing());
        // The worker decodes the new "before"; then a result of it shows as a result.
        for _ in 0..500 {
            harness.step();
            if harness.state().compare.sizes()[0].is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(harness.state().compare.sizes()[0], Some([256, 192]));
        give_result(&mut harness);
        assert!(!on_edit(&harness));
    }

    #[test]
    fn a_run_that_starts_mid_drag_ends_the_drag_and_a_failed_run_returns_to_edit() {
        let dir = tempfile::tempdir().unwrap();
        let mut harness = edit_harness(dir.path());
        let layout = harness.state().compare.edit_layout().unwrap();
        let at = Handle::Right.position(layout.canvas);
        harness.hover_at(at);
        harness.drag_at(at);
        harness.hover_at(at + egui::vec2(20.0, 0.0));
        harness.step();
        let mid = sides(&harness.state().state.settings);
        assert!(mid[2] > 32.0);
        // A run starts (the shortcut works while the button is held): the stage is Result, and
        // the pointer no longer moves the margin.
        harness.state_mut().state.status = Status::Running {
            file: 0,
            files: 1,
            percent: 0.0,
            phase: "Starting".into(),
        };
        harness.hover_at(at + egui::vec2(60.0, 0.0));
        harness.step();
        assert!(!on_edit(&harness));
        assert_eq!(sides(&harness.state().state.settings), mid);
        // It fails before a result: back to Edit, the button still held, nothing dragged.
        harness.state_mut().state.status = Status::Error("the model could not be loaded".into());
        harness.step();
        assert!(on_edit(&harness));
        harness.hover_at(at + egui::vec2(90.0, 0.0));
        harness.step();
        assert_eq!(sides(&harness.state().state.settings), mid);
        harness.drop_at(at + egui::vec2(90.0, 0.0));
        harness.step();
        assert_eq!(sides(&harness.state().state.settings), mid);
        // The canvas is fitted again, inside the image area.
        let now = harness.state().compare.edit_layout().unwrap();
        assert!(now.canvas.max.x < 1200.0 && now.scale < layout.scale);
    }

    #[test]
    fn a_percent_drag_settles_with_one_write_per_pointer_move() {
        let dir = tempfile::tempdir().unwrap();
        // 12.5% of 256 x 192 is 32 by 24 px.
        let settings = Settings {
            extend_mode: crate::state::ExtendMode::Percent,
            extend_left: 12.5,
            extend_top: 12.5,
            extend_right: 12.5,
            extend_bottom: 12.5,
            ..edit_settings()
        };
        let mut harness = edit_harness_with(dir.path(), settings);
        let layout = harness.state().compare.edit_layout().unwrap();
        let at = Handle::Right.position(layout.canvas);
        harness.hover_at(at);
        harness.drag_at(at);
        harness.step();
        assert_eq!(harness.state().discards, 0);
        let mut last = harness.state().state.settings.clone();
        for (i, dx) in [37.0, 50.0, 23.0].into_iter().enumerate() {
            harness.hover_at(at + egui::vec2(dx, 0.0));
            harness.step();
            assert_eq!(harness.state().discards, i + 1, "move {i}");
            let now = harness.state().state.settings.clone();
            assert_ne!(now, last);
            // The pointer stays put: nothing more is written or drawn again.
            for _ in 0..5 {
                harness.step();
            }
            assert_eq!(harness.state().discards, i + 1, "still after move {i}");
            assert_eq!(harness.state().state.settings, now);
            last = now;
            let right = crate::ui::canvas_edit::current_margins(&last, [256, 192])
                .unwrap()
                .right;
            assert_eq!(right, 32 + (dx / layout.scale).round() as u32, "move {i}");
        }
        harness.drop_at(at + egui::vec2(23.0, 0.0));
        harness.step();
        let s = &harness.state().state.settings;
        assert_eq!(s.extend_mode, crate::state::ExtendMode::Percent);
        assert_eq!(s.extend_left, 12.5);
    }

    #[test]
    fn pressing_a_handle_focuses_it() {
        let dir = tempfile::tempdir().unwrap();
        let mut harness = edit_harness(dir.path());
        let layout = harness.state().compare.edit_layout().unwrap();
        let at = Handle::Bottom.position(layout.canvas);
        harness.hover_at(at);
        harness.drag_at(at);
        harness.drop_at(at);
        harness.step();
        assert!(harness.get_by_label("Bottom edge").is_focused());
        assert_eq!(sides(&harness.state().state.settings), [32.0; 4]);
        assert_eq!(harness.state().discards, 0);
        // And the arrows move it from there.
        harness.key_press(egui::Key::ArrowDown);
        harness.step();
        assert_eq!(
            sides(&harness.state().state.settings),
            [32.0, 32.0, 32.0, 33.0]
        );
    }

    /// Writes a `w`×`h` PNG of one colour.
    fn png(path: &Path, [w, h]: [u32; 2], rgb: [u8; 3]) {
        let [r, g, b] = rgb;
        image::RgbaImage::from_pixel(w, h, image::Rgba([r, g, b, 255]))
            .save(path)
            .unwrap();
    }

    /// Three steps with real files in `dir`, each input and output of its own size, run with
    /// scale 2, 3 and 4. The middle one is current.
    fn three_steps(dir: &Path) -> Vec<Step> {
        (0..3u32)
            .map(|n| {
                let input = dir.join(format!("in{n}.png"));
                let output = dir.join(format!("out{n}.png"));
                let source_size = [8 + n, 6];
                let result_size = [16 + 2 * n, 12];
                png(&input, source_size, [40, 40, 40]);
                png(&output, result_size, [200, 60 * n as u8, 40]);
                Step {
                    input,
                    output,
                    settings: Settings {
                        scale: Some(2 + n),
                        ..Settings::default()
                    },
                    source_size,
                    result_size,
                    margins: Margins::default(),
                    label: format!("Upscale {}× · step {n}", 2 + n),
                }
            })
            .collect()
    }

    /// A window with `steps` in its history and step `at` current, its previews not asked for.
    fn history_harness(
        dir: &Path,
        settings: Settings,
        steps: &[Step],
        at: usize,
    ) -> egui_kittest::Harness<'static, App> {
        let mut app = app_with(dir, settings);
        for s in steps {
            app.state.history.push(s.clone());
        }
        app.state.history.go(at);
        app.state.mode = Some(Mode::Single(steps[at].input.clone()));
        let mut harness = egui_kittest::Harness::builder()
            .with_size([1200.0, 800.0])
            .build_ui_state(|ui, app: &mut App| app.frame(ui), app);
        harness.step();
        harness
    }

    /// Steps the window until the previews of the step asked for have arrived.
    fn settle(harness: &mut egui_kittest::Harness<'_, App>) {
        for _ in 0..1000 {
            harness.step();
            if harness.state().showing.is_none() {
                harness.step();
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("the worker did not send the step's previews");
    }

    /// Step `at` of `steps` is current and on screen.
    fn assert_shows(app: &App, steps: &[Step], at: usize) {
        assert_eq!(app.state.history.at(), Some(at));
        assert_on_screen(app, &steps[at]);
    }

    /// `s` is on screen: its input is the selected image, its pair is shown.
    fn assert_on_screen(app: &App, s: &Step) {
        assert_eq!(app.state.mode, Some(Mode::Single(s.input.clone())));
        assert_eq!(app.state.source_size, Some(s.source_size));
        assert_eq!(
            app.compare.sizes(),
            [Some(s.source_size), Some(s.result_size)]
        );
    }

    /// The outputs of the steps the worker was asked to show, in order.
    fn asked(app: &App) -> Vec<PathBuf> {
        app.asked.clone()
    }

    #[test]
    fn back_forward_and_a_click_walk_the_history() {
        let dir = tempfile::tempdir().unwrap();
        let steps = three_steps(dir.path());
        let mut harness = history_harness(dir.path(), Settings::default(), &steps, 1);
        assert_eq!(harness.state().strip.thumb_count(), 0);

        harness.get_by_label("Back").click();
        harness.step();
        assert_eq!(asked(harness.state()), [steps[0].output.clone()]);
        settle(&mut harness);
        assert_shows(harness.state(), &steps, 0);
        // The result preview made the step's thumbnail.
        assert_eq!(harness.state().strip.thumb_count(), 1);
        // At the first step Back is disabled.
        harness.get_by_label("Back").click();
        harness.step();
        assert_eq!(harness.state().state.history.at(), Some(0));
        assert_eq!(asked(harness.state()).len(), 1);

        harness.get_by_label("Forward").click();
        harness.step();
        harness.get_by_label("Forward").click();
        harness.step();
        let want = [0, 1, 2].map(|i| steps[i].output.clone());
        assert_eq!(asked(harness.state()), want);
        settle(&mut harness);
        assert_shows(harness.state(), &steps, 2);

        harness.get_by_label("Step 2: Upscale 3× · step 1").click();
        harness.step();
        assert_eq!(asked(harness.state()).last(), Some(&steps[1].output));
        settle(&mut harness);
        assert_shows(harness.state(), &steps, 1);
        assert!(harness.query_by_label("2 of 3").is_some());
    }

    #[test]
    fn a_run_started_after_going_back_reads_the_steps_input() {
        let dir = tempfile::tempdir().unwrap();
        let steps = three_steps(dir.path());
        let mut harness = history_harness(dir.path(), Settings::default(), &steps, 2);
        // An unrelated image is selected; going back makes the step's input the image.
        let other = dir.path().join("other.png");
        png(&other, [5, 5], [0, 0, 0]);
        harness.state_mut().select(&other);
        harness.step();
        harness.key_press_modifiers(egui::Modifiers::ALT, egui::Key::ArrowLeft);
        settle(&mut harness);
        assert_shows(harness.state(), &steps, 1);
        // The size line and Start see the step's input, and a result shows as a result.
        assert!(harness.state().compare.edit_layout().is_none());
        harness.state_mut().start();
        let Some(Mode::Single(p)) = &harness.state().state.mode else {
            panic!("no image")
        };
        assert_eq!(*p, steps[1].input);
        assert!(harness.state().state.status.is_running());
    }

    #[test]
    fn remove_takes_the_step_out_and_keeps_its_file() {
        let dir = tempfile::tempdir().unwrap();
        let steps = three_steps(dir.path());
        let mut harness = history_harness(dir.path(), Settings::default(), &steps, 1);
        harness.get_by_label("Remove").click();
        harness.step();
        let h = &harness.state().state.history;
        assert_eq!(h.steps().len(), 2);
        assert_eq!(h.at(), Some(1));
        assert_eq!(h.current().unwrap().output, steps[2].output);
        assert!(steps[1].output.is_file());
        // The view moves to the step the cursor moved to.
        settle(&mut harness);
        assert_on_screen(harness.state(), &steps[2]);
        assert_eq!(harness.state().asked.last(), Some(&steps[2].output));
        // The last ones go too: the strip goes, the selected image stays on screen alone.
        harness.get_by_label("Remove").click();
        harness.step();
        harness.get_by_label("Remove").click();
        harness.step();
        harness.step();
        assert!(harness.state().state.history.steps().is_empty());
        assert!(harness.query_by_label("Back").is_none());
        assert_eq!(harness.state().strip.thumb_count(), 0);
        let mode = harness.state().state.mode.clone();
        assert_eq!(mode, Some(Mode::Single(steps[0].input.clone())));
        for _ in 0..1000 {
            harness.step();
            if harness.state().compare.sizes()[0].is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(
            harness.state().compare.sizes(),
            [Some(steps[0].source_size), None]
        );
    }

    #[test]
    fn use_as_input_selects_the_result_and_restore_brings_its_settings_back() {
        let dir = tempfile::tempdir().unwrap();
        let steps = three_steps(dir.path());
        let mut harness = history_harness(dir.path(), Settings::default(), &steps, 1);
        harness.get_by_label("Restore settings").click();
        harness.step();
        assert_eq!(harness.state().state.settings.scale, Some(3));

        harness.get_by_label("Use as input").click();
        harness.step();
        let mode = harness.state().state.mode.clone();
        assert_eq!(mode, Some(Mode::Single(steps[1].output.clone())));
        for _ in 0..1000 {
            harness.step();
            if harness.state().state.source_size.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        harness.step();
        // The result is now the image: alone, as a new image, with its size in the sidebar.
        assert_eq!(
            harness.state().compare.sizes(),
            [Some(steps[1].result_size), None]
        );
        assert_eq!(
            harness.state().state.source_size,
            Some(steps[1].result_size)
        );
        // The history is unchanged.
        assert_eq!(harness.state().state.history.steps().len(), 3);
    }

    #[test]
    fn history_keys_walk_but_not_in_a_text_field_or_while_running() {
        let dir = tempfile::tempdir().unwrap();
        let steps = three_steps(dir.path());
        let settings = Settings {
            extend_enabled: true,
            extend_linked: false,
            ..Settings::default()
        };
        let mut harness = history_harness(dir.path(), settings, &steps, 1);
        let at = |h: &egui_kittest::Harness<'_, App>| h.state().state.history.at();
        let cmd = egui::Modifiers::COMMAND;
        let shift = egui::Modifiers::SHIFT;
        let keys = [
            (egui::Modifiers::ALT, egui::Key::ArrowLeft, 0),
            (egui::Modifiers::ALT, egui::Key::ArrowRight, 1),
            (cmd, egui::Key::Y, 2),
            (cmd, egui::Key::Z, 1),
            (cmd | shift, egui::Key::Z, 2),
        ];
        for (modifiers, key, want) in keys {
            harness.key_press_modifiers(modifiers, key);
            harness.step();
            assert_eq!(at(&harness), Some(want), "{modifiers:?} {key:?}");
        }
        settle(&mut harness);

        // A text field keeps its own undo.
        harness.get_by_label("Right").click();
        harness.step();
        harness.step();
        assert!(harness.ctx.text_edit_focused());
        harness.key_press_modifiers(cmd, egui::Key::Z);
        harness.step();
        harness.key_press_modifiers(egui::Modifiers::ALT, egui::Key::ArrowLeft);
        harness.step();
        assert_eq!(at(&harness), Some(2));
        assert_eq!(asked(harness.state()).len(), keys.len());
        harness.ctx.memory_mut(|m| {
            if let Some(id) = m.focused() {
                m.surrender_focus(id);
            }
        });
        harness.step();
        assert!(!harness.ctx.text_edit_focused());

        // While a run goes, neither the keys nor the strip move.
        harness.state_mut().state.status = Status::Running {
            file: 0,
            files: 1,
            percent: 10.0,
            phase: "Upscaling".into(),
        };
        harness.step();
        harness.key_press_modifiers(cmd, egui::Key::Z);
        harness.step();
        harness.get_by_label("Back").click();
        harness.step();
        harness.get_by_label("Step 1: Upscale 2× · step 0").click();
        harness.step();
        assert_eq!(at(&harness), Some(2));
    }

    #[test]
    fn a_focused_canvas_handle_keeps_its_keys() {
        let dir = tempfile::tempdir().unwrap();
        let mut harness = edit_harness(dir.path());
        let steps = three_steps(dir.path());
        for s in &steps {
            harness.state_mut().state.history.push(s.clone());
        }
        harness.step();
        harness.get_by_label("Right edge").focus();
        harness.step();
        harness.key_press_modifiers(egui::Modifiers::ALT, egui::Key::ArrowLeft);
        harness.step();
        harness.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::Z);
        harness.step();
        assert_eq!(harness.state().state.history.at(), Some(2));
        assert!(harness.get_by_label("Right edge").is_focused());
    }

    #[test]
    fn a_step_whose_file_is_gone_stays_and_says_so() {
        let dir = tempfile::tempdir().unwrap();
        let steps = three_steps(dir.path());
        std::fs::remove_file(&steps[0].output).unwrap();
        let mut harness = history_harness(dir.path(), Settings::default(), &steps, 1);
        harness.get_by_label("Back").click();
        settle(&mut harness);
        let app = harness.state();
        assert_eq!(app.state.history.steps().len(), 3);
        assert_eq!(app.state.history.at(), Some(0));
        let why = app.step_error.clone().unwrap();
        assert!(why.ends_with("no longer exists"), "{why}");
        assert!(
            harness
                .query_by_label("This step cannot be shown")
                .is_some()
        );
        // Another step shows again.
        harness.get_by_label("Forward").click();
        settle(&mut harness);
        assert_shows(harness.state(), &steps, 1);
        assert!(harness.state().step_error.is_none());
    }

    /// `n` steps made from the fixture photo, written to `dir`: upscales, extensions and a grey
    /// version in turn, so the thumbnails differ in shape and tone.
    fn photo_steps(dir: &Path, n: usize) -> Vec<Step> {
        let photo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/images/photo_like_256x192.jpg");
        let img = image::open(&photo).unwrap().into_rgba8();
        let input = dir.join("harbour at night.png");
        img.save(&input).unwrap();
        let filter = image::imageops::FilterType::Triangle;
        (0..n)
            .map(|i| {
                let (out, label, margins): (image::RgbaImage, &str, Margins) = match i % 3 {
                    0 => (
                        image::imageops::resize(&img, 512, 384, filter),
                        "Upscale 2× · Real-ESRGAN General v3 (fast)",
                        Margins::default(),
                    ),
                    1 => {
                        let ground = image::Rgba([96, 104, 120, 255]);
                        let mut wide = image::RgbaImage::from_pixel(384, 192, ground);
                        image::imageops::overlay(&mut wide, &img, 64, 0);
                        let m = Margins {
                            left: 64,
                            top: 0,
                            right: 64,
                            bottom: 0,
                        };
                        (wide, "Extend +64 px · LaMa (512)", m)
                    }
                    _ => {
                        let big = image::imageops::resize(&img, 1024, 768, filter);
                        let grey = image::DynamicImage::ImageRgba8(big).grayscale();
                        (
                            grey.into_rgba8(),
                            "Upscale 4× · Real-ESRGAN General v3 (fast)",
                            Margins::default(),
                        )
                    }
                };
                let output = dir.join(format!("harbour at night_scaleup_{i}.png"));
                out.save(&output).unwrap();
                Step {
                    input: input.clone(),
                    output,
                    settings: Settings::default(),
                    source_size: [256, 192],
                    result_size: [out.width(), out.height()],
                    margins,
                    label: label.into(),
                }
            })
            .collect()
    }

    /// Puts `steps` in the history with their thumbnails, made from the files as a run makes
    /// them, then goes to step `at` (the worker sends its previews).
    fn with_history(app: &mut App, ctx: &egui::Context, steps: &[Step], at: usize) {
        for s in steps {
            app.state.history.push(s.clone());
            let img = image::open(&s.output).unwrap().into_rgba8();
            let size = [img.width() as usize, img.height() as usize];
            let color = egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw());
            app.strip
                .note_result(ctx, &app.state.history, &s.output, &color);
        }
        app.history_action(HistoryAction::Go(at));
    }

    fn mirror_settings() -> Settings {
        Settings {
            extend_enabled: true,
            extend_linked: false,
            extend_left: 64.0,
            extend_right: 64.0,
            extend_fill: "mirror".into(),
            ..Settings::default()
        }
    }

    /// Draws the whole window in its main states, dark and light, into `$SCALEUP_RENDER_DIR/*.png`
    /// for a person to look at. Run with `-- --ignored`; skipped when the variable is unset.
    #[test]
    #[ignore = "writes PNG files for a manual look; needs SCALEUP_RENDER_DIR and a GPU adapter"]
    fn window_renders() {
        if std::env::var_os("SCALEUP_RENDER_DIR").is_none() {
            eprintln!("skipped: SCALEUP_RENDER_DIR is not set");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let big = [1200.0, 800.0];
        let folder = |a: &mut App, _: &egui::Context| {
            a.state.mode = Some(Mode::Batch(PathBuf::from(r"C:\Users\alain\Pictures\Scans")));
        };
        for (theme, t) in [(egui::Theme::Dark, "dark"), (egui::Theme::Light, "light")] {
            let new = || app(dir.path());
            render(new(), big, theme, &format!("empty-{t}"), |_, _| {});
            render(new(), big, theme, &format!("folder-{t}"), folder);
            render(new(), big, theme, &format!("loaded-{t}"), |a, c| {
                with_images(a, c, false);
            });
            render(new(), big, theme, &format!("running-{t}"), |a, c| {
                with_images(a, c, false);
                a.state.status = Status::Running {
                    file: 0,
                    files: 1,
                    percent: 42.0,
                    phase: "Upscaling".into(),
                };
            });
            render(new(), big, theme, &format!("done-{t}"), |a, c| {
                with_images(a, c, true);
                a.state.status = Status::Done(PathBuf::from(r"C:\out\harbour at night_4x.png"));
                a.toasts
                    .push(Toast::success("Saved harbour at night_4x.png"));
            });
            let extend = Settings {
                extend_enabled: true,
                extend_left: 128.0,
                ..Settings::default()
            };
            let open = app_with(dir.path(), extend.clone());
            render(open, big, theme, &format!("extend-{t}"), |a, c| {
                with_images(a, c, false);
            });
            let small = app_with(dir.path(), extend);
            render(
                small,
                [900.0, 600.0],
                theme,
                &format!("small-{t}"),
                |a, c| {
                    with_images(a, c, false);
                    a.state.status = Status::Error(
                        "Cannot read harbour at night.jpg: unexpected end of file".into(),
                    );
                    a.log_open = true;
                    a.state.log = (0..6).map(|i| format!("tile {i}/24 done")).collect();
                },
            );
            // The edit stage with each fill, 96 px on every side.
            for (fill, name) in [
                ("ai", "ai"),
                ("mirror", "mirror"),
                ("edge", "edge"),
                ("#d8c8a8", "colour"),
                ("transparent", "transparent"),
            ] {
                let s = Settings {
                    extend_enabled: true,
                    extend_left: 96.0,
                    extend_fill: fill.into(),
                    ..Settings::default()
                };
                render(
                    app_with(dir.path(), s),
                    big,
                    theme,
                    &format!("edit-{name}-{t}"),
                    |a, c| with_images(a, c, false),
                );
            }
            let uneven = Settings {
                extend_enabled: true,
                extend_linked: false,
                extend_left: 160.0,
                extend_top: 24.0,
                extend_right: 48.0,
                extend_bottom: 96.0,
                ..Settings::default()
            };
            render(
                app_with(dir.path(), uneven.clone()),
                big,
                theme,
                &format!("edit-uneven-{t}"),
                |a, c| with_images(a, c, false),
            );
            let mirror = Settings {
                extend_fill: "mirror".into(),
                ..uneven.clone()
            };
            render(
                app_with(dir.path(), mirror.clone()),
                [900.0, 600.0],
                theme,
                &format!("edit-small-{t}"),
                |a, c| with_images(a, c, false),
            );
            render_then(
                app_with(dir.path(), mirror.clone()),
                big,
                theme,
                &format!("edit-hover-{t}"),
                |a, c| with_images(a, c, false),
                |h| {
                    let l = h.state().compare.edit_layout().unwrap();
                    h.hover_at(Handle::Right.position(l.canvas));
                },
            );
            let invalid = Settings {
                extend_left: 2000.0,
                ..mirror.clone()
            };
            render(
                app_with(dir.path(), invalid),
                big,
                theme,
                &format!("edit-invalid-{t}"),
                |a, c| with_images(a, c, false),
            );
            let aspect = Settings {
                extend_mode: crate::state::ExtendMode::Aspect,
                extend_aspect: (16, 9),
                ..mirror.clone()
            };
            render(
                app_with(dir.path(), aspect),
                big,
                theme,
                &format!("edit-aspect-{t}"),
                |a, c| with_images(a, c, false),
            );
            render(
                app_with(dir.path(), mirror),
                big,
                theme,
                &format!("result-adjust-{t}"),
                |a, c| {
                    with_images(a, c, true);
                    a.state.status = Status::Done(PathBuf::from(r"C:\out\harbour at night_4x.png"));
                },
            );
            render(new(), big, theme, &format!("models-{t}"), |a, c| {
                with_images(a, c, false);
                a.models_dialog.open = true;
            });
            render_then(
                new(),
                big,
                theme,
                &format!("advanced-{t}"),
                |a, c| {
                    with_images(a, c, false);
                    a.state.status = Status::Stopped { done: 3, total: 12 };
                    a.toasts.push(Toast::error(
                        "Cannot open C:\\out: the folder does not exist",
                    ));
                },
                |h| h.get_by_label("Advanced").click(),
            );
            render_then(
                new(),
                big,
                theme,
                &format!("picker-{t}"),
                |a, c| with_images(a, c, false),
                |h| h.get_by_label("Model").click(),
            );
            render(new(), big, theme, &format!("download-{t}"), |a, c| {
                with_images(a, c, false);
                a.state.status = Status::Downloading {
                    done: 21 << 20,
                    total: 64 << 20,
                };
            });
            render_then(
                new(),
                big,
                theme,
                &format!("license-{t}"),
                |a, _| a.models_dialog.open = true,
                |h| h.get_all_by_label("Show license").next().unwrap().click(),
            );
            render_then(
                new(),
                big,
                theme,
                &format!("drop-{t}"),
                |a, c| with_images(a, c, false),
                |h| {
                    h.input_mut().hovered_files.push(egui::HoveredFile {
                        path: Some(PathBuf::from("a.png")),
                        mime: String::new(),
                    });
                },
            );
            // The history strip: three steps with the middle one current; twelve, scrolled to the
            // current one; while running; at the smallest window with the log open.
            let three = photo_steps(dir.path(), 3);
            let twelve = photo_steps(dir.path(), 12);
            let running = |a: &mut App| {
                a.state.status = Status::Running {
                    file: 0,
                    files: 1,
                    percent: 42.0,
                    phase: "Upscaling".into(),
                };
            };
            // Name, window size, steps, current step, running, log open.
            type Case<'a> = (&'a str, [f32; 2], &'a [Step], usize, bool, bool);
            let cases: [Case<'_>; 4] = [
                ("history", big, &three, 1, false, false),
                ("history-many", big, &twelve, 10, false, false),
                ("history-running", big, &three, 1, true, false),
                ("history-small", [900.0, 600.0], &twelve, 4, false, true),
            ];
            for (name, size, steps, at, run, log) in cases {
                render_then(
                    new(),
                    size,
                    theme,
                    &format!("{name}-{t}"),
                    |a, c| {
                        with_history(a, c, steps, at);
                        a.log_open = log;
                        a.state.log = (0..6).map(|i| format!("tile {i}/24 done")).collect();
                    },
                    |h| {
                        settle(h);
                        if run {
                            running(h.state_mut());
                        }
                    },
                );
            }
            // Use as input on an extension, at the smallest window: the edit stage of the result.
            render_then(
                app_with(dir.path(), mirror_settings()),
                [900.0, 600.0],
                theme,
                &format!("history-edit-small-{t}"),
                |a, c| {
                    with_history(a, c, &three, 2);
                    a.log_open = true;
                },
                |h| {
                    settle(h);
                    h.get_by_label("Use as input").click();
                    for _ in 0..500 {
                        h.step();
                        if h.state().compare.edit_layout().is_some() {
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                },
            );
        }
    }
}
