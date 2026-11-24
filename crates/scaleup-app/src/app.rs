//! The eframe application shell: routes input to state, state to the worker, events back to state.

use std::path::{Path, PathBuf};

use eframe::egui;
use scaleup_core::Registry;

use crate::paste;
use crate::state::Mode;
use crate::state::{AppState, Settings, Status, apply_event, mode_from_path};
use crate::ui::compare::{CompareView, StageInput};
use crate::ui::hub_dialog::{HubAction, HubDialog};
use crate::ui::models_dialog::{DialogAction, ModelsDialog};
use crate::ui::toast::{Toast, Toasts};
use crate::ui::{sidebar, statusbar, theme};
use crate::worker::{self, UiMsg, Which, WorkerEvent, WorkerHandle};

const SETTINGS_KEY: &str = "settings";
const SIDEBAR_WIDTH: f32 = 320.0;

pub struct App {
    pub state: AppState,
    registry: Registry,
    worker: WorkerHandle,
    toasts: Toasts,
    log_open: bool,
    compare: CompareView,
    models_dialog: ModelsDialog,
    hub_dialog: HubDialog,
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
                image,
                size,
                margins,
            } => {
                self.state.note_preview(which, size);
                self.compare.set(ctx, which, image, size, margins);
                if which == Which::After {
                    self.compare.result_settings(&self.state.settings);
                }
                return;
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
    }

    fn open_output_folder(&mut self) {
        let Status::Done(path) = &self.state.status else {
            return;
        };
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

        self.toasts.show(&ctx);
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
    }

    /// An app on the edit stage of the fixture photo (256×192) with 32 px on every side.
    fn edit_harness(dir: &Path) -> egui_kittest::Harness<'static, App> {
        let settings = Settings {
            extend_enabled: true,
            extend_linked: false,
            extend_left: 32.0,
            extend_top: 32.0,
            extend_right: 32.0,
            extend_bottom: 32.0,
            extend_fill: "mirror".into(),
            ..Settings::default()
        };
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
        }
    }
}
