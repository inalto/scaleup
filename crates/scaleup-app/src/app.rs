//! The eframe application shell: routes input to state, state to the worker, events back to state.

use std::path::{Path, PathBuf};

use eframe::egui;
use scaleup_core::Registry;

use crate::paste;
use crate::state::Mode;
use crate::state::{AppState, Settings, Status, apply_event, mode_from_path};
use crate::ui::compare::CompareView;
use crate::ui::hub_dialog::{HubAction, HubDialog};
use crate::ui::models_dialog::{DialogAction, ModelsDialog};
use crate::ui::toast::{Toast, Toasts};
use crate::ui::{sidebar, statusbar};
use crate::worker::{self, UiMsg, WorkerEvent, WorkerHandle};

const SETTINGS_KEY: &str = "settings";

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
        if let Some(Err(e)) = self.state.settings.extend_summary(self.state.source_size) {
            self.toasts
                .push(Toast::error(format!("Extend canvas: {e}")));
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

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
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
                            for line in &self.state.log {
                                ui.monospace(line);
                            }
                        });
                });
        }

        let side = egui::Panel::left("sidebar")
            .exact_size(300.0)
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

        egui::CentralPanel::default().show(ui, |ui| {
            if self.compare.is_empty() {
                ui.centered_and_justified(|ui| {
                    match &self.state.mode {
                        Some(Mode::Batch(p)) => {
                            ui.weak(format!("Folder selected: {}", p.display()))
                        }
                        Some(Mode::Single(_)) if matches!(self.state.status, Status::Error(_)) => {
                            ui.weak("This image could not be opened")
                        }
                        Some(Mode::Single(_)) => ui.weak("Loading preview…"),
                        None => ui.weak("Drop an image or a folder here"),
                    };
                });
            } else {
                self.compare.show(ui);
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

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if let Ok(json) = serde_json::to_string(&self.state.settings) {
            storage.set_string(SETTINGS_KEY, json);
        }
    }
}
