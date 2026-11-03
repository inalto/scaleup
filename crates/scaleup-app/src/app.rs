//! The eframe application shell: routes input to state, state to the worker, events back to state.

use std::path::{Path, PathBuf};

use eframe::egui;
use scaleup_core::Registry;

use crate::state::{AppState, Settings, Status, apply_event, mode_from_path};
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
        self.state.status = Status::Running {
            file: 0,
            files: 1,
            percent: 0.0,
            phase: "Starting".into(),
        };
        self.send(UiMsg::Start {
            mode,
            settings: self.state.settings.clone(),
        });
    }

    fn drain_events(&mut self) {
        while let Ok(ev) = self.worker.rx.try_recv() {
            self.handle_event(ev);
        }
    }

    fn handle_event(&mut self, ev: WorkerEvent) {
        if let Some(toast) = apply_event(&mut self.state, ev) {
            self.toasts.push(toast);
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
        self.drain_events();

        let dropped = ctx.input(|i| i.raw.dropped_files.first().and_then(|f| f.path.clone()));
        if let Some(path) = dropped {
            self.select(&path);
        }
        if self.state.status.is_running() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        let mut open_folder = false;
        egui::Panel::bottom("status").show_inside(ui, |ui| {
            ui.add_space(4.0);
            open_folder = statusbar::show(ui, &self.state, &mut self.log_open);
            ui.add_space(4.0);
        });
        if open_folder {
            self.open_output_folder();
        }

        let entries = self.registry.entries().to_vec();
        let side = egui::Panel::left("sidebar")
            .exact_size(300.0)
            .resizable(false)
            .show_inside(ui, |ui| sidebar::show(ui, &mut self.state, &entries))
            .inner;
        if let Some(p) = side.picked {
            self.select(&p);
        }
        if side.device_changed {
            self.send(UiMsg::ChangeDevice(self.state.settings.gpu.clone()));
        }
        if side.start {
            self.start();
        }
        if side.stop {
            self.send(UiMsg::Stop);
        }

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.centered_and_justified(|ui| {
                ui.weak("Drop an image or a folder here");
            });
        });

        self.toasts.show(&ctx);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if let Ok(json) = serde_json::to_string(&self.state.settings) {
            storage.set_string(SETTINGS_KEY, json);
        }
    }
}
