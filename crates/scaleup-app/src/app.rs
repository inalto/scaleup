//! The eframe application shell.

use eframe::egui;

use crate::state::{AppState, Settings};

const SETTINGS_KEY: &str = "settings";

pub struct App {
    pub state: AppState,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let settings = cc
            .storage
            .and_then(|s| s.get_string(SETTINGS_KEY))
            .and_then(|json| serde_json::from_str::<Settings>(&json).ok())
            .unwrap_or_default();
        Self {
            state: AppState {
                settings,
                ..AppState::default()
            },
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.label("ScaleUP");
        });
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if let Ok(json) = serde_json::to_string(&self.state.settings) {
            storage.set_string(SETTINGS_KEY, json);
        }
    }
}
