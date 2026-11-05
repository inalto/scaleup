//! The "Models" window: the catalogue with licenses, downloads and deletion.

use std::collections::HashSet;

use eframe::egui;
use scaleup_core::registry::SOURCE_HUB;
use scaleup_core::{ModelEntry, Registry};

include!(concat!(env!("OUT_DIR"), "/licenses.rs"));

/// What the user asked for this frame.
#[derive(Debug, PartialEq, Eq)]
pub enum DialogAction {
    Download(String),
    Delete(String),
    /// An OpenModelDB model: removed with its folder and its entry in `installed.json`.
    RemoveHub(String),
}

/// What Delete does for `e`: hub models go through the hub, so `installed.json` stays in step.
fn delete_action(e: &ModelEntry) -> DialogAction {
    if e.source == SOURCE_HUB {
        DialogAction::RemoveHub(e.id.clone())
    } else {
        DialogAction::Delete(e.id.clone())
    }
}

#[derive(Default)]
pub struct ModelsDialog {
    pub open: bool,
    /// Rows with their license text expanded.
    expanded: HashSet<String>,
}

fn size_label(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
}

fn license_badge(ui: &mut egui::Ui, entry: &ModelEntry) {
    let (fill, text) = if entry.commercial {
        (egui::Color32::from_rgb(28, 110, 60), entry.license.clone())
    } else {
        (
            egui::Color32::from_rgb(190, 40, 40),
            format!("NC {}", entry.license),
        )
    };
    ui.label(
        egui::RichText::new(format!(" {text} "))
            .small()
            .color(egui::Color32::WHITE)
            .background_color(fill),
    );
}

impl ModelsDialog {
    /// `busy` disables the buttons that would queue work behind a running job.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        registry: &Registry,
        busy: bool,
    ) -> Option<DialogAction> {
        let mut action = None;
        let mut open = self.open;
        egui::Window::new("Models")
            .open(&mut open)
            .default_width(640.0)
            .default_height(460.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    egui::Grid::new("models_grid")
                        .striped(true)
                        .num_columns(6)
                        .show(ui, |ui| {
                            for e in registry.entries() {
                                self.row(ui, registry, e, busy, &mut action);
                                ui.end_row();
                                if self.expanded.contains(&e.id) {
                                    ui.label("");
                                    let text = license_text(&e.id).unwrap_or("No license text.");
                                    ui.add(
                                        egui::Label::new(egui::RichText::new(text).small()).wrap(),
                                    );
                                    ui.end_row();
                                }
                            }
                        });
                });
            });
        self.open = open;
        action
    }

    fn row(
        &mut self,
        ui: &mut egui::Ui,
        registry: &Registry,
        e: &ModelEntry,
        busy: bool,
        action: &mut Option<DialogAction>,
    ) {
        ui.label(&e.name).on_hover_text(&e.id);
        ui.label(format!("{}× {}", e.scale, e.arch));
        license_badge(ui, e);
        ui.label(size_label(e.size_bytes));
        if registry.is_downloaded(&e.id) {
            ui.horizontal(|ui| {
                ui.label("Downloaded");
                if ui.add_enabled(!busy, egui::Button::new("Delete")).clicked() {
                    *action = Some(delete_action(e));
                }
            });
        } else if ui
            .add_enabled(!busy, egui::Button::new("Download"))
            .clicked()
        {
            *action = Some(DialogAction::Download(e.id.clone()));
        }
        if ui.small_button("Show license").clicked() && !self.expanded.remove(&e.id) {
            self.expanded.insert(e.id.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deleting_a_hub_model_goes_through_the_hub() {
        let dir = tempfile::tempdir().unwrap();
        let mut registry = Registry::with_dir(dir.path().to_path_buf());
        let builtin = registry.entries()[0].clone();
        assert_eq!(
            delete_action(&builtin),
            DialogAction::Delete(builtin.id.clone())
        );
        let src = dir.path().join("mine-2x.onnx");
        std::fs::write(&src, b"x").unwrap();
        let id = registry.import_onnx(&src).unwrap();
        let imported = registry.get(&id).unwrap();
        assert_eq!(delete_action(imported), DialogAction::Delete(id.clone()));
        let hub = ModelEntry {
            source: SOURCE_HUB.into(),
            ..builtin
        };
        assert_eq!(delete_action(&hub), DialogAction::RemoveHub(hub.id.clone()));
    }

    #[test]
    fn license_lookup_has_every_manifest_id() {
        let dir = tempfile::tempdir().unwrap();
        let registry = Registry::with_dir(dir.path().to_path_buf());
        assert!(!registry.entries().is_empty());
        for e in registry.entries() {
            let text = license_text(&e.id).unwrap_or_else(|| panic!("no license for {}", e.id));
            assert!(!text.trim().is_empty(), "{}", e.id);
        }
        assert_eq!(license_text("no-such-model"), None);
    }
}
