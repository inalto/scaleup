//! The "Models" window: the catalogue with licenses, downloads and deletion.

use eframe::egui;
use scaleup_core::registry::SOURCE_HUB;
use scaleup_core::{ModelEntry, ModelKind, Registry};

use crate::ui::theme;

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
    /// The model whose license text shows under the list.
    license_of: Option<String>,
}

/// What a model does: its scale for an upscaler, "Extend" for an inpainting model (scale 1).
fn kind_label(e: &ModelEntry) -> String {
    match e.kind {
        ModelKind::Upscale => format!("{}× {}", e.scale, e.arch),
        ModelKind::Inpaint => format!("Extend · {}", e.arch),
    }
}

/// The widest the name cell grows: the "good for" line wraps there instead of widening the grid.
const GOOD_FOR_WIDTH: f32 = 270.0;

fn size_label(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
}

fn license_badge(ui: &mut egui::Ui, entry: &ModelEntry) {
    if entry.commercial {
        theme::badge(ui, &entry.license, theme::SUCCESS).on_hover_text("Free for any use");
    } else {
        theme::badge(ui, &format!("NC {}", entry.license), theme::DANGER)
            .on_hover_text("Not for commercial use");
    }
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
            .default_width(680.0)
            .default_height(460.0)
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(ctx.content_rect().center())
            .collapsible(false)
            .show(ctx, |ui| {
                let shown = self
                    .license_of
                    .as_ref()
                    .and_then(|id| registry.get(id))
                    .cloned();
                // With a license open the list keeps the upper part, the text the rest.
                let list_height = if shown.is_some() {
                    (ui.available_height() * 0.45).max(120.0)
                } else {
                    f32::INFINITY
                };
                egui::ScrollArea::vertical()
                    .id_salt("models_list")
                    .max_height(list_height)
                    .show(ui, |ui| {
                        egui::Grid::new("models_grid")
                            .striped(true)
                            .num_columns(6)
                            .show(ui, |ui| {
                                for title in ["Model", "Type", "License", "Size", "", ""] {
                                    ui.strong(title);
                                }
                                ui.end_row();
                                for e in registry.entries() {
                                    self.row(ui, registry, e, busy, &mut action);
                                    ui.end_row();
                                }
                            });
                    });
                if let Some(e) = shown {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.strong(format!("License of {}", e.name));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.link("Hide").clicked() {
                                self.license_of = None;
                            }
                        });
                    });
                    let text = license_text(&e.id).unwrap_or("No license text.");
                    egui::ScrollArea::vertical()
                        .id_salt("models_license")
                        .auto_shrink(false)
                        .show(ui, |ui| {
                            ui.add(egui::Label::new(text).wrap());
                        });
                }
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
        // An inpainting model says what it suits under its name, as the pickers do.
        ui.vertical(|ui| {
            ui.set_max_width(GOOD_FOR_WIDTH);
            ui.label(&e.name).on_hover_text(&e.id);
            if !e.good_for.is_empty() {
                ui.weak(&e.good_for);
            }
        });
        ui.label(kind_label(e));
        // In a row of three lines the badge keeps its own height, level with the name.
        ui.horizontal(|ui| license_badge(ui, e));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(size_label(e.size_bytes));
        });
        if registry.is_downloaded(&e.id) {
            ui.horizontal(|ui| {
                ui.weak("Downloaded");
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
        let shown = self.license_of.as_deref() == Some(e.id.as_str());
        let text = if shown {
            "Hide license"
        } else {
            "Show license"
        };
        if ui.link(text).clicked() {
            self.license_of = (!shown).then(|| e.id.clone());
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
    fn inpainting_models_are_labelled_extend_not_by_scale() {
        let dir = tempfile::tempdir().unwrap();
        let registry = Registry::with_dir(dir.path().to_path_buf());
        let lama = registry.get("lama-512").unwrap();
        assert_eq!(kind_label(lama), "Extend · LaMa");
        let up = registry.upscalers().next().unwrap();
        assert_eq!(kind_label(up), format!("{}× {}", up.scale, up.arch));
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
