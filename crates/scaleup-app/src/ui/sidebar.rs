//! The left column: input, model, output options and the Upscale / Stop button.

use std::path::{Path, PathBuf};

use eframe::egui;
use scaleup_core::registry::SOURCE_HUB;
use scaleup_core::{DeviceInfo, DeviceKind, ModelEntry};

use crate::state::{AppState, Mode};

/// What the user asked for this frame.
#[derive(Default)]
pub struct SidebarOutput {
    pub picked: Option<PathBuf>,
    pub start: bool,
    pub stop: bool,
    pub device_changed: bool,
    pub open_models: bool,
    pub open_hub: bool,
}

/// Text for a path that must fit `max` characters: keeps both ends.
pub fn elide_middle(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max || max < 3 {
        return s.to_string();
    }
    let tail = (max - 1) / 2;
    let head = max - 1 - tail;
    chars[..head]
        .iter()
        .chain(std::iter::once(&'…'))
        .chain(&chars[chars.len() - tail..])
        .collect()
}

/// The native "open image" dialog.
pub fn pick_image() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter(
            "Images",
            &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        )
        .pick_file()
}

fn nc_badge(ui: &mut egui::Ui) {
    ui.label(
        egui::RichText::new(" NC ")
            .small()
            .strong()
            .color(egui::Color32::WHITE)
            .background_color(egui::Color32::from_rgb(190, 40, 40)),
    );
}

fn model_row(ui: &mut egui::Ui, selected: &mut String, entry: &ModelEntry) {
    ui.horizontal(|ui| {
        if !entry.commercial {
            nc_badge(ui);
        }
        ui.selectable_value(selected, entry.id.clone(), &entry.name);
        ui.weak(&entry.license);
    });
}

/// The name shown for the selected model: catalogue name, or a custom file name.
fn model_title(model_id: &str, entries: &[ModelEntry]) -> String {
    match entries.iter().find(|e| e.id == model_id) {
        Some(e) => e.name.clone(),
        None => Path::new(model_id)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| model_id.to_string()),
    }
}

/// The model list in its three groups: free for any use, non-commercial (both built in or
/// imported), and installed from OpenModelDB.
fn groups(entries: &[ModelEntry]) -> [Vec<&ModelEntry>; 3] {
    let mut out = [Vec::new(), Vec::new(), Vec::new()];
    for e in entries {
        let group = if e.source == SOURCE_HUB {
            2
        } else if e.commercial {
            0
        } else {
            1
        };
        out[group].push(e);
    }
    out
}

fn model_picker(ui: &mut egui::Ui, st: &mut AppState, entries: &[ModelEntry]) {
    let s = &mut st.settings;
    egui::ComboBox::from_id_salt("model")
        .width(ui.available_width())
        .selected_text(model_title(&s.model_id, entries))
        .height(420.0)
        .show_ui(ui, |ui| {
            ui.set_min_width(330.0);
            let [free, nc_models, hub] = groups(entries);
            ui.weak("Free for any use");
            for e in free {
                model_row(ui, &mut s.model_id, e);
            }
            ui.separator();
            let nc = egui::CollapsingHeader::new("Non-commercial")
                .open(Some(s.show_nc_models))
                .show(ui, |ui| {
                    ui.weak("These licenses forbid commercial use of the results.");
                    for e in nc_models {
                        model_row(ui, &mut s.model_id, e);
                    }
                });
            if nc.header_response.clicked() {
                s.show_nc_models = !s.show_nc_models;
            }
            if !hub.is_empty() {
                ui.separator();
                ui.weak("From OpenModelDB");
                for e in hub {
                    model_row(ui, &mut s.model_id, e);
                }
            }
        });
    ui.horizontal(|ui| {
        if ui.small_button("Custom model file…").clicked()
            && let Some(p) = rfd::FileDialog::new()
                .add_filter("Model", &["onnx", "param"])
                .pick_file()
        {
            s.model_id = p.to_string_lossy().into_owned();
        }
    });
    if let Some(e) = entries.iter().find(|e| e.id == s.model_id)
        && !e.commercial
    {
        ui.horizontal(|ui| {
            nc_badge(ui);
            ui.small(format!("{}: not for commercial use", e.license));
        });
    }
}

fn output_options(ui: &mut egui::Ui, st: &mut AppState) {
    let s = &mut st.settings;
    ui.add_enabled_ui(!s.use_width, |ui| {
        ui.horizontal(|ui| {
            ui.label("Scale");
            for v in [2, 3, 4] {
                ui.radio_value(&mut s.scale, v, format!("{v}×"));
            }
        });
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut s.use_width, "Custom width");
        ui.add_enabled(
            s.use_width,
            egui::DragValue::new(&mut s.width)
                .range(16..=32768)
                .suffix(" px"),
        );
    });
    ui.horizontal(|ui| {
        ui.label("Format");
        egui::ComboBox::from_id_salt("format")
            .selected_text(s.format.as_str())
            .show_ui(ui, |ui| {
                for f in ["png", "jpg", "webp"] {
                    ui.selectable_value(&mut s.format, f.to_string(), f);
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label("Quality");
        ui.add(egui::Slider::new(&mut s.quality, 1..=100))
            .on_hover_text("JPEG and WebP quality. For PNG, higher means smaller and slower.");
    });
}

fn gpu_label(gpu: &str, devices: &[DeviceInfo]) -> String {
    match gpu {
        "auto" => "Auto".into(),
        "cpu" => "CPU".into(),
        other => devices
            .iter()
            .find(|d| d.kind.is_gpu() && gpu_value(d) == other)
            .map_or_else(|| format!("GPU {other}"), |d| d.name.clone()),
    }
}

/// The settings value that selects device `d`.
fn gpu_value(d: &DeviceInfo) -> String {
    if d.kind == DeviceKind::Vulkan {
        format!("vk:{}", d.index)
    } else {
        d.index.to_string()
    }
}

fn advanced(ui: &mut egui::Ui, st: &mut AppState, out: &mut SidebarOutput) {
    let devices = st.devices.clone();
    let s = &mut st.settings;
    egui::CollapsingHeader::new("Advanced").show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("Tile size");
            egui::ComboBox::from_id_salt("tile")
                .selected_text(if s.tile == 0 {
                    "Auto".to_string()
                } else {
                    s.tile.to_string()
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut s.tile, 0, "Auto");
                    for t in [128, 256, 512, 1024] {
                        ui.selectable_value(&mut s.tile, t, t.to_string());
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label("GPU");
            let before = s.gpu.clone();
            egui::ComboBox::from_id_salt("gpu")
                .width(ui.available_width())
                .selected_text(gpu_label(&s.gpu, &devices))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut s.gpu, "auto".to_string(), "Auto");
                    ui.selectable_value(&mut s.gpu, "cpu".to_string(), "CPU");
                    for d in devices.iter().filter(|d| d.kind.is_gpu()) {
                        ui.selectable_value(&mut s.gpu, gpu_value(d), &d.name);
                    }
                });
            out.device_changed = s.gpu != before;
        });
        ui.checkbox(&mut s.tta, "Test-time augmentation (8× slower)");
        ui.checkbox(&mut s.double, "Double upscale");
        ui.checkbox(&mut s.overwrite, "Overwrite existing files");
        ui.label("Output folder");
        ui.horizontal(|ui| {
            let label = s
                .output_dir
                .as_ref()
                .map_or("Same as input".to_string(), |d| {
                    elide_middle(&d.to_string_lossy(), 22)
                });
            if ui.button(label).clicked()
                && let Some(d) = rfd::FileDialog::new().pick_folder()
            {
                s.output_dir = Some(d);
            }
            if s.output_dir.is_some() && ui.small_button("Same as input").clicked() {
                s.output_dir = None;
            }
        });
    });
}

pub fn show(ui: &mut egui::Ui, st: &mut AppState, entries: &[ModelEntry]) -> SidebarOutput {
    let mut out = SidebarOutput::default();
    let running = st.status.is_running();

    egui::Panel::bottom("sidebar_actions")
        .show_separator_line(false)
        .show_inside(ui, |ui| {
            ui.add_space(8.0);
            let size = egui::vec2(ui.available_width(), 40.0);
            if running {
                let text = if st.stopping { "Stopping…" } else { "Stop" };
                let enabled = !st.stopping;
                if ui
                    .add_enabled(enabled, |ui: &mut egui::Ui| {
                        ui.add_sized(size, egui::Button::new(text))
                    })
                    .clicked()
                {
                    out.stop = true;
                }
            } else {
                let text = egui::RichText::new("Upscale").size(18.0).strong();
                let button = egui::Button::new(text);
                if ui
                    .add_enabled(st.mode.is_some(), |ui: &mut egui::Ui| {
                        ui.add_sized(size, button)
                    })
                    .clicked()
                {
                    out.start = true;
                }
            }
            ui.add_space(8.0);
        });

    egui::CentralPanel::default().show_inside(ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.add_enabled_ui(!running, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Select Image").clicked()
                        && let Some(p) = pick_image()
                    {
                        out.picked = Some(p);
                    }
                    if ui.button("Select Folder").clicked()
                        && let Some(p) = rfd::FileDialog::new().pick_folder()
                    {
                        out.picked = Some(p);
                    }
                });
                match &st.mode {
                    Some(Mode::Single(p)) | Some(Mode::Batch(p)) => {
                        let full = p.to_string_lossy();
                        ui.label(elide_middle(&full, 38))
                            .on_hover_text(full.as_ref());
                    }
                    None => {
                        ui.weak("No image selected");
                    }
                }
                ui.separator();
                ui.horizontal(|ui| {
                    ui.strong("Model");
                    if ui.small_button("Manage…").clicked() {
                        out.open_models = true;
                    }
                    if ui.small_button("Browse OpenModelDB…").clicked() {
                        out.open_hub = true;
                    }
                });
                model_picker(ui, st, entries);
                ui.separator();
                output_options(ui, st);
                ui.separator();
                advanced(ui, st, &mut out);
            });
        });
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_value_uses_the_vk_prefix_for_vulkan_devices() {
        let d = |kind, index| DeviceInfo {
            kind,
            index,
            name: "x".into(),
        };
        assert_eq!(gpu_value(&d(DeviceKind::Vulkan, 1)), "vk:1");
        assert_eq!(gpu_value(&d(DeviceKind::DirectMl, 0)), "0");
    }

    #[test]
    fn hub_models_get_their_own_group() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("hub").join("h").join("m.onnx");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"x").unwrap();
        let hub_model = |id: &str, license: &str| scaleup_core::hub::InstalledModel {
            id: id.into(),
            name: id.into(),
            arch: "span".into(),
            scale: 4,
            license: Some(license.into()),
            format: "onnx".into(),
            file: file.clone(),
            sha256: None,
            verified: true,
            source_url: String::new(),
            installed_at: String::new(),
        };
        scaleup_core::hub::Installed {
            models: vec![
                hub_model("h-free", "MIT"),
                hub_model("h-nc", "CC-BY-NC-4.0"),
            ],
        }
        .save(dir.path())
        .unwrap();
        let reg = scaleup_core::Registry::with_dir(dir.path().to_path_buf());
        let [free, nc, hub] = groups(reg.entries());
        let ids = |g: &[&ModelEntry]| g.iter().map(|e| e.id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&hub), ["h-free", "h-nc"]);
        assert!(free.iter().chain(&nc).all(|e| e.source != SOURCE_HUB));
        assert!(free.iter().all(|e| e.commercial) && nc.iter().all(|e| !e.commercial));
        assert_eq!(free.len() + nc.len(), reg.entries().len() - 2);
    }

    #[test]
    fn elide_middle_keeps_both_ends() {
        assert_eq!(elide_middle("short", 10), "short");
        assert_eq!(elide_middle("abcdefghij", 5), "ab…ij");
    }
}
