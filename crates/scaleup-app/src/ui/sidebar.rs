//! The left column: input, model, output options and the Upscale / Stop button.

use std::path::{Path, PathBuf};

use eframe::egui;
use scaleup_core::registry::SOURCE_HUB;
use scaleup_core::{DeviceInfo, DeviceKind, Fill, ModelEntry, Registry};

use crate::state::{AppState, ExtendMode, Mode, Settings};

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

/// The Start button's text: it says when the run will download a model first.
pub fn start_label(s: &Settings, registry: &Registry) -> &'static str {
    let missing = |id: &str| registry.get(id).is_some() && !registry.is_downloaded(id);
    let ai = s.extend_enabled && Fill::parse(&s.extend_fill) == Ok(Fill::Ai);
    if missing(&s.model_id) || (ai && missing(&s.extend_model)) {
        "Download and start"
    } else {
        "Upscale"
    }
}

/// The fill choices of the Extend section: label, and the settings value it selects.
const FILLS: [(&str, &str); 5] = [
    ("AI (inpainting)", "ai"),
    ("Mirror", "mirror"),
    ("Edge pixels", "edge"),
    ("Colour", "#ffffff"),
    ("Transparent", "transparent"),
];

fn fill_label(fill: &str) -> &'static str {
    match Fill::parse(fill) {
        Ok(Fill::Color(_)) => "Colour",
        _ => FILLS
            .iter()
            .find(|(_, v)| v.eq_ignore_ascii_case(fill.trim()))
            .map_or("?", |(label, _)| label),
    }
}

fn side_value(ui: &mut egui::Ui, v: &mut f32, mode: ExtendMode) {
    let drag = match mode {
        ExtendMode::Percent => egui::DragValue::new(v)
            .range(0.0..=400.0)
            .speed(0.5)
            .max_decimals(1)
            .suffix(" %"),
        _ => egui::DragValue::new(v)
            .range(0.0..=16384.0)
            .speed(1.0)
            .fixed_decimals(0)
            .suffix(" px"),
    };
    ui.add(drag);
}

const ASPECTS: [(u32, u32); 5] = [(1, 1), (4, 3), (3, 2), (16, 9), (9, 16)];

fn extend_section(ui: &mut egui::Ui, st: &mut AppState, registry: &Registry) {
    let source = st.source_size;
    let s = &mut st.settings;
    egui::CollapsingHeader::new("Extend canvas")
        .default_open(s.extend_enabled)
        .show(ui, |ui| {
            ui.checkbox(&mut s.extend_enabled, "Add canvas before upscaling");
            ui.add_enabled_ui(s.extend_enabled, |ui| {
                ui.horizontal(|ui| {
                    ui.radio_value(&mut s.extend_mode, ExtendMode::Pixels, "Pixels");
                    ui.radio_value(&mut s.extend_mode, ExtendMode::Percent, "Percent");
                    ui.radio_value(&mut s.extend_mode, ExtendMode::Aspect, "Aspect");
                });
                if s.extend_mode == ExtendMode::Aspect {
                    ui.horizontal_wrapped(|ui| {
                        for (w, h) in ASPECTS {
                            ui.selectable_value(&mut s.extend_aspect, (w, h), format!("{w}:{h}"));
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Custom");
                        ui.add(egui::DragValue::new(&mut s.extend_aspect.0).range(1..=100));
                        ui.label(":");
                        ui.add(egui::DragValue::new(&mut s.extend_aspect.1).range(1..=100));
                    });
                } else {
                    ui.checkbox(&mut s.extend_linked, "Same on every side");
                    let mode = s.extend_mode;
                    if s.extend_linked {
                        ui.horizontal(|ui| {
                            ui.label("Each side");
                            side_value(ui, &mut s.extend_left, mode);
                        });
                    } else {
                        egui::Grid::new("extend_sides").show(ui, |ui| {
                            ui.label("Left");
                            side_value(ui, &mut s.extend_left, mode);
                            ui.label("Top");
                            side_value(ui, &mut s.extend_top, mode);
                            ui.end_row();
                            ui.label("Right");
                            side_value(ui, &mut s.extend_right, mode);
                            ui.label("Bottom");
                            side_value(ui, &mut s.extend_bottom, mode);
                            ui.end_row();
                        });
                    }
                }
                ui.horizontal(|ui| {
                    ui.label("Fill");
                    egui::ComboBox::from_id_salt("extend_fill")
                        .selected_text(fill_label(&s.extend_fill))
                        .show_ui(ui, |ui| {
                            let current = fill_label(&s.extend_fill);
                            for (label, value) in FILLS {
                                if ui.selectable_label(current == label, label).clicked()
                                    && current != label
                                {
                                    s.extend_fill = value.into();
                                }
                            }
                        });
                    if let Ok(Fill::Color(mut rgb)) = Fill::parse(&s.extend_fill)
                        && ui.color_edit_button_srgb(&mut rgb).changed()
                    {
                        let [r, g, b] = rgb;
                        s.extend_fill = format!("#{r:02x}{g:02x}{b:02x}");
                    }
                });
                if Fill::parse(&s.extend_fill) == Ok(Fill::Ai) {
                    inpainter_picker(ui, &mut s.extend_model, registry);
                }
            });
            match s.extend_summary(source) {
                Some(Ok(text)) => {
                    ui.label(text);
                }
                Some(Err(e)) => {
                    ui.colored_label(ui.visuals().error_fg_color, e);
                }
                None => {}
            }
        });
}

fn inpainter_picker(ui: &mut egui::Ui, selected: &mut String, registry: &Registry) {
    let entry = registry.get(selected);
    ui.horizontal(|ui| {
        ui.label("Model");
        egui::ComboBox::from_id_salt("extend_model")
            .width(ui.available_width())
            .selected_text(entry.map_or(selected.as_str(), |e| e.name.as_str()))
            .show_ui(ui, |ui| {
                for e in registry.inpainters() {
                    ui.selectable_value(selected, e.id.clone(), &e.name)
                        .on_hover_text(&e.notes);
                }
            });
    });
    ui.small("Suits modest extensions of soft or plain backgrounds.");
    if let Some(e) = entry
        && !registry.is_downloaded(&e.id)
    {
        ui.small(format!(
            "Downloads {:.0} MB on first use.",
            e.size_bytes as f64 / (1024.0 * 1024.0)
        ));
    }
}

pub fn show(ui: &mut egui::Ui, st: &mut AppState, registry: &Registry) -> SidebarOutput {
    let mut out = SidebarOutput::default();
    let running = st.status.is_running();
    let entries: Vec<ModelEntry> = registry.upscalers().cloned().collect();
    let invalid = match st.settings.extend_summary(st.source_size) {
        Some(Err(e)) => Some(e),
        _ => None,
    };

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
                let label = start_label(&st.settings, registry);
                let text = egui::RichText::new(label).size(18.0).strong();
                let button = egui::Button::new(text);
                let resp = ui.add_enabled(
                    st.mode.is_some() && invalid.is_none(),
                    |ui: &mut egui::Ui| ui.add_sized(size, button),
                );
                if let Some(e) = &invalid {
                    resp.clone()
                        .on_disabled_hover_text(format!("Extend canvas: {e}"));
                }
                if resp.clicked() {
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
                model_picker(ui, st, &entries);
                ui.separator();
                output_options(ui, st);
                ui.separator();
                extend_section(ui, st, registry);
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
        // The picker lists the upscalers, as the app passes them.
        let upscalers: Vec<ModelEntry> = reg.upscalers().cloned().collect();
        let [free, nc, hub] = groups(&upscalers);
        let ids = |g: &[&ModelEntry]| g.iter().map(|e| e.id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&hub), ["h-free", "h-nc"]);
        assert!(free.iter().chain(&nc).all(|e| e.source != SOURCE_HUB));
        assert!(free.iter().all(|e| e.commercial) && nc.iter().all(|e| !e.commercial));
        assert_eq!(free.len() + nc.len(), upscalers.len() - 2);
        assert!(reg.inpainters().count() > 0);
        assert!(
            free.iter()
                .chain(&nc)
                .all(|e| e.kind == scaleup_core::ModelKind::Upscale)
        );
    }

    #[test]
    fn start_says_when_a_model_will_be_downloaded_first() {
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::with_dir(dir.path().to_path_buf());
        let mut s = Settings::default();
        assert_eq!(start_label(&s, &reg), "Download and start");
        s.model_id = dir.path().join("mine.onnx").to_string_lossy().into_owned();
        assert_eq!(start_label(&s, &reg), "Upscale");
        s.extend_enabled = true;
        assert_eq!(s.extend_fill, "ai");
        assert_eq!(start_label(&s, &reg), "Download and start");
        s.extend_fill = "mirror".into();
        assert_eq!(start_label(&s, &reg), "Upscale");
        s.extend_fill = "ai".into();
        std::fs::write(reg.local_path("lama-512"), b"x").unwrap();
        assert_eq!(start_label(&s, &reg), "Upscale");
        s.extend_model = "migan-512-places2".into();
        assert_eq!(start_label(&s, &reg), "Download and start");
    }

    #[test]
    fn fill_labels_name_every_choice() {
        for (label, value) in FILLS {
            assert_eq!(fill_label(value), label);
        }
        assert_eq!(fill_label("#102030"), "Colour");
        assert_eq!(fill_label("AI"), "AI (inpainting)");
    }

    /// Draws the sidebar with the Extend section open and the compare view of an extended result
    /// headless (egui_kittest + wgpu) into `$SCALEUP_RENDER_DIR/*.png` for a person to look at.
    /// The result is real: the fixture photo extended by 128 px on each side with the mirror fill
    /// and upscaled 4× through the nearest-neighbour fixture model on the CPU. Run with
    /// `-- --ignored`; skipped when the variable is unset.
    #[test]
    #[ignore = "writes PNG files for a manual look; needs SCALEUP_RENDER_DIR and a GPU adapter"]
    fn extend_renders() {
        use crate::ui::compare::CompareView;
        use crate::worker::Which;
        use scaleup_core::{
            CancelToken, DevicePreference, ExtendOptions, FileJob, MarginSpec, Margins,
            OutputFormat, Upscaler,
        };

        let Some(out) = std::env::var_os("SCALEUP_RENDER_DIR").map(PathBuf::from) else {
            eprintln!("skipped: SCALEUP_RENDER_DIR is not set");
            return;
        };
        std::fs::create_dir_all(&out).unwrap();
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
        let photo = root.join("images/photo_like_256x192.jpg");
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::with_dir(dir.path().to_path_buf());

        let settings = Settings {
            extend_enabled: true,
            extend_left: 128.0,
            ..Settings::default()
        };
        let mut st = AppState {
            settings,
            mode: Some(Mode::Single(photo.clone())),
            source_size: Some([256, 192]),
            ..AppState::default()
        };
        let mut harness = egui_kittest::Harness::builder()
            .with_size([300.0, 760.0])
            .wgpu()
            .build_ui_state(
                |ui, (st, reg): &mut (AppState, Registry)| {
                    show(ui, st, reg);
                },
                (st.clone(), reg.clone()),
            );
        for _ in 0..4 {
            harness.step();
        }
        let file = out.join("extend-sidebar.png");
        harness.render().expect("render").save(&file).unwrap();
        eprintln!("wrote {}", file.display());

        // The over-the-limit case: the reason in red and Start disabled.
        st.settings.extend_left = 2000.0;
        let mut harness = egui_kittest::Harness::builder()
            .with_size([300.0, 760.0])
            .wgpu()
            .build_ui_state(
                |ui, (st, reg): &mut (AppState, Registry)| {
                    show(ui, st, reg);
                },
                (st, reg),
            );
        for _ in 0..4 {
            harness.step();
        }
        let file = out.join("extend-sidebar-invalid.png");
        harness.render().expect("render").save(&file).unwrap();
        eprintln!("wrote {}", file.display());

        let up = Upscaler::new(
            DevicePreference::Cpu,
            Registry::with_dir(dir.path().to_path_buf()),
        )
        .unwrap();
        let model = up
            .load(
                root.join("nearest2x.onnx").to_str().unwrap(),
                &mut |_, _| {},
            )
            .unwrap();
        let spec = MarginSpec::Pixels(Margins {
            left: 128,
            top: 128,
            right: 128,
            bottom: 128,
        });
        let job = FileJob {
            input: photo.clone(),
            output: dir.path().join("extended.png"),
            format: OutputFormat::Png,
            quality: 90,
            overwrite: true,
            double: false,
            extend: Some(ExtendOptions {
                margins: spec.clone(),
                fill: Fill::Mirror,
            }),
        };
        let opts = scaleup_core::UpscaleOptions {
            output_scale: Some(4),
            ..Default::default()
        };
        let result = up
            .run_file(
                Some(model.as_ref()),
                None,
                &job,
                &opts,
                &mut |_| {},
                &CancelToken::new(),
            )
            .unwrap();
        let margins = spec.resolve(256, 192).unwrap();
        let color = |p: &Path| {
            let img = scaleup_core::load_image(p).unwrap();
            let size = [img.width(), img.height()];
            let dims = [size[0] as usize, size[1] as usize];
            (
                egui::ColorImage::from_rgba_unmultiplied(dims, img.as_raw()),
                size,
            )
        };
        let (before, before_size) = color(&photo);
        let (after, after_size) = color(&result);
        assert_eq!(after_size, [(256 + 256) * 4, (192 + 256) * 4]);
        let mut harness = egui_kittest::Harness::builder()
            .with_size([900.0, 700.0])
            .wgpu()
            .build_ui_state(
                |ui, view: &mut CompareView| view.show(ui),
                CompareView::default(),
            );
        let ctx = harness.ctx.clone();
        harness
            .state_mut()
            .set(&ctx, Which::Before, before, before_size, Margins::default());
        harness
            .state_mut()
            .set(&ctx, Which::After, after, after_size, margins);
        for _ in 0..4 {
            harness.step();
        }
        let file = out.join("extend-compare.png");
        harness.render().expect("render").save(&file).unwrap();
        eprintln!("wrote {}", file.display());
    }

    #[test]
    fn elide_middle_keeps_both_ends() {
        assert_eq!(elide_middle("short", 10), "short");
        assert_eq!(elide_middle("abcdefghij", 5), "ab…ij");
    }
}
