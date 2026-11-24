//! The left column: input, model, output options and the Upscale / Stop button.

use std::path::{Path, PathBuf};

use eframe::egui;
use scaleup_core::registry::SOURCE_HUB;
use scaleup_core::{DeviceInfo, DeviceKind, Fill, ModelEntry, Registry};

use crate::state::{AppState, ExtendMode, FillChoice, Mode, Settings, TrySeveral};
use crate::ui::theme;

/// What the user asked for this frame.
#[derive(Default)]
pub struct SidebarOutput {
    pub picked: Option<PathBuf>,
    pub start: bool,
    /// The fills the Try several button runs, in list order.
    pub try_fills: Option<Vec<FillChoice>>,
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
    theme::badge(ui, "NC", theme::DANGER).on_hover_text("Not for commercial use");
}

/// The label of a control row: every row's control starts at the same place.
fn row_label(ui: &mut egui::Ui, text: &str) -> egui::Response {
    ui.add_sized(
        [LABEL_WIDTH, ui.spacing().interact_size.y],
        egui::Label::new(text).halign(egui::Align::Min),
    )
}

const LABEL_WIDTH: f32 = 56.0;
/// Space between two groups of controls; rows inside a group sit closer.
const SECTION_GAP: f32 = 10.0;

/// The name shown for an output format.
fn format_label(format: &str) -> &'static str {
    match format {
        "jpg" => "JPEG",
        "webp" => "WebP",
        _ => "PNG",
    }
}

/// One model of a picker: the NC badge when it is not for commercial use, the name, the licence.
/// Returns the response of the name, for a hover text.
fn model_row(ui: &mut egui::Ui, selected: &mut String, entry: &ModelEntry) -> egui::Response {
    ui.horizontal(|ui| {
        if !entry.commercial {
            nc_badge(ui);
        }
        let name = ui.selectable_value(selected, entry.id.clone(), &entry.name);
        ui.weak(&entry.license);
        name
    })
    .inner
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

/// A combo box as wide as the space left in the row.
fn fill_width(ui: &egui::Ui) -> f32 {
    ui.available_width() - 2.0 * ui.spacing().button_padding.x
}

fn model_picker(
    ui: &mut egui::Ui,
    st: &mut AppState,
    entries: &[ModelEntry],
    out: &mut SidebarOutput,
) {
    let s = &mut st.settings;
    ui.add_space(4.0);
    let title = ui
        .horizontal(|ui| {
            let title = ui.strong("Model");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .link("Manage…")
                    .on_hover_text("Download or delete the models of the catalogue")
                    .clicked()
                {
                    out.open_models = true;
                }
            });
            title
        })
        .inner;
    egui::ComboBox::from_id_salt("model")
        .width(fill_width(ui))
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
        })
        .response
        .labelled_by(title.id);
    ui.horizontal(|ui| {
        if ui.link("Browse OpenModelDB…").clicked() {
            out.open_hub = true;
        }
        if ui
            .link("Custom file…")
            .on_hover_text("Use a model file from this computer (.onnx or .param)")
            .clicked()
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
    theme::section(ui, "Output");
    ui.add_enabled_ui(!s.use_width, |ui| {
        ui.horizontal(|ui| {
            row_label(ui, "Scale");
            for v in [2, 3, 4] {
                ui.radio_value(&mut s.scale, Some(v), format!("{v}×"));
            }
            ui.radio_value(&mut s.scale, None, "None")
                .on_hover_text("Only extend the canvas: nothing is upscaled.");
        });
    });
    ui.horizontal(|ui| {
        row_label(ui, "Width");
        ui.checkbox(&mut s.use_width, "Custom");
        let width = egui::DragValue::new(&mut s.width)
            .range(16..=32768)
            .suffix(" px");
        ui.add_enabled(s.use_width, width)
            .on_hover_text("Output width in pixels; the height follows the image.")
            .on_disabled_hover_text("Tick Custom to set the output width instead of a scale.");
    });
    ui.horizontal(|ui| {
        let label = row_label(ui, "Format");
        egui::ComboBox::from_id_salt("format")
            .selected_text(format_label(&s.format))
            .show_ui(ui, |ui| {
                for f in ["png", "jpg", "webp"] {
                    ui.selectable_value(&mut s.format, f.to_string(), format_label(f));
                }
            })
            .response
            .labelled_by(label.id);
    });
    ui.horizontal(|ui| {
        let label = row_label(ui, "Quality");
        ui.add(egui::Slider::new(&mut s.quality, 1..=100))
            .labelled_by(label.id)
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
    egui::CollapsingHeader::new(egui::RichText::new("Advanced").strong()).show(ui, |ui| {
        ui.horizontal(|ui| {
            let label = row_label(ui, "Tile size");
            egui::ComboBox::from_id_salt("tile")
                .selected_text(if s.tile == 0 {
                    "Auto".to_string()
                } else {
                    format!("{} px", s.tile)
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut s.tile, 0, "Auto");
                    for t in [128, 256, 512, 1024] {
                        ui.selectable_value(&mut s.tile, t, format!("{t} px"));
                    }
                })
                .response
                .labelled_by(label.id)
                .on_hover_text("Smaller tiles need less GPU memory and run slower.");
        });
        ui.horizontal(|ui| {
            let label = row_label(ui, "Device");
            let before = s.gpu.clone();
            egui::ComboBox::from_id_salt("gpu")
                .width(fill_width(ui))
                .selected_text(gpu_label(&s.gpu, &devices))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut s.gpu, "auto".to_string(), "Auto");
                    ui.selectable_value(&mut s.gpu, "cpu".to_string(), "CPU");
                    for d in devices.iter().filter(|d| d.kind.is_gpu()) {
                        ui.selectable_value(&mut s.gpu, gpu_value(d), &d.name);
                    }
                })
                .response
                .labelled_by(label.id);
            out.device_changed = s.gpu != before;
        });
        ui.checkbox(&mut s.tta, "Test-time augmentation (8× slower)")
            .on_hover_text("Runs the model on eight flipped and rotated copies and averages them.");
        ui.checkbox(&mut s.double, "Double upscale")
            .on_hover_text("Runs the model twice: a 4× model gives 16×.");
        ui.checkbox(&mut s.overwrite, "Overwrite existing files");
        ui.horizontal(|ui| {
            row_label(ui, "Save to");
            if ui.button("Choose…").clicked()
                && let Some(d) = rfd::FileDialog::new().pick_folder()
            {
                s.output_dir = Some(d);
            }
            if s.output_dir.is_some()
                && ui
                    .link("Reset")
                    .on_hover_text("Save next to the input again")
                    .clicked()
            {
                s.output_dir = None;
            }
        });
        match &s.output_dir {
            Some(d) => {
                let full = d.to_string_lossy();
                ui.label(elide_middle(&full, 36))
                    .on_hover_text(full.as_ref());
            }
            None => {
                ui.weak("The folder of the input");
            }
        }
    });
}

/// The Start button's text: `Extend` when nothing is upscaled, and it says when the run will
/// download a model first.
pub fn start_label(s: &Settings, registry: &Registry) -> &'static str {
    let missing = |id: &str| registry.get(id).is_some() && !registry.is_downloaded(id);
    let ai = s.extend_enabled && Fill::parse(&s.extend_fill) == Ok(Fill::Ai);
    if !s.upscales() {
        return if ai && missing(&s.extend_model) {
            "Download and extend"
        } else {
            "Extend"
        };
    }
    if missing(&s.model_id) || (ai && missing(&s.extend_model)) {
        "Download and start"
    } else {
        "Upscale"
    }
}

/// One row of the Try several list.
#[derive(Clone, Debug, PartialEq)]
pub struct FillOffer {
    pub choice: FillChoice,
    pub name: String,
    /// The catalogue entry of an AI fill: its licence, what it suits and its size.
    pub entry: Option<ModelEntry>,
}

/// The fills Try several offers: every inpainting model of the catalogue, then Mirror and Edge
/// pixels. The list has no colour or transparent fill: they have one look only.
pub fn offered_fills(registry: &Registry) -> Vec<FillOffer> {
    let models = registry.inpainters().map(|e| FillOffer {
        choice: FillChoice::Ai(e.id.clone()),
        name: e.name.clone(),
        entry: Some(e.clone()),
    });
    let plain = [Fill::Mirror, Fill::Edge].map(|fill| {
        let choice = FillChoice::Plain(fill);
        FillOffer {
            name: choice.title(registry),
            choice,
            entry: None,
        }
    });
    models.chain(plain).collect()
}

/// The Try several button's text: `Try 3 fills`, and it says when a ticked model, or the
/// upscaling model, is downloaded first, as the Start button does.
pub fn try_label(s: &Settings, fills: &[FillChoice], registry: &Registry) -> String {
    let missing = |id: &str| registry.get(id).is_some() && !registry.is_downloaded(id);
    let download = fills
        .iter()
        .any(|f| matches!(f, FillChoice::Ai(id) if missing(id)))
        || (s.upscales() && missing(&s.model_id));
    let n = fills.len();
    let fills = match n {
        0 => "several fills".into(),
        1 => "1 fill".into(),
        n => format!("{n} fills"),
    };
    if download {
        format!("Download and try {fills}")
    } else {
        format!("Try {fills}")
    }
}

/// Why Try several cannot run, `None` when it can. `ticked` is how many fills are ticked.
pub fn try_blocker(
    s: &Settings,
    mode: Option<&Mode>,
    source: Option<[u32; 2]>,
    running: bool,
    ticked: usize,
) -> Option<String> {
    if running {
        return Some("A run is in progress".into());
    }
    if !matches!(mode, Some(Mode::Single(_))) {
        return Some("Try several needs a single image".into());
    }
    if !s.extend_enabled {
        return Some("Turn on Extend canvas to try fills".into());
    }
    if let Some(why) = s.start_blocker(source) {
        return Some(why);
    }
    (ticked < 2).then(|| "Tick at least two fills".into())
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

fn side_value(ui: &mut egui::Ui, name: &str, v: &mut f32, mode: ExtendMode) {
    let label = ui.label(name);
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
    ui.add(drag).labelled_by(label.id);
}

const ASPECTS: [(u32, u32); 5] = [(1, 1), (4, 3), (3, 2), (16, 9), (9, 16)];

fn extend_section(
    ui: &mut egui::Ui,
    st: &mut AppState,
    registry: &Registry,
    out: &mut SidebarOutput,
) {
    let source = st.source_size;
    let running = st.status.is_running();
    let mode = st.mode.clone();
    let try_several_state = &mut st.try_several;
    let s = &mut st.settings;
    egui::CollapsingHeader::new(egui::RichText::new("Extend canvas").strong())
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
                        let label = row_label(ui, "Custom");
                        ui.add(egui::DragValue::new(&mut s.extend_aspect.0).range(1..=100))
                            .labelled_by(label.id);
                        ui.label(":");
                        ui.add(egui::DragValue::new(&mut s.extend_aspect.1).range(1..=100))
                            .labelled_by(label.id);
                    });
                } else {
                    ui.checkbox(&mut s.extend_linked, "Same on every side");
                    let mode = s.extend_mode;
                    if s.extend_linked {
                        ui.horizontal(|ui| {
                            side_value(ui, "Each side", &mut s.extend_left, mode);
                        });
                    } else {
                        egui::Grid::new("extend_sides").show(ui, |ui| {
                            side_value(ui, "Left", &mut s.extend_left, mode);
                            side_value(ui, "Top", &mut s.extend_top, mode);
                            ui.end_row();
                            side_value(ui, "Right", &mut s.extend_right, mode);
                            side_value(ui, "Bottom", &mut s.extend_bottom, mode);
                            ui.end_row();
                        });
                    }
                }
                ui.horizontal(|ui| {
                    let label = row_label(ui, "Fill");
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
                        })
                        .response
                        .labelled_by(label.id);
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
                try_several(
                    ui,
                    s,
                    try_several_state,
                    registry,
                    TryContext {
                        mode: mode.as_ref(),
                        source,
                        running,
                    },
                    out,
                );
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
        let label = row_label(ui, "Model");
        egui::ComboBox::from_id_salt("extend_model")
            .width(fill_width(ui))
            .selected_text(entry.map_or(selected.as_str(), |e| e.name.as_str()))
            .show_ui(ui, |ui| {
                for e in registry.inpainters() {
                    model_row(ui, selected, e).on_hover_text(&e.notes);
                }
            })
            .response
            .labelled_by(label.id);
    });
    if let Some(e) = entry {
        if !e.good_for.is_empty() {
            ui.weak(&e.good_for);
        }
        if !registry.is_downloaded(&e.id) {
            ui.weak(download_note(e));
        }
    }
}

/// `Downloads 196 MB on first use.`
fn download_note(e: &ModelEntry) -> String {
    format!(
        "Downloads {:.0} MB on first use.",
        e.size_bytes as f64 / (1024.0 * 1024.0)
    )
}

/// What the Try several button needs to know besides the settings.
struct TryContext<'a> {
    mode: Option<&'a Mode>,
    source: Option<[u32; 2]>,
    running: bool,
}

/// The Try several list, a collapsing area under the Fill rows (a popup would cover the controls
/// it belongs to and close on every stray click): one row with a checkbox per fill, and the button
/// that runs the ticked ones on the same extension.
fn try_several(
    ui: &mut egui::Ui,
    s: &Settings,
    state: &mut TrySeveral,
    registry: &Registry,
    cx: TryContext<'_>,
    out: &mut SidebarOutput,
) {
    let offers = offered_fills(registry);
    let open = state.open;
    let list = egui::CollapsingHeader::new("Try several fills")
        .open(Some(open))
        .show(ui, |ui| {
            ui.weak("Runs each ticked fill on this extension.");
            for offer in &offers {
                let mut on = state.is_ticked(&offer.choice, s);
                if fill_row(ui, offer, &mut on, registry) {
                    state.set(&offer.choice, on, s);
                }
            }
            let fills: Vec<FillChoice> = offers
                .iter()
                .filter(|o| state.is_ticked(&o.choice, s))
                .map(|o| o.choice.clone())
                .collect();
            let blocker = try_blocker(s, cx.mode, cx.source, cx.running, fills.len());
            let size = egui::vec2(ui.available_width(), 28.0);
            let button = egui::Button::new(try_label(s, &fills, registry));
            let resp = ui.add_enabled(blocker.is_none(), |ui: &mut egui::Ui| {
                ui.add_sized(size, button)
            });
            if resp
                .on_disabled_hover_text(blocker.unwrap_or_default())
                .clicked()
            {
                out.try_fills = Some(fills);
            }
        });
    if list.header_response.clicked() {
        state.open = !open;
        // The list is long and the Extend section low in the sidebar: bring it up to the top.
        if state.open {
            list.header_response.scroll_to_me(Some(egui::Align::TOP));
        }
    }
}

/// One fill of the list: a checkbox with its name and licence, and under it, weak, what a model
/// suits and the download it needs. True when the checkbox was clicked.
fn fill_row(ui: &mut egui::Ui, offer: &FillOffer, on: &mut bool, registry: &Registry) -> bool {
    let changed = ui
        .horizontal(|ui| {
            let changed = ui.checkbox(on, &offer.name).changed();
            if let Some(e) = &offer.entry {
                if !e.commercial {
                    nc_badge(ui);
                }
                ui.weak(&e.license);
            }
            changed
        })
        .inner;
    if let Some(e) = &offer.entry {
        let mut detail = e.good_for.clone();
        if !registry.is_downloaded(&e.id) {
            let note = download_note(e);
            detail = if detail.is_empty() {
                note
            } else {
                format!("{detail} {note}")
            };
        }
        if !detail.is_empty() {
            // Level with the name, not with the box.
            let indent = ui.spacing().icon_width + ui.spacing().icon_spacing;
            ui.horizontal_top(|ui| {
                ui.add_space(indent);
                ui.add(egui::Label::new(egui::RichText::new(detail).weak()).wrap());
            });
        }
    }
    changed
}

pub fn show(ui: &mut egui::Ui, st: &mut AppState, registry: &Registry) -> SidebarOutput {
    let mut out = SidebarOutput::default();
    let running = st.status.is_running();
    let entries: Vec<ModelEntry> = registry.upscalers().cloned().collect();
    let invalid = st.settings.start_blocker(st.source_size);

    egui::Panel::bottom("sidebar_actions")
        .show_separator_line(false)
        .show(ui, |ui| {
            ui.add_space(8.0);
            let size = egui::vec2(ui.available_width(), 40.0);
            if running {
                let text = if st.stopping { "Stopping…" } else { "Stop" };
                let button = egui::Button::new(egui::RichText::new(text).size(16.0).strong());
                let enabled = !st.stopping;
                if ui
                    .add_enabled(enabled, |ui: &mut egui::Ui| ui.add_sized(size, button))
                    .on_hover_text("Stop the run (Esc)")
                    .clicked()
                {
                    out.stop = true;
                }
            } else {
                let label = start_label(&st.settings, registry);
                let enabled = st.mode.is_some() && invalid.is_none();
                // The accent marks an action that can be taken; a disabled one stays neutral.
                let button = if enabled {
                    theme::primary_button(label)
                } else {
                    egui::Button::new(egui::RichText::new(label).size(16.0).strong())
                };
                let resp = ui.add_enabled(enabled, |ui: &mut egui::Ui| ui.add_sized(size, button));
                let why = invalid
                    .clone()
                    .unwrap_or_else(|| "Select an image or a folder first".into());
                if resp.on_disabled_hover_text(why).clicked() {
                    out.start = true;
                }
            }
            ui.add_space(8.0);
        });

    egui::CentralPanel::default().show(ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.add_enabled_ui(!running, |ui| {
                theme::section(ui, "Image");
                ui.columns(2, |cols| {
                    let size = egui::vec2(cols[0].available_width(), 28.0);
                    if cols[0]
                        .add_sized(size, egui::Button::new("Select Image"))
                        .clicked()
                        && let Some(p) = pick_image()
                    {
                        out.picked = Some(p);
                    }
                    if cols[1]
                        .add_sized(size, egui::Button::new("Select Folder"))
                        .clicked()
                        && let Some(p) = rfd::FileDialog::new().pick_folder()
                    {
                        out.picked = Some(p);
                    }
                });
                match &st.mode {
                    Some(Mode::Single(p)) | Some(Mode::Batch(p)) => {
                        let full = p.to_string_lossy();
                        ui.label(elide_middle(&full, 40))
                            .on_hover_text(full.as_ref());
                    }
                    None => {
                        ui.weak("No image selected");
                    }
                }
                ui.add_space(SECTION_GAP);
                model_picker(ui, st, &entries, &mut out);
                ui.add_space(SECTION_GAP);
                output_options(ui, st);
                ui.add_space(SECTION_GAP);
                ui.separator();
                extend_section(ui, st, registry, &mut out);
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
    fn start_says_extend_when_nothing_is_upscaled() {
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::with_dir(dir.path().to_path_buf());
        // The upscaling model is missing too, but an extend-only run never loads it.
        let mut s = Settings {
            scale: None,
            extend_enabled: true,
            ..Settings::default()
        };
        assert_eq!(start_label(&s, &reg), "Download and extend");
        std::fs::write(reg.local_path("lama-512"), b"x").unwrap();
        assert_eq!(start_label(&s, &reg), "Extend");
        s.extend_fill = "mirror".into();
        s.extend_model = "migan-512-places2".into();
        assert_eq!(start_label(&s, &reg), "Extend");
        // A custom width upscales: the model is needed again.
        s.use_width = true;
        assert_eq!(start_label(&s, &reg), "Download and start");
    }

    #[test]
    fn try_several_offers_every_inpainting_model_then_the_plain_fills() {
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::with_dir(dir.path().to_path_buf());
        let offers = offered_fills(&reg);
        let models: Vec<&str> = reg.inpainters().map(|e| e.id.as_str()).collect();
        assert_eq!(models, ["lama-512", "migan-512-places2", "lama-manga"]);
        let choices: Vec<FillChoice> = offers.iter().map(|o| o.choice.clone()).collect();
        let mut expected: Vec<FillChoice> = models
            .iter()
            .map(|id| FillChoice::Ai((*id).into()))
            .collect();
        expected.extend([
            FillChoice::Plain(Fill::Mirror),
            FillChoice::Plain(Fill::Edge),
        ]);
        assert_eq!(choices, expected);
        let names: Vec<&str> = offers.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(names[0], "LaMa (512)");
        assert_eq!(&names[3..], ["Mirror", "Edge pixels"]);
        assert!(offers[..3].iter().all(|o| o.entry.is_some()));
        assert!(offers[3..].iter().all(|o| o.entry.is_none()));
    }

    #[test]
    fn the_try_button_counts_the_fills_and_says_when_it_downloads() {
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::with_dir(dir.path().to_path_buf());
        let mirror = FillChoice::Plain(Fill::Mirror);
        let edge = FillChoice::Plain(Fill::Edge);
        let lama = FillChoice::Ai("lama-512".into());
        let migan = FillChoice::Ai("migan-512-places2".into());
        // Extend only: the upscaling model is not needed.
        let s = Settings {
            scale: None,
            extend_enabled: true,
            ..Settings::default()
        };
        assert_eq!(try_label(&s, &[], &reg), "Try several fills");
        assert_eq!(
            try_label(&s, std::slice::from_ref(&mirror), &reg),
            "Try 1 fill"
        );
        assert_eq!(
            try_label(&s, &[mirror.clone(), edge.clone()], &reg),
            "Try 2 fills"
        );
        assert_eq!(
            try_label(&s, &[mirror.clone(), lama.clone(), edge.clone()], &reg),
            "Download and try 3 fills"
        );
        std::fs::write(reg.local_path("lama-512"), b"x").unwrap();
        assert_eq!(
            try_label(&s, &[mirror.clone(), lama.clone(), edge.clone()], &reg),
            "Try 3 fills"
        );
        assert_eq!(
            try_label(&s, &[lama.clone(), migan], &reg),
            "Download and try 2 fills"
        );
        // With an upscale the model of the upscale is downloaded first too.
        let up = Settings {
            scale: Some(2),
            ..s
        };
        assert_eq!(
            try_label(&up, &[mirror, edge], &reg),
            "Download and try 2 fills"
        );
    }

    #[test]
    fn the_try_button_says_why_it_is_disabled() {
        let single = Mode::Single(PathBuf::from("a.png"));
        let s = Settings {
            extend_enabled: true,
            ..Settings::default()
        };
        let why = |s: &Settings, mode: Option<&Mode>, running, n| {
            try_blocker(s, mode, Some([100, 100]), running, n)
        };
        assert_eq!(why(&s, Some(&single), false, 2), None);
        assert_eq!(why(&s, Some(&single), false, 5), None);
        assert_eq!(
            why(&s, Some(&single), false, 1).as_deref(),
            Some("Tick at least two fills")
        );
        assert_eq!(
            why(&s, Some(&single), false, 0).as_deref(),
            Some("Tick at least two fills")
        );
        let folder = Mode::Batch(PathBuf::from("dir"));
        for mode in [Some(&folder), None] {
            assert_eq!(
                why(&s, mode, false, 3).as_deref(),
                Some("Try several needs a single image")
            );
        }
        assert_eq!(
            why(&s, Some(&single), true, 3).as_deref(),
            Some("A run is in progress")
        );
        let off = Settings {
            extend_enabled: false,
            ..s.clone()
        };
        assert_eq!(
            why(&off, Some(&single), false, 3).as_deref(),
            Some("Turn on Extend canvas to try fills")
        );
        // An extension that cannot be made gives the reason Start gives.
        let wide = Settings {
            extend_left: 5000.0,
            extend_linked: false,
            ..s
        };
        let reason = wide.start_blocker(Some([100, 100])).unwrap();
        assert_eq!(why(&wide, Some(&single), false, 3), Some(reason));
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
            .with_size([320.0, 760.0])
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
            .with_size([320.0, 760.0])
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
        let settings = Settings {
            extend_enabled: true,
            extend_left: 128.0,
            extend_fill: "mirror".into(),
            ..Settings::default()
        };
        let registry = Registry::with_dir(dir.path().to_path_buf());
        let mut harness = egui_kittest::Harness::builder()
            .with_size([900.0, 700.0])
            .wgpu()
            .build_ui_state(
                |ui, (view, settings): &mut (CompareView, Settings)| {
                    view.show(
                        ui,
                        crate::ui::compare::StageInput {
                            settings,
                            registry: &registry,
                            single: true,
                            running: false,
                            step: None,
                        },
                    );
                },
                (CompareView::default(), settings),
            );
        let ctx = harness.ctx.clone();
        harness
            .state_mut()
            .0
            .set(&ctx, Which::Before, before, before_size, Margins::default());
        harness
            .state_mut()
            .0
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
