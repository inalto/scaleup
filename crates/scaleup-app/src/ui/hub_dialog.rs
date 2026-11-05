//! The "Browse OpenModelDB" window: search and filters over the catalogue, each model's
//! readiness on this machine, install / import / convert / remove, the details of a model with
//! its before/after thumbnails, and "Recommend…". All work happens on the worker.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use eframe::egui;
use egui::Color32;
use scaleup_core::hub::{
    Ask, Capabilities, Catalog, Content, Filter, Goal, Installed, InstalledModel, LicenseClass,
    ModelCard, PIP_INSTALL, PytorchSource, Readiness, Recommendation, SpeedPref,
    expected_file_name, pick_resource, pytorch_source, readiness, recommend,
};

use crate::state::HubFilters;

/// The scale chips.
pub const SCALES: [u32; 5] = [1, 2, 3, 4, 8];
/// The content chips: catalogue tags.
pub const CONTENT_TAGS: [&str; 7] = [
    "photo",
    "anime",
    "cartoon",
    "pixel-art",
    "game-textures",
    "text",
    "faces",
];
const ROW_HEIGHT: f32 = 24.0;
const THUMB_WIDTH: f32 = 160.0;
const WARN: Color32 = Color32::from_rgb(200, 140, 20);

const CONTENTS: [(Content, &str); 10] = [
    (Content::Any, "Anything"),
    (Content::Photo, "Photos"),
    (Content::Anime, "Anime"),
    (Content::Cartoon, "Cartoons"),
    (Content::PixelArt, "Pixel art"),
    (Content::GameTexture, "Game textures"),
    (Content::Text, "Text"),
    (Content::Cgi, "CGI"),
    (Content::AiGenerated, "AI-generated images"),
    (Content::Faces, "Faces"),
];
const GOALS: [(Goal, &str); 6] = [
    (Goal::Upscale, "Upscale"),
    (Goal::Restore, "Restore at the same size"),
    (Goal::Denoise, "Remove noise"),
    (Goal::Dejpeg, "Remove JPEG artefacts"),
    (Goal::Deblur, "Remove blur"),
    (Goal::Any, "Anything"),
];
const SPEEDS: [(SpeedPref, &str); 3] = [
    (SpeedPref::Fastest, "Fastest"),
    (SpeedPref::Balanced, "Balanced"),
    (SpeedPref::Quality, "Best quality"),
];

/// What the user asked for this frame; the app sends each to the worker.
#[derive(Debug, PartialEq, Eq)]
pub enum HubAction {
    Refresh { force: bool },
    Install(String),
    Convert(String),
    Import { id: String, file: PathBuf },
    Remove(String),
    Thumbs(String),
}

/// Badge colour of a license class: green when commercial use is plainly fine, amber when it
/// comes with share-alike or copyleft duties, red when forbidden, grey when unknown.
pub fn license_colour(class: LicenseClass) -> Color32 {
    match class {
        LicenseClass::Permissive | LicenseClass::Attribution => Color32::from_rgb(28, 110, 60),
        LicenseClass::ShareAlike | LicenseClass::Copyleft => Color32::from_rgb(180, 120, 10),
        LicenseClass::NonCommercial => Color32::from_rgb(190, 40, 40),
        LicenseClass::Unknown => Color32::from_rgb(110, 110, 110),
    }
}

fn license_words(class: LicenseClass) -> &'static str {
    match class {
        LicenseClass::Permissive => "Commercial use allowed.",
        LicenseClass::Attribution => "Commercial use allowed; credit the author.",
        LicenseClass::ShareAlike => "Commercial use allowed; credit the author and share alike.",
        LicenseClass::Copyleft => {
            "Commercial use allowed; derived works must use the same license."
        }
        LicenseClass::NonCommercial => "Not for commercial use.",
        LicenseClass::Unknown => "License not stated or not recognised: assume no commercial use.",
    }
}

/// Icon and name of a readiness state. Each of the five is distinct.
pub fn readiness_label(r: Readiness) -> (&'static str, &'static str) {
    match r {
        Readiness::Installed => ("✔", "Installed"),
        Readiness::Downloadable => ("⬇", "Downloadable"),
        Readiness::ManualDownload => ("🌐", "Manual download"),
        Readiness::NeedsConversion => ("⚙", "Needs conversion"),
        Readiness::Unsupported => ("⛔", "Unsupported"),
    }
}

fn readiness_colour(r: Readiness) -> Color32 {
    match r {
        Readiness::Installed => Color32::from_rgb(40, 150, 80),
        Readiness::Downloadable => Color32::from_rgb(60, 130, 210),
        Readiness::ManualDownload => WARN,
        Readiness::NeedsConversion => Color32::from_rgb(150, 100, 200),
        Readiness::Unsupported => Color32::from_rgb(120, 120, 120),
    }
}

fn readiness_help(r: Readiness) -> &'static str {
    match r {
        Readiness::Installed => "Installed: pick it in the model list under From OpenModelDB.",
        Readiness::Downloadable => "Runs here; ScaleUP can download it for you.",
        Readiness::ManualDownload => {
            "Runs here, but its file is on a site you have to open in a browser."
        }
        Readiness::NeedsConversion => {
            "Only a PyTorch file is available, which has to be converted to ONNX first."
        }
        Readiness::Unsupported => "No file this machine can run, or not an RGB model.",
    }
}

/// Whether a model counts as "runnable on this machine": installed, or usable without conversion.
fn runnable(r: Readiness) -> bool {
    matches!(
        r,
        Readiness::Installed | Readiness::Downloadable | Readiness::ManualDownload
    )
}

/// The models matching the search text and the toggles, with their readiness, in search order.
pub fn hub_rows<'a>(
    cat: &'a Catalog,
    text: &str,
    f: &HubFilters,
    caps: &Capabilities,
    installed: &Installed,
) -> Vec<(&'a ModelCard, Readiness)> {
    let filter = Filter {
        tags_any: f.content.clone(),
        arch: f.arch.clone(),
        commercial_only: f.commercial_only,
        ..Filter::default()
    };
    cat.search(text, &filter)
        .into_iter()
        .filter(|m| f.scales.is_empty() || f.scales.contains(&m.scale))
        .map(|m| (m, readiness(m, caps, installed)))
        .filter(|(_, r)| !f.runnable_only || runnable(*r))
        .collect()
}

fn toggle_in<T: PartialEq>(list: &mut Vec<T>, value: T, on: bool) {
    list.retain(|v| *v != value);
    if on {
        list.push(value);
    }
}

/// `s` cut to `max` characters with an ellipsis.
fn cut(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_owned()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

fn license_badge(ui: &mut egui::Ui, card: &ModelCard) {
    let class = card.license_class();
    ui.label(
        egui::RichText::new(format!(
            " {} ",
            card.license.as_deref().unwrap_or("unknown")
        ))
        .small()
        .color(Color32::WHITE)
        .background_color(license_colour(class)),
    )
    .on_hover_text(license_words(class));
}

/// The page to open for a model whose file the user downloads by hand (`https` only).
fn download_page(card: &ModelCard, caps: &Capabilities) -> Option<String> {
    pick_resource(card, caps)
        .map(|(r, idx)| r.urls[idx].clone())
        .or_else(|| match pytorch_source(card) {
            PytorchSource::Manual { url, .. } => Some(url),
            _ => None,
        })
        .filter(|u| u.starts_with("https://"))
}

fn pick_model_file() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Model", &["onnx", "pth", "safetensors"])
        .pick_file()
}

/// Content, goal, speed, the (commercial, runnable) toggles and the installed generation.
type RecommendKey = (Content, Goal, SpeedPref, (bool, bool), u64);

/// The question and answers of the "Recommend…" window.
struct RecommendState {
    open: bool,
    content: Content,
    goal: Goal,
    speed: SpeedPref,
    /// The ask the results answer, with the installed generation.
    key: Option<RecommendKey>,
    results: Vec<Recommendation>,
}

impl Default for RecommendState {
    fn default() -> Self {
        Self {
            open: false,
            content: Content::Any,
            goal: Goal::Upscale,
            speed: SpeedPref::Balanced,
            key: None,
            results: Vec::new(),
        }
    }
}

/// Rows of the list: index into the catalogue and readiness.
type Rows = Vec<(usize, Readiness)>;

/// Where the catalogue request of the window stands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum CatalogLoad {
    /// Not asked for since the window was opened.
    #[default]
    NotRequested,
    Loading,
    /// The last request failed: only Refresh (or opening the window again) asks again.
    Failed,
}

/// A worker action the window can ask for, for [`action_enabled`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionKind {
    Install,
    Convert,
    Import,
    Remove,
    Refresh,
}

/// Whether a button may be used. While a download, conversion, import or upscale runs
/// (`busy`), nothing that queues work on the worker is; Refresh also waits for the catalogue
/// request on its way (`loading`).
pub fn action_enabled(kind: ActionKind, busy: bool, loading: bool) -> bool {
    match kind {
        ActionKind::Refresh => !busy && !loading,
        ActionKind::Install | ActionKind::Convert | ActionKind::Import | ActionKind::Remove => {
            !busy
        }
    }
}

/// Recommendations for `ask`, top five; only installed or usable-as-is models when
/// `runnable_only`, as the list's "Runnable on this machine" toggle.
fn recommend_for(cat: &Catalog, ask: &Ask, runnable_only: bool) -> Vec<Recommendation> {
    recommend(cat, ask, cat.models.len())
        .into_iter()
        .filter(|r| !runnable_only || runnable(r.readiness))
        .take(5)
        .collect()
}

/// Loosens the filters that hide `card` (readiness `r`) from the list, leaving the others.
fn relax_for(card: &ModelCard, r: Readiness, f: &mut HubFilters) {
    if !f.scales.is_empty() && !f.scales.contains(&card.scale) {
        f.scales.clear();
    }
    if !f.content.is_empty()
        && !f
            .content
            .iter()
            .any(|t| card.tags.iter().any(|c| c.eq_ignore_ascii_case(t)))
    {
        f.content.clear();
    }
    if f.arch
        .as_ref()
        .is_some_and(|a| !a.eq_ignore_ascii_case(&card.architecture))
    {
        f.arch = None;
    }
    if f.commercial_only && !scaleup_core::hub::commercial_ok(card.license_class()) {
        f.commercial_only = false;
    }
    if f.runnable_only && !runnable(r) {
        f.runnable_only = false;
    }
}

#[derive(Default)]
pub struct HubDialog {
    pub open: bool,
    catalog: Option<Arc<Catalog>>,
    /// Catalogue index by id.
    index: HashMap<String, usize>,
    archs: Vec<String>,
    installed: Installed,
    caps: Capabilities,
    load: CatalogLoad,
    search: String,
    selected: Option<String>,
    /// Bumped when the catalogue or the installed list changes.
    generation: u64,
    rows: Option<((String, HubFilters, u64), Rows)>,
    thumbs: HashMap<String, [Option<egui::TextureHandle>; 2]>,
    thumbs_requested: HashSet<String>,
    /// Models whose download page was opened: they now offer "Import file…".
    opened_pages: HashSet<String>,
    scroll_to: Option<String>,
    rec: RecommendState,
}

impl HubDialog {
    /// Opens the window with the installed list of `models_dir`.
    pub fn open_with(&mut self, models_dir: &Path) {
        self.open = true;
        self.installed = Installed::load(models_dir);
        self.caps = Capabilities::current();
        self.generation += 1;
        // A failed load is tried once more per opening.
        if self.load == CatalogLoad::Failed {
            self.load = CatalogLoad::NotRequested;
        }
    }

    /// The catalogue request to send now, if any: automatically (`force == false`) only once
    /// per opening while there is no catalogue; from the Refresh button whenever none is on its
    /// way.
    fn catalog_request(&mut self, force: bool) -> Option<HubAction> {
        let ask = if force {
            self.load != CatalogLoad::Loading
        } else {
            self.catalog.is_none() && self.load == CatalogLoad::NotRequested
        };
        ask.then(|| {
            self.load = CatalogLoad::Loading;
            HubAction::Refresh { force }
        })
    }

    /// The catalogue request failed.
    pub fn on_catalog_failed(&mut self) {
        self.load = CatalogLoad::Failed;
    }

    fn loading(&self) -> bool {
        self.load == CatalogLoad::Loading
    }

    /// Selects `id` and makes sure the list shows it: clears the search and loosens the filters
    /// that hide it.
    fn reveal(&mut self, id: &str, f: &mut HubFilters) {
        let Some(card) = self.catalog.as_ref().and_then(|c| c.get(id)) else {
            return;
        };
        let r = readiness(card, &self.caps, &self.installed);
        relax_for(card, r, f);
        self.search.clear();
        self.selected = Some(id.to_owned());
        self.scroll_to = Some(id.to_owned());
    }

    pub fn set_catalog(&mut self, cat: Arc<Catalog>) {
        self.load = CatalogLoad::NotRequested;
        self.index = cat
            .models
            .iter()
            .enumerate()
            .map(|(i, m)| (m.id.clone(), i))
            .collect();
        let mut archs: Vec<String> = cat
            .models
            .iter()
            .map(|m| m.architecture.to_lowercase())
            .collect();
        archs.sort();
        archs.dedup();
        self.archs = archs;
        self.catalog = Some(cat);
        self.generation += 1;
    }

    pub fn on_installed(&mut self, m: InstalledModel) {
        self.opened_pages.remove(&m.id);
        self.installed.upsert(m);
        self.generation += 1;
    }

    pub fn on_removed(&mut self, id: &str) {
        self.installed.remove(id);
        self.generation += 1;
    }

    pub fn set_thumbs(
        &mut self,
        ctx: &egui::Context,
        id: String,
        img: [Option<egui::ColorImage>; 2],
    ) {
        let [lr, sr] = img;
        let tex = |side: &str, img: Option<egui::ColorImage>| {
            img.map(|img| {
                ctx.load_texture(
                    format!("hub-thumb-{side}-{id}"),
                    img,
                    egui::TextureOptions::LINEAR,
                )
            })
        };
        let pair = [tex("lr", lr), tex("sr", sr)];
        self.thumbs.insert(id, pair);
    }

    /// `busy` disables the buttons that would queue work behind a running job.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        filters: &mut HubFilters,
        busy: bool,
    ) -> Vec<HubAction> {
        let mut actions = Vec::new();
        if !self.open {
            return actions;
        }
        actions.extend(self.catalog_request(false));
        let mut open = self.open;
        // Never larger than the app's window: at 800×600 a fixed size would hang off the edge.
        let room =
            (ctx.content_rect().size() - egui::vec2(24.0, 24.0)).max(egui::vec2(320.0, 240.0));
        egui::Window::new("Browse OpenModelDB")
            .open(&mut open)
            .default_size(egui::vec2(1040.0, 640.0).min(room))
            .min_width(room.x.min(600.0))
            .min_height(room.y.min(360.0))
            .max_size(room)
            .show(ctx, |ui| {
                self.top_bar(ui, busy, &mut actions);
                self.filter_row(ui, filters);
                ui.separator();
                let Some(cat) = self.catalog.clone() else {
                    ui.horizontal(|ui| {
                        if self.loading() {
                            ui.spinner();
                            ui.weak("Loading the OpenModelDB catalogue…");
                        } else {
                            ui.weak("The catalogue could not be loaded. Try Refresh.");
                        }
                    });
                    return;
                };
                self.refresh_rows(&cat, filters);
                egui::Panel::right("hub_details")
                    .resizable(true)
                    .default_size(360.0)
                    .min_size(280.0)
                    .show_inside(ui, |ui| self.details(ui, &cat, busy, &mut actions));
                egui::CentralPanel::default()
                    .show_inside(ui, |ui| self.list(ui, &cat, busy, &mut actions));
            });
        self.open = open;
        self.recommend_window(ctx, filters);
        if let Some(id) = &self.selected
            && self.thumbs_requested.insert(id.clone())
        {
            actions.push(HubAction::Thumbs(id.clone()));
        }
        actions
    }

    fn top_bar(&mut self, ui: &mut egui::Ui, busy: bool, actions: &mut Vec<HubAction>) {
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .hint_text("Search name, author, tags…")
                    .desired_width(300.0),
            );
            if ui.button("Recommend…").clicked() {
                self.rec.open = !self.rec.open;
            }
            let enabled = action_enabled(ActionKind::Refresh, busy, self.loading());
            if ui
                .add_enabled(enabled, egui::Button::new("Refresh"))
                .on_hover_text("Fetch the catalogue from openmodeldb.info again")
                .clicked()
            {
                actions.extend(self.catalog_request(true));
            }
            if self.loading() {
                ui.spinner();
            }
            if let (Some(cat), Some((_, rows))) = (&self.catalog, &self.rows) {
                ui.weak(format!("{} of {} models", rows.len(), cat.models.len()));
            }
        });
    }

    fn filter_row(&mut self, ui: &mut egui::Ui, f: &mut HubFilters) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Scale");
            for s in SCALES {
                let mut on = f.scales.contains(&s);
                if ui.toggle_value(&mut on, format!("{s}×")).clicked() {
                    toggle_in(&mut f.scales, s, on);
                }
            }
            ui.separator();
            ui.label("Content");
            for tag in CONTENT_TAGS {
                let mut on = f.content.iter().any(|t| t == tag);
                if ui.toggle_value(&mut on, tag).clicked() {
                    toggle_in(&mut f.content, tag.to_owned(), on);
                }
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut f.commercial_only, "Commercial use OK")
                .on_hover_text("Only licenses that clearly allow commercial use");
            ui.checkbox(&mut f.runnable_only, "Runnable on this machine")
                .on_hover_text("Installed, or runs here without conversion");
            ui.separator();
            ui.label("Architecture");
            egui::ComboBox::from_id_salt("hub_arch")
                .selected_text(f.arch.as_deref().unwrap_or("Any"))
                .height(360.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut f.arch, None, "Any");
                    for a in &self.archs {
                        ui.selectable_value(&mut f.arch, Some(a.clone()), a);
                    }
                });
        });
    }

    fn refresh_rows(&mut self, cat: &Catalog, f: &HubFilters) {
        let key = (self.search.clone(), f.clone(), self.generation);
        if self.rows.as_ref().is_some_and(|(k, _)| *k == key) {
            return;
        }
        let rows = hub_rows(cat, &self.search, f, &self.caps, &self.installed)
            .into_iter()
            .filter_map(|(m, r)| Some((*self.index.get(&m.id)?, r)))
            .collect();
        self.rows = Some((key, rows));
    }

    fn list(&mut self, ui: &mut egui::Ui, cat: &Catalog, busy: bool, actions: &mut Vec<HubAction>) {
        let rows = self
            .rows
            .as_ref()
            .map(|(_, r)| r.clone())
            .unwrap_or_default();
        if rows.is_empty() {
            ui.weak("No model matches. Clear the search or loosen the filters.");
            return;
        }
        let mut area = egui::ScrollArea::vertical()
            .id_salt("hub_rows")
            .auto_shrink(false);
        if let Some(id) = self.scroll_to.take()
            && let Some(pos) = rows.iter().position(|(i, _)| cat.models[*i].id == id)
        {
            area = area
                .vertical_scroll_offset(pos as f32 * (ROW_HEIGHT + ui.spacing().item_spacing.y));
        }
        area.show_rows(ui, ROW_HEIGHT, rows.len(), |ui, range| {
            for &(i, r) in &rows[range] {
                self.row(ui, &cat.models[i], r, busy, actions);
            }
        });
    }

    fn row(
        &mut self,
        ui: &mut egui::Ui,
        card: &ModelCard,
        r: Readiness,
        busy: bool,
        actions: &mut Vec<HubAction>,
    ) {
        ui.horizontal(|ui| {
            ui.set_height(ROW_HEIGHT);
            let (icon, name) = readiness_label(r);
            ui.label(egui::RichText::new(icon).color(readiness_colour(r)))
                .on_hover_text(name);
            let selected = self.selected.as_deref() == Some(card.id.as_str());
            if ui
                .selectable_label(selected, cut(&card.name, 34))
                .on_hover_text(&card.id)
                .clicked()
            {
                self.selected = Some(card.id.clone());
            }
            ui.weak(cut(&card.authors().join(", "), 16));
            ui.label(format!("{}×", card.scale));
            ui.weak(&card.architecture);
            license_badge(ui, card);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                self.primary_button(ui, card, r, busy, actions);
            });
        });
    }

    /// The one button of a list row.
    fn primary_button(
        &mut self,
        ui: &mut egui::Ui,
        card: &ModelCard,
        r: Readiness,
        busy: bool,
        actions: &mut Vec<HubAction>,
    ) {
        match r {
            Readiness::Installed => {
                ui.add_enabled(false, egui::Button::new("Installed ✓"));
            }
            Readiness::Downloadable => {
                if ui
                    .add_enabled(
                        action_enabled(ActionKind::Install, busy, false),
                        egui::Button::new("Install"),
                    )
                    .clicked()
                {
                    actions.push(HubAction::Install(card.id.clone()));
                }
            }
            Readiness::ManualDownload => {
                if self.opened_pages.contains(&card.id) {
                    self.import_button(ui, card, busy, actions);
                } else {
                    self.open_page_button(ui, card);
                }
            }
            Readiness::NeedsConversion => match pytorch_source(card) {
                PytorchSource::Download => {
                    if ui
                        .add_enabled(
                            action_enabled(ActionKind::Convert, busy, false),
                            egui::Button::new("Convert"),
                        )
                        .on_hover_text(
                            "Download the PyTorch file and convert it to ONNX with Python",
                        )
                        .clicked()
                    {
                        actions.push(HubAction::Convert(card.id.clone()));
                    }
                }
                PytorchSource::Manual { .. } if self.opened_pages.contains(&card.id) => {
                    self.import_button(ui, card, busy, actions);
                }
                PytorchSource::Manual { .. } => self.open_page_button(ui, card),
                PytorchSource::None => {
                    ui.add_enabled(false, egui::Button::new("No file"));
                }
            },
            Readiness::Unsupported => {
                ui.add_enabled(false, egui::Button::new("Unsupported"));
            }
        }
    }

    fn open_page_button(&mut self, ui: &mut egui::Ui, card: &ModelCard) {
        let Some(url) = download_page(card, &self.caps) else {
            ui.add_enabled(false, egui::Button::new("No download page"));
            return;
        };
        if ui
            .button("Open download page")
            .on_hover_text(&url)
            .clicked()
        {
            // Failing to start a browser leaves the URL in the details to copy.
            let _ = open::that(&url);
            self.opened_pages.insert(card.id.clone());
            self.selected = Some(card.id.clone());
        }
    }

    fn import_button(
        &mut self,
        ui: &mut egui::Ui,
        card: &ModelCard,
        busy: bool,
        actions: &mut Vec<HubAction>,
    ) {
        if ui
            .add_enabled(
                action_enabled(ActionKind::Import, busy, false),
                egui::Button::new("Import file…"),
            )
            .on_hover_text("Choose the file you downloaded; it is checked against the catalogue")
            .clicked()
            && let Some(file) = pick_model_file()
        {
            actions.push(HubAction::Import {
                id: card.id.clone(),
                file,
            });
        }
    }

    fn details(
        &mut self,
        ui: &mut egui::Ui,
        cat: &Catalog,
        busy: bool,
        actions: &mut Vec<HubAction>,
    ) {
        let Some(card) = self.selected.as_deref().and_then(|id| cat.get(id)) else {
            ui.weak("Select a model to see its details.");
            return;
        };
        let r = readiness(card, &self.caps, &self.installed);
        egui::ScrollArea::vertical()
            .id_salt("hub_details_scroll")
            .auto_shrink(false)
            .show(ui, |ui| {
                ui.heading(&card.name);
                ui.weak(&card.id);
                ui.label(format!("by {} · {}", card.authors().join(", "), card.date));
                ui.label(format!(
                    "{}× · {} · {} in, {} out",
                    card.scale, card.architecture, card.in_ch, card.out_ch
                ));
                ui.horizontal_wrapped(|ui| {
                    license_badge(ui, card);
                    ui.label(license_words(card.license_class()));
                });
                ui.separator();
                self.readiness_block(ui, card, r, busy, actions);
                self.thumbnails(ui, card);
                if !card.tags.is_empty() {
                    ui.horizontal_wrapped(|ui| {
                        ui.weak("Tags");
                        for t in &card.tags {
                            ui.label(egui::RichText::new(t).small());
                        }
                    });
                }
                if !card.description.is_empty() {
                    ui.separator();
                    ui.label(&card.description);
                }
                ui.separator();
                ui.weak("Files");
                if card.resources.is_empty() {
                    ui.weak("None listed.");
                }
                for res in &card.resources {
                    let size = res.size.map_or("size unknown".into(), |s| {
                        format!("{:.1} MB", s as f64 / 1e6)
                    });
                    let sha = if res.sha256.as_deref().is_some_and(|s| !s.trim().is_empty()) {
                        "sha256 listed"
                    } else {
                        "no sha256"
                    };
                    ui.label(format!(
                        "{} {} ({size}, {sha})",
                        res.platform.as_deref().unwrap_or("?"),
                        res.kind
                    ));
                    for url in &res.urls {
                        ui.label(egui::RichText::new(url).small().weak());
                    }
                }
            });
    }

    fn readiness_block(
        &mut self,
        ui: &mut egui::Ui,
        card: &ModelCard,
        r: Readiness,
        busy: bool,
        actions: &mut Vec<HubAction>,
    ) {
        let (icon, name) = readiness_label(r);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(icon).color(readiness_colour(r)));
            ui.strong(name);
        });
        ui.label(readiness_help(r));
        match r {
            Readiness::Installed => {
                if let Some(m) = self.installed.get(&card.id) {
                    if !m.verified {
                        ui.colored_label(
                            WARN,
                            "⚠ Not verified: OpenModelDB lists no sha256 for this file.",
                        );
                    }
                    ui.label(
                        egui::RichText::new(m.file.display().to_string())
                            .small()
                            .weak(),
                    );
                }
                if ui
                    .add_enabled(
                        action_enabled(ActionKind::Remove, busy, false),
                        egui::Button::new("Remove"),
                    )
                    .clicked()
                {
                    actions.push(HubAction::Remove(card.id.clone()));
                }
            }
            Readiness::Downloadable => {
                if ui
                    .add_enabled(
                        action_enabled(ActionKind::Install, busy, false),
                        egui::Button::new("Install"),
                    )
                    .clicked()
                {
                    actions.push(HubAction::Install(card.id.clone()));
                }
            }
            Readiness::ManualDownload => {
                if let Some((res, idx)) = pick_resource(card, &self.caps) {
                    let url = &res.urls[idx];
                    let name = expected_file_name(url, &card.id, &res.kind);
                    ui.label(format!(
                        "Save the file as '{name}', then choose Import file…. It is checked against the catalogue's sha256 before it is installed."
                    ));
                    match res.sha256.as_deref().filter(|s| !s.trim().is_empty()) {
                        Some(sha) => {
                            ui.label(egui::RichText::new(format!("sha256 {sha}")).small().weak());
                        }
                        None => {
                            ui.colored_label(
                                WARN,
                                "The catalogue lists no sha256 for this file: it will be installed unverified.",
                            );
                        }
                    }
                    ui.label(egui::RichText::new(url).small().weak());
                }
                ui.horizontal(|ui| {
                    self.open_page_button(ui, card);
                    self.import_button(ui, card, busy, actions);
                });
            }
            Readiness::NeedsConversion => match pytorch_source(card) {
                PytorchSource::Download => {
                    ui.label(
                        "Convert downloads the PyTorch file, checks its sha256 and converts it to ONNX with Python 3, which needs:",
                    );
                    ui.monospace(PIP_INSTALL);
                    if ui
                        .add_enabled(
                            action_enabled(ActionKind::Convert, busy, false),
                            egui::Button::new("Convert"),
                        )
                        .clicked()
                    {
                        actions.push(HubAction::Convert(card.id.clone()));
                    }
                }
                PytorchSource::Manual {
                    url,
                    file_name,
                    sha256,
                } => {
                    ui.label(format!(
                        "The PyTorch file is on a site ScaleUP cannot download from. Save it as '{file_name}', then choose Import file…: it is checked against the catalogue's sha256 and converted to ONNX with Python 3, which needs:"
                    ));
                    ui.monospace(PIP_INSTALL);
                    match sha256 {
                        Some(sha) => {
                            ui.label(egui::RichText::new(format!("sha256 {sha}")).small().weak());
                        }
                        None => {
                            ui.colored_label(
                                WARN,
                                "The catalogue lists no sha256 for this file: it will be installed unverified.",
                            );
                        }
                    }
                    ui.label(egui::RichText::new(url).small().weak());
                    ui.horizontal(|ui| {
                        self.open_page_button(ui, card);
                        self.import_button(ui, card, busy, actions);
                    });
                }
                PytorchSource::None => {
                    ui.label("The catalogue lists no PyTorch file for it.");
                }
            },
            Readiness::Unsupported => {}
        }
        ui.separator();
    }

    fn thumbnails(&self, ui: &mut egui::Ui, card: &ModelCard) {
        match self.thumbs.get(&card.id) {
            Some(pair) if pair.iter().any(Option::is_some) => {
                ui.horizontal(|ui| {
                    for (tex, caption) in pair.iter().zip(["Before", "After"]) {
                        let Some(tex) = tex else { continue };
                        ui.vertical(|ui| {
                            ui.add(
                                egui::Image::from_texture(egui::load::SizedTexture::from_handle(
                                    tex,
                                ))
                                .max_width(THUMB_WIDTH),
                            );
                            ui.weak(caption);
                        });
                    }
                });
                ui.separator();
            }
            // Nothing to show, or not yet: a missing picture is just missing.
            _ => {}
        }
    }

    fn recommend_window(&mut self, ctx: &egui::Context, f: &mut HubFilters) {
        if !self.rec.open {
            return;
        }
        let Some(cat) = self.catalog.clone() else {
            return;
        };
        let mut open = true;
        egui::Window::new("Recommend a model")
            .open(&mut open)
            .collapsible(false)
            .default_width(480.0)
            .show(ctx, |ui| {
                egui::Grid::new("hub_recommend_ask")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("What is in the image?");
                        pick(ui, "rec_content", &mut self.rec.content, &CONTENTS);
                        ui.end_row();
                        ui.label("What should happen?");
                        pick(ui, "rec_goal", &mut self.rec.goal, &GOALS);
                        ui.end_row();
                        ui.label("Speed");
                        pick(ui, "rec_speed", &mut self.rec.speed, &SPEEDS);
                        ui.end_row();
                    });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut f.commercial_only, "Commercial use OK");
                    ui.checkbox(&mut f.runnable_only, "Runnable on this machine");
                });
                let key = (
                    self.rec.content,
                    self.rec.goal,
                    self.rec.speed,
                    (f.commercial_only, f.runnable_only),
                    self.generation,
                );
                if self.rec.key != Some(key) {
                    let ask = Ask {
                        content: self.rec.content,
                        goal: self.rec.goal,
                        scale: None,
                        speed: self.rec.speed,
                        commercial_only: f.commercial_only,
                        caps: self.caps.clone(),
                        installed: self.installed.clone(),
                    };
                    self.rec.results = recommend_for(&cat, &ask, f.runnable_only);
                    self.rec.key = Some(key);
                }
                ui.separator();
                if self.rec.results.is_empty() {
                    ui.weak("No model matches.");
                }
                let mut picked = None;
                for (n, rec) in self.rec.results.iter().enumerate() {
                    let (icon, ready) = readiness_label(rec.readiness);
                    ui.horizontal(|ui| {
                        ui.strong(format!("{}. {}", n + 1, rec.card.name));
                        ui.label(egui::RichText::new(icon).color(readiness_colour(rec.readiness)))
                            .on_hover_text(ready);
                        ui.weak(format!("{}× {}", rec.card.scale, rec.card.architecture));
                        if ui.small_button("Select").clicked() {
                            picked = Some(rec.card.id.clone());
                        }
                    });
                    ui.label(egui::RichText::new(rec.reasons.join("; ")).small());
                }
                if let Some(id) = picked {
                    self.reveal(&id, f);
                }
            });
        self.rec.open = open;
    }
}

/// A combo box over `choices`.
fn pick<T: Copy + PartialEq>(ui: &mut egui::Ui, id: &str, value: &mut T, choices: &[(T, &str)]) {
    let current = choices
        .iter()
        .find(|(v, _)| v == value)
        .map_or("", |(_, l)| l);
    egui::ComboBox::from_id_salt(id)
        .selected_text(current)
        .show_ui(ui, |ui| {
            for (v, label) in choices {
                ui.selectable_value(value, *v, *label);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = include_str!("../../../../tests/fixtures/hub/models.sample.json");

    #[test]
    fn license_badge_colour_table() {
        use LicenseClass::*;
        let green = license_colour(Permissive);
        let amber = license_colour(ShareAlike);
        let red = license_colour(NonCommercial);
        let grey = license_colour(Unknown);
        assert_eq!(license_colour(Attribution), green);
        assert_eq!(license_colour(Copyleft), amber);
        let all = [green, amber, red, grey];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a, b);
            }
        }
        // Hue checks: green has the most green, red the most red, amber red and green over blue,
        // grey equal channels.
        assert!(green.g() > green.r() && green.g() > green.b());
        assert!(red.r() > red.g() && red.r() > red.b());
        assert!(amber.r() > amber.b() && amber.g() > amber.b());
        assert!(grey.r() == grey.g() && grey.g() == grey.b());
    }

    #[test]
    fn hub_rows_filter_by_toggles() {
        let cat = Catalog::parse(SAMPLE).unwrap();
        let caps = Capabilities {
            onnx: true,
            native_archs: vec!["esrgan".into()],
        };
        let none = Installed::default();
        let all = HubFilters {
            commercial_only: false,
            runnable_only: false,
            ..HubFilters::default()
        };
        let rows = |f: &HubFilters, text: &str| hub_rows(&cat, text, f, &caps, &none);
        let ids = |f: &HubFilters| -> Vec<String> {
            rows(f, "").iter().map(|(m, _)| m.id.clone()).collect()
        };
        assert_eq!(rows(&all, "").len(), 40);

        // Commercial use OK: the 27 commercial licenses, never NonCommercial or Unknown.
        let commercial = HubFilters {
            commercial_only: true,
            ..all.clone()
        };
        let found = rows(&commercial, "");
        assert_eq!(found.len(), 27);
        assert!(found.iter().all(|(m, _)| {
            !matches!(
                m.license_class(),
                LicenseClass::NonCommercial | LicenseClass::Unknown
            )
        }));

        // Scale chips: any of the chosen scales (24 x1, 6 x4, 1 x8 in the fixture).
        let scales = |s: &[u32]| HubFilters {
            scales: s.to_vec(),
            ..all.clone()
        };
        assert_eq!(rows(&scales(&[4]), "").len(), 6);
        assert_eq!(rows(&scales(&[1, 8]), "").len(), 25);
        assert_eq!(
            rows(
                &HubFilters {
                    scales: vec![4],
                    ..commercial.clone()
                },
                ""
            )
            .len(),
            5
        );

        // Runnable on this machine: no conversion, no unsupported model.
        let runnable = HubFilters {
            runnable_only: true,
            ..all.clone()
        };
        let found = rows(&runnable, "");
        assert!(found.iter().all(|(_, r)| runnable_ok(*r)), "{found:?}");
        let found_ids = ids(&runnable);
        assert!(found_ids.contains(&"1x-Anti-Aliasing".to_string()));
        assert!(!found_ids.contains(&"4x-DRCT-L".to_string()));
        assert!(ids(&all).contains(&"4x-DRCT-L".to_string()));

        // Content chips: any of the chosen tags; architecture; search text.
        let anime = HubFilters {
            content: vec!["anime".into()],
            ..all.clone()
        };
        let found = rows(&anime, "");
        assert!(!found.is_empty() && found.len() < 40);
        assert!(
            found
                .iter()
                .all(|(m, _)| m.tags.iter().any(|t| t == "anime"))
        );
        let span = HubFilters {
            arch: Some("span".into()),
            ..all.clone()
        };
        assert_eq!(rows(&span, "").len(), 1);
        assert_eq!(rows(&all, "jingyunliang").len(), 1);

        // An installed model is Installed, whatever its files.
        let mut inst = Installed::default();
        inst.models.push(InstalledModel {
            id: "4x-DRCT-L".into(),
            name: "n".into(),
            arch: "drct".into(),
            scale: 4,
            license: None,
            format: "onnx".into(),
            file: PathBuf::from("x.onnx"),
            sha256: None,
            verified: true,
            source_url: String::new(),
            installed_at: String::new(),
        });
        let found = hub_rows(&cat, "DRCT-L", &runnable, &caps, &inst);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].1, Readiness::Installed);
    }

    fn runnable_ok(r: Readiness) -> bool {
        runnable(r)
    }

    #[test]
    fn the_five_readiness_states_look_different() {
        let all = [
            Readiness::Installed,
            Readiness::Downloadable,
            Readiness::ManualDownload,
            Readiness::NeedsConversion,
            Readiness::Unsupported,
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(readiness_label(*a).0, readiness_label(*b).0);
                assert_ne!(readiness_label(*a).1, readiness_label(*b).1);
                assert_ne!(readiness_colour(*a), readiness_colour(*b));
                assert_ne!(readiness_help(*a), readiness_help(*b));
            }
        }
        // Only Installed reads as ready.
        for r in all {
            assert_eq!(
                readiness_label(r).1 == "Installed",
                r == Readiness::Installed
            );
        }
    }

    fn sample() -> Arc<Catalog> {
        Arc::new(Catalog::parse(SAMPLE).unwrap())
    }

    #[test]
    fn catalogue_is_asked_for_once_per_opening_and_refresh_retries() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = HubDialog::default();
        d.open_with(dir.path());
        assert_eq!(
            d.catalog_request(false),
            Some(HubAction::Refresh { force: false })
        );
        // While it is on its way: no second request, from any side.
        assert_eq!(d.catalog_request(false), None);
        assert_eq!(d.catalog_request(true), None);
        assert!(!action_enabled(ActionKind::Refresh, false, d.loading()));
        // Failed: the next frames do not ask again.
        d.on_catalog_failed();
        for _ in 0..3 {
            assert_eq!(d.catalog_request(false), None);
        }
        // Refresh does.
        assert_eq!(
            d.catalog_request(true),
            Some(HubAction::Refresh { force: true })
        );
        d.on_catalog_failed();
        // Opening the window again tries once more.
        d.open_with(dir.path());
        assert!(d.catalog_request(false).is_some());
        assert_eq!(d.catalog_request(false), None);
        // Loaded: no automatic request; Refresh still works.
        d.set_catalog(sample());
        assert_eq!(d.catalog_request(false), None);
        assert!(d.catalog_request(true).is_some());
    }

    #[test]
    fn busy_disables_every_worker_action() {
        use ActionKind::*;
        for kind in [Install, Convert, Import, Remove, Refresh] {
            assert!(action_enabled(kind, false, false), "{kind:?}");
            assert!(!action_enabled(kind, true, false), "{kind:?}");
        }
        assert!(!action_enabled(Refresh, false, true));
        assert!(action_enabled(Install, false, true));
    }

    #[test]
    fn recommend_respects_runnable_and_commercial() {
        let cat = sample();
        let caps = Capabilities {
            onnx: true,
            native_archs: vec!["esrgan".into()],
        };
        let ask = |commercial_only| Ask {
            content: Content::Any,
            goal: Goal::Any,
            scale: None,
            speed: SpeedPref::Balanced,
            commercial_only,
            caps: caps.clone(),
            installed: Installed::default(),
        };
        let all = recommend_for(&cat, &ask(false), false);
        assert!(all.len() == 5);
        assert!(
            recommend(&cat, &ask(false), 40)
                .iter()
                .any(|r| r.readiness == Readiness::NeedsConversion),
            "the fixture has conversion-only picks to filter"
        );
        let runnable_only = recommend_for(&cat, &ask(true), true);
        assert!(!runnable_only.is_empty() && runnable_only.len() <= 5);
        for r in &runnable_only {
            assert!(runnable(r.readiness), "{}", r.card.id);
            assert!(
                scaleup_core::hub::commercial_ok(r.card.license_class()),
                "{}",
                r.card.id
            );
        }
    }

    #[test]
    fn a_picked_recommendation_is_always_in_the_list() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = HubDialog::default();
        d.open_with(dir.path());
        d.caps = Capabilities {
            onnx: true,
            native_archs: vec!["esrgan".into()],
        };
        d.set_catalog(sample());
        let cat = d.catalog.clone().unwrap();
        // A conversion-only, non-commercial-or-unknown x4 card, hidden by nearly every filter.
        let hidden = "4x-DRCT-L";
        let mut f = HubFilters {
            scales: vec![1],
            content: vec!["pixel-art".into()],
            arch: Some("span".into()),
            ..HubFilters::default()
        };
        d.search = "zzz".into();
        let in_rows = |d: &HubDialog, f: &HubFilters| {
            hub_rows(&cat, &d.search, f, &d.caps, &d.installed)
                .iter()
                .any(|(m, _)| m.id == hidden)
        };
        assert!(!in_rows(&d, &f));
        d.reveal(hidden, &mut f);
        assert!(in_rows(&d, &f), "{f:?}");
        assert_eq!(d.selected.as_deref(), Some(hidden));
        assert_eq!(d.scroll_to.as_deref(), Some(hidden));
        // Filters that did not hide it are kept.
        let mut kept = HubFilters {
            runnable_only: false,
            commercial_only: false,
            scales: vec![4, 2],
            ..HubFilters::default()
        };
        d.reveal(hidden, &mut kept);
        assert_eq!(kept.scales, [4, 2]);
    }

    /// Draws the window headless (egui_kittest + wgpu) into `$SCALEUP_RENDER_DIR/*.png` for a
    /// person to look at. Run with `-- --ignored`; skipped when the variable is unset.
    #[test]
    #[ignore = "writes PNG files for a manual look; needs SCALEUP_RENDER_DIR and a GPU adapter"]
    fn hub_dialog_renders() {
        let Some(out) = std::env::var_os("SCALEUP_RENDER_DIR").map(PathBuf::from) else {
            eprintln!("skipped: SCALEUP_RENDER_DIR is not set");
            return;
        };
        std::fs::create_dir_all(&out).unwrap();
        let caps = Capabilities {
            onnx: true,
            native_archs: vec!["esrgan".into(), "compact".into()],
        };
        let dialog = |loaded: bool| {
            let mut d = HubDialog {
                open: true,
                caps: caps.clone(),
                ..HubDialog::default()
            };
            if loaded {
                d.set_catalog(sample());
            } else {
                d.load = CatalogLoad::Failed;
            }
            d
        };
        let render = |name: &str, size: [f32; 2], d: HubDialog, filters: HubFilters| {
            let mut harness = egui_kittest::Harness::builder()
                .with_size(size)
                .wgpu()
                .build_ui_state(
                    |ui, (d, f): &mut (HubDialog, HubFilters)| {
                        let ctx = ui.ctx().clone();
                        d.show(&ctx, f, false);
                    },
                    (d, filters),
                );
            for _ in 0..4 {
                harness.step();
            }
            let img = harness.render().expect("render");
            let file = out.join(format!("{name}.png"));
            img.save(&file).unwrap();
            eprintln!("wrote {}", file.display());
        };
        let all = HubFilters {
            commercial_only: false,
            runnable_only: false,
            ..HubFilters::default()
        };
        render(
            "a-list-1200x800",
            [1200.0, 800.0],
            dialog(true),
            HubFilters::default(),
        );
        let mut selected = dialog(true);
        selected.selected = Some("4x-DRCT-L".into());
        render("b-details-1200x800", [1200.0, 800.0], selected, all.clone());
        let mut rec = dialog(true);
        rec.rec.open = true;
        render(
            "c-recommend-1200x800",
            [1200.0, 800.0],
            rec,
            HubFilters::default(),
        );
        render("d-list-800x600", [800.0, 600.0], dialog(true), all);
        render(
            "e-failed-1200x800",
            [1200.0, 800.0],
            dialog(false),
            HubFilters::default(),
        );
    }

    #[test]
    fn toggle_in_adds_and_removes_once() {
        let mut v = vec![2];
        toggle_in(&mut v, 4, true);
        toggle_in(&mut v, 4, true);
        assert_eq!(v, [2, 4]);
        toggle_in(&mut v, 2, false);
        assert_eq!(v, [4]);
    }

    #[test]
    fn cut_keeps_short_text() {
        assert_eq!(cut("abc", 5), "abc");
        assert_eq!(cut("abcdef", 4), "abc…");
    }

    #[test]
    fn download_page_is_https_only() {
        let card: ModelCard = serde_json::from_str(
            r#"{"name":"n","license":null,"architecture":"compact","scale":2,
                "inputChannels":3,"outputChannels":3,
                "resources":[{"platform":"onnx","type":"onnx","urls":["https://mega.nz/x"]}]}"#,
        )
        .unwrap();
        let caps = Capabilities {
            onnx: true,
            native_archs: Vec::new(),
        };
        assert_eq!(
            download_page(&card, &caps).as_deref(),
            Some("https://mega.nz/x")
        );
        let mut plain = card.clone();
        plain.resources[0].urls = vec!["http://mega.nz/x".into()];
        assert_eq!(download_page(&plain, &caps), None);
    }
}
