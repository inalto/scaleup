//! The bottom line: device, what is happening, progress, and the way to the results.

use eframe::egui;

use crate::state::{AppState, Status};

/// The bar position for the current status, if a bar applies.
fn fraction(status: &Status) -> Option<f32> {
    match status {
        Status::Running { percent, .. } => Some(percent / 100.0),
        Status::Downloading { done, total } if *total > 0 => Some(*done as f32 / *total as f32),
        _ => None,
    }
}

/// Returns true when the user asked to open the output folder.
pub fn show(ui: &mut egui::Ui, st: &AppState, log_open: &mut bool) -> bool {
    let mut open_folder = false;
    ui.horizontal(|ui| {
        ui.toggle_value(log_open, "Log");
        if st.device_name.is_empty() {
            ui.weak("Detecting device…");
        } else {
            ui.weak(&st.device_name);
        }
        ui.separator();
        match &st.status {
            Status::Error(_) => {
                ui.colored_label(ui.visuals().error_fg_color, st.status.label());
            }
            status => {
                ui.label(status.label());
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if matches!(st.status, Status::Done(_)) && ui.button("Open folder").clicked() {
                open_folder = true;
            }
            if let Some(f) = fraction(&st.status) {
                ui.add(egui::ProgressBar::new(f).show_percentage());
            }
        });
    });
    open_folder
}
