//! The bottom line: device, what is happening, progress, and the way to the results.

use eframe::egui;

use crate::state::{AppState, Status};
use crate::ui::theme;

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
        ui.toggle_value(log_open, "Log")
            .on_hover_text("Show or hide the messages of the runs");
        if st.device_name.is_empty() {
            ui.weak("Detecting device…");
        } else {
            ui.weak(&st.device_name)
                .on_hover_text("The device runs use. Change it under Advanced.");
        }
        ui.separator();
        match &st.status {
            Status::Error(_) => {
                let text = egui::RichText::new(st.status_line()).color(ui.visuals().error_fg_color);
                ui.add(egui::Label::new(text).truncate())
                    .on_hover_text(st.status_line());
            }
            Status::Done(path) => {
                ui.label(st.status_line())
                    .on_hover_text(path.display().to_string());
            }
            _ => {
                ui.label(st.status_line());
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if matches!(st.status, Status::Done(_)) && ui.button("Open folder").clicked() {
                open_folder = true;
            }
            if let Some(f) = fraction(&st.status) {
                // White on the accent while the bar is under the number, the text colour after.
                let color = if f > 0.08 {
                    egui::Color32::WHITE
                } else {
                    ui.visuals().text_color()
                };
                let percent = egui::RichText::new(format!("{:.0}%", f * 100.0)).color(color);
                ui.add(
                    egui::ProgressBar::new(f)
                        .text(percent)
                        .desired_height(18.0)
                        .fill(theme::ACCENT),
                );
            }
        });
    });
    open_folder
}
