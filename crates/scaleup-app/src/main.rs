//! ScaleUP desktop app.
#![cfg_attr(not(test), windows_subsystem = "windows")]

mod app;
mod state;
mod worker;

use eframe::egui;

fn icon() -> egui::IconData {
    let img = image::load_from_memory(include_bytes!("../assets/icon.png"))
        .expect("embedded icon is a valid PNG")
        .into_rgba8();
    let (width, height) = img.dimensions();
    egui::IconData {
        rgba: img.into_raw(),
        width,
        height,
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([900.0, 600.0])
            .with_icon(icon()),
        ..Default::default()
    };
    eframe::run_native(
        "ScaleUP",
        options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc)))),
    )
}
