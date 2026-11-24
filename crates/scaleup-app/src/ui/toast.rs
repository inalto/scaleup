//! Short messages stacked in the corner of the window.

use eframe::egui;

use crate::ui::theme;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToastKind {
    Success,
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Toast {
    pub kind: ToastKind,
    pub text: String,
}

impl Toast {
    pub fn success(text: impl Into<String>) -> Self {
        Self {
            kind: ToastKind::Success,
            text: text.into(),
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            kind: ToastKind::Error,
            text: text.into(),
        }
    }

    fn lifetime(&self) -> f64 {
        match self.kind {
            ToastKind::Success => 3.0,
            ToastKind::Error => 6.0,
        }
    }
}

/// The toasts on screen, each with the egui time it expires at.
#[derive(Default)]
pub struct Toasts {
    active: Vec<(Toast, Option<f64>)>,
}

impl Toasts {
    pub fn push(&mut self, toast: Toast) {
        self.active.push((toast, None));
    }

    /// Draws the toasts in the bottom right corner, `bottom` points above the window's bottom edge
    /// (above the status bar, or above the history strip when it shows).
    pub fn show(&mut self, ctx: &egui::Context, bottom: f32) {
        let now = ctx.input(|i| i.time);
        self.active.retain_mut(|(toast, expires)| {
            let e = *expires.get_or_insert(now + toast.lifetime());
            e > now
        });
        if self.active.is_empty() {
            return;
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(250));
        egui::Area::new(egui::Id::unique("toasts"))
            .order(egui::Order::Foreground)
            .anchor(egui::Align2::RIGHT_BOTTOM, [-12.0, -bottom])
            .interactable(false)
            .show(ctx, |ui| {
                ui.set_max_width(420.0);
                for (toast, _) in &self.active {
                    let fill = match toast.kind {
                        ToastKind::Success => theme::SUCCESS,
                        ToastKind::Error => theme::DANGER,
                    };
                    egui::Frame::popup(ui.style())
                        .fill(fill)
                        .stroke(egui::Stroke::NONE)
                        .inner_margin(egui::Margin::symmetric(12, 8))
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new(&toast.text).color(egui::Color32::WHITE));
                        });
                }
            });
    }
}
