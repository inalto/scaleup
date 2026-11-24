//! The look every window shares: colours with one meaning each, spacing, and the few widgets
//! built from them.

use eframe::egui::{self, Color32, CornerRadius, Stroke, Theme, vec2};

/// The blue of the app icon: the primary action, selection and progress.
pub const ACCENT: Color32 = Color32::from_rgb(47, 91, 207);
/// Free to use, installed, saved. White text reads on it.
pub const SUCCESS: Color32 = Color32::from_rgb(28, 110, 60);
/// Non-commercial licences and failures. White text reads on it.
pub const DANGER: Color32 = Color32::from_rgb(180, 40, 40);

const RADIUS: u8 = 6;

/// Sets spacing and colours for the dark and the light theme. Call once, at startup.
pub fn apply(ctx: &egui::Context) {
    for theme in [Theme::Dark, Theme::Light] {
        ctx.style_mut_of(theme, |style| {
            let s = &mut style.spacing;
            s.item_spacing = vec2(8.0, 6.0);
            s.button_padding = vec2(8.0, 4.0);
            s.interact_size.y = 24.0;
            s.slider_width = 120.0;

            let dark = theme == Theme::Dark;
            let v = &mut style.visuals;
            // Secondary text keeps a readable contrast on the panels (the default fades too far).
            v.weak_text_color = Some(if dark {
                Color32::from_gray(150)
            } else {
                Color32::from_gray(95)
            });
            v.error_fg_color = if dark {
                Color32::from_rgb(255, 120, 110)
            } else {
                Color32::from_rgb(176, 30, 30)
            };
            v.selection.bg_fill = if dark {
                ACCENT
            } else {
                Color32::from_rgb(196, 212, 250)
            };
            v.selection.stroke = Stroke::new(
                1.0,
                if dark {
                    Color32::WHITE
                } else {
                    Color32::from_rgb(20, 44, 120)
                },
            );
            v.hyperlink_color = if dark {
                Color32::from_rgb(140, 170, 255)
            } else {
                ACCENT
            };
            v.slider_trailing_fill = true;
            let radius = CornerRadius::same(RADIUS);
            v.window_corner_radius = radius;
            v.menu_corner_radius = radius;
            for w in [
                &mut v.widgets.noninteractive,
                &mut v.widgets.inactive,
                &mut v.widgets.hovered,
                &mut v.widgets.active,
                &mut v.widgets.open,
            ] {
                w.corner_radius = radius;
            }
        });
    }
}

/// The background of the image area: darker than the panels, so the picture stands out.
pub fn stage_fill(visuals: &egui::Visuals) -> Color32 {
    if visuals.dark_mode {
        Color32::from_gray(16)
    } else {
        Color32::from_gray(214)
    }
}

/// The translucent dark ground of labels and chips drawn on the picture; white text reads on it.
pub const OVERLAY: Color32 = Color32::from_black_alpha(140);
/// The dark edge under white marks on the picture (split line, knob, handles, outlines): they stay
/// visible on a bright picture.
pub const ON_IMAGE_EDGE: Color32 = Color32::from_black_alpha(110);
/// The dark line under the white outline of the canvas and of the picture in the edit stage:
/// denser than [`ON_IMAGE_EDGE`], because a 1 px line over a mirrored copy of the picture has
/// nothing else to tell it apart.
pub const OUTLINE_EDGE: Color32 = Color32::from_black_alpha(170);

/// The two squares of the checkerboard that stands for transparency: close in tone, so it stays
/// quiet next to the picture.
pub fn checker_colors(visuals: &egui::Visuals) -> [Color32; 2] {
    if visuals.dark_mode {
        [Color32::from_gray(52), Color32::from_gray(68)]
    } else {
        [Color32::from_gray(222), Color32::from_gray(242)]
    }
}

/// The tint and the hatch lines of canvas that AI inpainting will fill: a faint grey over the
/// stage, so it reads as "to be made" and not as a colour.
pub fn hatch_colors(visuals: &egui::Visuals) -> (Color32, Color32) {
    if visuals.dark_mode {
        (Color32::from_white_alpha(10), Color32::from_white_alpha(26))
    } else {
        (Color32::from_black_alpha(14), Color32::from_black_alpha(30))
    }
}

/// A chip on the picture: white text on the overlay ground, or on `fill` (`DANGER` for a problem).
pub fn overlay_chip(ui: &mut egui::Ui, text: &str, fill: Color32) -> egui::Response {
    egui::Frame::new()
        .fill(fill)
        .corner_radius(CornerRadius::same(4))
        .inner_margin(egui::Margin::symmetric(6, 3))
        .show(ui, |ui| {
            ui.add(
                egui::Label::new(egui::RichText::new(text).color(Color32::WHITE)).selectable(false),
            )
        })
        .response
}

/// A button on the picture, in the chips' look.
pub fn overlay_button<'a>(text: &str) -> egui::Button<'a> {
    egui::Button::new(egui::RichText::new(text).color(Color32::WHITE))
        .fill(OVERLAY)
        .stroke(Stroke::NONE)
}

/// A small label on a coloured, rounded ground (licence class, "NC").
pub fn badge(ui: &mut egui::Ui, text: &str, fill: Color32) -> egui::Response {
    egui::Frame::new()
        .fill(fill)
        .corner_radius(CornerRadius::same(4))
        .inner_margin(egui::Margin::symmetric(5, 1))
        .show(ui, |ui| {
            ui.add(
                egui::Label::new(
                    egui::RichText::new(text)
                        .small()
                        .strong()
                        .color(Color32::WHITE),
                )
                .selectable(false),
            )
        })
        .response
}

/// The one action a view is for.
pub fn primary_button<'a>(text: &str) -> egui::Button<'a> {
    egui::Button::new(
        egui::RichText::new(text)
            .size(16.0)
            .strong()
            .color(Color32::WHITE),
    )
    .fill(ACCENT)
}

/// The title of a group of controls in a side panel.
pub fn section(ui: &mut egui::Ui, title: &str) -> egui::Response {
    ui.add_space(4.0);
    ui.strong(title)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WCAG relative luminance contrast of two opaque colours.
    fn contrast(a: Color32, b: Color32) -> f32 {
        let lum = |c: Color32| {
            let ch = |v: u8| {
                let v = f32::from(v) / 255.0;
                if v <= 0.04045 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                }
            };
            0.2126 * ch(c.r()) + 0.7152 * ch(c.g()) + 0.0722 * ch(c.b())
        };
        let (hi, lo) = (lum(a).max(lum(b)), lum(a).min(lum(b)));
        (hi + 0.05) / (lo + 0.05)
    }

    #[test]
    fn white_text_reads_on_every_coloured_ground() {
        for fill in [ACCENT, SUCCESS, DANGER] {
            assert!(contrast(Color32::WHITE, fill) >= 4.5, "{fill:?}");
        }
    }

    #[test]
    fn secondary_and_error_text_read_on_the_panels() {
        let ctx = egui::Context::default();
        apply(&ctx);
        for theme in [Theme::Dark, Theme::Light] {
            let style = ctx.style_of(theme);
            let v = &style.visuals;
            let weak = v.weak_text_color.unwrap();
            assert!(contrast(weak, v.panel_fill) >= 4.5, "{theme:?} weak");
            assert!(
                contrast(v.error_fg_color, v.panel_fill) >= 4.5,
                "{theme:?} error"
            );
        }
    }
}
