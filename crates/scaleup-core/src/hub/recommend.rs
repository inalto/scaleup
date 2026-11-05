//! "Choose a model for me": turns what the user is upscaling and what they want into a ranked
//! shortlist with the reasons for each pick.
//!
//! Score of a model, summed over the rules that fire (each fired rule adds a line to `reasons`):
//!
//! | Rule | Points |
//! |---|---|
//! | tag matches the content (`photo`, `anime`, `cartoon`, `pixel-art`, `game-textures`, `text`, `cgi`, `ai-generated`, `faces`) | +3 |
//! | tag `general-upscaler`, content Any or Photo | +2 |
//! | tag `pretrained`, content Any | +1 |
//! | goal Denoise / Dejpeg / Deblur: tag `denoise` / `jpeg` or `compression-removal` / `deblur` | +2 |
//! | speed Fastest, by `speed_tier` 0..3 | +3 / +1 / -2 / -4 |
//! | speed Balanced | +1 / +2 / 0 / -2 |
//! | speed Quality | 0 / +1 / +3 / +2 |
//! | readiness Installed / Downloadable / ManualDownload / NeedsConversion | +3 / +1 / -1 / -3 |
//! | licence not usable commercially (when not `commercial_only`) | -1 |
//! | has preview images | +1 |
//! | dated 2024 or later | +0.5 |
//!
//! Excluded: goal Restore with a scale other than 1, goal Upscale with scale 1; a scale different from the asked one; readiness
//! Unsupported; NonCommercial or Unknown licence when `commercial_only`. Ties go to the newer model.

use super::catalog::{Capabilities, Catalog, ModelCard, Readiness, readiness, speed_tier};
use super::installed::Installed;
use super::license::{LicenseClass, commercial_ok};

/// What kind of image is being upscaled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Content {
    Photo,
    Anime,
    Cartoon,
    PixelArt,
    GameTexture,
    Text,
    Cgi,
    AiGenerated,
    Faces,
    Any,
}

/// What the user wants done to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Goal {
    Upscale,
    /// Clean up at the same size (1x models only).
    Restore,
    Denoise,
    Dejpeg,
    Deblur,
    Any,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeedPref {
    Fastest,
    Balanced,
    Quality,
}

/// Lowercase, `_` as `-`.
fn normalize(s: &str) -> String {
    s.trim().to_lowercase().replace('_', "-")
}

impl Content {
    const NAMES: &'static str =
        "photo, anime, cartoon, pixel-art, game-textures, text, cgi, ai-generated, faces, any";

    /// Parses a CLI value, case-insensitive, `-` or `_`.
    pub fn parse(s: &str) -> Result<Self, String> {
        Ok(match normalize(s).as_str() {
            "photo" => Self::Photo,
            "anime" => Self::Anime,
            "cartoon" => Self::Cartoon,
            "pixel-art" => Self::PixelArt,
            "game-textures" => Self::GameTexture,
            "text" => Self::Text,
            "cgi" => Self::Cgi,
            "ai-generated" => Self::AiGenerated,
            "faces" => Self::Faces,
            "any" => Self::Any,
            _ => {
                return Err(format!(
                    "unknown content {s:?}; expected one of: {}",
                    Self::NAMES
                ));
            }
        })
    }

    /// The catalogue tag that marks this content; `None` for Any.
    fn tag(self) -> Option<&'static str> {
        match self {
            Self::Photo => Some("photo"),
            Self::Anime => Some("anime"),
            Self::Cartoon => Some("cartoon"),
            Self::PixelArt => Some("pixel-art"),
            Self::GameTexture => Some("game-textures"),
            Self::Text => Some("text"),
            Self::Cgi => Some("cgi"),
            Self::AiGenerated => Some("ai-generated"),
            Self::Faces => Some("faces"),
            Self::Any => None,
        }
    }
}

impl Goal {
    const NAMES: &'static str = "upscale, restore, denoise, dejpeg, deblur, any";

    /// Parses a CLI value, case-insensitive, `-` or `_`.
    pub fn parse(s: &str) -> Result<Self, String> {
        Ok(match normalize(s).as_str() {
            "upscale" => Self::Upscale,
            "restore" => Self::Restore,
            "denoise" => Self::Denoise,
            "dejpeg" => Self::Dejpeg,
            "deblur" => Self::Deblur,
            "any" => Self::Any,
            _ => {
                return Err(format!(
                    "unknown goal {s:?}; expected one of: {}",
                    Self::NAMES
                ));
            }
        })
    }
}

impl SpeedPref {
    const NAMES: &'static str = "fastest, balanced, quality";

    /// Parses a CLI value, case-insensitive.
    pub fn parse(s: &str) -> Result<Self, String> {
        Ok(match normalize(s).as_str() {
            "fastest" => Self::Fastest,
            "balanced" => Self::Balanced,
            "quality" => Self::Quality,
            _ => {
                return Err(format!(
                    "unknown speed {s:?}; expected one of: {}",
                    Self::NAMES
                ));
            }
        })
    }

    /// Points by `speed_tier` 0..3.
    fn points(self) -> [f32; 4] {
        match self {
            Self::Fastest => [3.0, 1.0, -2.0, -4.0],
            Self::Balanced => [1.0, 2.0, 0.0, -2.0],
            Self::Quality => [0.0, 1.0, 3.0, 2.0],
        }
    }
}

/// The question put to [`recommend`].
#[derive(Clone, Debug)]
pub struct Ask {
    pub content: Content,
    pub goal: Goal,
    /// Only models of this scale; any scale when `None`.
    pub scale: Option<u32>,
    pub speed: SpeedPref,
    /// Leave out models whose licence does not clearly allow commercial use.
    pub commercial_only: bool,
    pub caps: Capabilities,
    pub installed: Installed,
}

#[derive(Clone, Debug)]
pub struct Recommendation {
    pub card: ModelCard,
    pub score: f32,
    /// Whether the model can be used now; anything but `Installed` still needs a step from the user.
    pub readiness: Readiness,
    /// Each rule that raised or lowered the score, in words.
    pub reasons: Vec<String>,
}

fn has_tag(card: &ModelCard, tag: &str) -> bool {
    card.tags.iter().any(|t| t.eq_ignore_ascii_case(tag))
}

/// Scores one model; `None` when the ask rules it out.
fn score(card: &ModelCard, ask: &Ask) -> Option<Recommendation> {
    if (ask.goal == Goal::Restore && card.scale != 1)
        || (ask.goal == Goal::Upscale && card.scale == 1)
    {
        return None;
    }
    if ask.scale.is_some_and(|s| s != card.scale) {
        return None;
    }
    let ready = readiness(card, &ask.caps, &ask.installed);
    if ready == Readiness::Unsupported {
        return None;
    }
    let class = card.license_class();
    let commercial = commercial_ok(class);
    if ask.commercial_only && !commercial {
        return None;
    }

    let mut score = 0.0_f32;
    let mut reasons = Vec::new();
    let mut add = |points: f32, why: String| {
        score += points;
        reasons.push(why);
    };

    if let Some(tag) = ask.content.tag()
        && has_tag(card, tag)
    {
        add(3.0, format!("tagged {tag}"));
    }
    if matches!(ask.content, Content::Any | Content::Photo) && has_tag(card, "general-upscaler") {
        add(2.0, "general-purpose upscaler".into());
    }
    if ask.content == Content::Any && has_tag(card, "pretrained") {
        add(1.0, "pretrained model".into());
    }

    let (wanted, why): (&[&str], &str) = match ask.goal {
        Goal::Denoise => (&["denoise"], "removes noise"),
        Goal::Dejpeg => (&["jpeg", "compression-removal"], "removes JPEG artefacts"),
        Goal::Deblur => (&["deblur"], "removes blur"),
        Goal::Upscale | Goal::Restore | Goal::Any => (&[], ""),
    };
    if wanted.iter().any(|t| has_tag(card, t)) {
        add(2.0, why.into());
    }

    let tier = speed_tier(&card.architecture);
    let points = ask.speed.points()[usize::from(tier.min(3))];
    let tier_name = ["fast", "moderate", "heavy", "slow or unknown"][usize::from(tier.min(3))];
    add(
        points,
        format!("{tier_name} architecture ({})", card.architecture),
    );

    let (points, why) = match ready {
        Readiness::Installed => (3.0, "already installed"),
        Readiness::Downloadable => (1.0, "can be downloaded here"),
        Readiness::ManualDownload => (-1.0, "needs a manual download"),
        Readiness::NeedsConversion => (-3.0, "needs conversion to ONNX"),
        Readiness::Unsupported => unreachable!("excluded above"),
    };
    add(points, why.into());

    if !commercial {
        add(
            -1.0,
            match class {
                LicenseClass::NonCommercial => "non-commercial licence".into(),
                _ => "licence unclear for commercial use".into(),
            },
        );
    }
    if !card.images.is_empty() {
        add(1.0, "has comparison images".into());
    }
    if card.date.as_str() >= "2024" {
        add(0.5, "recent (2024 or later)".into());
    }

    Some(Recommendation {
        card: card.clone(),
        score,
        readiness: ready,
        reasons,
    })
}

/// The best `n` models for `ask`, highest score first, newer first among equals.
pub fn recommend(cat: &Catalog, ask: &Ask, n: usize) -> Vec<Recommendation> {
    let mut out: Vec<Recommendation> = cat.models.iter().filter_map(|m| score(m, ask)).collect();
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.card.date.cmp(&a.card.date))
    });
    out.truncate(n);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::InstalledModel;

    const SAMPLE: &str = include_str!("../../../../tests/fixtures/hub/models.sample.json");

    fn ask() -> Ask {
        Ask {
            content: Content::Any,
            goal: Goal::Any,
            scale: None,
            speed: SpeedPref::Balanced,
            commercial_only: false,
            caps: Capabilities {
                onnx: true,
                native_archs: Vec::new(),
            },
            installed: Installed::default(),
        }
    }

    /// An RGB ONNX model on a direct host (Downloadable) with the given properties.
    fn card(
        id: &str,
        arch: &str,
        scale: u32,
        license: &str,
        tags: &[&str],
        date: &str,
    ) -> ModelCard {
        let tags = serde_json::to_string(tags).unwrap();
        let mut c: ModelCard = serde_json::from_str(&format!(
            r#"{{"name":"{id}","author":"a","license":"{license}","tags":{tags},"description":"",
                "date":"{date}","architecture":"{arch}","scale":{scale},"inputChannels":3,
                "outputChannels":3,"resources":[{{"platform":"onnx","type":"onnx","size":1,
                "sha256":null,"urls":["https://github.com/x/y/m.onnx"]}}]}}"#
        ))
        .unwrap();
        c.id = id.into();
        c
    }

    fn catalog(models: Vec<ModelCard>) -> Catalog {
        Catalog {
            models,
            fetched_at: std::time::SystemTime::now(),
        }
    }

    fn installed_entry(id: &str) -> InstalledModel {
        InstalledModel {
            id: id.into(),
            name: id.into(),
            arch: "esrgan".into(),
            scale: 4,
            license: Some("MIT".into()),
            format: "onnx".into(),
            file: "m.onnx".into(),
            sha256: None,
            verified: false,
            source_url: String::new(),
            installed_at: String::new(),
        }
    }

    #[test]
    fn restore_goal_only_returns_1x() {
        let cat = Catalog::parse(SAMPLE).unwrap();
        let mut a = ask();
        a.goal = Goal::Restore;
        a.caps.native_archs = vec!["esrgan".into(), "compact".into(), "span".into()];
        let got = recommend(&cat, &a, 100);
        assert!(!got.is_empty(), "the sample has 1x models");
        assert!(got.iter().all(|r| r.card.scale == 1));
    }

    #[test]
    fn upscale_goal_excludes_1x() {
        let cat = Catalog::parse(SAMPLE).unwrap();
        let mut a = ask();
        a.goal = Goal::Upscale;
        a.caps.native_archs = vec!["esrgan".into(), "compact".into(), "span".into()];
        let got = recommend(&cat, &a, 100);
        assert!(!got.is_empty(), "the sample has upscalers");
        assert!(got.iter().all(|r| r.card.scale > 1));
    }

    #[test]
    fn fastest_prefers_compact_over_dat() {
        let cat = catalog(vec![
            card("4x-dat", "dat", 4, "MIT", &[], "2023-01-01"),
            card("4x-compact", "compact", 4, "MIT", &[], "2023-01-01"),
        ]);
        let mut a = ask();
        a.speed = SpeedPref::Fastest;
        let got = recommend(&cat, &a, 5);
        assert_eq!(got[0].card.id, "4x-compact");
        assert!(got[0].score > got[1].score);
        a.speed = SpeedPref::Quality;
        assert_eq!(recommend(&cat, &a, 5)[0].card.id, "4x-dat");
    }

    #[test]
    fn commercial_only_excludes_nc() {
        let cat = catalog(vec![
            card("nc", "esrgan", 4, "CC-BY-NC-4.0", &[], "2023-01-01"),
            card("unknown", "esrgan", 4, "Custom", &[], "2023-01-01"),
            card("mit", "esrgan", 4, "MIT", &[], "2023-01-01"),
        ]);
        let mut a = ask();
        a.commercial_only = true;
        let ids: Vec<_> = recommend(&cat, &a, 5)
            .into_iter()
            .map(|r| r.card.id)
            .collect();
        assert_eq!(ids, ["mit"]);
        a.commercial_only = false;
        let got = recommend(&cat, &a, 5);
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].card.id, "mit", "the penalty puts NC last");
    }

    #[test]
    fn installed_ranks_above_equal_downloadable() {
        let cat = catalog(vec![
            card("a", "esrgan", 4, "MIT", &[], "2023-01-01"),
            card("b", "esrgan", 4, "MIT", &[], "2023-01-01"),
        ]);
        let mut a = ask();
        a.installed.upsert(installed_entry("b"));
        let got = recommend(&cat, &a, 5);
        assert_eq!(got[0].card.id, "b");
        assert_eq!(got[0].readiness, Readiness::Installed);
        assert_eq!(got[1].readiness, Readiness::Downloadable);
        assert!(got[0].score > got[1].score);
    }

    #[test]
    fn unsupported_never_returned() {
        let mut gray = card("gray", "esrgan", 4, "MIT", &[], "2023-01-01");
        gray.in_ch = 1;
        let cat = catalog(vec![
            gray,
            card("ok", "esrgan", 4, "MIT", &[], "2023-01-01"),
        ]);
        let got = recommend(&cat, &ask(), 5);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].card.id, "ok");
        let sample = Catalog::parse(SAMPLE).unwrap();
        assert!(
            recommend(&sample, &ask(), 100)
                .iter()
                .all(|r| r.readiness != Readiness::Unsupported)
        );
    }

    #[test]
    fn reasons_non_empty() {
        let cat = Catalog::parse(SAMPLE).unwrap();
        let mut a = ask();
        a.content = Content::Anime;
        a.caps.native_archs = vec!["esrgan".into(), "compact".into()];
        let got = recommend(&cat, &a, 5);
        assert!(!got.is_empty());
        assert!(got.iter().all(|r| !r.reasons.is_empty()));
        assert!(got[0].reasons.iter().any(|r| r == "tagged anime"));
    }

    #[test]
    fn scale_mismatch_is_excluded_and_n_truncates() {
        let cat = catalog(vec![
            card("x2", "esrgan", 2, "MIT", &[], "2023-01-01"),
            card("x4a", "esrgan", 4, "MIT", &[], "2023-01-01"),
            card("x4b", "esrgan", 4, "MIT", &[], "2024-01-01"),
        ]);
        let mut a = ask();
        a.scale = Some(4);
        let got = recommend(&cat, &a, 10);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].card.id, "x4b");
        assert_eq!(recommend(&cat, &a, 1).len(), 1);
    }

    #[test]
    fn ties_go_to_the_newer_model() {
        // Both dated before 2024, so no date bonus: the tie is broken by the date alone.
        let cat = catalog(vec![
            card("old", "esrgan", 4, "MIT", &[], "2021-01-01"),
            card("new", "esrgan", 4, "MIT", &[], "2023-06-01"),
        ]);
        let got = recommend(&cat, &ask(), 2);
        assert_eq!(got[0].score, got[1].score);
        assert_eq!(got[0].card.id, "new");
    }

    #[test]
    fn content_goal_and_bonus_rules_score() {
        let score_of = |tags: &[&str], date: &str, a: &Ask| {
            let cat = catalog(vec![card("m", "esrgan", 4, "MIT", tags, date)]);
            recommend(&cat, a, 1)[0].score
        };
        let mut a = ask();
        let base = score_of(&[], "2023-01-01", &a);
        assert_eq!(score_of(&[], "2024-03-01", &a) - base, 0.5);
        assert_eq!(score_of(&["pretrained"], "2023-01-01", &a) - base, 1.0);

        a.content = Content::Photo;
        let photo = score_of(&[], "2023-01-01", &a);
        assert_eq!(
            score_of(&["general-upscaler"], "2023-01-01", &a) - photo,
            2.0
        );
        assert_eq!(score_of(&["photo"], "2023-01-01", &a) - photo, 3.0);
        assert_eq!(score_of(&["pretrained"], "2023-01-01", &a) - photo, 0.0);

        a.content = Content::Anime;
        let anime = score_of(&[], "2023-01-01", &a);
        assert_eq!(score_of(&["anime"], "2023-01-01", &a) - anime, 3.0);
        assert_eq!(
            score_of(&["general-upscaler"], "2023-01-01", &a) - anime,
            0.0
        );

        a.content = Content::Any;
        for (goal, tags) in [
            (Goal::Denoise, vec!["denoise"]),
            (Goal::Dejpeg, vec!["jpeg"]),
            (Goal::Dejpeg, vec!["compression-removal"]),
            (Goal::Deblur, vec!["deblur"]),
        ] {
            a.goal = goal;
            let none = score_of(&[], "2023-01-01", &a);
            assert_eq!(score_of(&tags, "2023-01-01", &a) - none, 2.0, "{goal:?}");
        }
    }

    #[test]
    fn images_and_licence_rules_score() {
        let mut with_images = card("m", "esrgan", 4, "MIT", &[], "2023-01-01");
        let plain = recommend(&catalog(vec![with_images.clone()]), &ask(), 1)[0].score;
        with_images.images = vec![serde_json::json!({})];
        let got = recommend(&catalog(vec![with_images]), &ask(), 1)[0].score;
        assert_eq!(got - plain, 1.0);

        let nc = card("m", "esrgan", 4, "CC-BY-NC-4.0", &[], "2023-01-01");
        let nc_score = recommend(&catalog(vec![nc]), &ask(), 1)[0].score;
        assert_eq!(plain - nc_score, 1.0);
    }

    #[test]
    fn needs_conversion_and_manual_download_are_labelled_and_penalised() {
        let mut pth = card("pth", "esrgan", 4, "MIT", &[], "2023-01-01");
        pth.resources[0].platform = Some("pytorch".into());
        pth.resources[0].kind = "pth".into();
        let mut manual = card("manual", "esrgan", 4, "MIT", &[], "2023-01-01");
        manual.resources[0].urls = vec!["https://mega.nz/file/x".into()];
        let ok = card("ok", "esrgan", 4, "MIT", &[], "2023-01-01");
        let got = recommend(&catalog(vec![pth, manual, ok]), &ask(), 5);
        let ready: Vec<_> = got
            .iter()
            .map(|r| (r.card.id.as_str(), r.readiness))
            .collect();
        assert_eq!(
            ready,
            [
                ("ok", Readiness::Downloadable),
                ("manual", Readiness::ManualDownload),
                ("pth", Readiness::NeedsConversion)
            ]
        );
    }

    #[test]
    fn parse_content() {
        assert_eq!(Content::parse("Photo"), Ok(Content::Photo));
        assert_eq!(Content::parse("pixel-art"), Ok(Content::PixelArt));
        assert_eq!(Content::parse("PIXEL_ART"), Ok(Content::PixelArt));
        assert_eq!(Content::parse("game_textures"), Ok(Content::GameTexture));
        assert_eq!(Content::parse("ai-generated"), Ok(Content::AiGenerated));
        let err = Content::parse("sketch").unwrap_err();
        assert!(
            err.contains("sketch") && err.contains("game-textures"),
            "{err}"
        );
    }

    #[test]
    fn parse_goal() {
        assert_eq!(Goal::parse("Upscale"), Ok(Goal::Upscale));
        assert_eq!(Goal::parse("dejpeg"), Ok(Goal::Dejpeg));
        assert_eq!(Goal::parse("ANY"), Ok(Goal::Any));
        let err = Goal::parse("sharpen").unwrap_err();
        assert!(err.contains("sharpen") && err.contains("deblur"), "{err}");
    }

    #[test]
    fn parse_speed() {
        assert_eq!(SpeedPref::parse("Fastest"), Ok(SpeedPref::Fastest));
        assert_eq!(SpeedPref::parse("balanced"), Ok(SpeedPref::Balanced));
        assert_eq!(SpeedPref::parse("QUALITY"), Ok(SpeedPref::Quality));
        let err = SpeedPref::parse("slow").unwrap_err();
        assert!(err.contains("slow") && err.contains("quality"), "{err}");
    }
}
