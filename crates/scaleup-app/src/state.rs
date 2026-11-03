//! Everything the UI remembers: persisted [`Settings`] and the transient [`AppState`].

use std::path::{Path, PathBuf};

use scaleup_core::UpscaleOptions;
use serde::{Deserialize, Serialize};

/// Every setting visible in the sidebar; survives restarts.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Settings {
    /// Catalogue id, or a path to a model file.
    pub model_id: String,
    pub scale: u32,
    pub use_width: bool,
    pub width: u32,
    pub format: String,
    pub quality: u8,
    /// 0 = automatic.
    pub tile: u32,
    /// `auto`, `cpu` or a GPU index.
    pub gpu: String,
    pub tta: bool,
    pub double: bool,
    pub overwrite: bool,
    pub output_dir: Option<PathBuf>,
    pub show_nc_models: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model_id: "realesr-general-x4v3".into(),
            scale: 4,
            use_width: false,
            width: 1920,
            format: "png".into(),
            quality: 90,
            tile: 0,
            gpu: "auto".into(),
            tta: false,
            double: false,
            overwrite: false,
            output_dir: None,
            show_nc_models: false,
        }
    }
}

impl Settings {
    /// The core options these settings stand for. A custom width overrides the scale.
    pub fn to_options(&self) -> UpscaleOptions {
        UpscaleOptions {
            tile_size: self.tile,
            tta: self.tta,
            output_scale: (!self.use_width).then_some(self.scale),
            output_width: self.use_width.then_some(self.width),
            ..UpscaleOptions::default()
        }
    }

    /// The configured output folder, else the folder `input` is in.
    pub fn output_dir_for(&self, input: &Path) -> PathBuf {
        self.output_dir
            .clone()
            .unwrap_or_else(|| input.parent().map(Path::to_path_buf).unwrap_or_default())
    }
}

/// What the next run will process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    Single(PathBuf),
    Batch(PathBuf),
}

#[derive(Clone, Debug, PartialEq, Default)]
pub enum Status {
    #[default]
    Idle,
    Downloading {
        done: u64,
        total: u64,
    },
    Running {
        file: usize,
        files: usize,
        percent: f32,
        phase: String,
    },
    Done(PathBuf),
    Error(String),
    Stopped {
        done: usize,
        total: usize,
    },
}

#[derive(Clone, Debug, Default)]
pub struct AppState {
    pub settings: Settings,
    pub mode: Option<Mode>,
    pub status: Status,
    pub last_result: Option<PathBuf>,
    pub device_name: String,
    pub log: Vec<String>,
}

impl AppState {
    pub fn to_options(&self) -> UpscaleOptions {
        self.settings.to_options()
    }

    /// The configured output folder, else the folder `input` is in.
    pub fn output_dir_for(&self, input: &Path) -> PathBuf {
        self.settings.output_dir_for(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_default_values() {
        let s = Settings::default();
        assert_eq!(s.model_id, "realesr-general-x4v3");
        assert_eq!(s.scale, 4);
        assert!(!s.use_width);
        assert_eq!(s.width, 1920);
        assert_eq!(s.format, "png");
        assert_eq!(s.quality, 90);
        assert_eq!(s.tile, 0);
        assert_eq!(s.gpu, "auto");
        assert!(!s.tta && !s.double && !s.overwrite && !s.show_nc_models);
        assert_eq!(s.output_dir, None);
    }

    #[test]
    fn settings_json_roundtrip() {
        let s = Settings {
            model_id: "x".into(),
            scale: 2,
            use_width: true,
            width: 800,
            format: "webp".into(),
            quality: 40,
            tile: 256,
            gpu: "cpu".into(),
            tta: true,
            double: true,
            overwrite: true,
            output_dir: Some(PathBuf::from("out")),
            show_nc_models: true,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<Settings>(&json).unwrap(), s);
    }

    #[test]
    fn settings_from_older_json_fill_missing_fields() {
        let s: Settings = serde_json::from_str(r#"{"scale":2}"#).unwrap();
        assert_eq!(s.scale, 2);
        assert_eq!(s.format, "png");
    }

    #[test]
    fn to_options_width_overrides_scale() {
        let mut st = AppState::default();
        st.settings.tile = 128;
        st.settings.tta = true;
        let o = st.to_options();
        assert_eq!(o.output_scale, Some(4));
        assert_eq!(o.output_width, None);
        assert_eq!(o.tile_size, 128);
        assert!(o.tta);
        st.settings.use_width = true;
        let o = st.to_options();
        assert_eq!(o.output_scale, None);
        assert_eq!(o.output_width, Some(1920));
    }

    #[test]
    fn output_dir_defaults_to_input_parent() {
        let mut st = AppState::default();
        assert_eq!(
            st.output_dir_for(Path::new("a/b/c.png")),
            PathBuf::from("a/b")
        );
        st.settings.output_dir = Some(PathBuf::from("out"));
        assert_eq!(
            st.output_dir_for(Path::new("a/b/c.png")),
            PathBuf::from("out")
        );
    }
}
