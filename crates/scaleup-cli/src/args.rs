//! Command line definition, compatible with the `upscayl-bin` short flags.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use scaleup_core::DevicePreference;

#[derive(Parser, Debug)]
#[command(
    name = "scaleup",
    version,
    about = "AI image upscaler",
    subcommand_negates_reqs = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Option<Cmd>,
    /// Bare `scaleup -i .. -o ..` is the same as `scaleup upscale`.
    #[command(flatten)]
    pub upscale: Option<UpscaleArgs>,
}

#[derive(Subcommand, Debug)]
pub enum Cmd {
    /// Upscale an image or a folder of images
    Upscale(Box<UpscaleArgs>),
    /// List, inspect and download models
    Models {
        #[command(subcommand)]
        cmd: ModelsCmd,
    },
    /// List the devices inference can run on
    Devices,
}

#[derive(Subcommand, Debug)]
pub enum ModelsCmd {
    /// Table of all catalogue models
    List,
    /// Details and license of one model
    Info { id: String },
    /// Download a model (verified against its sha256)
    Download { id: String },
    /// Print where the model file lives (or would live)
    Path { id: String },
    /// Copy a folder of ncnn .param/.bin pairs into the models folder and list them
    Import { dir: PathBuf },
}

#[derive(Args, Debug, Clone)]
pub struct UpscaleArgs {
    /// Input image or folder
    #[arg(short = 'i', long)]
    pub input: PathBuf,
    /// Output file or folder
    #[arg(short = 'o', long)]
    pub output: PathBuf,
    /// Model id from the catalogue, or path to an .onnx file
    #[arg(short = 'n', long, default_value = "realesr-general-x4v3")]
    pub model: String,
    /// Models directory
    #[arg(short = 'm', long)]
    pub models_dir: Option<PathBuf>,
    /// Output scale 2..4 (default: the model's scale)
    #[arg(short = 's', long, value_parser = clap::value_parser!(u32).range(2..=4))]
    pub scale: Option<u32>,
    /// Output width in pixels (overrides -s)
    #[arg(short = 'w', long, value_parser = clap::value_parser!(u32).range(1..))]
    pub width: Option<u32>,
    /// Output format: png, jpg or webp (default: output extension, else png)
    #[arg(short = 'f', long)]
    pub format: Option<String>,
    /// Quality 0..100 for jpg; PNG compression level
    #[arg(short = 'c', long, default_value_t = 90, value_parser = clap::value_parser!(u8).range(0..=100))]
    pub quality: u8,
    /// Tile size, 0 = automatic, otherwise at least 32
    #[arg(short = 't', long, default_value_t = 0, value_parser = parse_tile)]
    pub tile: u32,
    /// Pixels of context around each tile, capped at a quarter tile (default: what the model asks for)
    #[arg(long)]
    pub tile_overlap: Option<u32>,
    /// Device: auto, cpu or a GPU index
    #[arg(short = 'g', long, default_value = "auto", value_parser = validate_gpu)]
    pub gpu: String,
    /// Test-time augmentation (8x slower, slightly better)
    #[arg(short = 'x', long)]
    pub tta: bool,
    /// Run the model twice
    #[arg(long)]
    pub double: bool,
    /// Replace existing output files
    #[arg(long)]
    pub overwrite: bool,
    /// Files processed in parallel in folder mode
    #[arg(long, default_value_t = 1, value_parser = parse_jobs)]
    pub jobs: usize,
    /// Print phases and device information
    #[arg(short = 'v', long)]
    pub verbose: bool,
    /// One JSON object per line on stderr
    #[arg(long)]
    pub json: bool,
    /// Accepted for upscayl-bin compatibility, ignored
    #[arg(short = 'z', long = "model-scale")]
    pub model_scale: Option<u32>,
    /// upscayl-bin `load:proc:save` thread counts; accepted and ignored
    #[arg(short = 'j', long)]
    pub threads: Option<String>,
}

fn parse_jobs(s: &str) -> Result<usize, String> {
    match s.parse::<usize>() {
        Ok(n) if n >= 1 => Ok(n),
        _ => Err(format!("'{s}' is not a positive number")),
    }
}

fn parse_tile(s: &str) -> Result<u32, String> {
    let t: u32 = s.parse().map_err(|_| format!("'{s}' is not a number"))?;
    if t == 0 || t >= 32 {
        Ok(t)
    } else {
        Err("tile size must be 0 (auto) or at least 32".into())
    }
}

fn validate_gpu(s: &str) -> Result<String, String> {
    parse_gpu(s).map(|_| s.to_string())
}

/// `auto`, `cpu` or a single digit GPU index.
pub fn parse_gpu(s: &str) -> Result<DevicePreference, String> {
    DevicePreference::parse(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_gpu_cases() {
        assert_eq!(parse_gpu("auto"), Ok(DevicePreference::Auto));
        assert_eq!(parse_gpu("CPU"), Ok(DevicePreference::Cpu));
        assert_eq!(parse_gpu("0"), Ok(DevicePreference::Gpu(0)));
        assert_eq!(parse_gpu("9"), Ok(DevicePreference::Gpu(9)));
        assert_eq!(parse_gpu("vk:1"), Ok(DevicePreference::Vulkan(1)));
        assert!(parse_gpu("10").is_err());
        assert!(parse_gpu("bogus").is_err());
        assert!(parse_gpu("").is_err());
    }

    #[test]
    fn tile_rules() {
        assert_eq!(parse_tile("0"), Ok(0));
        assert_eq!(parse_tile("32"), Ok(32));
        assert!(parse_tile("16").is_err());
        assert!(parse_tile("x").is_err());
    }
}
