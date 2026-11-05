//! `scaleup models ...` and registry construction.

use std::path::Path;

use scaleup_core::registry::parse_manifest;
use scaleup_core::{CancelToken, Registry, ScaleUpError};

use crate::args::ModelsCmd;

/// Registry for `dir` (or the default models dir). `SCALEUP_TEST_MANIFEST` replaces the catalogue
/// with a manifest file; it exists for tests only.
pub fn registry_for(dir: Option<&Path>) -> Registry {
    let base = match dir {
        Some(d) => Registry::with_dir(d.to_path_buf()),
        None => Registry::builtin(),
    };
    let Some(path) = std::env::var_os("SCALEUP_TEST_MANIFEST") else {
        return base;
    };
    let entries = std::fs::read_to_string(path)
        .map_err(ScaleUpError::from)
        .and_then(|s| parse_manifest(&s))
        .expect("SCALEUP_TEST_MANIFEST must be a readable manifest");
    Registry::from_entries(entries, base.models_dir().to_path_buf())
}

macro_rules! licenses {
    ($($id:literal),* $(,)?) => {
        fn license_text(id: &str) -> Option<&'static str> {
            match id {
                $($id => Some(include_str!(concat!("../../../models/licenses/", $id, ".txt"))),)*
                _ => None,
            }
        }
    };
}

licenses!(
    "4x-UltraSharp",
    "4xNomosUni_span_multijpg",
    "4xRealWebPhoto_v4_dat2",
    "4x_foolhardy_Remacri",
    "realesr-animevideov3",
    "realesr-general-x4v3",
    "realesrgan-x2plus",
    "realesrgan-x4plus-anime",
    "realesrgan-x4plus",
);

fn unknown(id: &str) -> ScaleUpError {
    ScaleUpError::Model(format!("unknown model '{id}'"))
}

pub fn yes_no(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}

/// Prints `head` and `rows` as plain text with left-aligned columns two spaces apart.
pub fn print_table(head: &[&str], rows: Vec<Vec<String>>) {
    let head: Vec<String> = head.iter().map(|h| (*h).to_owned()).collect();
    let mut widths = vec![0usize; head.len()];
    for row in std::iter::once(&head).chain(rows.iter()) {
        for (w, cell) in widths.iter_mut().zip(row) {
            *w = (*w).max(cell.chars().count());
        }
    }
    for row in std::iter::once(&head).chain(rows.iter()) {
        let line: Vec<String> = row
            .iter()
            .zip(&widths)
            .map(|(c, w)| format!("{c:<w$}"))
            .collect();
        println!("{}", line.join("  ").trim_end());
    }
}

pub fn run_models(cmd: &ModelsCmd, cancel: &CancelToken) -> Result<i32, ScaleUpError> {
    let mut reg = registry_for(None);
    match cmd {
        ModelsCmd::List => {
            let rows = reg
                .entries()
                .iter()
                .map(|e| {
                    vec![
                        e.id.clone(),
                        format!("{}x", e.scale),
                        e.license.clone(),
                        yes_no(e.commercial).into(),
                        yes_no(reg.is_downloaded(&e.id)).into(),
                    ]
                })
                .collect();
            print_table(
                &["ID", "SCALE", "LICENSE", "COMMERCIAL", "DOWNLOADED"],
                rows,
            );
        }
        ModelsCmd::Info { id } => {
            let e = reg.get(id).ok_or_else(|| unknown(id))?;
            println!("id:          {}", e.id);
            println!("name:        {}", e.name);
            println!("arch:        {}", e.arch);
            println!("scale:       {}x", e.scale);
            println!("license:     {}", e.license);
            println!("commercial:  {}", yes_no(e.commercial));
            println!("downloaded:  {}", yes_no(reg.is_downloaded(id)));
            println!("path:        {}", reg.local_path(id).display());
            if !e.notes.is_empty() {
                println!("notes:       {}", e.notes);
            }
            if let Some(first) = license_text(id).and_then(|t| t.lines().next()) {
                println!("\n{first}");
            }
        }
        ModelsCmd::Download { id } => {
            reg.get(id).ok_or_else(|| unknown(id))?;
            let mut last = u64::MAX;
            let path = reg.ensure_cancellable(
                id,
                &mut |done, total| {
                    if total > 0 && done * 100 / total != last {
                        last = done * 100 / total;
                        eprintln!("{last}%");
                    }
                },
                cancel,
            )?;
            println!("{}", path.display());
        }
        ModelsCmd::Import { dir } => {
            for id in reg.import_ncnn(dir)? {
                println!("{id}");
            }
        }
        ModelsCmd::Convert { file, out } => {
            return crate::convert::run_convert(file, out.as_deref());
        }
        ModelsCmd::Path { id } => {
            reg.get(id).ok_or_else(|| unknown(id))?;
            println!("{}", reg.local_path(id).display());
        }
    }
    Ok(0)
}
