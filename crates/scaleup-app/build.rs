//! Embeds `models/licenses/<id>.txt` as a `match` from model id to license text.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let dir = manifest.join("../../models/licenses");
    println!("cargo:rerun-if-changed={}", dir.display());

    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .expect("models/licenses exists")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "txt"))
        .collect();
    files.sort();

    let mut code = String::from(
        "/// The license text shipped for model `id`.\npub fn license_text(id: &str) -> Option<&'static str> {\n    match id {\n",
    );
    for file in &files {
        let id = file.file_stem().unwrap().to_string_lossy();
        let path = fs::canonicalize(file).unwrap();
        writeln!(code, "        {id:?} => Some(include_str!({path:?})),").unwrap();
    }
    code.push_str("        _ => None,\n    }\n}\n");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::write(Path::new(&out).join("licenses.rs"), code).unwrap();
}
