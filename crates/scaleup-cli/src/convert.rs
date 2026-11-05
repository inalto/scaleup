//! `scaleup models convert`: a PyTorch checkpoint to ONNX through the user's own Python.

use std::fs;
use std::path::Path;

use scaleup_core::ScaleUpError;
use scaleup_core::hub::{check_checkpoint, convert_to_onnx, find_python};

use crate::models::registry_for;

/// Converts `file` into `<out or models folder>/<stem>.onnx` and lists it as a local model.
/// Everything that can be checked without Python is checked first.
pub fn run_convert(file: &Path, out: Option<&Path>) -> Result<i32, ScaleUpError> {
    check_checkpoint(file)?;
    let mut reg = registry_for(None);
    let id = file
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    if reg.get(&id).is_some() && !reg.is_imported(&id) {
        return Err(ScaleUpError::InvalidOption(format!(
            "'{id}' is already the id of another model; rename {} and try again",
            file.display()
        )));
    }
    let python = find_python()?;
    let dir = out.unwrap_or(reg.models_dir()).to_path_buf();
    fs::create_dir_all(&dir)?;
    let onnx = dir.join(format!("{id}.onnx"));
    eprintln!("converting with {}", python.path().display());
    convert_to_onnx(&python, file, &onnx)?;
    let id = reg.import_onnx(&onnx)?;
    eprintln!(
        "listed as model '{id}'; its license is unknown, so it is marked non-commercial until you check"
    );
    println!("{}", reg.local_path(&id).display());
    Ok(0)
}
