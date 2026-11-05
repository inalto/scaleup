//! Optional PyTorch → ONNX conversion through the user's own Python (spandrel, torch, onnx).
//!
//! Python only ever runs because the user asked for a conversion; nothing here is called
//! implicitly. The script is `tools/convert_to_onnx.py`, embedded and written to a private
//! temporary folder for each run, so installed packages need not ship it.

use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use super::catalog::{ModelCard, Resource};
use super::download::{Fetched, Host, Http, check_id, classify_url, fetch, register};
use super::installed::InstalledModel;
use crate::{Result, ScaleUpError};

/// The line that installs what a conversion needs. Error messages carry it on a line of its own.
pub const PIP_INSTALL: &str = "pip install spandrel torch onnx";
/// A path to the Python executable to use; wins over `python3` / `python` on `PATH`.
pub const PYTHON_ENV: &str = "SCALEUP_PYTHON";
const SCRIPT: &str = include_str!("../../../../tools/convert_to_onnx.py");
const SCRIPT_NAME: &str = "convert_to_onnx.py";
const CHECKPOINT_EXTENSIONS: [&str; 2] = ["pth", "safetensors"];

/// A Python 3 that imports spandrel, torch and onnx.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Python {
    exe: PathBuf,
}

impl Python {
    pub fn path(&self) -> &Path {
        &self.exe
    }
}

/// A fresh folder of our own under the temporary directory, removed on drop. Python runs inside
/// it, so the script's folder and the working directory hold nothing that could be imported.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Result<Scratch> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        let dir =
            std::env::temp_dir().join(format!("scaleup-convert-{}-{nanos}", std::process::id()));
        // `create_dir`, not `create_dir_all`: a folder someone prepared is not ours.
        fs::create_dir(&dir)?;
        Ok(Scratch(dir))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn python_command(exe: &OsStr, cwd: &Path) -> Command {
    let mut cmd = Command::new(exe);
    cmd.current_dir(cwd);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW: the desktop app has no console for Python to borrow.
        cmd.creation_flags(0x0800_0000);
    }
    cmd
}

fn run_code(exe: &OsStr, cwd: &Path, code: &str) -> Option<Output> {
    python_command(exe, cwd).args(["-c", code]).output().ok()
}

/// [`find_python_from`] with `SCALEUP_PYTHON` when it is set.
pub fn find_python() -> Result<Python> {
    find_python_from(std::env::var_os(PYTHON_ENV).filter(|v| !v.is_empty()))
}

/// The first of `explicit` (alone, when given) or `python3`, `python` on `PATH` that is Python 3
/// and imports spandrel, torch and onnx. Fails with [`ScaleUpError::Model`] carrying
/// [`PIP_INSTALL`] on a line of its own.
pub fn find_python_from(explicit: Option<OsString>) -> Result<Python> {
    let scratch = Scratch::new()?;
    let candidates: Vec<OsString> = match &explicit {
        Some(p) => vec![p.clone()],
        None => vec!["python3".into(), "python".into()],
    };
    let mut without_modules = None;
    for exe in candidates {
        // The Windows Store stub answers to `python3` without being Python.
        let is_python3 = run_code(&exe, &scratch.0, "import sys; print(sys.version_info[0])")
            .is_some_and(|o| {
                o.status.success() && String::from_utf8_lossy(&o.stdout).trim() == "3"
            });
        if !is_python3 {
            continue;
        }
        if run_code(&exe, &scratch.0, "import spandrel, torch, onnx")
            .is_some_and(|o| o.status.success())
        {
            return Ok(Python { exe: exe.into() });
        }
        without_modules.get_or_insert(exe);
    }
    Err(ScaleUpError::Model(match (without_modules, explicit) {
        (Some(exe), _) => format!(
            "{} lacks spandrel, torch or onnx, which converting a model needs. Install them with:\n{PIP_INSTALL}",
            PathBuf::from(exe).display()
        ),
        (None, Some(exe)) => format!(
            "{PYTHON_ENV}={} is not a Python 3 that runs. Converting a model needs Python 3 with:\n{PIP_INSTALL}",
            PathBuf::from(exe).display()
        ),
        (None, None) => format!(
            "converting a model needs Python 3, and neither python3 nor python is on PATH (or set {PYTHON_ENV}). Install Python, then:\n{PIP_INSTALL}"
        ),
    }))
}

/// Checks that `input` is an existing `.pth` or `.safetensors` file, before any Python runs.
pub fn check_checkpoint(input: &Path) -> Result<()> {
    if !input.is_file() {
        return Err(ScaleUpError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("{}: no such file", input.display()),
        )));
    }
    let ext = input
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if CHECKPOINT_EXTENSIONS.contains(&ext.as_str()) {
        Ok(())
    } else {
        Err(ScaleUpError::InvalidOption(format!(
            "{}: not a PyTorch checkpoint (expected .pth or .safetensors)",
            input.display()
        )))
    }
}

/// Converts checkpoint `input` to the ONNX file `output` (input `input` float32 [1,3,H,W] with
/// dynamic H/W, output `output`, opset 17). Fails with the script's one-line reason.
pub fn convert_to_onnx(python: &Python, input: &Path, output: &Path) -> Result<()> {
    check_checkpoint(input)?;
    let input = std::path::absolute(input)?;
    let output = std::path::absolute(output)?;
    let scratch = Scratch::new()?;
    let script = scratch.0.join(SCRIPT_NAME);
    fs::write(&script, SCRIPT)?;
    let run = python_command(python.exe.as_os_str(), &scratch.0)
        .arg(&script)
        .arg(&input)
        .arg(&output)
        .output()
        .map_err(|e| ScaleUpError::Model(format!("{}: {e}", python.exe.display())))?;
    if !run.status.success() {
        let stderr = String::from_utf8_lossy(&run.stderr);
        let reason = stderr
            .lines()
            .rev()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .map_or_else(
                || format!("Python exited with {}", run.status),
                str::to_owned,
            );
        return Err(ScaleUpError::Model(format!(
            "converting {}: {reason}",
            input.display()
        )));
    }
    if !output.is_file() {
        return Err(ScaleUpError::Model(format!(
            "converting {}: Python reported success but wrote no {}",
            input.display(),
            output.display()
        )));
    }
    Ok(())
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    std::io::copy(&mut File::open(path)?, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// The PyTorch file of `card` to convert and its URL index: one on a host the app downloads
/// from, safetensors before pth.
fn pick_pytorch(card: &ModelCard) -> Option<(&Resource, usize)> {
    card.resources
        .iter()
        .filter(|r| {
            r.platform.as_deref() == Some("pytorch")
                && CHECKPOINT_EXTENSIONS
                    .iter()
                    .any(|e| r.kind.eq_ignore_ascii_case(e))
        })
        .filter_map(|r| {
            let idx = r
                .urls
                .iter()
                .position(|u| classify_url(u) != Host::Manual)?;
            Some((
                u8::from(!r.kind.eq_ignore_ascii_case("safetensors")),
                r,
                idx,
            ))
        })
        .min_by_key(|(rank, _, _)| *rank)
        .map(|(_, r, idx)| (r, idx))
}

/// Downloads the PyTorch file of `card` into `<models_dir>/hub/<id>/`, verifies it, converts it
/// with `python` and records the ONNX file in `installed.json` (`format: "onnx"`, `verified` as
/// the download was, the ONNX file's SHA-256). `converting` is called between the two steps.
/// The PyTorch file is deleted afterwards, and on failure.
pub fn install_converted(
    card: &ModelCard,
    models_dir: &Path,
    python: &Python,
    http: &dyn Http,
    progress: &mut dyn FnMut(u64, Option<u64>),
    converting: &mut dyn FnMut(),
) -> Result<InstalledModel> {
    check_id(&card.id)?;
    if card.in_ch != 3 || card.out_ch != 3 {
        return Err(ScaleUpError::Model(format!(
            "'{}' is not an RGB model ({} in, {} out)",
            card.id, card.in_ch, card.out_ch
        )));
    }
    let Some((resource, idx)) = pick_pytorch(card) else {
        return Err(ScaleUpError::Download(format!(
            "'{}' has no PyTorch file the app can download; get one by hand and run `scaleup models convert <file>`",
            card.id
        )));
    };
    let Fetched {
        file,
        sha256: _,
        verified,
        url,
        ..
    } = fetch(card, resource, idx, models_dir, http, progress)?;
    converting();
    let onnx = file.with_extension("onnx");
    let converted = convert_to_onnx(python, &file, &onnx).and_then(|()| file_sha256(&onnx));
    let _ = fs::remove_file(&file);
    let sha = match converted {
        Ok(sha) => sha,
        Err(e) => {
            let _ = fs::remove_file(&onnx);
            return Err(e);
        }
    };
    register(card, models_dir, onnx, "onnx".into(), sha, verified, &url)
}

#[cfg(test)]
mod tests {
    use super::super::download::testing::{Fake, card, resource, resp, sha};
    use super::super::installed::Installed;
    use super::*;

    const GH_PTH: &str = "https://github.com/a/b/releases/download/v1/4x-Test.pth";

    fn bogus_python() -> Python {
        Python {
            exe: PathBuf::from("scaleup-no-such-python-xyz"),
        }
    }

    fn has_pip_line(msg: &str) -> bool {
        msg.lines().any(|l| l.trim() == PIP_INSTALL)
    }

    #[test]
    fn missing_python_names_the_pip_line() {
        let err = find_python_from(Some("scaleup-no-such-python-xyz".into())).unwrap_err();
        assert!(matches!(err, ScaleUpError::Model(_)), "{err:?}");
        let msg = err.to_string();
        assert!(has_pip_line(&msg), "{msg}");
        assert!(
            msg.contains("SCALEUP_PYTHON=scaleup-no-such-python-xyz"),
            "{msg}"
        );
    }

    #[test]
    fn check_checkpoint_wants_an_existing_pytorch_file() {
        let dir = tempfile::tempdir().unwrap();
        let pth = dir.path().join("m.PTH");
        let st = dir.path().join("m.safetensors");
        let onnx = dir.path().join("m.onnx");
        for f in [&pth, &st, &onnx] {
            fs::write(f, b"x").unwrap();
        }
        check_checkpoint(&pth).unwrap();
        check_checkpoint(&st).unwrap();
        assert!(matches!(
            check_checkpoint(&onnx),
            Err(ScaleUpError::InvalidOption(_))
        ));
        let missing = check_checkpoint(&dir.path().join("gone.pth")).unwrap_err();
        assert!(matches!(missing, ScaleUpError::Io(_)), "{missing:?}");
        assert!(missing.to_string().contains("gone.pth"), "{missing}");
    }

    #[test]
    fn convert_with_a_python_that_does_not_run_fails_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        let pth = dir.path().join("m.pth");
        fs::write(&pth, b"x").unwrap();
        let out = dir.path().join("m.onnx");
        let err = convert_to_onnx(&bogus_python(), &pth, &out).unwrap_err();
        assert!(
            err.to_string().contains("scaleup-no-such-python-xyz"),
            "{err}"
        );
        assert!(!out.exists());
    }

    #[test]
    fn pick_pytorch_prefers_downloadable_safetensors() {
        let mega = "https://mega.nz/x";
        let c = card(
            "dat",
            vec![
                resource("onnx", "onnx", None, &[GH_PTH]),
                resource("pth", "pytorch", None, &[GH_PTH]),
                resource("safetensors", "pytorch", None, &[mega, GH_PTH]),
            ],
        );
        let (r, idx) = pick_pytorch(&c).unwrap();
        assert_eq!((r.kind.as_str(), idx), ("safetensors", 1));
        let only_manual = card("dat", vec![resource("pth", "pytorch", None, &[mega])]);
        assert!(pick_pytorch(&only_manual).is_none());
    }

    #[test]
    fn install_converted_without_a_pytorch_file_makes_no_request() {
        let dir = tempfile::tempdir().unwrap();
        let c = card("dat", vec![resource("onnx", "onnx", None, &[GH_PTH])]);
        let http = Fake::default();
        let err = install_converted(
            &c,
            dir.path(),
            &bogus_python(),
            &http,
            &mut |_, _| {},
            &mut || {},
        )
        .unwrap_err();
        assert!(err.to_string().contains("scaleup models convert"), "{err}");
        assert!(http.requests().is_empty());
    }

    #[test]
    fn install_converted_sha_mismatch_stops_before_converting() {
        let dir = tempfile::tempdir().unwrap();
        let c = card(
            "dat",
            vec![resource("pth", "pytorch", Some(sha(b"right")), &[GH_PTH])],
        );
        let http = Fake::with(vec![resp(b"wrong", "application/octet-stream")]);
        let mut converted = false;
        let err = install_converted(
            &c,
            dir.path(),
            &bogus_python(),
            &http,
            &mut |_, _| {},
            &mut || converted = true,
        )
        .unwrap_err();
        assert!(err.to_string().contains("sha256 mismatch"), "{err}");
        assert!(!converted);
        assert!(Installed::load(dir.path()).models.is_empty());
    }

    #[test]
    fn install_converted_failure_leaves_nothing_behind() {
        let dir = tempfile::tempdir().unwrap();
        let body = b"checkpoint".to_vec();
        let c = card(
            "dat",
            vec![resource("pth", "pytorch", Some(sha(&body)), &[GH_PTH])],
        );
        let http = Fake::with(vec![resp(&body, "application/octet-stream")]);
        let mut converted = false;
        let err = install_converted(
            &c,
            dir.path(),
            &bogus_python(),
            &http,
            &mut |_, _| {},
            &mut || converted = true,
        )
        .unwrap_err();
        assert!(
            converted,
            "the download was verified, so conversion started"
        );
        assert!(matches!(err, ScaleUpError::Model(_)), "{err:?}");
        let folder = dir.path().join("hub").join("4x-Test");
        assert_eq!(fs::read_dir(&folder).unwrap().count(), 0);
        assert!(Installed::load(dir.path()).models.is_empty());
    }

    /// Run by hand: needs `tools/.venv` and `models/cache/src/realesr-animevideov3.pth`.
    #[test]
    #[ignore = "needs the tools/.venv Python and a real checkpoint"]
    fn convert_real_checkpoint_with_venv_python() {
        use crate::backend::{Backend, OrtBackend};
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let venv = if cfg!(windows) {
            root.join("tools/.venv/Scripts/python.exe")
        } else {
            root.join("tools/.venv/bin/python")
        };
        let pth = root.join("models/cache/src/realesr-animevideov3.pth");
        if !venv.is_file() || !pth.is_file() {
            eprintln!(
                "skipped: {} or {} is missing",
                venv.display(),
                pth.display()
            );
            return;
        }
        let python = find_python_from(Some(venv.into_os_string())).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("animevideov3.onnx");
        convert_to_onnx(&python, &pth, &out).unwrap();
        let backend = OrtBackend::new(crate::DevicePreference::Cpu).unwrap();
        let model = backend.load_model(&out, None).unwrap();
        assert_eq!(model.scale(), 4);
        let tile = crate::Image32::new(16, 16);
        let up = model.run(&tile).unwrap();
        assert_eq!((up.width, up.height), (64, 64));
        eprintln!(
            "converted {} -> {} ({} bytes), scale {}",
            pth.display(),
            out.display(),
            fs::metadata(&out).unwrap().len(),
            model.scale()
        );
    }
}
