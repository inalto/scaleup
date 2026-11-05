//! Optional PyTorch → ONNX conversion through the user's own Python (spandrel, torch, onnx).
//!
//! Python only ever runs because the user asked for a conversion; nothing here is called
//! implicitly. The script is `tools/convert_to_onnx.py`, embedded and written to a private
//! temporary folder for each run, so installed packages need not ship it.

use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use super::catalog::{Capabilities, ModelCard, Resource, is_native};
use super::download::{
    Checked, Fetched, Host, Http, check_id, check_import, classify_url, fetch,
    manual_pytorch_error, register,
};
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

/// How often a running conversion checks whether it was cancelled.
const POLL: Duration = Duration::from_millis(100);

/// The file the script writes before renaming it to `output`.
fn part_of(output: &Path) -> PathBuf {
    let mut name = output.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    output.with_file_name(name)
}

/// Converts checkpoint `input` to the ONNX file `output` (input `input` float32 [1,3,H,W] with
/// dynamic H/W, output `output`, opset 17). Fails with the script's one-line reason.
///
/// `cancelled` is asked about every 100 ms while Python runs; when it answers true, Python is
/// killed, `output` and its `.part` are removed and the result is [`ScaleUpError::Cancelled`].
pub fn convert_to_onnx(
    python: &Python,
    input: &Path,
    output: &Path,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<()> {
    check_checkpoint(input)?;
    let input = std::path::absolute(input)?;
    let output = std::path::absolute(output)?;
    let scratch = Scratch::new()?;
    let script = scratch.0.join(SCRIPT_NAME);
    fs::write(&script, SCRIPT)?;
    // A file, not a pipe: a pipe nobody reads while polling could fill up and stall Python.
    let log = scratch.0.join("stderr.txt");
    let mut child = python_command(python.exe.as_os_str(), &scratch.0)
        .arg(&script)
        .arg(&input)
        .arg(&output)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(File::create(&log)?)
        .spawn()
        .map_err(|e| ScaleUpError::Model(format!("{}: {e}", python.exe.display())))?;
    let discard = || {
        let _ = fs::remove_file(&output);
        let _ = fs::remove_file(part_of(&output));
    };
    let status = loop {
        if cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            discard();
            return Err(ScaleUpError::Cancelled);
        }
        match child.try_wait()? {
            Some(status) => break status,
            None => std::thread::sleep(POLL),
        }
    };
    // Ctrl-C in a console reaches Python too, which then fails: that is a cancel, not an error.
    if cancelled() {
        discard();
        return Err(ScaleUpError::Cancelled);
    }
    if !status.success() {
        discard();
        let stderr = fs::read_to_string(&log).unwrap_or_default();
        let reason = stderr
            .lines()
            .rev()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .map_or_else(|| format!("Python exited with {status}"), str::to_owned);
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

fn pytorch_resources(card: &ModelCard) -> impl Iterator<Item = &Resource> {
    card.resources.iter().filter(|r| {
        r.platform.as_deref() == Some("pytorch")
            && CHECKPOINT_EXTENSIONS
                .iter()
                .any(|e| r.kind.eq_ignore_ascii_case(e))
    })
}

/// A PyTorch file of `card` that is only on hosts the user opens in a browser, and the index of
/// its first `https` URL (else its first URL). `None` when one can be downloaded automatically.
pub(crate) fn manual_pytorch(card: &ModelCard) -> Option<(&Resource, usize)> {
    if pick_pytorch(card).is_some() {
        return None;
    }
    let r = pytorch_resources(card).find(|r| !r.urls.is_empty())?;
    let idx = r
        .urls
        .iter()
        .position(|u| u.starts_with("https://"))
        .unwrap_or(0);
    Some((r, idx))
}

/// Where the PyTorch file to convert comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PytorchSource {
    /// The app downloads it ([`install_converted`]).
    Download,
    /// The user opens `url` in a browser, saves the file as `file_name` and imports it
    /// ([`import_converted`]).
    Manual {
        url: String,
        file_name: String,
        sha256: Option<String>,
    },
    /// The card lists no PyTorch file.
    None,
}

/// How the PyTorch file of `card` can be had: the same "non-manual host" test as
/// [`install_converted`] and the error of [`super::install`].
pub fn pytorch_source(card: &ModelCard) -> PytorchSource {
    if pick_pytorch(card).is_some() {
        return PytorchSource::Download;
    }
    match manual_pytorch(card) {
        Some((r, idx)) => PytorchSource::Manual {
            url: r.urls[idx].clone(),
            file_name: super::download::expected_file_name(
                &r.urls[idx],
                &card.id,
                &r.kind.to_ascii_lowercase(),
            ),
            sha256: r.sha256.clone().filter(|s| !s.trim().is_empty()),
        },
        None => PytorchSource::None,
    }
}

/// Whether importing `file` for `card` means converting it: a PyTorch checkpoint of an
/// architecture this host does not run natively.
pub fn import_needs_conversion(card: &ModelCard, file: &Path, caps: &Capabilities) -> bool {
    let ext = file
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    CHECKPOINT_EXTENSIONS.contains(&ext.as_str()) && !is_native(card, caps)
}

/// The PyTorch file of `card` to convert and its URL index: one on a host the app downloads
/// from, safetensors before pth.
pub(crate) fn pick_pytorch(card: &ModelCard) -> Option<(&Resource, usize)> {
    pytorch_resources(card)
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
/// the download was, the ONNX file's SHA-256). `converting` is called between the two steps;
/// `cancelled` is asked while Python runs (see [`convert_to_onnx`]). The PyTorch file is deleted
/// afterwards, and on failure or cancel. A card whose PyTorch file is only on manual hosts fails
/// with where to get it and the `hub import` to run.
pub fn install_converted(
    card: &ModelCard,
    models_dir: &Path,
    python: &Python,
    http: &dyn Http,
    progress: &mut dyn FnMut(u64, Option<u64>),
    converting: &mut dyn FnMut(),
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<InstalledModel> {
    check_rgb(card)?;
    let Some((resource, idx)) = pick_pytorch(card) else {
        return Err(if manual_pytorch(card).is_some() {
            manual_pytorch_error(card)
        } else {
            ScaleUpError::Download(format!("'{}' lists no PyTorch file to convert", card.id))
        });
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
    let converted = convert_and_hash(python, &file, &onnx, cancelled);
    let _ = fs::remove_file(&file);
    register(
        card,
        models_dir,
        onnx,
        "onnx".into(),
        converted?,
        verified,
        &url,
    )
}

/// Checks a PyTorch file the user downloaded for `card` against the catalogue (as
/// [`super::import_file`] does), converts it into `<models_dir>/hub/<id>/<stem>.onnx` and records
/// that. The user's file stays where it is.
pub fn import_converted(
    card: &ModelCard,
    file: &Path,
    models_dir: &Path,
    python: &Python,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<InstalledModel> {
    check_rgb(card)?;
    check_checkpoint(file)?;
    let Checked {
        name,
        sha256: _,
        verified,
        source_url,
        ..
    } = check_import(card, file)?;
    let dir = models_dir.join("hub").join(&card.id);
    fs::create_dir_all(&dir)?;
    let stem = Path::new(&name)
        .file_stem()
        .map_or_else(|| card.id.clone(), |s| s.to_string_lossy().into_owned());
    let onnx = dir.join(format!("{stem}.onnx"));
    let sha = convert_and_hash(python, file, &onnx, cancelled)?;
    register(
        card,
        models_dir,
        onnx,
        "onnx".into(),
        sha,
        verified,
        &source_url,
    )
}

fn check_rgb(card: &ModelCard) -> Result<()> {
    check_id(&card.id)?;
    if card.in_ch != 3 || card.out_ch != 3 {
        return Err(ScaleUpError::Model(format!(
            "'{}' is not an RGB model ({} in, {} out)",
            card.id, card.in_ch, card.out_ch
        )));
    }
    Ok(())
}

/// [`convert_to_onnx`], then the SHA-256 of the result; `onnx` is removed on failure.
fn convert_and_hash(
    python: &Python,
    input: &Path,
    onnx: &Path,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<String> {
    let result = convert_to_onnx(python, input, onnx, cancelled).and_then(|()| file_sha256(onnx));
    if result.is_err() {
        let _ = fs::remove_file(onnx);
    }
    result
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
        let err = convert_to_onnx(&bogus_python(), &pth, &out, &mut || false).unwrap_err();
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
            &mut || false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("lists no PyTorch file"), "{err}");
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
            &mut || false,
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
            &mut || false,
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

    /// A stand-in for Python: answers the `-c` probes with "3". As the converter it writes
    /// `<output>`, or (`slow`) writes `<output>.part`, sleeps about 5 s and fails.
    fn fake_python(dir: &Path, slow: bool) -> PathBuf {
        #[cfg(windows)]
        let (name, text) = (
            "python.cmd",
            if slow {
                "@echo off\r\nif \"%~1\"==\"-c\" (\r\necho 3\r\nexit /b 0\r\n)\r\necho partial> \"%~3.part\"\r\nping -n 6 127.0.0.1 >nul\r\nexit /b 1\r\n"
            } else {
                "@echo off\r\nif \"%~1\"==\"-c\" (\r\necho 3\r\nexit /b 0\r\n)\r\necho onnx> \"%~3\"\r\nexit /b 0\r\n"
            },
        );
        #[cfg(not(windows))]
        let (name, text) = (
            "python",
            if slow {
                "#!/bin/sh\nif [ \"$1\" = \"-c\" ]; then echo 3; exit 0; fi\necho partial > \"$3.part\"\nsleep 5\nexit 1\n"
            } else {
                "#!/bin/sh\nif [ \"$1\" = \"-c\" ]; then echo 3; exit 0; fi\necho onnx > \"$3\"\nexit 0\n"
            },
        );
        let path = dir.join(name);
        fs::write(&path, text).unwrap();
        #[cfg(not(windows))]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }

    /// Answers false for `after`, then true.
    fn cancel_after(after: std::time::Duration) -> impl FnMut() -> bool {
        let start = std::time::Instant::now();
        move || start.elapsed() >= after
    }

    #[test]
    fn fake_python_passes_the_probes() {
        let dir = tempfile::tempdir().unwrap();
        let exe = fake_python(dir.path(), false);
        let python = find_python_from(Some(exe.clone().into_os_string())).unwrap();
        assert_eq!(python.path(), exe);
    }

    #[test]
    fn cancel_kills_python_and_removes_partial_output() {
        let dir = tempfile::tempdir().unwrap();
        let python = Python {
            exe: fake_python(dir.path(), true),
        };
        let pth = dir.path().join("m.pth");
        fs::write(&pth, b"x").unwrap();
        let out = dir.path().join("m.onnx");
        let started = std::time::Instant::now();
        let mut cancelled = cancel_after(Duration::from_millis(1500));
        let err = convert_to_onnx(&python, &pth, &out, &mut cancelled).unwrap_err();
        assert!(matches!(err, ScaleUpError::Cancelled), "{err:?}");
        assert!(
            started.elapsed() < Duration::from_secs(4),
            "{:?}",
            started.elapsed()
        );
        assert!(!out.exists());
        assert!(!part_of(&out).exists(), "the .part file is removed");
        assert!(pth.exists(), "the input is not ours to delete");
    }

    #[test]
    fn install_converted_cancel_leaves_nothing_behind() {
        let dir = tempfile::tempdir().unwrap();
        let python = Python {
            exe: fake_python(dir.path(), true),
        };
        let models = dir.path().join("models");
        let body = b"checkpoint".to_vec();
        let c = card(
            "dat",
            vec![resource("pth", "pytorch", Some(sha(&body)), &[GH_PTH])],
        );
        let http = Fake::with(vec![resp(&body, "application/octet-stream")]);
        let err = install_converted(
            &c,
            &models,
            &python,
            &http,
            &mut |_, _| {},
            &mut || {},
            &mut cancel_after(Duration::from_millis(1500)),
        )
        .unwrap_err();
        assert!(matches!(err, ScaleUpError::Cancelled), "{err:?}");
        let folder = models.join("hub").join("4x-Test");
        let left: Vec<_> = fs::read_dir(&folder).unwrap().flatten().collect();
        assert!(left.is_empty(), "{left:?}");
        assert!(Installed::load(&models).models.is_empty());
    }

    #[test]
    fn install_converted_registers_the_onnx() {
        let dir = tempfile::tempdir().unwrap();
        let python = Python {
            exe: fake_python(dir.path(), false),
        };
        let models = dir.path().join("models");
        let body = b"checkpoint".to_vec();
        let c = card(
            "dat",
            vec![resource("pth", "pytorch", Some(sha(&body)), &[GH_PTH])],
        );
        let http = Fake::with(vec![resp(&body, "application/octet-stream")]);
        let m = install_converted(
            &c,
            &models,
            &python,
            &http,
            &mut |_, _| {},
            &mut || {},
            &mut || false,
        )
        .unwrap();
        assert_eq!(m.format, "onnx");
        assert!(m.verified);
        assert_eq!(
            m.file,
            models.join("hub").join("4x-Test").join("4x-Test.onnx")
        );
        assert_eq!(m.sha256, Some(file_sha256(&m.file).unwrap()));
        assert!(
            !m.file.with_extension("pth").exists(),
            "the PyTorch file is dropped"
        );
    }

    #[test]
    fn manual_pytorch_cards_point_at_import_everywhere() {
        let mega = "https://mega.nz/file/x/4x-Test.pth";
        let c = card(
            "dat",
            vec![resource("pth", "pytorch", Some(sha(b"w")), &[mega])],
        );
        assert_eq!(
            pytorch_source(&c),
            PytorchSource::Manual {
                url: mega.into(),
                file_name: "4x-Test.pth".into(),
                sha256: Some(sha(b"w")),
            }
        );
        let dir = tempfile::tempdir().unwrap();
        let http = Fake::default();
        let err = install_converted(
            &c,
            dir.path(),
            &bogus_python(),
            &http,
            &mut |_, _| {},
            &mut || {},
            &mut || false,
        )
        .unwrap_err()
        .to_string();
        let caps = Capabilities {
            onnx: true,
            native_archs: Vec::new(),
        };
        let plain = super::super::install(&c, dir.path(), &caps, &http, &mut |_, _| {})
            .unwrap_err()
            .to_string();
        for msg in [&err, &plain] {
            assert!(msg.contains(mega), "{msg}");
            assert!(msg.contains("scaleup hub import 4x-Test <file>"), "{msg}");
            assert!(!msg.contains("--convert"), "{msg}");
        }
        assert!(http.requests().is_empty());
        let gh = card("dat", vec![resource("pth", "pytorch", None, &[GH_PTH])]);
        assert_eq!(pytorch_source(&gh), PytorchSource::Download);
        let none = card("dat", vec![resource("onnx", "onnx", None, &[GH_PTH])]);
        assert_eq!(pytorch_source(&none), PytorchSource::None);
    }

    #[test]
    fn import_needs_conversion_for_pytorch_files_of_non_native_archs() {
        let c = card("dat", Vec::new());
        let none = Capabilities::default();
        let dat = Capabilities {
            onnx: true,
            native_archs: vec!["DAT".into()],
        };
        assert!(import_needs_conversion(&c, Path::new("a.pth"), &none));
        assert!(import_needs_conversion(
            &c,
            Path::new("a.SafeTensors"),
            &none
        ));
        assert!(!import_needs_conversion(&c, Path::new("a.onnx"), &none));
        assert!(!import_needs_conversion(&c, Path::new("a.pth"), &dat));
    }

    #[test]
    fn import_converted_checks_converts_and_keeps_the_users_file() {
        let dir = tempfile::tempdir().unwrap();
        let python = Python {
            exe: fake_python(dir.path(), false),
        };
        let models = dir.path().join("models");
        let src = dir.path().join("Downloaded.pth");
        fs::write(&src, b"weights").unwrap();
        let c = card(
            "dat",
            vec![resource(
                "pth",
                "pytorch",
                Some(sha(b"weights")),
                &["https://mega.nz/x"],
            )],
        );
        let m = import_converted(&c, &src, &models, &python, &mut || false).unwrap();
        assert!(m.verified);
        assert_eq!(m.format, "onnx");
        assert_eq!(
            m.file,
            models.join("hub").join("4x-Test").join("Downloaded.onnx")
        );
        assert!(src.exists());
        let wrong = dir.path().join("wrong.pth");
        fs::write(&wrong, b"other").unwrap();
        let err = import_converted(&c, &wrong, &models, &python, &mut || false).unwrap_err();
        assert!(err.to_string().contains("sha256 mismatch"), "{err}");
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
        convert_to_onnx(&python, &pth, &out, &mut || false).unwrap();
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
