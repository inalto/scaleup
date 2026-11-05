mod common;
use common::*;
use predicates::prelude::*;

const CATALOG: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/hub_catalog.json"
);
const PIP: &str = "pip install spandrel torch onnx";

/// `scaleup` with no Python to find: `PATH` empty and no `SCALEUP_PYTHON`.
fn without_python(models: &std::path::Path) -> assert_cmd::Command {
    let mut cmd = scaleup();
    cmd.env("PATH", "")
        .env_remove("SCALEUP_PYTHON")
        .env("SCALEUP_MODELS_DIR", models)
        .env("SCALEUP_HUB_CATALOG", CATALOG);
    cmd
}

#[test]
fn convert_without_python_exits_3_with_pip_hint() {
    let models = tempfile::tempdir().unwrap();
    let src = tempfile::tempdir().unwrap();
    let pth = src.path().join("mine-2x.pth");
    std::fs::write(&pth, b"not really a checkpoint").unwrap();
    let a = without_python(models.path())
        .args(["models", "convert"])
        .arg(&pth)
        .assert()
        .code(3);
    let err = String::from_utf8(a.get_output().stderr.clone()).unwrap();
    assert!(err.lines().any(|l| l.trim() == PIP), "{err}");
    assert!(!models.path().join("mine-2x.onnx").exists());
    assert!(!models.path().join("local.json").exists());
}

#[test]
fn convert_rejects_missing_and_non_pytorch_files() {
    let models = tempfile::tempdir().unwrap();
    without_python(models.path())
        .args(["models", "convert", "no-such-file.pth"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("no-such-file.pth"));
    without_python(models.path())
        .args(["models", "convert"])
        .arg(fixture_model())
        .assert()
        .code(1)
        .stderr(predicate::str::contains(".pth or .safetensors"));
}

#[test]
fn convert_refuses_a_catalogue_id_before_running_python() {
    let models = tempfile::tempdir().unwrap();
    let src = tempfile::tempdir().unwrap();
    let pth = src.path().join("realesr-general-x4v3.pth");
    std::fs::write(&pth, b"x").unwrap();
    without_python(models.path())
        .args(["models", "convert"])
        .arg(&pth)
        .assert()
        .code(1)
        .stderr(predicate::str::contains("realesr-general-x4v3"))
        .stderr(predicate::str::contains("rename"));
}

#[test]
fn hub_install_of_a_pytorch_only_card_mentions_convert() {
    let models = tempfile::tempdir().unwrap();
    without_python(models.path())
        .args(["hub", "install", "4x-DRCT-L"])
        .assert()
        .code(3)
        .stderr(predicate::str::contains(
            "scaleup hub install 4x-DRCT-L --convert",
        ));
}

#[test]
fn hub_install_convert_without_python_downloads_nothing() {
    let models = tempfile::tempdir().unwrap();
    let a = without_python(models.path())
        .args(["hub", "install", "4x-DRCT-L", "--convert"])
        .assert()
        .code(3);
    let err = String::from_utf8(a.get_output().stderr.clone()).unwrap();
    assert!(err.lines().any(|l| l.trim() == PIP), "{err}");
    assert!(!models.path().join("hub").join("4x-DRCT-L").exists());
}

#[test]
fn hub_import_of_a_pytorch_file_converts_and_needs_python() {
    let models = tempfile::tempdir().unwrap();
    let src = tempfile::tempdir().unwrap();
    let pth = src.path().join("4x-DRCT-L.pth");
    std::fs::write(&pth, b"weights").unwrap();
    let a = without_python(models.path())
        .args(["hub", "import", "4x-DRCT-L"])
        .arg(&pth)
        .assert()
        .code(3);
    let err = String::from_utf8(a.get_output().stderr.clone()).unwrap();
    assert!(err.lines().any(|l| l.trim() == PIP), "{err}");
    assert!(pth.exists(), "the user's file stays where it is");
    assert!(!models.path().join("hub").join("4x-DRCT-L").exists());
}
