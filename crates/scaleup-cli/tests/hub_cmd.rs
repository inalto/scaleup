mod common;
use common::*;
use predicates::prelude::*;

const CATALOG: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/hub_catalog.json"
);

fn hub(models: &std::path::Path) -> assert_cmd::Command {
    let mut cmd = scaleup();
    cmd.env("SCALEUP_HUB_CATALOG", CATALOG)
        .env("SCALEUP_MODELS_DIR", models)
        .arg("hub");
    cmd
}

fn out(a: assert_cmd::assert::Assert) -> String {
    String::from_utf8(a.get_output().stdout.clone()).unwrap()
}

fn models_list(models: &std::path::Path) -> String {
    out(scaleup()
        .env("SCALEUP_MODELS_DIR", models)
        .args(["models", "list"])
        .assert()
        .code(0))
}

#[test]
fn hub_search_filters_scale() {
    let models = tempfile::tempdir().unwrap();
    let text = out(hub(models.path())
        .args(["search", "--scale", "2"])
        .assert()
        .code(0));
    let mut lines = text.lines();
    let head: Vec<_> = lines.next().unwrap().split_whitespace().collect();
    assert_eq!(head, ["ID", "SCALE", "ARCH", "LICENSE", "READY", "TAGS"]);
    let rows: Vec<Vec<&str>> = lines.map(|l| l.split_whitespace().collect()).collect();
    let ids: Vec<&str> = rows.iter().map(|r| r[0]).collect();
    assert!(ids.contains(&"2x-MLP-StarSample-V1-0") && ids.contains(&"test-nearest2x"));
    assert!(rows.iter().all(|r| r[1] == "2x"), "{text}");
    assert!(!text.contains("4x-DRCT-L"));

    // Text and flags: commercial licenses only, installable or installed only.
    let text = out(hub(models.path())
        .args(["search", "starsample", "--commercial", "--runnable"])
        .assert()
        .code(0));
    assert!(text.contains("2x-MLP-StarSample-V1-0"), "{text}");
    assert!(
        !text.contains("2x-StarSample-V2-HQ"),
        "non-commercial: {text}"
    );
    let text = out(hub(models.path())
        .args(["search", "--runnable"])
        .assert()
        .code(0));
    assert!(
        !text.contains("4x-DRCT-L") && !text.contains("1x-MangaJPEGHQ"),
        "{text}"
    );
    assert!(
        text.contains("1x-BroadcastToStudio-SPAN"),
        "manual hosts count: {text}"
    );
}

#[test]
fn hub_search_limit_and_tags() {
    let models = tempfile::tempdir().unwrap();
    let a = hub(models.path())
        .args(["search", "--limit", "2"])
        .assert()
        .code(0)
        .stderr(predicate::str::contains("showing 2 of 10"));
    assert_eq!(out(a).lines().count(), 3);
    let text = out(hub(models.path())
        .args(["search", "--tag", "general-upscaler"])
        .assert()
        .code(0));
    assert!(
        text.contains("2x-MLP-StarSample-V1-0") && !text.contains("4x-DRCT-L"),
        "{text}"
    );
}

#[test]
fn hub_info_shows_readiness() {
    let models = tempfile::tempdir().unwrap();
    let text = out(hub(models.path())
        .args(["info", "2x-MLP-StarSample-V1-0"])
        .assert()
        .code(0));
    assert!(text.contains("downloadable"), "{text}");
    assert!(text.contains("CC0-1.0"), "{text}");
    assert!(text.contains("[0] pytorch pth"), "{text}");
    assert!(text.contains("[direct]"), "host class: {text}");
    let text = out(hub(models.path())
        .args(["info", "1x-BroadcastToStudio-SPAN"])
        .assert()
        .code(0));
    assert!(
        text.contains(
            "CC-BY-NC-SA-4.0 — non-commercial use only; attribution and share-alike required"
        ),
        "{text}"
    );
    assert!(text.contains("manual"), "{text}");
    let text = out(hub(models.path())
        .args(["info", "4x-DRCT-L"])
        .assert()
        .code(0));
    assert!(text.contains("needs-conversion"), "{text}");
    hub(models.path())
        .args(["info", "no-such-model"])
        .assert()
        .code(3)
        .stderr(predicate::str::contains("no-such-model"));
}

#[test]
fn hub_install_manual_host_exit_3_with_hint() {
    let models = tempfile::tempdir().unwrap();
    hub(models.path())
        .args(["install", "1x-BroadcastToStudio-SPAN"])
        .assert()
        .code(3)
        .stderr(predicate::str::contains("https://mega.nz/"))
        // SPAN runs natively, so the PyTorch file is preferred to the card's ONNX one.
        .stderr(predicate::str::contains(
            "save the file as '1x-BroadcastToStudio-SPAN.pth' (sha256 77e2656651cf3e2558a7dd8756b47013f5dc20dc24c310801aaabe96cc9ed6ab)",
        ))
        .stderr(predicate::str::contains(
            "`scaleup hub import 1x-BroadcastToStudio-SPAN <file>`",
        ));
    // No resources at all: no URL to open, but the file name and the way in are named.
    hub(models.path())
        .args(["install", "test-nofiles"])
        .assert()
        .code(3)
        .stderr(predicate::str::contains(
            "lists no download for 'test-nofiles'",
        ))
        .stderr(predicate::str::contains(
            "expected name 'test-nofiles.onnx'",
        ))
        .stderr(predicate::str::contains(
            "`scaleup hub import test-nofiles <file>`",
        ));
    hub(models.path())
        .args(["install", "2x-MLP-StarSample-V1-0", "--resource", "9"])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("--resource 9"));
}

#[test]
fn hub_import_then_upscale_resolves_id() {
    let models = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    let file = work.path().join("nearest2x.onnx");
    std::fs::copy(fixture_model(), &file).unwrap();
    let path = out(hub(models.path())
        .args(["import", "test-nearest2x"])
        .arg(&file)
        .assert()
        .code(0));
    let installed = models
        .path()
        .join("hub")
        .join("test-nearest2x")
        .join("nearest2x.onnx");
    assert_eq!(std::path::PathBuf::from(path.trim_end()), installed);
    assert!(installed.is_file());

    assert!(models_list(models.path()).contains("test-nearest2x"));
    let search = out(hub(models.path())
        .args(["search", "nearest"])
        .assert()
        .code(0));
    assert!(search.contains("installed"), "{search}");

    let input = work.path().join("in.png");
    let output = work.path().join("out.png");
    write_png(&input);
    scaleup()
        .env("SCALEUP_MODELS_DIR", models.path())
        .args(["upscale", "-g", "cpu", "-n", "test-nearest2x", "-i"])
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .assert()
        .code(0);
    assert_eq!(image::open(&output).unwrap().width(), 8);

    // Removing it takes it out of the list again.
    hub(models.path())
        .args(["remove", "test-nearest2x"])
        .assert()
        .code(0);
    assert!(!models.path().join("hub").join("test-nearest2x").exists());
    assert!(!models_list(models.path()).contains("test-nearest2x"));
    hub(models.path())
        .args(["remove", "test-nearest2x"])
        .assert()
        .code(3);
}

#[test]
fn hub_import_rejects_a_file_of_another_model() {
    let models = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    let file = work.path().join("other.onnx");
    std::fs::write(&file, b"not the model").unwrap();
    hub(models.path())
        .args(["import", "test-nearest2x"])
        .arg(&file)
        .assert()
        .code(3)
        .stderr(predicate::str::contains("sha256 mismatch"));
}

#[test]
fn hub_recommend_table() {
    let models = tempfile::tempdir().unwrap();
    let text = out(hub(models.path())
        .args([
            "recommend",
            "--content",
            "anime",
            "--goal",
            "upscale",
            "-n",
            "3",
        ])
        .assert()
        .code(0));
    let mut lines = text.lines();
    let head: Vec<_> = lines.next().unwrap().split_whitespace().collect();
    assert_eq!(
        head,
        [
            "#", "ID", "SCALE", "ARCH", "LICENSE", "READY", "SCORE", "WHY"
        ]
    );
    let rows: Vec<_> = lines.collect();
    assert!(!rows.is_empty() && rows.len() <= 3, "{text}");
    assert!(
        rows.iter().all(|r| !r.contains(" 1x ")),
        "upscale goal skips 1x: {text}"
    );
    hub(models.path())
        .args(["recommend", "--content", "bogus"])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("bogus"));
}

#[test]
fn unknown_model_name_hints_at_the_hub() {
    let models = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    let input = work.path().join("in.png");
    write_png(&input);
    scaleup()
        .env("SCALEUP_MODELS_DIR", models.path())
        .args(["upscale", "-g", "cpu", "-n", "nope-2x", "-i"])
        .arg(&input)
        .arg("-o")
        .arg(work.path().join("out.png"))
        .assert()
        .code(3)
        .stderr(predicate::str::contains(
            "neither a known model id nor an existing file",
        ))
        .stderr(predicate::str::contains("`scaleup hub search`"));
}

#[test]
fn hub_refresh_reports_the_catalogue_size() {
    let models = tempfile::tempdir().unwrap();
    hub(models.path())
        .arg("refresh")
        .assert()
        .code(0)
        .stdout(predicate::str::contains("10 models"));
}

#[test]
fn hub_import_without_catalogue_hash_warns_and_is_unverified() {
    let models = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    let file = work.path().join("nearest2x.onnx");
    std::fs::copy(fixture_model(), &file).unwrap();
    hub(models.path())
        .args(["import", "test-nohash"])
        .arg(&file)
        .assert()
        .code(0)
        .stderr(predicate::str::contains(
            "warning: OpenModelDB lists no sha256",
        ));
    let json = std::fs::read_to_string(models.path().join("hub").join("installed.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    let m = &v["models"][0];
    assert_eq!(m["id"], "test-nohash");
    assert_eq!(m["verified"], false);
}
