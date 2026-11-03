mod common;
use common::*;
use predicates::prelude::*;
use std::io::{Read, Write};
use std::net::TcpListener;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

fn manifest_ids() -> Vec<String> {
    let text = std::fs::read_to_string(format!("{ROOT}/../../models/manifest.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    v["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn models_import_copies_pairs_and_lists_them() {
    let models = tempfile::tempdir().unwrap();
    let src = tempfile::tempdir().unwrap();
    std::fs::write(src.path().join("mine-4x.param"), b"7767517").unwrap();
    std::fs::write(src.path().join("mine-4x.bin"), b"w").unwrap();
    let out = scaleup()
        .env("SCALEUP_MODELS_DIR", models.path())
        .args(["models", "import"])
        .arg(src.path())
        .assert()
        .code(0);
    let text = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    assert!(text.contains("mine-4x"), "{text}");
    assert!(models.path().join("mine-4x.bin").is_file());
    let out = scaleup()
        .env("SCALEUP_MODELS_DIR", models.path())
        .args(["models", "list"])
        .assert()
        .code(0);
    let text = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    assert!(text.contains("mine-4x"), "{text}");
}

#[test]
fn models_list_table() {
    let dir = tempfile::tempdir().unwrap();
    let out = scaleup()
        .env("SCALEUP_MODELS_DIR", dir.path())
        .args(["models", "list"])
        .assert()
        .code(0);
    let text = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let mut lines = text.lines();
    let head: Vec<_> = lines.next().unwrap().split_whitespace().collect();
    assert_eq!(head, ["ID", "SCALE", "LICENSE", "COMMERCIAL", "DOWNLOADED"]);
    let ids = manifest_ids();
    let rows: Vec<_> = lines.collect();
    assert_eq!(rows.len(), ids.len());
    for id in ids {
        assert!(
            rows.iter()
                .any(|r| r.split_whitespace().next() == Some(id.as_str())),
            "{id}"
        );
    }
}

#[test]
fn models_info_prints_license() {
    let license = std::fs::read_to_string(format!(
        "{ROOT}/../../models/licenses/realesr-general-x4v3.txt"
    ))
    .unwrap();
    let first = license.lines().next().unwrap().to_string();
    scaleup()
        .args(["models", "info", "realesr-general-x4v3"])
        .assert()
        .code(0)
        .stdout(predicate::str::contains(first));
}

#[test]
fn models_path_and_unknown() {
    let dir = tempfile::tempdir().unwrap();
    let expected = dir.path().join("realesrgan-x4plus.onnx");
    scaleup()
        .env("SCALEUP_MODELS_DIR", dir.path())
        .args(["models", "path", "realesrgan-x4plus"])
        .assert()
        .code(0)
        .stdout(predicate::str::contains(expected.display().to_string()));
    scaleup().args(["models", "path", "nope"]).assert().code(3);
}

#[test]
fn devices_lists_cpu() {
    scaleup()
        .arg("devices")
        .assert()
        .code(0)
        .stdout(predicate::str::contains("cpu"));
}

#[test]
fn ctrlc_removes_part_file() {
    // Sends headers and a few bytes, then stalls for a while.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut s, _)) = listener.accept() {
            let mut buf = [0u8; 1024];
            let _ = s.read(&mut buf);
            let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100000\r\n\r\nabcdef");
            let _ = s.flush();
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    });
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("manifest.json");
    std::fs::write(
        &manifest,
        format!(
            r#"{{"version":1,"models":[{{"id":"slow","name":"Slow","arch":"x","scale":2,
            "license":"MIT","commercial":true,"size_bytes":100000,
            "url":"http://127.0.0.1:{port}/slow.onnx","sha256":"{}","tags":[],"notes":""}}]}}"#,
            "00".repeat(32)
        ),
    )
    .unwrap();
    let models = dir.path().join("models");
    scaleup()
        .env("SCALEUP_MODELS_DIR", &models)
        .env("SCALEUP_TEST_MANIFEST", &manifest)
        .env("SCALEUP_TEST_CANCEL_AFTER_MS", "100")
        .args(["models", "download", "slow"])
        .assert()
        .code(4);
    let leftovers: Vec<_> = std::fs::read_dir(&models)
        .map(|d| d.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}
