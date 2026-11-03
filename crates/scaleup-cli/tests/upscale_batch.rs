mod common;
use common::*;
use predicates::prelude::*;
use std::path::{Path, PathBuf};

fn setup() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in");
    std::fs::create_dir(&input).unwrap();
    for n in ["a", "b", "c"] {
        write_png(&input.join(format!("{n}.png")));
    }
    std::fs::write(input.join("notes.txt"), "x").unwrap();
    let out = dir.path().join("out");
    (dir, input, out)
}

fn run(input: &Path, out: &Path) -> assert_cmd::Command {
    let mut c = scaleup();
    c.arg("-i")
        .arg(input)
        .arg("-o")
        .arg(out)
        .arg("-n")
        .arg(fixture_model())
        .args(["-g", "cpu"]);
    c
}

#[test]
fn dir_to_dir_creates_outputs() {
    let (_d, input, out) = setup();
    run(&input, &out)
        .assert()
        .code(0)
        .stderr(predicate::str::contains("3 done, 0 failed, 1 skipped"));
    for n in ["a", "b", "c"] {
        assert!(out.join(format!("{n}_scaleup_2x_nearest2x.png")).is_file());
    }
    assert_eq!(std::fs::read_dir(&out).unwrap().count(), 3);
}

#[test]
fn jobs_2_same_result() {
    let (_d, input, out) = setup();
    run(&input, &out)
        .args(["--jobs", "2"])
        .assert()
        .code(0)
        .stderr(predicate::str::contains("3 done, 0 failed, 1 skipped"));
    let img = image::open(out.join("a_scaleup_2x_nearest2x.png")).unwrap();
    assert_eq!((img.width(), img.height()), (8, 8));
}

#[test]
fn one_corrupt_file_does_not_abort() {
    let (_d, input, out) = setup();
    std::fs::write(input.join("bad.png"), b"not a png").unwrap();
    let assert = run(&input, &out).arg("--json").assert().code(0);
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    let events: Vec<serde_json::Value> = stderr
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(events.iter().filter(|e| e["event"] == "error").count(), 1);
    let summary = events.iter().find(|e| e["event"] == "summary").unwrap();
    assert_eq!(summary["done"].as_u64(), Some(3));
    assert_eq!(summary["failed"].as_u64(), Some(1));
}
