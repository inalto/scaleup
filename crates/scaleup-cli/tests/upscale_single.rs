mod common;
use common::*;
use predicates::prelude::*;

fn base(input: &std::path::Path, out: &std::path::Path) -> assert_cmd::Command {
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
fn single_png_2x() {
    let dir = tempfile::tempdir().unwrap();
    let (input, out) = (dir.path().join("in.png"), dir.path().join("out.png"));
    write_png(&input);
    base(&input, &out)
        .assert()
        .code(0)
        .stderr(predicate::str::contains("100.00%"));
    assert_eq!(image::open(&out).unwrap().to_rgba8().dimensions(), (8, 8));
}

#[test]
fn output_dir_naming() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("pic.png");
    write_png(&input);
    let out = dir.path().join("out");
    std::fs::create_dir(&out).unwrap();
    base(&input, &out).assert().code(0);
    assert!(out.join("pic_scaleup_2x_nearest2x.png").is_file());
}

#[test]
fn missing_input_exit_2() {
    let dir = tempfile::tempdir().unwrap();
    base(&dir.path().join("nope.png"), &dir.path().join("o.png"))
        .assert()
        .code(2);
}

#[test]
fn missing_output_parent_exit_2() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.png");
    write_png(&input);
    base(&input, &dir.path().join("nodir").join("o.png"))
        .assert()
        .code(2);
}

#[test]
fn trailing_separator_creates_output_dir() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.png");
    write_png(&input);
    let out = format!("{}/new/", dir.path().display());
    base(&input, std::path::Path::new(&out)).assert().code(0);
    assert!(dir.path().join("new/in_scaleup_2x_nearest2x.png").is_file());
}

#[test]
fn format_from_extension() {
    let dir = tempfile::tempdir().unwrap();
    let (input, out) = (dir.path().join("in.png"), dir.path().join("out.jpg"));
    write_png(&input);
    base(&input, &out).assert().code(0);
    assert_eq!(&std::fs::read(&out).unwrap()[..3], &[0xFF, 0xD8, 0xFF]);
}

#[test]
fn no_extension_defaults_to_png() {
    let dir = tempfile::tempdir().unwrap();
    let (input, out) = (dir.path().join("in.png"), dir.path().join("out"));
    write_png(&input);
    base(&input, &out).assert().code(0);
    assert_eq!(&std::fs::read(&out).unwrap()[..4], b"\x89PNG");
}

#[test]
fn relative_model_path_with_space() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::copy(fixture_model(), dir.path().join("my model.onnx")).unwrap();
    write_png(&dir.path().join("in.png"));
    scaleup()
        .current_dir(dir.path())
        .args([
            "-i",
            "in.png",
            "-o",
            "out.png",
            "-n",
            "my model.onnx",
            "-g",
            "cpu",
        ])
        .assert()
        .code(0);
    assert!(dir.path().join("out.png").is_file());
}

#[test]
fn existing_output_is_kept_without_overwrite() {
    let dir = tempfile::tempdir().unwrap();
    let (input, out) = (dir.path().join("in.png"), dir.path().join("out.png"));
    write_png(&input);
    std::fs::write(&out, b"keep").unwrap();
    base(&input, &out).assert().code(0);
    assert_eq!(std::fs::read(&out).unwrap(), b"keep");
}
