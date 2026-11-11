mod common;
use common::*;
use predicates::prelude::*;
use scaleup_core::MarginSpec;
use std::path::{Path, PathBuf};

const W: u32 = 8;
const H: u32 = 4;

fn write_input(path: &Path) {
    image::RgbaImage::from_fn(W, H, |x, y| {
        image::Rgba([x as u8 * 30, y as u8 * 60, 128, 255])
    })
    .save(path)
    .unwrap();
}

fn setup() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.png");
    write_input(&input);
    let out = dir.path().join("out.png");
    (dir, input, out)
}

fn extend(input: &Path, out: &Path) -> assert_cmd::Command {
    let mut c = scaleup();
    c.args(["extend", "-g", "cpu", "-i"])
        .arg(input)
        .arg("-o")
        .arg(out);
    c
}

fn size_of(path: &Path) -> (u32, u32) {
    image::image_dimensions(path).unwrap()
}

/// The size a spec grows the test image to, from the same resolver the engine uses.
fn expected(spec: &str) -> (u32, u32) {
    let m = MarginSpec::parse(spec).unwrap().resolve(W, H).unwrap();
    (W + m.left + m.right, H + m.top + m.bottom)
}

#[test]
fn every_extend_spec_form_grows_the_canvas() {
    for spec in ["3", "1,2,3,4", "50%", "10%,25%,50%,100%"] {
        let (_d, input, out) = setup();
        extend(&input, &out)
            .args(["--extend", spec, "--extend-fill", "mirror"])
            .assert()
            .code(0);
        assert_eq!(size_of(&out), expected(spec), "--extend {spec}");
        assert_ne!(size_of(&out), (W, H), "--extend {spec} changed nothing");
    }
}

#[test]
fn aspect_grows_to_the_ratio() {
    let (_d, input, out) = setup();
    extend(&input, &out)
        .args(["--aspect", "1:1", "--extend-fill", "edge"])
        .assert()
        .code(0);
    assert_eq!(size_of(&out), (8, 8));
}

#[test]
fn colour_and_transparent_fills_work() {
    for fill in ["#ff0000", "transparent"] {
        let (_d, input, out) = setup();
        extend(&input, &out)
            .args(["--extend", "2", "--extend-fill", fill])
            .assert()
            .code(0);
        assert_eq!(size_of(&out), (12, 8));
    }
}

#[test]
fn extend_and_aspect_conflict() {
    let (_d, input, out) = setup();
    extend(&input, &out)
        .args(["--extend", "4", "--aspect", "1:1"])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("cannot be used with"));
    assert!(!out.exists());
}

#[test]
fn extend_command_needs_extend_or_aspect() {
    let (_d, input, out) = setup();
    extend(&input, &out).assert().code(1);
    assert!(!out.exists());
}

#[test]
fn extend_rejects_a_ratio_and_aspect_rejects_margins() {
    let (_d, input, out) = setup();
    extend(&input, &out)
        .args(["--extend", "16:9"])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("--aspect"));
    extend(&input, &out)
        .args(["--aspect", "64"])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("W:H"));
    extend(&input, &out).args(["--extend=-5"]).assert().code(1);
    extend(&input, &out)
        .args(["--extend", "4", "--extend-fill", "plaid"])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("#rrggbb"));
}

#[test]
fn upscale_fill_and_model_flags_need_an_extension() {
    let (_d, input, out) = setup();
    for flags in [["--extend-fill", "mirror"], ["--extend-model", "lama-512"]] {
        scaleup()
            .args(["upscale", "-i"])
            .arg(&input)
            .arg("-o")
            .arg(&out)
            .arg("-n")
            .arg(fixture_model())
            .args(flags)
            .assert()
            .code(1)
            .stderr(predicate::str::contains("--extend"));
    }
    assert!(!out.exists());
}

#[test]
fn upscale_with_extend_grows_then_upscales() {
    let (dir, input, _) = setup();
    let out = dir.path().join("o");
    scaleup()
        .arg("-i")
        .arg(&input)
        .arg("-o")
        .arg(format!("{}/", out.display()))
        .arg("-n")
        .arg(fixture_model())
        .args(["-g", "cpu", "--extend", "2", "--extend-fill", "edge"])
        .assert()
        .code(0);
    let named = out.join("in_scaleup_ext_2x_nearest2x.png");
    assert!(
        named.is_file(),
        "{:?}",
        std::fs::read_dir(&out).unwrap().collect::<Vec<_>>()
    );
    assert_eq!(size_of(&named), (24, 16));
}

#[test]
fn folder_mode_names_files_by_fill() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in");
    std::fs::create_dir(&input).unwrap();
    for n in ["a", "b"] {
        write_input(&input.join(format!("{n}.png")));
    }
    let out = dir.path().join("out");
    scaleup()
        .args(["extend", "-g", "cpu", "-i"])
        .arg(&input)
        .arg("-o")
        .arg(&out)
        .args(["--extend", "2", "--extend-fill", "#00ff7f"])
        .assert()
        .code(0)
        .stderr(predicate::str::contains("2 done, 0 failed, 0 skipped"));
    for n in ["a", "b"] {
        let p = out.join(format!("{n}_scaleup_ext_00ff7f.png"));
        assert_eq!(size_of(&p), (12, 8));
    }
}

#[test]
fn json_lines_carry_the_extending_phase() {
    let (_d, input, out) = setup();
    let assert = extend(&input, &out)
        .args(["--extend", "2", "--extend-fill", "mirror", "-v", "--json"])
        .assert()
        .code(0);
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    let phases: Vec<String> = stderr
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
        .filter(|v| v["event"] == "phase")
        .map(|v| v["phase"].as_str().unwrap().to_owned())
        .collect();
    assert!(phases.contains(&"extending".to_owned()), "{phases:?}");
    assert!(stderr.contains(r#""phase":"extending""#));
}

#[test]
fn existing_output_is_skipped_without_overwrite() {
    let (_d, input, out) = setup();
    std::fs::write(&out, "keep").unwrap();
    extend(&input, &out)
        .args(["--extend", "2", "--extend-fill", "edge"])
        .assert()
        .code(0)
        .stderr(predicate::str::contains("exists, skipping"));
    assert_eq!(std::fs::read(&out).unwrap(), b"keep");
}

#[test]
fn a_plain_fill_never_looks_at_the_inpainting_model() {
    let (_d, input, out) = setup();
    extend(&input, &out)
        .args([
            "--extend",
            "2",
            "--extend-fill",
            "edge",
            "--extend-model",
            "no-such-model",
        ])
        .assert()
        .code(0);
}

#[test]
fn an_unknown_inpainting_model_is_a_model_error() {
    let (_d, input, out) = setup();
    extend(&input, &out)
        .args(["--extend", "2", "--extend-model", "no-such-model"])
        .assert()
        .code(3)
        .stderr(predicate::str::contains("no-such-model"));
    assert!(!out.exists());
}

#[test]
fn an_upscaling_model_is_refused_as_inpainter() {
    let (_d, input, out) = setup();
    extend(&input, &out)
        .args(["--extend", "2", "--extend-model", "realesrgan-x4plus"])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("inpainting"));
}

#[test]
fn models_list_hides_inpainters_unless_asked() {
    scaleup()
        .args(["models", "list"])
        .assert()
        .code(0)
        .stdout(predicate::str::contains("realesrgan-x4plus"))
        .stdout(predicate::str::contains("lama-512").not())
        .stdout(predicate::str::contains("migan-512-places2").not());
    scaleup()
        .args(["models", "list", "--kind", "inpaint"])
        .assert()
        .code(0)
        .stdout(predicate::str::contains("lama-512"))
        .stdout(predicate::str::contains("migan-512-places2"))
        .stdout(predicate::str::contains("realesrgan-x4plus").not());
    scaleup()
        .args(["models", "list", "--kind", "all"])
        .assert()
        .code(0)
        .stdout(predicate::str::contains("lama-512"))
        .stdout(predicate::str::contains("realesrgan-x4plus"));
}

fn inpaint_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/inpaint_mean_lama.onnx")
}

/// `got`'s pixels at `(left, top)` and on are `input`'s, each repeated `scale`×`scale` times.
fn assert_known_region(got: &Path, input: &Path, left: u32, top: u32, scale: u32) {
    let got = image::open(got).unwrap().to_rgba8();
    let src = image::open(input).unwrap().to_rgba8();
    for y in 0..src.height() * scale {
        for x in 0..src.width() * scale {
            assert_eq!(
                got.get_pixel(left + x, top + y),
                src.get_pixel(x / scale, y / scale),
                "at {x},{y}"
            );
        }
    }
}

#[test]
fn default_ai_fill_extends_with_the_inpainting_model() {
    let (_d, input, out) = setup();
    extend(&input, &out)
        .args(["--extend", "2", "--extend-model"])
        .arg(inpaint_fixture())
        .assert()
        .code(0);
    assert_eq!(size_of(&out), (12, 8));
    assert_known_region(&out, &input, 2, 2, 1);
}

#[test]
fn upscale_with_the_default_ai_fill_extends_then_upscales() {
    let (_d, input, out) = setup();
    scaleup()
        .args(["upscale", "-g", "cpu", "-i"])
        .arg(&input)
        .arg("-o")
        .arg(&out)
        .arg("-n")
        .arg(fixture_model())
        .args(["--extend", "2", "--extend-model"])
        .arg(inpaint_fixture())
        .assert()
        .code(0);
    assert_eq!(size_of(&out), (24, 16));
    assert_known_region(&out, &input, 4, 4, 2);
}

#[test]
fn extend_says_extended_and_upscale_keeps_the_upscayl_line() {
    let (dir, input, out) = setup();
    let stderr =
        |a: assert_cmd::assert::Assert| String::from_utf8(a.get_output().stderr.clone()).unwrap();
    let extended = stderr(
        extend(&input, &out)
            .args(["--extend", "2", "--extend-fill", "edge"])
            .assert()
            .code(0),
    );
    assert_eq!(extended, " Extended Successfully!\n");
    // `upscale`, with and without an extension, keeps the upscayl-bin line.
    let no_flags: &[&str] = &[];
    for (name, flags) in [
        ("plain.png", no_flags),
        ("ext.png", &["--extend", "2", "--extend-fill", "edge"]),
    ] {
        let upscaled = stderr(
            scaleup()
                .args(["upscale", "-g", "cpu", "-i"])
                .arg(&input)
                .arg("-o")
                .arg(dir.path().join(name))
                .arg("-n")
                .arg(fixture_model())
                .args(flags)
                .assert()
                .code(0),
        );
        assert!(
            upscaled.ends_with(" Upscayled Successfully!\n") && !upscaled.contains("Extended"),
            "{name}: {upscaled:?}"
        );
    }
}

/// Serves `body` once with HTTP 200 on a loopback port; returns its URL.
fn serve_once(body: Vec<u8>) -> String {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut s, _)) = listener.accept() {
            let mut buf = [0u8; 1024];
            let _ = s.read(&mut buf);
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = s.write_all(head.as_bytes());
            let _ = s.write_all(&body);
        }
    });
    format!("http://127.0.0.1:{port}/fill.onnx")
}

#[test]
fn inpainter_download_reports_its_progress() {
    use sha2::{Digest, Sha256};
    let body = std::fs::read(inpaint_fixture()).unwrap();
    let sha: String = Sha256::digest(&body)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let size = body.len();
    let url = serve_once(body);
    let (dir, input, out) = setup();
    let manifest = dir.path().join("manifest.json");
    std::fs::write(
        &manifest,
        format!(
            r#"{{"version":1,"models":[{{"id":"test-fill","name":"Fill","arch":"lama","scale":1,
            "license":"MIT","commercial":true,"size_bytes":{size},"kind":"inpaint",
            "url":"{url}","sha256":"{sha}","tags":[],"notes":""}}]}}"#
        ),
    )
    .unwrap();
    let models = dir.path().join("models");
    extend(&input, &out)
        .env("SCALEUP_TEST_MANIFEST", &manifest)
        .arg("-m")
        .arg(&models)
        .args(["--extend", "2", "--extend-model", "test-fill", "-v"])
        .assert()
        .code(0)
        .stderr(predicate::str::contains("downloading model test-fill"))
        .stderr(predicate::str::contains("download 100%"));
    assert!(models.join("test-fill.onnx").is_file());
    assert_eq!(size_of(&out), (12, 8));
}

#[test]
fn ctrlc_during_inpainter_download_exits_4_and_leaves_no_file() {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
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
    let (dir, input, out) = setup();
    let manifest = dir.path().join("manifest.json");
    std::fs::write(
        &manifest,
        format!(
            r#"{{"version":1,"models":[{{"id":"slow-fill","name":"Slow","arch":"lama","scale":1,
            "license":"MIT","commercial":true,"size_bytes":100000,"kind":"inpaint",
            "url":"http://127.0.0.1:{port}/slow.onnx","sha256":"{}","tags":[],"notes":""}}]}}"#,
            "00".repeat(32)
        ),
    )
    .unwrap();
    let models = dir.path().join("models");
    extend(&input, &out)
        .env("SCALEUP_TEST_MANIFEST", &manifest)
        .env("SCALEUP_TEST_CANCEL_AFTER_MS", "300")
        .arg("-m")
        .arg(&models)
        .args(["--extend", "2", "--extend-model", "slow-fill"])
        .assert()
        .code(4);
    assert!(!out.exists());
    let leftovers: Vec<_> = std::fs::read_dir(&models)
        .map(|d| d.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}
