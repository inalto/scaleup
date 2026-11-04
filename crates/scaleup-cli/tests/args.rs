use assert_cmd::Command;
use predicates::prelude::*;

fn scaleup() -> Command {
    Command::cargo_bin("scaleup").unwrap()
}

#[test]
fn subcommand_help_exits_0() {
    scaleup().args(["upscale", "--help"]).assert().code(0);
}

#[test]
fn bad_gpu_is_usage_error() {
    scaleup()
        .args(["-i", "x", "-o", "y", "-g", "bogus"])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("-g"));
}

#[test]
fn out_of_range_values_are_usage_errors() {
    for bad in [["-s", "5"], ["-c", "101"], ["-t", "16"]] {
        scaleup()
            .args(["-i", "x", "-o", "y"])
            .args(bad)
            .assert()
            .code(1);
    }
}

#[test]
fn bare_flags_reach_the_upscaler() {
    // Parsed as a bare upscale; the missing input is an I/O error (2), not a usage error.
    scaleup()
        .args(["-i", "definitely-missing.png", "-o", "out.png", "-g", "cpu"])
        .assert()
        .code(2);
}

#[test]
fn tile_overlap_flag_is_accepted() {
    // Parsed; the missing input is an I/O error (2), not a usage error.
    scaleup()
        .args(["-i", "definitely-missing.png", "-o", "out.png", "-g", "cpu"])
        .args(["--tile-overlap", "32"])
        .assert()
        .code(2);
}
