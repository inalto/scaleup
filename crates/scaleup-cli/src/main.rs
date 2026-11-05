mod args;
mod convert;
mod devices;
mod hub;
mod models;
mod report;
mod upscale;

use std::process::exit;
use std::time::Duration;

use clap::{CommandFactory, Parser};
use scaleup_core::CancelToken;

use args::{Cli, Cmd};
use report::Reporter;
use upscale::{EXIT_CANCELLED, EXIT_USAGE, exit_code_for};

fn main() {
    scaleup_wgpu::register();
    let mut cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            let code = if e.use_stderr() { EXIT_USAGE } else { 0 };
            let _ = e.print();
            exit(code);
        }
    };

    // Bare `scaleup -i .. -o ..` is `scaleup upscale ..`; no subcommand and no flags is a usage error.
    if cli.cmd.is_none() {
        match cli.upscale.take() {
            Some(a) => cli.cmd = Some(Cmd::Upscale(Box::new(a))),
            None => {
                let _ = Cli::command().print_help();
                exit(EXIT_USAGE);
            }
        }
    }

    let cancel = CancelToken::new();
    {
        let cancel = cancel.clone();
        let _ = ctrlc::set_handler(move || {
            // A second Ctrl-C while a stalled download blocks the first one: give up now.
            if cancel.is_cancelled() {
                exit(EXIT_CANCELLED);
            }
            cancel.cancel();
        });
    }
    // Test hook: cancel after N milliseconds without needing a console signal.
    if let Some(ms) = std::env::var("SCALEUP_TEST_CANCEL_AFTER_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        let cancel = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(ms));
            cancel.cancel();
        });
    }

    let (json, verbose) = match &cli.cmd {
        Some(Cmd::Upscale(a)) => (a.json, a.verbose),
        _ => (false, false),
    };
    let rep = Reporter::new(json, verbose);

    let result = match &cli.cmd {
        None => unreachable!("a missing subcommand was resolved above"),
        Some(Cmd::Upscale(a)) => upscale::run_upscale(a, &rep, &cancel),
        Some(Cmd::Models { cmd }) => models::run_models(cmd, &cancel),
        Some(Cmd::Hub { cmd }) => hub::run_hub(cmd),
        Some(Cmd::Devices) => devices::run_devices(),
    };
    match result {
        Ok(code) => exit(code),
        Err(scaleup_core::ScaleUpError::Cancelled) => {
            rep.error("cancelled");
            exit(EXIT_CANCELLED);
        }
        Err(e) => {
            rep.error(&e.to_string());
            exit(exit_code_for(&e));
        }
    }
}
