//! `scaleup hub ...`: search, install and recommend models from OpenModelDB.

use std::fs;
use std::path::{Path, PathBuf};

use scaleup_core::ScaleUpError;
use scaleup_core::hub::{
    Ask, Capabilities, Catalog, Content, Filter, Goal, Host, Installed, InstalledModel,
    LicenseClass, ModelCard, Readiness, SpeedPref, UreqHttp, classify_license, classify_url,
    commercial_ok, import_file, install, readiness, recommend,
};

use crate::args::HubCmd;
use crate::models::{print_table, registry_for, yes_no};

/// Test hook: a catalogue file that replaces the network fetch and the cache entirely.
const CATALOG_ENV: &str = "SCALEUP_HUB_CATALOG";
/// Width the TAGS column is cut to.
const TAGS_WIDTH: usize = 40;

fn models_dir() -> PathBuf {
    registry_for(None).models_dir().to_path_buf()
}

fn load_catalog(models_dir: &Path, refresh: bool) -> Result<Catalog, ScaleUpError> {
    match std::env::var_os(CATALOG_ENV) {
        Some(path) => {
            let text = fs::read_to_string(&path).map_err(|e| {
                ScaleUpError::Model(format!(
                    "{CATALOG_ENV} {}: {e}",
                    PathBuf::from(&path).display()
                ))
            })?;
            Catalog::parse(&text)
        }
        None => Catalog::fetch(models_dir, refresh),
    }
}

fn find<'a>(cat: &'a Catalog, id: &str) -> Result<&'a ModelCard, ScaleUpError> {
    cat.get(id).ok_or_else(|| {
        ScaleUpError::Model(format!(
            "unknown model '{id}' in the OpenModelDB catalogue; try `scaleup hub search`"
        ))
    })
}

fn ready_name(r: Readiness) -> &'static str {
    match r {
        Readiness::Installed => "installed",
        Readiness::Downloadable => "downloadable",
        Readiness::ManualDownload => "manual",
        Readiness::NeedsConversion => "needs-conversion",
        Readiness::Unsupported => "unsupported",
    }
}

fn ready_help(r: Readiness, id: &str) -> String {
    match r {
        Readiness::Installed => "installed and ready to use".into(),
        Readiness::Downloadable => format!("can be installed with `scaleup hub install {id}`"),
        Readiness::ManualDownload => {
            format!("download a file in a browser, then run `scaleup hub import {id} <file>`")
        }
        Readiness::NeedsConversion => {
            "only a PyTorch file, which this machine cannot run yet".into()
        }
        Readiness::Unsupported => "no file this machine can run, or not an RGB model".into(),
    }
}

/// The SPDX id and what the license allows, in words.
fn license_sentence(license: Option<&str>) -> String {
    let what = match classify_license(license) {
        LicenseClass::Permissive => "permissive; commercial use allowed",
        LicenseClass::Attribution => "commercial use allowed; attribution required",
        LicenseClass::ShareAlike => "commercial use allowed; attribution and share-alike required",
        LicenseClass::Copyleft => {
            "commercial use allowed; derived works must be shared under the same license"
        }
        LicenseClass::NonCommercial if license.is_some_and(|l| l.contains("-SA")) => {
            "non-commercial use only; attribution and share-alike required"
        }
        LicenseClass::NonCommercial => "non-commercial use only; attribution required",
        LicenseClass::Unknown => {
            "license not stated or not recognised; assume no commercial use until you have checked"
        }
    };
    format!("{} — {what}", license.unwrap_or("unknown"))
}

/// Tags joined with commas, cut at a tag boundary to at most `max` characters.
fn cut_tags(tags: &[String], max: usize) -> String {
    let all = tags.join(",");
    if all.chars().count() <= max {
        return all;
    }
    let mut out = String::new();
    for tag in tags {
        let next = if out.is_empty() {
            tag.clone()
        } else {
            format!("{out},{tag}")
        };
        if next.chars().count() + 4 > max {
            break;
        }
        out = next;
    }
    if out.is_empty() {
        out = all.chars().take(max.saturating_sub(3)).collect();
        return format!("{out}...");
    }
    format!("{out},...")
}

fn host_name(url: &str) -> &'static str {
    match classify_url(url) {
        Host::Direct => "direct",
        Host::GoogleDrive { .. } => "google-drive",
        Host::Manual => "manual (open in a browser)",
    }
}

/// `card` with only its resource `index`, so that `install` can pick nothing else.
fn only_resource(card: &ModelCard, index: usize) -> Result<ModelCard, ScaleUpError> {
    let resource = card.resources.get(index).ok_or_else(|| {
        ScaleUpError::InvalidOption(format!(
            "--resource {index}: '{}' has {} resources",
            card.id,
            card.resources.len()
        ))
    })?;
    let mut narrowed = card.clone();
    narrowed.resources = vec![resource.clone()];
    Ok(narrowed)
}

fn warn_if_unverified(model: &InstalledModel) {
    if !model.verified {
        eprintln!(
            "warning: OpenModelDB lists no sha256 for this file, so it could not be verified"
        );
    }
}

fn size_text(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / 1e6)
}

fn search(
    models_dir: &Path,
    filters: &Filter,
    text: &str,
    runnable: bool,
    refresh: bool,
    limit: usize,
) -> Result<(), ScaleUpError> {
    let cat = load_catalog(models_dir, refresh)?;
    let caps = Capabilities::current();
    let installed = Installed::load(models_dir);
    let hits: Vec<(&ModelCard, Readiness)> = cat
        .search(text, filters)
        .into_iter()
        .map(|c| (c, readiness(c, &caps, &installed)))
        .filter(|(_, r)| {
            !runnable
                || matches!(
                    r,
                    Readiness::Installed | Readiness::Downloadable | Readiness::ManualDownload
                )
        })
        .collect();
    let total = hits.len();
    let rows = hits
        .into_iter()
        .take(limit)
        .map(|(c, r)| {
            vec![
                c.id.clone(),
                format!("{}x", c.scale),
                c.architecture.clone(),
                c.license.clone().unwrap_or_else(|| "unknown".into()),
                ready_name(r).into(),
                cut_tags(&c.tags, TAGS_WIDTH),
            ]
        })
        .collect::<Vec<_>>();
    let shown = rows.len();
    print_table(&["ID", "SCALE", "ARCH", "LICENSE", "READY", "TAGS"], rows);
    if total == 0 {
        eprintln!("no model matches");
    } else if shown < total {
        eprintln!("showing {shown} of {total}; use --limit to see more");
    }
    Ok(())
}

fn info(models_dir: &Path, id: &str) -> Result<(), ScaleUpError> {
    let cat = load_catalog(models_dir, false)?;
    let card = find(&cat, id)?;
    let caps = Capabilities::current();
    let installed = Installed::load(models_dir);
    let ready = readiness(card, &caps, &installed);
    println!("id:          {}", card.id);
    println!("name:        {}", card.name);
    println!("author:      {}", card.authors().join(", "));
    println!("date:        {}", card.date);
    println!("arch:        {}", card.architecture);
    println!("scale:       {}x", card.scale);
    println!("channels:    {} in, {} out", card.in_ch, card.out_ch);
    println!("license:     {}", license_sentence(card.license.as_deref()));
    println!(
        "commercial:  {}",
        yes_no(commercial_ok(card.license_class()))
    );
    println!("tags:        {}", card.tags.join(", "));
    println!(
        "ready:       {} ({})",
        ready_name(ready),
        ready_help(ready, id)
    );
    if let Some(m) = installed.get(id) {
        let state = if m.file.is_file() {
            ""
        } else {
            " (file missing; reinstall it)"
        };
        println!("installed:   {}{state}", m.file.display());
        println!("verified:    {}", yes_no(m.verified));
    }
    println!("resources:");
    if card.resources.is_empty() {
        println!("  none listed");
    }
    for (i, r) in card.resources.iter().enumerate() {
        let size = r
            .size
            .map(size_text)
            .unwrap_or_else(|| "size unknown".into());
        println!(
            "  [{i}] {} {} ({size}), sha256 {}",
            r.platform.as_deref().unwrap_or("?"),
            r.kind,
            r.sha256.as_deref().unwrap_or("not listed")
        );
        for url in &r.urls {
            println!("      {url} [{}]", host_name(url));
        }
    }
    if !card.description.is_empty() {
        println!("\n{}", card.description);
    }
    Ok(())
}

fn do_install(models_dir: &Path, id: &str, resource: Option<usize>) -> Result<(), ScaleUpError> {
    let cat = load_catalog(models_dir, false)?;
    let mut card = find(&cat, id)?.clone();
    if let Some(index) = resource {
        card = only_resource(&card, index)?;
    }
    let mut last = u64::MAX;
    let model = install(
        &card,
        models_dir,
        &Capabilities::current(),
        &UreqHttp,
        &mut |done, total| {
            if let Some(total) = total.filter(|t| *t > 0)
                && done * 100 / total != last
            {
                last = done * 100 / total;
                eprintln!("{last}%");
            }
        },
    )?;
    warn_if_unverified(&model);
    println!("{}", model.file.display());
    Ok(())
}

fn remove(models_dir: &Path, id: &str) -> Result<(), ScaleUpError> {
    let mut installed = Installed::load(models_dir);
    if installed.get(id).is_none() {
        return Err(ScaleUpError::Model(format!(
            "'{id}' is not an installed OpenModelDB model"
        )));
    }
    if id.is_empty() || id.starts_with('.') || id.contains(['/', '\\']) {
        return Err(ScaleUpError::Model(format!(
            "refusing to delete the folder of '{id}': not a plain model id"
        )));
    }
    match fs::remove_dir_all(models_dir.join("hub").join(id)) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
        _ => {}
    }
    installed.remove(id);
    installed.save(models_dir)?;
    println!("removed {id}");
    Ok(())
}

fn do_recommend(
    models_dir: &Path,
    (content, goal, speed): (&str, &str, &str),
    scale: Option<u32>,
    commercial: bool,
    n: usize,
) -> Result<(), ScaleUpError> {
    let invalid = |e: String| ScaleUpError::InvalidOption(e);
    let ask = Ask {
        content: Content::parse(content).map_err(invalid)?,
        goal: Goal::parse(goal).map_err(invalid)?,
        scale,
        speed: SpeedPref::parse(speed).map_err(invalid)?,
        commercial_only: commercial,
        caps: Capabilities::current(),
        installed: Installed::load(models_dir),
    };
    let cat = load_catalog(models_dir, false)?;
    let picks = recommend(&cat, &ask, n);
    let rows = picks
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let c = &r.card;
            vec![
                (i + 1).to_string(),
                c.id.clone(),
                format!("{}x", c.scale),
                c.architecture.clone(),
                c.license.clone().unwrap_or_else(|| "unknown".into()),
                ready_name(r.readiness).into(),
                format!("{:.1}", r.score),
                r.reasons
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("; "),
            ]
        })
        .collect();
    print_table(
        &[
            "#", "ID", "SCALE", "ARCH", "LICENSE", "READY", "SCORE", "WHY",
        ],
        rows,
    );
    if picks.is_empty() {
        eprintln!("no model matches");
    }
    Ok(())
}

pub fn run_hub(cmd: &HubCmd) -> Result<i32, ScaleUpError> {
    let dir = models_dir();
    match cmd {
        HubCmd::Search {
            text,
            scale,
            tag,
            arch,
            commercial,
            runnable,
            refresh,
            limit,
        } => {
            let filters = Filter {
                scale: *scale,
                tags_all: tag.clone(),
                arch: arch.clone(),
                commercial_only: *commercial,
                ..Filter::default()
            };
            search(
                &dir,
                &filters,
                text.as_deref().unwrap_or(""),
                *runnable,
                *refresh,
                *limit,
            )?;
        }
        HubCmd::Info { id } => info(&dir, id)?,
        HubCmd::Install { id, resource } => do_install(&dir, id, *resource)?,
        HubCmd::Import { id, file } => {
            let cat = load_catalog(&dir, false)?;
            let model = import_file(find(&cat, id)?, file, &dir)?;
            warn_if_unverified(&model);
            println!("{}", model.file.display());
        }
        HubCmd::Remove { id } => remove(&dir, id)?,
        HubCmd::Recommend {
            content,
            goal,
            scale,
            speed,
            commercial,
            n,
        } => do_recommend(&dir, (content, goal, speed), *scale, *commercial, *n)?,
        HubCmd::Refresh => {
            let cat = load_catalog(&dir, true)?;
            println!("{} models", cat.models.len());
        }
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(t: &[&str]) -> Vec<String> {
        t.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn short_tag_lists_are_kept() {
        assert_eq!(cut_tags(&tags(&["anime", "cartoon"]), 40), "anime,cartoon");
        assert_eq!(cut_tags(&[], 40), "");
    }

    #[test]
    fn long_tag_lists_are_cut_at_a_tag() {
        let t = tags(&[
            "compression-removal",
            "restoration",
            "general-upscaler",
            "photo",
        ]);
        let cut = cut_tags(&t, 40);
        assert_eq!(cut, "compression-removal,restoration,...");
        assert!(cut.chars().count() <= 40);
        assert_eq!(
            cut_tags(&tags(&["a-very-long-single-tag-name"]), 12),
            "a-very-lo..."
        );
    }

    #[test]
    fn license_sentences_say_what_the_class_means() {
        assert_eq!(
            license_sentence(Some("CC-BY-NC-SA-4.0")),
            "CC-BY-NC-SA-4.0 — non-commercial use only; attribution and share-alike required"
        );
        assert!(license_sentence(Some("CC-BY-NC-4.0")).ends_with("only; attribution required"));
        assert!(license_sentence(Some("MIT")).contains("commercial use allowed"));
        assert!(license_sentence(Some("CC-BY-4.0")).contains("attribution required"));
        assert!(license_sentence(Some("CC-BY-SA-4.0")).contains("share-alike"));
        assert!(license_sentence(Some("GPL-3.0-only")).contains("same license"));
        assert!(license_sentence(None).starts_with("unknown — "));
    }

    #[test]
    fn only_resource_keeps_the_chosen_one() {
        let card: ModelCard = serde_json::from_str(
            r#"{"name":"n","license":null,"architecture":"compact","scale":2,
                "inputChannels":3,"outputChannels":3,
                "resources":[{"platform":"onnx","type":"onnx","urls":["u1"]},
                             {"platform":"pytorch","type":"pth","urls":["u2"]}]}"#,
        )
        .unwrap();
        let one = only_resource(&card, 1).unwrap();
        assert_eq!(one.resources.len(), 1);
        assert_eq!(one.resources[0].kind, "pth");
        let err = only_resource(&card, 2).unwrap_err();
        assert!(matches!(err, ScaleUpError::InvalidOption(_)), "{err:?}");
    }

    #[test]
    fn readiness_names_are_the_five_states() {
        let names = [
            Readiness::Installed,
            Readiness::Downloadable,
            Readiness::ManualDownload,
            Readiness::NeedsConversion,
            Readiness::Unsupported,
        ]
        .map(ready_name);
        assert_eq!(
            names,
            [
                "installed",
                "downloadable",
                "manual",
                "needs-conversion",
                "unsupported"
            ]
        );
    }
}
