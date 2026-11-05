//! The OpenModelDB catalogue: parsing, the on-disk cache, filters, search and per-host readiness.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::installed::Installed;
use super::license::{LicenseClass, classify_license, commercial_ok};
use crate::{Result, ScaleUpError};

const CATALOG_URL: &str = "https://openmodeldb.info/api/v1/models.json";
/// A cached catalogue older than this is refreshed.
const CACHE_TTL: Duration = Duration::from_secs(24 * 3600);
const TIMEOUT: Duration = Duration::from_secs(30);
/// The live file is 1.5 MB; this leaves room to grow.
const MAX_BODY: u64 = 64 * 1024 * 1024;

/// One downloadable file of a model.
#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct Resource {
    /// `"pytorch"` or `"onnx"`.
    pub platform: Option<String>,
    /// File type: `"pth"`, `"safetensors"` or `"onnx"`.
    #[serde(rename = "type")]
    pub kind: String,
    pub size: Option<u64>,
    pub sha256: Option<String>,
    #[serde(default)]
    pub urls: Vec<String>,
}

/// One model of the catalogue.
#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct ModelCard {
    /// The key of the model in the catalogue (filled by [`Catalog::parse`]).
    #[serde(skip)]
    pub id: String,
    pub name: String,
    /// A string, or an array of strings.
    #[serde(default)]
    pub author: serde_json::Value,
    pub license: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub date: String,
    pub architecture: String,
    pub scale: u32,
    #[serde(rename = "inputChannels")]
    pub in_ch: u32,
    #[serde(rename = "outputChannels")]
    pub out_ch: u32,
    #[serde(default)]
    pub resources: Vec<Resource>,
    #[serde(default)]
    pub images: Vec<serde_json::Value>,
    pub thumbnail: Option<serde_json::Value>,
}

impl ModelCard {
    /// The authors as a list, whichever shape the catalogue used.
    pub fn authors(&self) -> Vec<String> {
        match &self.author {
            serde_json::Value::String(s) => vec![s.clone()],
            serde_json::Value::Array(a) => a
                .iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect(),
            _ => Vec::new(),
        }
    }

    pub fn license_class(&self) -> LicenseClass {
        classify_license(self.license.as_deref())
    }
}

/// The parsed catalogue and when it was downloaded.
#[derive(Clone, Debug, PartialEq)]
pub struct Catalog {
    /// Sorted by id.
    pub models: Vec<ModelCard>,
    pub fetched_at: SystemTime,
}

/// `<models_dir>/hub/catalog.json` (the body) and `catalog.meta.json` (its `ETag` and download time).
#[derive(Serialize, Deserialize)]
struct CacheMeta {
    etag: Option<String>,
    /// Seconds since the Unix epoch.
    fetched_at: u64,
}

fn hub_dir(models_dir: &Path) -> PathBuf {
    models_dir.join("hub")
}

fn secs(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn read_meta(models_dir: &Path) -> Option<CacheMeta> {
    let text = fs::read_to_string(hub_dir(models_dir).join("catalog.meta.json")).ok()?;
    serde_json::from_str(&text).ok()
}

/// Writes the body and its metadata; each file is replaced atomically.
fn store_cache(models_dir: &Path, body: &str, etag: Option<&str>, at: SystemTime) -> Result<()> {
    let dir = hub_dir(models_dir);
    fs::create_dir_all(&dir)?;
    let meta = CacheMeta {
        etag: etag.map(str::to_owned),
        fetched_at: secs(at),
    };
    let meta_json = serde_json::to_string(&meta).map_err(|e| ScaleUpError::Model(e.to_string()))?;
    for (name, text) in [("catalog.json", body), ("catalog.meta.json", &meta_json)] {
        let tmp = dir.join(format!("{name}.part"));
        fs::write(&tmp, text)?;
        fs::rename(&tmp, dir.join(name))?;
    }
    Ok(())
}

/// Whether a catalogue downloaded at `fetched_at` is still good at `now`. A date in the future
/// (the clock was set back) is not trusted.
pub fn cache_is_fresh(fetched_at: SystemTime, now: SystemTime) -> bool {
    now.duration_since(fetched_at)
        .is_ok_and(|age| age <= CACHE_TTL)
}

impl Catalog {
    /// Parses the catalogue: a JSON object keyed by model id.
    pub fn parse(json: &str) -> Result<Catalog> {
        let by_id: BTreeMap<String, ModelCard> = serde_json::from_str(json)
            .map_err(|e| ScaleUpError::Model(format!("OpenModelDB catalogue: {e}")))?;
        let models = by_id
            .into_iter()
            .map(|(id, mut card)| {
                card.id = id;
                card
            })
            .collect();
        Ok(Catalog {
            models,
            fetched_at: SystemTime::now(),
        })
    }

    /// The cached catalogue, however old. `None` when there is none or it cannot be read.
    pub fn load_cached(models_dir: &Path) -> Option<Catalog> {
        let file = hub_dir(models_dir).join("catalog.json");
        let text = fs::read_to_string(&file).ok()?;
        let mut cat = Catalog::parse(&text)
            .inspect_err(|e| tracing::warn!("ignoring {}: {e}", file.display()))
            .ok()?;
        cat.fetched_at = match read_meta(models_dir) {
            Some(meta) => UNIX_EPOCH + Duration::from_secs(meta.fetched_at),
            None => fs::metadata(&file)
                .and_then(|m| m.modified())
                .unwrap_or(UNIX_EPOCH),
        };
        Some(cat)
    }

    /// The catalogue from the cache when it is under 24 h old, else from openmodeldb.info (and
    /// cached). Without a network the stale cache is used; `force` always asks the server and fails
    /// with it.
    pub fn fetch(models_dir: &Path, force: bool) -> Result<Catalog> {
        fetch_from(CATALOG_URL, models_dir, force, SystemTime::now())
    }

    pub fn get(&self, id: &str) -> Option<&ModelCard> {
        self.models.iter().find(|m| m.id == id)
    }

    pub fn filter(&self, f: &Filter) -> Vec<&ModelCard> {
        self.models.iter().filter(|m| f.matches(m)).collect()
    }

    /// Models matching `f` and every word of `text`, case-insensitive, over id, name, description,
    /// tags and authors. Models whose id or name matches come first; catalogue order otherwise.
    pub fn search(&self, text: &str, f: &Filter) -> Vec<&ModelCard> {
        let words: Vec<String> = text.split_whitespace().map(str::to_lowercase).collect();
        let mut hits: Vec<(u8, &ModelCard)> = Vec::new();
        for m in self.models.iter().filter(|m| f.matches(m)) {
            let title = format!("{}\n{}", m.id, m.name).to_lowercase();
            let body = format!(
                "{}\n{}\n{}",
                m.description,
                m.tags.join("\n"),
                m.authors().join("\n")
            )
            .to_lowercase();
            if !words.iter().all(|w| title.contains(w) || body.contains(w)) {
                continue;
            }
            let rank = u8::from(!words.iter().all(|w| title.contains(w)));
            hits.push((rank, m));
        }
        hits.sort_by_key(|(rank, _)| *rank);
        hits.into_iter().map(|(_, m)| m).collect()
    }
}

/// [`Catalog::fetch`] against `url` at time `now`.
fn fetch_from(url: &str, models_dir: &Path, force: bool, now: SystemTime) -> Result<Catalog> {
    let cached = Catalog::load_cached(models_dir);
    if !force
        && let Some(cat) = cached.as_ref()
        && cache_is_fresh(cat.fetched_at, now)
    {
        return Ok(cat.clone());
    }
    match download(url, models_dir, cached.is_some(), now) {
        Ok(cat) => Ok(cat),
        Err(e) if !force && cached.is_some() => {
            tracing::warn!("using the cached OpenModelDB catalogue: {e}");
            Ok(cached.expect("checked above"))
        }
        Err(e) => Err(e),
    }
}

/// GETs the catalogue, revalidating with the stored `ETag` when a cache exists.
fn download(url: &str, models_dir: &Path, have_cache: bool, now: SystemTime) -> Result<Catalog> {
    let dl = |e: &dyn std::fmt::Display| ScaleUpError::Download(format!("{url}: {e}"));
    let agent = ureq::Agent::config_builder()
        .timeout_connect(Some(TIMEOUT))
        .timeout_recv_response(Some(TIMEOUT))
        .build()
        .new_agent();
    let mut request = agent.get(url);
    let etag = read_meta(models_dir).and_then(|m| m.etag);
    if have_cache && let Some(tag) = &etag {
        request = request.header("If-None-Match", tag);
    }
    let mut response = request.call().map_err(|e| dl(&e))?;
    if response.status().as_u16() == 304 {
        let mut cat = Catalog::load_cached(models_dir)
            .ok_or_else(|| dl(&"304 Not Modified but the cache is unreadable"))?;
        store_cache(
            models_dir,
            &fs::read_to_string(hub_dir(models_dir).join("catalog.json"))?,
            etag.as_deref(),
            now,
        )?;
        cat.fetched_at = now;
        return Ok(cat);
    }
    let new_etag = response
        .headers()
        .get("etag")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let body = response
        .body_mut()
        .with_config()
        .limit(MAX_BODY)
        .read_to_string()
        .map_err(|e| dl(&e))?;
    let mut cat = Catalog::parse(&body)?;
    store_cache(models_dir, &body, new_etag.as_deref(), now)?;
    cat.fetched_at = now;
    Ok(cat)
}

/// What the host can run.
#[derive(Clone, Debug, Default)]
pub struct Capabilities {
    /// An ONNX runtime is available.
    pub onnx: bool,
    /// Architectures of `.pth`/`.safetensors` files that run without conversion, lowercase
    /// (for example `esrgan`, `compact`, `span`, `realplksr`).
    pub native_archs: Vec<String>,
}

impl Capabilities {
    /// This process: ONNX is built in; native architectures come from the registered extra backend.
    pub fn current() -> Self {
        Self {
            onnx: true,
            native_archs: crate::backend::extra_backend()
                .map(|b| (b.native_archs)())
                .unwrap_or_default(),
        }
    }
}

/// Criteria for [`Catalog::filter`]; the default matches every model.
#[derive(Default, Clone, Debug)]
pub struct Filter {
    pub scale: Option<u32>,
    /// Every one of these tags (case-insensitive).
    pub tags_all: Vec<String>,
    /// At least one of these tags; no restriction when empty.
    pub tags_any: Vec<String>,
    pub arch: Option<String>,
    pub license: Option<Vec<LicenseClass>>,
    pub commercial_only: bool,
    /// Three channels in and out.
    pub rgb_only: bool,
    /// `"onnx"` or `"pytorch"`: has a file for that platform.
    pub platform: Option<String>,
    /// Can run on this host (ignores whether the model is already installed).
    pub runnable: Option<Capabilities>,
}

impl Filter {
    fn matches(&self, m: &ModelCard) -> bool {
        let has_tag = |t: &String| m.tags.iter().any(|x| x.eq_ignore_ascii_case(t));
        let class = m.license_class();
        self.scale.is_none_or(|s| m.scale == s)
            && self.tags_all.iter().all(has_tag)
            && (self.tags_any.is_empty() || self.tags_any.iter().any(has_tag))
            && self
                .arch
                .as_ref()
                .is_none_or(|a| m.architecture.eq_ignore_ascii_case(a))
            && self.license.as_ref().is_none_or(|l| l.contains(&class))
            && (!self.commercial_only || commercial_ok(class))
            && (!self.rgb_only || (m.in_ch == 3 && m.out_ch == 3))
            && self
                .platform
                .as_ref()
                .is_none_or(|p| m.resources.iter().any(|r| r.platform.as_deref() == Some(p)))
            && self
                .runnable
                .as_ref()
                .is_none_or(|caps| runnable_resources(m, caps).next().is_some())
    }
}

/// How a model stands on this host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Readiness {
    Installed,
    /// Runs as is and a file sits on a host the app can download from.
    Downloadable,
    /// Runs as is, but the files are on hosts the user has to open in a browser.
    ManualDownload,
    /// Only a PyTorch file this host cannot run directly; it has to be converted to ONNX first.
    NeedsConversion,
    Unsupported,
}

/// Whether the architecture of `m` runs from its PyTorch file on this host. Some DySample models
/// are filed under plain `realplksr` (their own architecture is `realplksr-dysample`); the native
/// builder cannot run their upsampler, so their id or name decides.
pub(crate) fn is_native(m: &ModelCard, caps: &Capabilities) -> bool {
    let dysample = |s: &str| s.to_ascii_lowercase().contains("dysample");
    if m.architecture.eq_ignore_ascii_case("realplksr") && (dysample(&m.id) || dysample(&m.name)) {
        return false;
    }
    caps.native_archs
        .iter()
        .any(|a| a.eq_ignore_ascii_case(&m.architecture))
}

/// The files of `m` this host can run: ONNX with an ONNX runtime, PyTorch files of a native
/// architecture. Models that are not RGB-in, RGB-out have none.
pub(crate) fn runnable_resources<'a>(
    m: &'a ModelCard,
    caps: &Capabilities,
) -> impl Iterator<Item = &'a Resource> {
    let rgb = m.in_ch == 3;
    let native = is_native(m, caps);
    m.resources.iter().filter(move |r| {
        rgb && match r.platform.as_deref() {
            Some("onnx") => caps.onnx,
            Some("pytorch") => native,
            _ => false,
        }
    })
}

pub fn readiness(card: &ModelCard, caps: &Capabilities, installed: &Installed) -> Readiness {
    if installed.get(&card.id).is_some() {
        return Readiness::Installed;
    }
    if card.in_ch != 3 {
        return Readiness::Unsupported;
    }
    let mut runnable = runnable_resources(card, caps).peekable();
    if runnable.peek().is_some() {
        let direct = runnable.any(|r| r.urls.iter().any(|u| is_direct_host(u)));
        return if direct {
            Readiness::Downloadable
        } else {
            Readiness::ManualDownload
        };
    }
    if card
        .resources
        .iter()
        .any(|r| r.platform.as_deref() == Some("pytorch"))
    {
        Readiness::NeedsConversion
    } else {
        Readiness::Unsupported
    }
}

/// The host of an `http(s)` URL, lowercase, without port or credentials.
pub(crate) fn url_host(url: &str) -> Option<String> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let authority = rest.split(['/', '?', '#']).next()?;
    let host = authority.rsplit('@').next()?;
    let host = host.split(':').next()?;
    (!host.is_empty()).then(|| host.to_lowercase())
}

/// Hosts the app downloads from without the user's help: GitHub, Oracle and Backblaze object
/// storage, Hugging Face and Google Drive.
pub(crate) fn is_direct_host(url: &str) -> bool {
    let Some(host) = url_host(url) else {
        return false;
    };
    host == "github.com"
        || host == "huggingface.co"
        || host == "drive.google.com"
        || host.ends_with(".backblazeb2.com")
        || (host.starts_with("objectstorage.") && host.ends_with(".oraclecloud.com"))
}

/// Relative cost of an architecture: 0 fastest, 3 slowest or unknown.
pub fn speed_tier(arch: &str) -> u8 {
    match arch.to_lowercase().as_str() {
        "compact" | "span" => 0,
        "esrgan" | "realplksr" | "spsr" | "esrgan+" => 1,
        "omnisr" | "swinir" | "srformer" | "dat" | "hat" | "rgt" | "drct" | "atd" | "grl" => 2,
        _ => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::{Installed, InstalledModel, LicenseClass};
    use std::time::{Duration, SystemTime};

    const SAMPLE: &str = include_str!("../../../../tests/fixtures/hub/models.sample.json");

    fn sample() -> Catalog {
        Catalog::parse(SAMPLE).unwrap()
    }

    fn card(json: &str) -> ModelCard {
        let mut c: ModelCard = serde_json::from_str(json).unwrap();
        c.id = "x".into();
        c
    }

    /// A minimal card with the given architecture, channels and resources.
    fn card_with(arch: &str, in_ch: u32, resources: &str) -> ModelCard {
        card(&format!(
            r#"{{"name":"n","author":"a","license":null,"tags":[],"description":"","date":"2024-01-01",
                "architecture":"{arch}","scale":2,"inputChannels":{in_ch},"outputChannels":3,
                "resources":{resources}}}"#
        ))
    }

    #[test]
    fn current_capabilities_have_onnx_and_the_native_list_of_the_backend() {
        let caps = Capabilities::current();
        assert!(caps.onnx);
        match crate::backend::extra_backend() {
            Some(b) => assert_eq!(caps.native_archs, (b.native_archs)()),
            None => assert!(caps.native_archs.is_empty()),
        }
    }

    #[test]
    fn parse_sample_has_40_ids() {
        let cat = sample();
        assert_eq!(cat.models.len(), 40);
        let m = cat.get("4x-DRCT-L").expect("known id");
        assert_eq!(m.id, "4x-DRCT-L");
        assert_eq!(m.architecture, "drct");
        assert!(cat.get("nope").is_none());
    }

    #[test]
    fn author_string_or_array_both_parse() {
        let cat = sample();
        let swinir = cat.get("8x-classicalSR-DF2K-s64w8-SwinIR-M").unwrap();
        assert!(swinir.author.is_array());
        assert_eq!(swinir.authors(), vec!["jingyunliang".to_string()]);
        let one = cat.get("4x-DRCT-L").unwrap();
        assert!(one.author.is_string());
        assert_eq!(one.authors().len(), 1);
        assert!(Catalog::parse("[]").is_err());
        assert!(Catalog::parse("not json").is_err());
    }

    #[test]
    fn license_classes_table() {
        use LicenseClass::*;
        let table = [
            ("MIT", Permissive),
            ("BSD-3-Clause", Permissive),
            ("Apache-2.0", Permissive),
            ("CC0-1.0", Permissive),
            ("Unlicense", Permissive),
            ("WTFPL", Permissive),
            ("CC-BY-4.0", Attribution),
            ("CC-BY-SA-4.0", ShareAlike),
            ("GPL-3.0-only", Copyleft),
            ("CC-BY-NC-4.0", NonCommercial),
            ("CC-BY-NC-SA-4.0", NonCommercial),
            ("Proprietary-Thing", Unknown),
        ];
        for (s, class) in table {
            assert_eq!(classify_license(Some(s)), class, "{s}");
        }
        assert_eq!(classify_license(None), Unknown);
        for c in [Permissive, Attribution, ShareAlike, Copyleft] {
            assert!(commercial_ok(c), "{c:?}");
        }
        for c in [NonCommercial, Unknown] {
            assert!(!commercial_ok(c), "{c:?}");
        }
    }

    #[test]
    fn filter_scale_and_commercial() {
        let cat = sample();
        // Hand count of the fixture: 24 x1, 7 x2, 6 x4, 1 x8, 2 x16 models.
        let scale = |s| Filter {
            scale: Some(s),
            ..Filter::default()
        };
        assert_eq!(cat.filter(&Filter::default()).len(), 40);
        assert_eq!(cat.filter(&scale(1)).len(), 24);
        assert_eq!(cat.filter(&scale(4)).len(), 6);
        assert_eq!(cat.filter(&scale(8)).len(), 1);
        // 27 of the 40 licenses allow commercial use; 5 of those are x4.
        let commercial = Filter {
            commercial_only: true,
            ..Filter::default()
        };
        assert_eq!(cat.filter(&commercial).len(), 27);
        let c4 = Filter {
            scale: Some(4),
            commercial_only: true,
            ..Filter::default()
        };
        assert_eq!(cat.filter(&c4).len(), 5);
        // 38 models are RGB in and out.
        let rgb = Filter {
            rgb_only: true,
            ..Filter::default()
        };
        assert_eq!(cat.filter(&rgb).len(), 38);
        let nc = Filter {
            license: Some(vec![LicenseClass::NonCommercial]),
            ..Filter::default()
        };
        assert_eq!(cat.filter(&nc).len(), 11);
        let onnx = Filter {
            platform: Some("onnx".into()),
            ..Filter::default()
        };
        assert_eq!(cat.filter(&onnx).len(), 4);
        let arch = Filter {
            arch: Some("SPAN".into()),
            ..Filter::default()
        };
        assert_eq!(cat.filter(&arch).len(), 1);
    }

    #[test]
    fn filter_tags_and_runnable() {
        let cat = sample();
        let tag = cat
            .models
            .iter()
            .flat_map(|m| m.tags.clone())
            .next()
            .expect("fixture has tags");
        let any = Filter {
            tags_any: vec![tag.to_uppercase(), "no-such-tag".into()],
            ..Filter::default()
        };
        assert!(!cat.filter(&any).is_empty());
        let all = Filter {
            tags_all: vec![tag, "no-such-tag".into()],
            ..Filter::default()
        };
        assert!(cat.filter(&all).is_empty());
        let caps = Capabilities {
            onnx: true,
            native_archs: vec!["esrgan".into()],
        };
        let runnable = Filter {
            runnable: Some(caps),
            ..Filter::default()
        };
        let found = cat.filter(&runnable);
        let ids: Vec<&str> = found.iter().map(|m| m.id.as_str()).collect();
        assert!(ids.contains(&"1x-BroadcastToStudio-SPAN"), "has ONNX");
        assert!(ids.contains(&"1x-Anti-Aliasing"), "native esrgan");
        assert!(!ids.contains(&"4x-DRCT-L"), "needs conversion");
        assert!(!ids.contains(&"1x-MangaJPEGHQ"), "one channel");
    }

    #[test]
    fn search_ranks_name_hits_first() {
        let cat = sample();
        let hits = cat.search("STAR", &Filter::default());
        assert_eq!(hits.len(), 4);
        for m in &hits[..2] {
            assert!(m.id.to_lowercase().contains("star"), "{}", m.id);
        }
        for m in &hits[2..] {
            assert!(!m.id.to_lowercase().contains("star"), "{}", m.id);
        }
        // Every word must match; the filter still applies.
        assert_eq!(cat.search("star hq", &Filter::default()).len(), 1);
        let x4 = Filter {
            scale: Some(4),
            ..Filter::default()
        };
        assert!(cat.search("star", &x4).iter().all(|m| m.scale == 4));
        assert_eq!(cat.search("  ", &Filter::default()).len(), 40);
        assert!(cat.search("zzzzqqq", &Filter::default()).is_empty());
        // Author matches count too.
        assert_eq!(cat.search("jingyunliang", &Filter::default()).len(), 1);
    }

    #[test]
    fn readiness_matrix() {
        let caps = Capabilities {
            onnx: true,
            native_archs: vec!["esrgan".into(), "span".into()],
        };
        let none = Installed::default();
        let gh = r#"["https://github.com/a/b/releases/download/v1/m.onnx"]"#;
        let mega = r#"["https://mega.nz/#!abc"]"#;
        let res = |platform: &str, kind: &str, urls: &str| {
            format!(
                r#"[{{"platform":"{platform}","type":"{kind}","size":1,"sha256":"00","urls":{urls}}}]"#
            )
        };
        let rd = |c: &ModelCard| readiness(c, &caps, &none);

        let onnx_gh = card_with("dat", 3, &res("onnx", "onnx", gh));
        assert_eq!(rd(&onnx_gh), Readiness::Downloadable);
        let onnx_mega = card_with("dat", 3, &res("onnx", "onnx", mega));
        assert_eq!(rd(&onnx_mega), Readiness::ManualDownload);
        let native = card_with("esrgan", 3, &res("pytorch", "pth", gh));
        assert_eq!(rd(&native), Readiness::Downloadable);
        let native_mega = card_with("esrgan", 3, &res("pytorch", "pth", mega));
        assert_eq!(rd(&native_mega), Readiness::ManualDownload);
        let conv = card_with("dat", 3, &res("pytorch", "pth", gh));
        assert_eq!(rd(&conv), Readiness::NeedsConversion);
        let one_ch = card_with("esrgan", 1, &res("pytorch", "pth", gh));
        assert_eq!(rd(&one_ch), Readiness::Unsupported);
        let empty = card_with("esrgan", 3, "[]");
        assert_eq!(rd(&empty), Readiness::Unsupported);
        // ONNX only, but the host has no ONNX runtime.
        let no_onnx = Capabilities {
            onnx: false,
            native_archs: vec![],
        };
        assert_eq!(readiness(&onnx_gh, &no_onnx, &none), Readiness::Unsupported);

        let mut inst = Installed::default();
        inst.models.push(InstalledModel {
            id: "x".into(),
            name: "n".into(),
            arch: "dat".into(),
            scale: 2,
            license: None,
            format: "onnx".into(),
            file: "x.onnx".into(),
            sha256: None,
            verified: false,
            source_url: String::new(),
            installed_at: String::new(),
        });
        assert_eq!(readiness(&onnx_mega, &caps, &inst), Readiness::Installed);
    }

    #[test]
    fn dysample_models_filed_under_realplksr_are_not_native() {
        let caps = Capabilities {
            onnx: true,
            native_archs: vec!["realplksr".into()],
        };
        let none = Installed::default();
        let gh_pth = r#"["https://github.com/a/b/releases/download/v1/m.pth"]"#;
        let gh_onnx = r#"["https://github.com/a/b/releases/download/v1/m.onnx"]"#;
        let pth = format!(
            r#"{{"platform":"pytorch","type":"pth","size":1,"sha256":"00","urls":{gh_pth}}}"#
        );
        let onnx = format!(
            r#"{{"platform":"onnx","type":"onnx","size":1,"sha256":"00","urls":{gh_onnx}}}"#
        );

        let plain = card_with("realplksr", 3, &format!("[{pth}]"));
        assert!(is_native(&plain, &caps));
        assert_eq!(readiness(&plain, &caps, &none), Readiness::Downloadable);

        let mut by_id = card_with("RealPLKSR", 3, &format!("[{pth}]"));
        by_id.id = "4x-Nomos2-realplksr-dysample".into();
        assert!(!is_native(&by_id, &caps));
        assert_eq!(readiness(&by_id, &caps, &none), Readiness::NeedsConversion);

        let mut by_name = card_with("realplksr", 3, &format!("[{pth},{onnx}]"));
        by_name.name = "4x mssim RealPLKSR DySample pretrain".into();
        assert!(!is_native(&by_name, &caps));
        assert_eq!(readiness(&by_name, &caps, &none), Readiness::Downloadable);
        let runnable: Vec<_> = runnable_resources(&by_name, &caps)
            .map(|r| r.platform.as_deref())
            .collect();
        assert_eq!(runnable, [Some("onnx")], "only the ONNX file runs");
    }

    #[test]
    fn download_hosts() {
        for ok in [
            "https://github.com/a/b",
            "https://objectstorage.us-phoenix-1.oraclecloud.com/n/x/o/f.pth",
            "https://f002.backblazeb2.com/file/b/f.pth",
            "https://huggingface.co/a/b/resolve/main/f.onnx",
            "https://drive.google.com/file/d/1/view",
        ] {
            assert!(is_direct_host(ok), "{ok}");
        }
        for bad in [
            "https://mega.nz/#!abc",
            "https://evilgithub.com/x",
            "https://github.com.evil.example/x",
            "https://example.com/?u=github.com",
            "not a url",
        ] {
            assert!(!is_direct_host(bad), "{bad}");
        }
    }

    #[test]
    fn speed_tier_table() {
        for a in ["compact", "span"] {
            assert_eq!(speed_tier(a), 0, "{a}");
        }
        for a in ["esrgan", "realplksr", "spsr", "esrgan+"] {
            assert_eq!(speed_tier(a), 1, "{a}");
        }
        for a in [
            "omnisr", "swinir", "srformer", "dat", "hat", "rgt", "drct", "atd", "grl",
        ] {
            assert_eq!(speed_tier(a), 2, "{a}");
        }
        assert_eq!(speed_tier("rife"), 3);
        assert_eq!(speed_tier("SPAN"), 0);
    }

    #[test]
    fn cache_roundtrip_in_tempdir() {
        let dir = tempfile::tempdir().unwrap();
        assert!(Catalog::load_cached(dir.path()).is_none());
        let when = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        store_cache(dir.path(), SAMPLE, Some("\"abc\""), when).unwrap();
        assert!(dir.path().join("hub").join("catalog.json").is_file());
        let cat = Catalog::load_cached(dir.path()).expect("cache loads");
        assert_eq!(cat.models.len(), 40);
        assert_eq!(cat.fetched_at, when);
        assert_eq!(
            read_meta(dir.path()).unwrap().etag.as_deref(),
            Some("\"abc\"")
        );

        let day = Duration::from_secs(24 * 3600);
        assert!(cache_is_fresh(when, when + day));
        assert!(!cache_is_fresh(when, when + day + Duration::from_secs(1)));
        // A cache dated in the future (clock change) is not trusted.
        assert!(!cache_is_fresh(when + day, when));
    }

    #[test]
    fn corrupt_cache_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("hub")).unwrap();
        std::fs::write(dir.path().join("hub").join("catalog.json"), "{ nope").unwrap();
        assert!(Catalog::load_cached(dir.path()).is_none());
    }

    /// Serves `replies` (raw HTTP responses) one per connection and returns the requests it saw.
    fn serve(replies: Vec<String>) -> (String, std::thread::JoinHandle<Vec<String>>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/models.json", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for reply in replies {
                let (mut s, _) = listener.accept().unwrap();
                let mut buf = vec![0u8; 8192];
                let n = s.read(&mut buf).unwrap();
                seen.push(String::from_utf8_lossy(&buf[..n]).into_owned());
                s.write_all(reply.as_bytes()).unwrap();
            }
            seen
        });
        (url, handle)
    }

    fn ok_reply(body: &str, etag: &str) -> String {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nETag: {etag}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    #[test]
    fn fetch_uses_cache_then_revalidates_with_etag() {
        let dir = tempfile::tempdir().unwrap();
        let t0 = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let not_modified =
            "HTTP/1.1 304 Not Modified\r\nETag: \"v1\"\r\nConnection: close\r\n\r\n".to_string();
        let (url, server) = serve(vec![ok_reply(SAMPLE, "\"v1\""), not_modified]);

        // Empty cache: downloads and stores.
        let cat = fetch_from(&url, dir.path(), false, t0).unwrap();
        assert_eq!(cat.models.len(), 40);
        assert_eq!(cat.fetched_at, t0);

        // Fresh cache: no request (the server only answers two).
        let hour = Duration::from_secs(3600);
        let cat = fetch_from(&url, dir.path(), false, t0 + hour).unwrap();
        assert_eq!(cat.fetched_at, t0);

        // Stale cache: conditional request, 304 keeps the body and renews the date.
        let later = t0 + Duration::from_secs(25 * 3600);
        let cat = fetch_from(&url, dir.path(), false, later).unwrap();
        assert_eq!(cat.models.len(), 40);
        assert_eq!(cat.fetched_at, later);
        assert_eq!(Catalog::load_cached(dir.path()).unwrap().fetched_at, later);

        let seen = server.join().unwrap();
        assert_eq!(seen.len(), 2);
        assert!(!seen[0].to_lowercase().contains("if-none-match"));
        assert!(seen[1].to_lowercase().contains("if-none-match: \"v1\""));
    }

    #[test]
    fn fetch_failure_falls_back_to_stale_cache_unless_forced() {
        let dir = tempfile::tempdir().unwrap();
        let t0 = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        store_cache(dir.path(), SAMPLE, None, t0).unwrap();
        // Nothing listens on port 1.
        let dead = "http://127.0.0.1:1/models.json";
        let later = t0 + Duration::from_secs(48 * 3600);
        let cat = fetch_from(dead, dir.path(), false, later).unwrap();
        assert_eq!(cat.fetched_at, t0);
        assert!(fetch_from(dead, dir.path(), true, later).is_err());
        let empty = tempfile::tempdir().unwrap();
        assert!(fetch_from(dead, empty.path(), false, later).is_err());
    }
}
