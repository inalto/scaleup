//! Getting a model's file onto this machine: download, verify, register.

use std::fs::{self, File};
use std::io::{BufWriter, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use super::catalog::{Capabilities, ModelCard, Resource, is_native, runnable_resources, url_host};
use super::installed::{Installed, InstalledModel};
use crate::{Result, ScaleUpError};

/// Connect and first-response timeout, as for the registry's downloads.
const TIMEOUT: Duration = Duration::from_secs(30);
/// How much of a body is looked at to tell a web page from a model file.
const SNIFF_LEN: usize = 512;
/// A Google Drive warning page is a few KB; this is a ceiling, not an expectation.
const MAX_PAGE: u64 = 2 * 1024 * 1024;
const MODEL_EXTENSIONS: [&str; 3] = ["onnx", "pth", "safetensors"];

/// How a URL can be fetched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Host {
    Direct,
    GoogleDrive {
        file_id: String,
    },
    /// The user has to open it in a browser.
    Manual,
}

/// A response whose body is read as a stream: model files are hundreds of megabytes.
pub struct HttpResponse {
    pub content_type: Option<String>,
    pub content_disposition: Option<String>,
    pub content_length: Option<u64>,
    /// `Set-Cookie` pairs, to send back on the next request.
    pub cookies: Vec<(String, String)>,
    pub body: Box<dyn Read + Send>,
}

pub trait Http {
    fn get(&self, url: &str, cookies: &[(String, String)]) -> Result<HttpResponse>;
}

/// The real client: `https` only, with the registry's timeouts and no total body timeout.
pub struct UreqHttp;

impl Http for UreqHttp {
    fn get(&self, url: &str, cookies: &[(String, String)]) -> Result<HttpResponse> {
        let dl = |e: &dyn std::fmt::Display| ScaleUpError::Download(format!("{url}: {e}"));
        if !url.starts_with("https://") {
            return Err(dl(&"only https downloads are allowed"));
        }
        let agent = ureq::Agent::config_builder()
            .timeout_connect(Some(TIMEOUT))
            .timeout_recv_response(Some(TIMEOUT))
            .https_only(true)
            .build()
            .new_agent();
        let mut request = agent.get(url);
        if !cookies.is_empty() {
            let header: Vec<String> = cookies.iter().map(|(k, v)| format!("{k}={v}")).collect();
            request = request.header("Cookie", header.join("; "));
        }
        let response = request.call().map_err(|e| dl(&e))?;
        let header = |name: &str| {
            response
                .headers()
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned)
        };
        let cookies = response
            .headers()
            .get_all("set-cookie")
            .iter()
            .filter_map(|v| v.to_str().ok())
            .filter_map(|c| c.split(';').next()?.split_once('='))
            .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
            .collect();
        let content_type = header("content-type");
        let content_disposition = header("content-disposition");
        // A decoded (compressed) body does not have the length of the header.
        let encoded = header("content-encoding").is_some();
        let body = response.into_body();
        Ok(HttpResponse {
            content_type,
            content_disposition,
            content_length: body.content_length().filter(|_| !encoded),
            cookies,
            body: Box::new(body.into_reader()),
        })
    }
}

/// Which way to fetch `url`. Google Drive links with a file id are handled; other hosts of
/// [`super::catalog`]'s direct list are plain downloads; everything else is for the user.
pub fn classify_url(url: &str) -> Host {
    if url_host(url).as_deref() == Some("drive.google.com") {
        return match drive_file_id(url) {
            Some(file_id) => Host::GoogleDrive { file_id },
            None => Host::Manual,
        };
    }
    if super::catalog::is_direct_host(url) {
        Host::Direct
    } else {
        Host::Manual
    }
}

fn is_id_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// `/file/d/<id>/â€¦` or an `id=<id>` query parameter.
fn drive_file_id(url: &str) -> Option<String> {
    let rest = url.split_once("://")?.1;
    let (path, query) = rest.split_once('?').unwrap_or((rest, ""));
    let query = query.split('#').next().unwrap_or("");
    let from_path = path
        .split_once("/file/d/")
        .map(|(_, tail)| tail.split(['/', '#']).next().unwrap_or(""));
    let from_query = query.split('&').find_map(|kv| kv.strip_prefix("id="));
    let id = from_path.filter(|s| !s.is_empty()).or(from_query)?;
    (!id.is_empty() && id.chars().all(is_id_char)).then(|| id.to_owned())
}

fn host_rank(host: &Host) -> u8 {
    match host {
        Host::Direct => 0,
        Host::GoogleDrive { .. } => 1,
        Host::Manual => 2,
    }
}

fn format_rank(card: &ModelCard, caps: &Capabilities, kind: &str) -> u8 {
    let pytorch_first = is_native(card, caps);
    match (kind, pytorch_first) {
        ("safetensors", true) => 0,
        ("pth", true) => 1,
        ("onnx", true) => 2,
        ("onnx", false) => 0,
        _ => 3,
    }
}

/// The file to install and the index of the URL to use. Only files this host can run count.
/// A file with a downloadable URL beats one that is only on manual hosts; then the format
/// (ONNX unless the architecture runs natively, then safetensors, pth, ONNX); within a file the
/// URL preference is Direct, Google Drive, Manual.
pub fn pick_resource<'a>(
    card: &'a ModelCard,
    caps: &Capabilities,
) -> Option<(&'a Resource, usize)> {
    runnable_resources(card, caps)
        .filter_map(|r| {
            let (idx, host) = r
                .urls
                .iter()
                .enumerate()
                .map(|(i, u)| (i, classify_url(u)))
                .min_by_key(|(_, h)| host_rank(h))?;
            let key = (host == Host::Manual, format_rank(card, caps, &r.kind));
            Some((key, r, idx))
        })
        .min_by_key(|(key, _, _)| *key)
        .map(|(_, r, idx)| (r, idx))
}

fn manual_error(
    card: &ModelCard,
    why: &str,
    url: Option<&str>,
    expected: &str,
    sha: Option<&str>,
) -> ScaleUpError {
    let mut msg = format!("{why}. ");
    match url {
        Some(url) => msg.push_str(&format!(
            "Open {url} in a browser, save the file as '{expected}'"
        )),
        None => msg.push_str(&format!(
            "Get the file elsewhere (expected name '{expected}')"
        )),
    }
    if let Some(sha) = sha {
        msg.push_str(&format!(" (sha256 {sha})"));
    }
    msg.push_str(&format!(
        ", then run `scaleup hub import {} <file>`",
        card.id
    ));
    ScaleUpError::Download(msg)
}

/// Nothing but letters, digits, `.`, `_` and `-`, and no leading dot: the id names a folder.
pub(crate) fn check_id(id: &str) -> Result<()> {
    let ok = !id.is_empty()
        && !id.starts_with('.')
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    if ok {
        Ok(())
    } else {
        Err(ScaleUpError::Download(format!(
            "'{id}' is not a usable model id"
        )))
    }
}

fn has_model_extension(name: &str) -> bool {
    name.rsplit_once('.')
        .is_some_and(|(_, ext)| MODEL_EXTENSIONS.iter().any(|e| ext.eq_ignore_ascii_case(e)))
}

/// The last path segment of `raw` when it is safe to use as a file name on every platform,
/// else an empty string. Names with a drive prefix or stream marker (`:`), characters Windows
/// forbids, control characters, a reserved device stem (`NUL`, `COM1`, ...) or a trailing dot
/// are refused rather than repaired.
fn sanitize_name(raw: &str) -> String {
    let last = raw.rsplit(['/', '\\']).next().unwrap_or("").trim();
    let stem = last.split('.').next().unwrap_or("").trim_end();
    let upper = stem.to_ascii_uppercase();
    let reserved = matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || ((upper.starts_with("COM") || upper.starts_with("LPT"))
            && upper.len() == 4
            && upper.as_bytes()[3].is_ascii_digit()
            && upper.as_bytes()[3] != b'0');
    let bad_char = |c: char| c.is_control() || matches!(c, ':' | '<' | '>' | '"' | '|' | '?' | '*');
    let one_component = {
        let mut parts = Path::new(last).components();
        matches!(parts.next(), Some(std::path::Component::Normal(_))) && parts.next().is_none()
    };
    if last.is_empty()
        || reserved
        || last.chars().any(bad_char)
        || last.ends_with(['.', ' '])
        || !one_component
    {
        String::new()
    } else {
        last.to_owned()
    }
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 2 < b.len()
            && let Some(Ok(v)) = s.get(i + 1..i + 3).map(|h| u8::from_str_radix(h, 16))
        {
            out.push(v);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The `filename` of a `Content-Disposition` header (`filename*=UTF-8''â€¦` wins).
fn disposition_name(header: &str) -> Option<String> {
    let lower = header.to_ascii_lowercase();
    if let Some(at) = lower.find("filename*=") {
        let value = header[at + 10..].split(';').next()?.trim();
        let encoded = value.rsplit_once('\'').map_or(value, |(_, v)| v);
        return Some(percent_decode(encoded.trim_matches('"')));
    }
    let at = lower.find("filename=")?;
    let value = header[at + 9..].split(';').next()?.trim();
    Some(value.trim_matches('"').to_owned())
}

/// The name to save under: the URL's last segment when it has a model extension, else the
/// `Content-Disposition` name when it has one, else `<id>.<kind>`.
fn file_name(url: &str, disposition: Option<&str>, id: &str, kind: &str) -> String {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let from_url = sanitize_name(path.rsplit('/').next().unwrap_or(""));
    if has_model_extension(&from_url) {
        return from_url;
    }
    if let Some(name) = disposition.and_then(disposition_name) {
        let name = sanitize_name(&name);
        if has_model_extension(&name) {
            return name;
        }
    }
    format!("{id}.{kind}")
}

fn valid_format(kind: &str) -> Result<String> {
    let kind = kind.to_ascii_lowercase();
    if MODEL_EXTENSIONS.contains(&kind.as_str()) {
        Ok(kind)
    } else {
        Err(ScaleUpError::Download(format!(
            "unsupported file type '{kind}' (expected onnx, pth or safetensors)"
        )))
    }
}

fn non_empty(s: &Option<String>) -> Option<&str> {
    s.as_deref().map(str::trim).filter(|s| !s.is_empty())
}

/// A response with its first bytes already read, so an HTML page can be told from a model.
struct Opened {
    content_disposition: Option<String>,
    content_length: Option<u64>,
    cookies: Vec<(String, String)>,
    html: bool,
    prefix: Vec<u8>,
    rest: Box<dyn Read + Send>,
}

impl Opened {
    fn reader(self) -> impl Read + Send {
        Cursor::new(self.prefix).chain(self.rest)
    }

    fn page(self, url: &str) -> Result<String> {
        let mut text = Vec::new();
        self.reader()
            .take(MAX_PAGE)
            .read_to_end(&mut text)
            .map_err(|e| ScaleUpError::Download(format!("{url}: {e}")))?;
        Ok(String::from_utf8_lossy(&text).into_owned())
    }
}

fn looks_like_html(content_type: Option<&str>, prefix: &[u8]) -> bool {
    if content_type.is_some_and(|c| c.trim_start().to_ascii_lowercase().starts_with("text/html")) {
        return true;
    }
    let head = String::from_utf8_lossy(prefix);
    let head = head.trim_start_matches(|c: char| c.is_whitespace() || c == '\u{feff}');
    let head = head
        .get(..head.len().min(15))
        .unwrap_or("")
        .to_ascii_lowercase();
    head.starts_with("<!doctype") || head.starts_with("<html")
}

fn open(http: &dyn Http, url: &str, cookies: &[(String, String)]) -> Result<Opened> {
    let mut response = http.get(url, cookies)?;
    let mut prefix = vec![0u8; SNIFF_LEN];
    let mut filled = 0;
    while filled < SNIFF_LEN {
        let n = response
            .body
            .read(&mut prefix[filled..])
            .map_err(|e| ScaleUpError::Download(format!("{url}: {e}")))?;
        if n == 0 {
            break;
        }
        filled += n;
    }
    prefix.truncate(filled);
    Ok(Opened {
        html: looks_like_html(response.content_type.as_deref(), &prefix),
        content_disposition: response.content_disposition,
        content_length: response.content_length,
        cookies: response.cookies,
        prefix,
        rest: response.body,
    })
}

/// The value of `name="â€¦"` inside one tag.
fn attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let key = format!(" {name}=\"");
    let start = tag.find(&key)? + key.len();
    let len = tag[start..].find('"')?;
    Some(&tag[start..start + len])
}

fn unescape(s: &str) -> String {
    s.replace("&amp;", "&")
}

fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn google_host(url: &str) -> bool {
    url.starts_with("https://")
        && url_host(url).is_some_and(|h| {
            h == "google.com"
                || h.ends_with(".google.com")
                || h == "googleusercontent.com"
                || h.ends_with(".googleusercontent.com")
        })
}

/// The token after `key` (`confirm=`, `uuid=`) in the page.
fn token_after<'a>(page: &'a str, key: &str) -> Option<&'a str> {
    let start = page.find(key)? + key.len();
    let len = page[start..]
        .find(|c: char| !is_id_char(c))
        .unwrap_or(page.len() - start);
    (len > 0).then(|| &page[start..start + len])
}

/// The request that gets past Google Drive's "can't scan this file for viruses" page: the
/// `download-form` with its hidden fields, or else the `confirm=` (and `uuid=`) of its link.
fn drive_followup(page: &str, file_id: &str) -> Option<String> {
    if let Some(form_at) = page.find("id=\"download-form\"") {
        let tag_start = page[..form_at].rfind("<form")?;
        let tag_end = form_at + page[form_at..].find('>')?;
        let action = unescape(attr(&page[tag_start..=tag_end], "action")?);
        let form_end = page[tag_end..]
            .find("</form>")
            .map_or(page.len(), |e| tag_end + e);
        let mut fields = Vec::new();
        for input in page[tag_end..form_end].split("<input").skip(1) {
            let tag = input.split('>').next().unwrap_or("");
            if let (Some(name), Some(value)) = (attr(tag, "name"), attr(tag, "value")) {
                fields.push(format!(
                    "{}={}",
                    url_encode(name),
                    url_encode(&unescape(value))
                ));
            }
        }
        if google_host(&action) && !fields.is_empty() {
            return Some(format!("{action}?{}", fields.join("&")));
        }
    }
    let confirm = token_after(page, "confirm=")?;
    let mut url = format!(
        "https://drive.google.com/uc?export=download&id={}&confirm={confirm}",
        url_encode(file_id)
    );
    if let Some(uuid) = token_after(page, "uuid=") {
        url.push_str(&format!("&uuid={uuid}"));
    }
    Some(url)
}

/// Fetches a Google Drive file, passing the large-file warning page when there is one.
fn open_drive(http: &dyn Http, file_id: &str, fail: &dyn Fn() -> ScaleUpError) -> Result<Opened> {
    let first_url = format!("https://drive.google.com/uc?export=download&id={file_id}");
    let first = open(http, &first_url, &[])?;
    if !first.html {
        return Ok(first);
    }
    let cookies = first.cookies.clone();
    let page = first.page(&first_url)?;
    let Some(next) = drive_followup(&page, file_id) else {
        return Err(fail());
    };
    let second = open(http, &next, &cookies)?;
    if second.html {
        return Err(fail());
    }
    Ok(second)
}

/// The most bytes accepted for a file the card gives no size for.
const MAX_UNKNOWN_SIZE: u64 = 4 * 1024 * 1024 * 1024;

/// The most bytes accepted for a file: the card's size with some margin, else a fixed ceiling.
fn byte_cap(size: Option<u64>) -> u64 {
    size.map_or(MAX_UNKNOWN_SIZE, |s| {
        s.saturating_add(s / 10).saturating_add(1024 * 1024)
    })
}

/// Streams the body to `dest`, hashing as it goes. Returns the SHA-256 in lowercase hex. Fails
/// when the body is longer than `cap` or shorter than its announced length.
fn write_hashed(
    opened: Opened,
    dest: &Path,
    total_hint: Option<u64>,
    cap: u64,
    progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<String> {
    let total = opened.content_length.or(total_hint);
    let announced = opened.content_length;
    let name = dest.display().to_string();
    let io_err = |e: std::io::Error| ScaleUpError::Download(format!("{name}: {e}"));
    let mut out = BufWriter::new(File::create(dest)?);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    let mut done = 0u64;
    let mut body = opened.reader();
    loop {
        let n = body.read(&mut buf).map_err(io_err)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        out.write_all(&buf[..n])?;
        done += n as u64;
        if done > cap {
            return Err(ScaleUpError::Download(format!(
                "{name}: the body is larger than the {cap} bytes expected"
            )));
        }
        progress(done, total);
    }
    if let Some(n) = announced
        && n != done
    {
        return Err(ScaleUpError::Download(format!(
            "{name}: truncated download, got {done} of {n} bytes"
        )));
    }
    out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Seconds since the epoch as `YYYY-MM-DDTHH:MM:SSZ` (proleptic Gregorian, UTC).
fn rfc3339_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Days since 1970-01-01 to a civil date (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

fn now_rfc3339() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    rfc3339_utc(secs)
}

/// Downloads the best file of `card` for this host into `<models_dir>/hub/<id>/`, checks its
/// SHA-256 against the card and records it in `installed.json` (replacing an earlier entry).
/// A card without a hash installs with `verified: false`. Hosts the app cannot fetch from, and
/// cards without a file, fail with the URL, the file name and hash to use with [`import_file`].
pub fn install(
    card: &ModelCard,
    models_dir: &Path,
    caps: &Capabilities,
    http: &dyn Http,
    progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<InstalledModel> {
    check_id(&card.id)?;
    let Some((resource, idx)) = pick_resource(card, caps) else {
        let convertible = card.in_ch == 3
            && card
                .resources
                .iter()
                .any(|r| r.platform.as_deref() == Some("pytorch") && !r.urls.is_empty());
        return Err(match card.resources.iter().find(|r| !r.urls.is_empty()) {
            Some(_) if convertible => ScaleUpError::Download(format!(
                "no file of '{}' ({}) can run on this machine as is; its PyTorch file can be converted to ONNX with `scaleup hub install {} --convert` (needs Python)",
                card.id, card.architecture, card.id
            )),
            Some(_) => ScaleUpError::Download(format!(
                "no file of '{}' ({}) can run on this machine",
                card.id, card.architecture
            )),
            None => {
                let kind = card.resources.first().map_or("onnx", |r| r.kind.as_str());
                let expected = format!("{}.{kind}", card.id);
                let sha = card.resources.first().and_then(|r| non_empty(&r.sha256));
                manual_error(
                    card,
                    &format!("OpenModelDB lists no download for '{}'", card.id),
                    None,
                    &expected,
                    sha,
                )
            }
        });
    };
    let fetched = fetch(card, resource, idx, models_dir, http, progress)?;
    register(
        card,
        models_dir,
        fetched.file,
        fetched.format,
        fetched.sha256,
        fetched.verified,
        &fetched.url,
    )
}

/// A downloaded file of a card, at its final name, not yet recorded in `installed.json`.
pub(crate) struct Fetched {
    pub file: PathBuf,
    pub format: String,
    /// SHA-256 of the file, lowercase hex.
    pub sha256: String,
    /// The hash matched the card's.
    pub verified: bool,
    pub url: String,
}

/// Downloads `card.resources[..]` `resource` from its URL `idx` into `<models_dir>/hub/<id>/` and
/// checks its SHA-256 against the card. A mismatch, a web page or a short body leave nothing.
pub(crate) fn fetch(
    card: &ModelCard,
    resource: &Resource,
    idx: usize,
    models_dir: &Path,
    http: &dyn Http,
    progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<Fetched> {
    check_id(&card.id)?;
    let format = valid_format(&resource.kind)?;
    let url = resource.urls[idx].as_str();
    let sha = non_empty(&resource.sha256);
    let expected_name = file_name(url, None, &card.id, &format);
    let host = classify_url(url);
    let manual = |why: &str| manual_error(card, why, Some(url), &expected_name, sha);
    let opened = match &host {
        Host::Manual => {
            return Err(manual(&format!(
                "'{}' cannot be downloaded automatically from this host",
                card.id
            )));
        }
        Host::Direct => open(http, url, &[])?,
        Host::GoogleDrive { file_id } => open_drive(http, file_id, &|| {
            manual("Google Drive did not return a file")
        })?,
    };
    if opened.html {
        return Err(ScaleUpError::Download(format!(
            "{url}: the server sent a web page, not a model file; open it in a browser and run `scaleup hub import {} <file>`",
            card.id
        )));
    }
    let name = file_name(
        url,
        opened.content_disposition.as_deref(),
        &card.id,
        &format,
    );
    let dir = models_dir.join("hub").join(&card.id);
    fs::create_dir_all(&dir)?;
    let dest = dir.join(&name);
    let part = dir.join(format!("{name}.part"));
    let actual = match write_hashed(
        opened,
        &part,
        resource.size,
        byte_cap(resource.size),
        progress,
    ) {
        Ok(actual) => actual,
        Err(e) => {
            let _ = fs::remove_file(&part);
            return Err(e);
        }
    };
    if let Some(expected) = sha
        && !actual.eq_ignore_ascii_case(expected)
    {
        let _ = fs::remove_file(&part);
        return Err(ScaleUpError::Download(format!(
            "{url}: sha256 mismatch: expected {expected}, got {actual}"
        )));
    }
    if let Err(e) = fs::rename(&part, &dest) {
        let _ = fs::remove_file(&part);
        return Err(e.into());
    }
    Ok(Fetched {
        file: dest,
        format,
        sha256: actual,
        verified: sha.is_some(),
        url: url.to_owned(),
    })
}

/// The name a hand-downloaded file of `card` is expected to have: the URL's last segment when it
/// names a model file, else `<id>.<kind>`.
pub fn expected_file_name(url: &str, id: &str, kind: &str) -> String {
    file_name(url, None, id, kind)
}

/// Deletes the folder of installed model `id` and its entry in `installed.json`.
pub fn uninstall(models_dir: &Path, id: &str) -> Result<()> {
    let mut installed = Installed::load(models_dir);
    if installed.get(id).is_none() {
        return Err(ScaleUpError::Model(format!(
            "'{id}' is not an installed OpenModelDB model"
        )));
    }
    check_id(id).map_err(|_| {
        ScaleUpError::Model(format!(
            "refusing to delete the folder of '{id}': not a plain model id"
        ))
    })?;
    match fs::remove_dir_all(models_dir.join("hub").join(id)) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
        _ => {}
    }
    installed.remove(id);
    installed.save(models_dir)
}

/// Records the file in `installed.json`, dropping the file of an earlier install of the same
/// model when it had another name.
pub(crate) fn register(
    card: &ModelCard,
    models_dir: &Path,
    file: PathBuf,
    format: String,
    sha256: String,
    verified: bool,
    source_url: &str,
) -> Result<InstalledModel> {
    let model = InstalledModel {
        id: card.id.clone(),
        name: card.name.clone(),
        arch: card.architecture.clone(),
        scale: card.scale,
        license: card.license.clone(),
        format,
        file,
        sha256: Some(sha256),
        verified,
        source_url: source_url.to_owned(),
        installed_at: now_rfc3339(),
    };
    let mut installed = Installed::load(models_dir);
    let old = installed.get(&card.id).map(|m| m.file.clone());
    installed.upsert(model.clone());
    installed.save(models_dir)?;
    if let Some(old) = old
        && old != model.file
        && old.starts_with(models_dir.join("hub").join(&card.id))
    {
        let _ = fs::remove_file(old);
    }
    Ok(model)
}

/// Registers a file the user downloaded by hand: its SHA-256 must match a file of the card
/// (a card without a hash for that file type accepts it as unverified). The file is moved to
/// `<models_dir>/hub/<id>/<its name>`; a file that does not match is left where it is.
pub fn import_file(card: &ModelCard, file: &Path, models_dir: &Path) -> Result<InstalledModel> {
    check_id(&card.id)?;
    let name = file
        .file_name()
        .map(|n| sanitize_name(&n.to_string_lossy()))
        .filter(|n| !n.is_empty())
        .ok_or_else(|| ScaleUpError::Download(format!("{}: not a file name", file.display())))?;
    let ext = valid_format(name.rsplit_once('.').map_or("", |(_, e)| e)).map_err(|_| {
        ScaleUpError::Download(format!(
            "{}: not a model file (expected .onnx, .pth or .safetensors)",
            file.display()
        ))
    })?;
    let mut hasher = Sha256::new();
    let mut reader =
        File::open(file).map_err(|e| ScaleUpError::Download(format!("{}: {e}", file.display())))?;
    std::io::copy(&mut reader, &mut hasher)?;
    let actual = format!("{:x}", hasher.finalize());
    let hash_matches =
        |r: &&Resource| non_empty(&r.sha256).is_some_and(|s| s.eq_ignore_ascii_case(&actual));
    let (resource, verified) = if let Some(r) = card.resources.iter().find(hash_matches) {
        (Some(r), true)
    } else if card.resources.is_empty() {
        (None, false)
    } else if let Some(r) = card
        .resources
        .iter()
        .find(|r| non_empty(&r.sha256).is_none() && r.kind.eq_ignore_ascii_case(&ext))
    {
        (Some(r), false)
    } else {
        let expected: Vec<&str> = card
            .resources
            .iter()
            .filter_map(|r| non_empty(&r.sha256))
            .collect();
        return Err(ScaleUpError::Download(format!(
            "{}: sha256 mismatch: expected {}, got {actual}",
            file.display(),
            expected.join(" or ")
        )));
    };
    let format = match resource {
        Some(r) => valid_format(&r.kind)?,
        None => ext,
    };
    let source_url = resource
        .and_then(|r| r.urls.first())
        .or_else(|| card.resources.iter().find_map(|r| r.urls.first()))
        .cloned()
        .unwrap_or_default();
    let dir = models_dir.join("hub").join(&card.id);
    fs::create_dir_all(&dir)?;
    let dest = dir.join(&name);
    if fs::rename(file, &dest).is_err() {
        // Another drive: copy next to the destination, rename, then drop the original.
        let part = dir.join(format!("{name}.part"));
        let copied = fs::copy(file, &part).and_then(|_| fs::rename(&part, &dest));
        if let Err(e) = copied {
            let _ = fs::remove_file(&part);
            return Err(e.into());
        }
        let _ = fs::remove_file(file);
    }
    register(
        card,
        models_dir,
        dest,
        format,
        actual,
        verified,
        &source_url,
    )
}

/// A scripted [`Http`] and card builders, shared by the tests of the hub modules.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::collections::VecDeque;
    use std::io::Cursor;
    use std::sync::Mutex;

    pub(crate) fn sha(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    pub(crate) type Request = (String, Vec<(String, String)>);

    #[derive(Default)]
    pub(crate) struct Fake {
        responses: Mutex<VecDeque<HttpResponse>>,
        requests: Mutex<Vec<Request>>,
    }

    impl Fake {
        pub(crate) fn with(responses: Vec<HttpResponse>) -> Fake {
            Fake {
                responses: Mutex::new(responses.into()),
                requests: Mutex::default(),
            }
        }
        pub(crate) fn requests(&self) -> Vec<Request> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl Http for Fake {
        fn get(&self, url: &str, cookies: &[(String, String)]) -> Result<HttpResponse> {
            self.requests
                .lock()
                .unwrap()
                .push((url.to_owned(), cookies.to_vec()));
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| ScaleUpError::Download(format!("{url}: unexpected request")))
        }
    }

    pub(crate) fn resp(body: &[u8], content_type: &str) -> HttpResponse {
        HttpResponse {
            content_type: Some(content_type.into()),
            content_disposition: None,
            content_length: Some(body.len() as u64),
            cookies: Vec::new(),
            body: Box::new(Cursor::new(body.to_vec())),
        }
    }

    pub(crate) fn resource(
        kind: &str,
        platform: &str,
        sha: Option<String>,
        urls: &[&str],
    ) -> Resource {
        Resource {
            platform: Some(platform.into()),
            kind: kind.into(),
            size: None,
            sha256: sha,
            urls: urls.iter().map(|u| (*u).to_owned()).collect(),
        }
    }

    pub(crate) fn card(arch: &str, resources: Vec<Resource>) -> ModelCard {
        ModelCard {
            id: "4x-Test".into(),
            name: "Test model".into(),
            author: serde_json::Value::Null,
            license: Some("MIT".into()),
            tags: Vec::new(),
            description: String::new(),
            date: String::new(),
            architecture: arch.into(),
            scale: 4,
            in_ch: 3,
            out_ch: 3,
            resources,
            images: Vec::new(),
            thumbnail: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::installed::Installed;
    use super::testing::*;
    use super::*;
    use crate::ScaleUpError;

    fn onnx_caps() -> Capabilities {
        Capabilities {
            onnx: true,
            native_archs: Vec::new(),
        }
    }

    const GH: &str = "https://github.com/a/b/releases/download/v1/4x-Test.onnx";
    const GD: &str = "https://drive.google.com/file/d/FILEID123/view?usp=sharing";

    fn leftovers(dir: &Path) -> Vec<String> {
        let mut found = Vec::new();
        if let Ok(rd) = std::fs::read_dir(dir.join("hub").join("4x-Test")) {
            for e in rd.flatten() {
                found.push(e.file_name().to_string_lossy().into_owned());
            }
        }
        found
    }

    #[test]
    fn direct_download_verifies_sha_and_registers() {
        let body = b"onnx-bytes".to_vec();
        let c = card(
            "span",
            vec![resource("onnx", "onnx", Some(sha(&body)), &[GH])],
        );
        let dir = tempfile::tempdir().unwrap();
        let http = Fake::with(vec![resp(&body, "application/octet-stream")]);
        let mut seen = Vec::new();
        let m = install(&c, dir.path(), &onnx_caps(), &http, &mut |d, t| {
            seen.push((d, t))
        })
        .unwrap();
        assert_eq!(m.id, "4x-Test");
        assert_eq!(m.format, "onnx");
        assert!(m.verified);
        assert_eq!(m.sha256.as_deref(), Some(sha(&body).as_str()));
        assert_eq!(m.source_url, GH);
        assert!(m.installed_at.ends_with('Z') && m.installed_at.len() == 20);
        assert_eq!(
            m.file,
            dir.path().join("hub").join("4x-Test").join("4x-Test.onnx")
        );
        assert_eq!(std::fs::read(&m.file).unwrap(), body);
        assert_eq!(leftovers(dir.path()), ["4x-Test.onnx"]);
        assert_eq!(seen.last(), Some(&(10, Some(10))));
        assert_eq!(Installed::load(dir.path()).get("4x-Test"), Some(&m));
    }

    #[test]
    fn card_without_sha_installs_unverified() {
        let c = card("span", vec![resource("onnx", "onnx", None, &[GH])]);
        let dir = tempfile::tempdir().unwrap();
        let http = Fake::with(vec![resp(b"abc", "application/octet-stream")]);
        let m = install(&c, dir.path(), &onnx_caps(), &http, &mut |_, _| {}).unwrap();
        assert!(!m.verified);
        // The hash of what was downloaded is kept even when the card has none to compare with.
        assert_eq!(m.sha256.as_deref(), Some(sha(b"abc").as_str()));
        let json = std::fs::read_to_string(dir.path().join("hub").join("installed.json")).unwrap();
        assert!(json.contains("\"verified\": false"));
    }

    #[test]
    fn sha_mismatch_removes_part_and_fails() {
        let c = card(
            "span",
            vec![resource("onnx", "onnx", Some(sha(b"expected")), &[GH])],
        );
        let dir = tempfile::tempdir().unwrap();
        let http = Fake::with(vec![resp(b"tampered", "application/octet-stream")]);
        let err = install(&c, dir.path(), &onnx_caps(), &http, &mut |_, _| {}).unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, ScaleUpError::Download(_)));
        assert!(
            msg.contains("sha256 mismatch") && msg.contains(&sha(b"expected")),
            "{msg}"
        );
        assert!(
            leftovers(dir.path()).is_empty(),
            "{:?}",
            leftovers(dir.path())
        );
        assert!(Installed::load(dir.path()).models.is_empty());
    }

    #[test]
    fn direct_html_is_not_saved_as_a_model() {
        let c = card("span", vec![resource("onnx", "onnx", None, &[GH])]);
        let dir = tempfile::tempdir().unwrap();
        // Wrong content type on purpose: the first bytes give it away.
        let http = Fake::with(vec![resp(
            b"\n <!DOCTYPE html><html>sign in</html>",
            "application/octet-stream",
        )]);
        let err = install(&c, dir.path(), &onnx_caps(), &http, &mut |_, _| {}).unwrap_err();
        assert!(err.to_string().contains("web page"), "{err}");
        assert!(leftovers(dir.path()).is_empty());
        assert!(Installed::load(dir.path()).models.is_empty());
    }

    #[test]
    fn gdrive_confirm_token_flow() {
        let body = b"big-model".to_vec();
        let c = card(
            "span",
            vec![resource("onnx", "onnx", Some(sha(&body)), &[GD])],
        );
        let dir = tempfile::tempdir().unwrap();
        let mut warning = resp(
            br#"<html><a href="/uc?export=download&amp;id=FILEID123&amp;confirm=t&amp;uuid=u-1_x">Download anyway</a></html>"#,
            "text/html; charset=utf-8",
        );
        warning.cookies = vec![("download_warning_abc".into(), "t".into())];
        let mut file = resp(&body, "application/octet-stream");
        file.content_disposition = Some("attachment; filename=\"Real Name.onnx\"".into());
        let http = Fake::with(vec![warning, file]);
        let m = install(&c, dir.path(), &onnx_caps(), &http, &mut |_, _| {}).unwrap();
        assert!(m.verified);
        assert_eq!(m.file.file_name().unwrap(), "Real Name.onnx");
        assert_eq!(std::fs::read(&m.file).unwrap(), body);
        let reqs = http.requests();
        assert_eq!(reqs.len(), 2);
        assert_eq!(
            reqs[0].0,
            "https://drive.google.com/uc?export=download&id=FILEID123"
        );
        assert!(reqs[1].0.contains("confirm=t"), "{}", reqs[1].0);
        assert!(reqs[1].0.contains("uuid=u-1_x"), "{}", reqs[1].0);
        assert!(reqs[1].0.contains("id=FILEID123"), "{}", reqs[1].0);
        assert_eq!(
            reqs[1].1,
            vec![("download_warning_abc".to_owned(), "t".to_owned())]
        );
    }

    #[test]
    fn gdrive_form_flow() {
        let body = b"form-model".to_vec();
        let c = card("span", vec![resource("onnx", "onnx", None, &[GD])]);
        let dir = tempfile::tempdir().unwrap();
        let page = br#"<form id="download-form" action="https://drive.usercontent.google.com/download" method="get">
<input type="hidden" name="id" value="FILEID123"><input type="hidden" name="export" value="download">
<input type="hidden" name="confirm" value="t"><input type="hidden" name="uuid" value="abc-9"></form>"#;
        let http = Fake::with(vec![
            resp(page, "text/html"),
            resp(&body, "application/octet-stream"),
        ]);
        let m = install(&c, dir.path(), &onnx_caps(), &http, &mut |_, _| {}).unwrap();
        assert_eq!(m.file.file_name().unwrap(), "4x-Test.onnx");
        let url = &http.requests()[1].0;
        assert!(url.starts_with("https://drive.usercontent.google.com/download?"));
        for part in ["id=FILEID123", "export=download", "confirm=t", "uuid=abc-9"] {
            assert!(url.contains(part), "{url}");
        }
    }

    #[test]
    fn gdrive_html_twice_fails_with_manual_hint() {
        let c = card(
            "span",
            vec![resource("onnx", "onnx", Some(sha(b"x")), &[GD])],
        );
        let dir = tempfile::tempdir().unwrap();
        let http = Fake::with(vec![
            resp(b"<html>confirm=t</html>", "text/html"),
            resp(
                b"<!DOCTYPE html><html>quota</html>",
                "application/octet-stream",
            ),
        ]);
        let err = install(&c, dir.path(), &onnx_caps(), &http, &mut |_, _| {}).unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, ScaleUpError::Download(_)));
        assert!(msg.contains("Google Drive did not return a file"), "{msg}");
        assert!(msg.contains(GD), "{msg}");
        assert!(msg.contains("scaleup hub import 4x-Test <file>"), "{msg}");
        assert!(
            msg.contains("4x-Test.onnx") && msg.contains(&sha(b"x")),
            "{msg}"
        );
        assert!(leftovers(dir.path()).is_empty());
        assert!(Installed::load(dir.path()).models.is_empty());
    }

    #[test]
    fn mega_url_is_manual_with_hint_containing_sha() {
        let url = "https://mega.nz/file/abc#key";
        let c = card(
            "span",
            vec![resource("onnx", "onnx", Some(sha(b"m")), &[url])],
        );
        let dir = tempfile::tempdir().unwrap();
        let http = Fake::default();
        let err = install(&c, dir.path(), &onnx_caps(), &http, &mut |_, _| {}).unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, ScaleUpError::Download(_)));
        assert!(msg.contains(url), "{msg}");
        assert!(msg.contains("4x-Test.onnx"), "{msg}");
        assert!(msg.contains(&sha(b"m")), "{msg}");
        assert!(msg.contains("scaleup hub import 4x-Test <file>"), "{msg}");
        assert!(http.requests().is_empty());
    }

    #[test]
    fn empty_resources_fail_with_import_hint() {
        let c = card("span", Vec::new());
        let dir = tempfile::tempdir().unwrap();
        let err = install(
            &c,
            dir.path(),
            &onnx_caps(),
            &Fake::default(),
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(matches!(err, ScaleUpError::Download(_)));
        assert!(
            err.to_string()
                .contains("scaleup hub import 4x-Test <file>")
        );
    }

    #[test]
    fn pick_resource_prefers_onnx_when_not_native() {
        let c = card(
            "span",
            vec![
                resource("pth", "pytorch", None, &[GH]),
                resource("onnx", "onnx", None, &[GH]),
            ],
        );
        let (r, i) = pick_resource(&c, &onnx_caps()).unwrap();
        assert_eq!((r.kind.as_str(), i), ("onnx", 0));
        // Not runnable at all: nothing to pick.
        assert!(pick_resource(&c, &Capabilities::default()).is_none());
    }

    #[test]
    fn pick_resource_prefers_safetensors_then_pth_when_native() {
        let c = card(
            "span",
            vec![
                resource("onnx", "onnx", None, &[GH]),
                resource("pth", "pytorch", None, &[GH]),
                resource("safetensors", "pytorch", None, &[GH]),
            ],
        );
        let caps = Capabilities {
            onnx: true,
            native_archs: vec!["span".into()],
        };
        assert_eq!(pick_resource(&c, &caps).unwrap().0.kind, "safetensors");
        let c = card("span", c.resources[..2].to_vec());
        assert_eq!(pick_resource(&c, &caps).unwrap().0.kind, "pth");
    }

    #[test]
    fn pick_resource_prefers_direct_url_then_drive_then_manual() {
        let mega = "https://mega.nz/x";
        let c = card(
            "span",
            vec![resource("onnx", "onnx", None, &[mega, GD, GH])],
        );
        assert_eq!(pick_resource(&c, &onnx_caps()).unwrap().1, 2);
        let c = card("span", vec![resource("onnx", "onnx", None, &[mega, GD])]);
        assert_eq!(pick_resource(&c, &onnx_caps()).unwrap().1, 1);
        // A downloadable file beats a preferred format that is only on a manual host.
        let c = card(
            "span",
            vec![
                resource("onnx", "onnx", None, &[mega]),
                resource("pth", "pytorch", None, &[GH]),
            ],
        );
        let caps = Capabilities {
            onnx: true,
            native_archs: vec!["span".into()],
        };
        assert_eq!(pick_resource(&c, &caps).unwrap().0.kind, "pth");
    }

    #[test]
    fn classify_url_hosts() {
        assert_eq!(classify_url(GH), Host::Direct);
        assert_eq!(
            classify_url("https://huggingface.co/a/b/resolve/main/x.pth"),
            Host::Direct
        );
        assert_eq!(
            classify_url(GD),
            Host::GoogleDrive {
                file_id: "FILEID123".into()
            }
        );
        for url in [
            "https://drive.google.com/open?id=ABC_-1",
            "https://drive.google.com/uc?export=download&id=ABC_-1",
        ] {
            assert_eq!(
                classify_url(url),
                Host::GoogleDrive {
                    file_id: "ABC_-1".into()
                },
                "{url}"
            );
        }
        for url in [
            "https://drive.google.com/drive/folders/XYZ",
            "https://mega.nz/#!abc",
            "https://evilgithub.com/x",
            "not a url",
        ] {
            assert_eq!(classify_url(url), Host::Manual, "{url}");
        }
    }

    #[test]
    fn import_file_verifies_and_moves() {
        let body = b"hand-downloaded".to_vec();
        let c = card(
            "span",
            vec![resource(
                "onnx",
                "onnx",
                Some(sha(&body)),
                &["https://mega.nz/file/abc"],
            )],
        );
        let dir = tempfile::tempdir().unwrap();
        let downloads = tempfile::tempdir().unwrap();
        let src = downloads.path().join("my download.onnx");
        std::fs::write(&src, &body).unwrap();
        let m = import_file(&c, &src, dir.path()).unwrap();
        assert!(m.verified);
        assert_eq!(m.format, "onnx");
        assert_eq!(m.source_url, "https://mega.nz/file/abc");
        assert_eq!(
            m.file,
            dir.path()
                .join("hub")
                .join("4x-Test")
                .join("my download.onnx")
        );
        assert_eq!(std::fs::read(&m.file).unwrap(), body);
        assert!(!src.exists(), "the file is moved, not copied");
        assert_eq!(Installed::load(dir.path()).get("4x-Test"), Some(&m));
    }

    #[test]
    fn import_file_rejects_wrong_hash_and_keeps_the_file() {
        let c = card(
            "span",
            vec![resource(
                "onnx",
                "onnx",
                Some(sha(b"right")),
                &["https://mega.nz/x"],
            )],
        );
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("wrong.onnx");
        std::fs::write(&src, b"wrong").unwrap();
        let err = import_file(&c, &src, dir.path()).unwrap_err();
        assert!(err.to_string().contains(&sha(b"right")), "{err}");
        assert!(src.exists());
        assert!(Installed::load(dir.path()).models.is_empty());
    }

    #[test]
    fn import_file_without_catalogue_hash_is_unverified() {
        let c = card("span", Vec::new());
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("m.safetensors");
        std::fs::write(&src, b"w").unwrap();
        let m = import_file(&c, &src, dir.path()).unwrap();
        assert!(!m.verified);
        assert_eq!(m.format, "safetensors");
        let bad = dir.path().join("m.zip");
        std::fs::write(&bad, b"w").unwrap();
        assert!(import_file(&c, &bad, dir.path()).is_err());
    }

    #[test]
    fn reinstall_replaces_the_entry() {
        let c = card("span", vec![resource("onnx", "onnx", None, &[GH])]);
        let dir = tempfile::tempdir().unwrap();
        for body in [&b"one"[..], &b"two"[..]] {
            let http = Fake::with(vec![resp(body, "application/octet-stream")]);
            install(&c, dir.path(), &onnx_caps(), &http, &mut |_, _| {}).unwrap();
        }
        let inst = Installed::load(dir.path());
        assert_eq!(inst.models.len(), 1);
        assert_eq!(std::fs::read(&inst.models[0].file).unwrap(), b"two");
    }

    #[test]
    fn unsafe_ids_are_rejected() {
        let mut c = card("span", vec![resource("onnx", "onnx", None, &[GH])]);
        c.id = "../evil".into();
        let dir = tempfile::tempdir().unwrap();
        let err = install(
            &c,
            dir.path(),
            &onnx_caps(),
            &Fake::default(),
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(err.to_string().contains("../evil"), "{err}");
    }

    #[test]
    fn file_names_come_from_url_then_header_then_id() {
        let id = "m";
        assert_eq!(
            file_name("https://x.org/a/b/Model.PTH?download=1", None, id, "pth"),
            "Model.PTH"
        );
        let cd = Some("attachment; filename=\"..\\..\\evil.onnx\"");
        assert_eq!(
            file_name("https://drive.google.com/file/d/Z/view", cd, id, "onnx"),
            "evil.onnx"
        );
        let cd = Some("attachment; filename*=UTF-8''my%20model.pth");
        assert_eq!(file_name("https://x.org/dl", cd, id, "pth"), "my model.pth");
        assert_eq!(file_name("https://x.org/dl", None, id, "onnx"), "m.onnx");
        assert_eq!(
            file_name(
                "https://x.org/dl",
                Some("attachment; filename=\"a.zip\""),
                id,
                "onnx"
            ),
            "m.onnx"
        );
    }

    #[test]
    fn unsafe_file_names_fall_back_to_id_and_stay_in_the_model_folder() {
        for bad in [
            "C:evil.onnx",
            "a:b.onnx",
            "NUL.onnx",
            "con",
            "COM1.onnx",
            "x.onnx.",
            "a*b.onnx",
        ] {
            let c = card(
                "span",
                vec![resource(
                    "onnx",
                    "onnx",
                    None,
                    &["https://github.com/a/b/dl"],
                )],
            );
            let dir = tempfile::tempdir().unwrap();
            let mut file = resp(b"w", "application/octet-stream");
            file.content_disposition = Some(format!("attachment; filename=\"{bad}\""));
            let http = Fake::with(vec![file]);
            let m = install(&c, dir.path(), &onnx_caps(), &http, &mut |_, _| {}).unwrap();
            let expected = dir.path().join("hub").join("4x-Test").join("4x-Test.onnx");
            assert_eq!(m.file, expected, "{bad}");
            assert!(m.file.is_file(), "{bad}");
        }
    }
    #[test]
    fn truncated_body_fails_even_without_a_card_hash() {
        let c = card("span", vec![resource("onnx", "onnx", None, &[GH])]);
        let dir = tempfile::tempdir().unwrap();
        let mut short = resp(b"half", "application/octet-stream");
        short.content_length = Some(100);
        let http = Fake::with(vec![short]);
        let err = install(&c, dir.path(), &onnx_caps(), &http, &mut |_, _| {}).unwrap_err();
        assert!(err.to_string().contains("truncated"), "{err}");
        assert!(leftovers(dir.path()).is_empty());
        assert!(Installed::load(dir.path()).models.is_empty());
    }

    #[test]
    fn body_over_the_size_cap_fails_and_leaves_no_part() {
        let mut r = resource("onnx", "onnx", None, &[GH]);
        r.size = Some(10);
        let c = card("span", vec![r]);
        let dir = tempfile::tempdir().unwrap();
        let mut big = resp(&vec![7u8; 2 * 1024 * 1024], "application/octet-stream");
        big.content_length = None;
        let http = Fake::with(vec![big]);
        let err = install(&c, dir.path(), &onnx_caps(), &http, &mut |_, _| {}).unwrap_err();
        assert!(err.to_string().contains("larger than"), "{err}");
        assert!(leftovers(dir.path()).is_empty());
        assert_eq!(byte_cap(None), 4 * 1024 * 1024 * 1024);
        assert_eq!(byte_cap(Some(1000)), 1100 + 1024 * 1024);
    }
    #[test]
    fn expected_file_name_follows_the_url_then_the_id() {
        assert_eq!(
            expected_file_name("https://mega.nz/file/x/My.pth", "id", "pth"),
            "My.pth"
        );
        assert_eq!(
            expected_file_name("https://mega.nz/file/abc#key", "id", "onnx"),
            "id.onnx"
        );
    }

    #[test]
    fn uninstall_deletes_the_folder_and_the_entry() {
        let c = card("span", vec![resource("onnx", "onnx", None, &[GH])]);
        let dir = tempfile::tempdir().unwrap();
        let http = Fake::with(vec![resp(b"w", "application/octet-stream")]);
        install(&c, dir.path(), &onnx_caps(), &http, &mut |_, _| {}).unwrap();
        uninstall(dir.path(), "4x-Test").unwrap();
        assert!(!dir.path().join("hub").join("4x-Test").exists());
        assert!(Installed::load(dir.path()).models.is_empty());
        let err = uninstall(dir.path(), "4x-Test").unwrap_err();
        assert!(err.to_string().contains("not an installed"), "{err}");
    }

    #[test]
    fn pytorch_only_card_points_at_convert() {
        let c = card("dat", vec![resource("pth", "pytorch", None, &[GH])]);
        let dir = tempfile::tempdir().unwrap();
        let err = install(
            &c,
            dir.path(),
            &onnx_caps(),
            &Fake::default(),
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("`scaleup hub install 4x-Test --convert`"),
            "{err}"
        );
    }

    #[test]
    fn rfc3339_known_dates() {
        assert_eq!(rfc3339_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_utc(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339_utc(1_700_000_000), "2023-11-14T22:13:20Z");
        assert_eq!(rfc3339_utc(4_102_444_799), "2099-12-31T23:59:59Z");
    }

    #[test]
    fn ureq_client_refuses_plain_http() {
        let err = UreqHttp
            .get("http://github.com/a/b.onnx", &[])
            .err()
            .expect("plain http is refused");
        assert!(err.to_string().contains("https"), "{err}");
    }
}
