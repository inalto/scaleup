//! Model catalogue (`models/manifest.json`, embedded) and verified model downloads.

use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::pipeline::CancelToken;
use crate::{Result, ScaleUpError};

const BUILTIN_MANIFEST: &str = include_str!("../../../models/manifest.json");

fn default_format() -> String {
    "onnx".into()
}

/// One model in the catalogue.
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, Eq)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub arch: String,
    pub scale: u32,
    pub license: String,
    /// Weights may be used commercially.
    pub commercial: bool,
    pub size_bytes: u64,
    pub url: String,
    /// Hex SHA-256 of the file at `url` (case-insensitive).
    pub sha256: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub notes: String,
    /// File format; only `"onnx"` in v1.
    #[serde(default = "default_format")]
    pub format: String,
}

#[derive(Deserialize)]
struct Manifest {
    models: Vec<ModelEntry>,
}

/// Parses a manifest document (`{"version": 1, "models": [...]}`).
pub fn parse_manifest(json: &str) -> Result<Vec<ModelEntry>> {
    serde_json::from_str::<Manifest>(json)
        .map(|m| m.models)
        .map_err(|e| ScaleUpError::Model(format!("invalid model manifest: {e}")))
}

/// A model ready to load: a local file, plus its scale and catalogue entry when known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedModel {
    pub path: PathBuf,
    /// From the manifest; `None` for a bare file path (the backend then probes the model).
    pub scale: Option<u32>,
    pub entry: Option<ModelEntry>,
}

/// The model catalogue plus the directory downloaded models live in.
#[derive(Clone, Debug)]
pub struct Registry {
    entries: Vec<ModelEntry>,
    models_dir: PathBuf,
    timeout: Duration,
}

/// Connect and first-response timeout for downloads.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// `SCALEUP_MODELS_DIR`, else `<data_dir>/scaleup/models`, else `./models`.
fn default_models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("SCALEUP_MODELS_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(dir);
    }
    dirs::data_dir()
        .map(|d| d.join("scaleup").join("models"))
        .unwrap_or_else(|| PathBuf::from("models"))
}

impl Registry {
    /// Embedded catalogue, default models directory.
    pub fn builtin() -> Self {
        Self::with_dir(default_models_dir())
    }

    /// Embedded catalogue, custom models directory.
    pub fn with_dir(dir: PathBuf) -> Self {
        let entries = parse_manifest(BUILTIN_MANIFEST).expect("embedded manifest is valid");
        Self::from_entries(entries, dir)
    }

    /// Arbitrary catalogue (tests, user manifests).
    pub fn from_entries(entries: Vec<ModelEntry>, models_dir: PathBuf) -> Self {
        Self {
            entries,
            models_dir,
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Download connect / response timeout (default 30 s).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn entries(&self) -> &[ModelEntry] {
        &self.entries
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub fn get(&self, id: &str) -> Option<&ModelEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// `<models_dir>/<id>.onnx`.
    pub fn local_path(&self, id: &str) -> PathBuf {
        self.models_dir.join(format!("{id}.onnx"))
    }

    pub fn is_downloaded(&self, id: &str) -> bool {
        self.local_path(id).is_file()
    }

    /// Local path of catalogue model `id`, downloading and verifying it first if missing.
    /// `progress(done, total)` reports bytes; `total` is 0 when unknown. A failed or corrupt
    /// download leaves no file behind.
    pub fn ensure(&self, id: &str, progress: &mut dyn FnMut(u64, u64)) -> Result<PathBuf> {
        self.ensure_cancellable(id, progress, &CancelToken::new())
    }

    /// Like [`Registry::ensure`], but a cancelled `cancel` aborts the download with
    /// [`ScaleUpError::Cancelled`] and removes the partial file.
    pub fn ensure_cancellable(
        &self,
        id: &str,
        progress: &mut dyn FnMut(u64, u64),
        cancel: &CancelToken,
    ) -> Result<PathBuf> {
        let entry = self
            .get(id)
            .ok_or_else(|| ScaleUpError::Model(format!("unknown model '{id}'")))?;
        let path = self.local_path(id);
        if path.is_file() {
            return Ok(path);
        }
        if entry.url.is_empty() {
            return Err(ScaleUpError::Download(format!(
                "model '{id}' has no download URL yet"
            )));
        }
        fs::create_dir_all(&self.models_dir)?;
        let part = self.models_dir.join(format!("{id}.onnx.part"));
        let result = download_verified(
            &entry.url,
            &entry.sha256,
            entry.size_bytes,
            self.timeout,
            &part,
            progress,
            cancel,
        )
        .and_then(|()| fs::rename(&part, &path).map_err(ScaleUpError::from));
        if result.is_err() {
            let _ = fs::remove_file(&part);
        }
        result.map(|()| path)
    }

    /// A catalogue id (downloaded on demand) or a path to an existing model file.
    pub fn resolve(&self, model: &str) -> Result<ResolvedModel> {
        if let Some(entry) = self.get(model) {
            let path = self.ensure(model, &mut |_, _| {})?;
            return Ok(ResolvedModel {
                path,
                scale: Some(entry.scale),
                entry: Some(entry.clone()),
            });
        }
        let path = PathBuf::from(model);
        if path.is_file() {
            return Ok(ResolvedModel {
                path,
                scale: None,
                entry: None,
            });
        }
        Err(ScaleUpError::Model(format!(
            "'{model}' is neither a known model id nor an existing file"
        )))
    }
}

fn download_verified(
    url: &str,
    expected_sha: &str,
    size_hint: u64,
    timeout: Duration,
    dest: &Path,
    progress: &mut dyn FnMut(u64, u64),
    cancel: &CancelToken,
) -> Result<()> {
    let dl = |e: &dyn std::fmt::Display| ScaleUpError::Download(format!("{url}: {e}"));
    // No total body timeout: large models on slow links legitimately take minutes.
    let agent = ureq::Agent::config_builder()
        .timeout_connect(Some(timeout))
        .timeout_recv_response(Some(timeout))
        .build()
        .new_agent();
    let mut response = agent.get(url).call().map_err(|e| dl(&e))?;
    let total = response.body().content_length().unwrap_or(size_hint);
    let mut reader = response.body_mut().as_reader();
    let mut out = BufWriter::new(File::create(dest)?);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    let mut done = 0u64;
    loop {
        if cancel.is_cancelled() {
            return Err(ScaleUpError::Cancelled);
        }
        let n = match reader.read(&mut buf) {
            Ok(n) => n,
            Err(_) if cancel.is_cancelled() => return Err(ScaleUpError::Cancelled),
            Err(e) => return Err(dl(&e)),
        };
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        out.write_all(&buf[..n])?;
        done += n as u64;
        progress(done, total);
    }
    out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    let actual = format!("{:x}", hasher.finalize());
    if !actual.eq_ignore_ascii_case(expected_sha.trim()) {
        return Err(dl(&format!(
            "sha256 mismatch: expected {expected_sha}, got {actual}"
        )));
    }
    Ok(())
}

/// Upscayl's rule: `x2`/`2x` → 2, `x3`/`3x` → 3, anything else → 4 (case-insensitive).
pub fn scale_from_name(name: &str) -> u32 {
    let n = name.to_ascii_lowercase();
    if n.contains("x2") || n.contains("2x") {
        2
    } else if n.contains("x3") || n.contains("3x") {
        3
    } else {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScaleUpError;
    use sha2::{Digest, Sha256};
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// Serves `body` with HTTP 200 to `count` connections on a loopback port; returns the base URL.
    fn serve(body: &'static [u8], count: usize) -> String {
        serve_claiming(body, body.len(), count)
    }

    /// Like `serve` but advertises `claimed_len` as Content-Length (to simulate truncation).
    fn serve_claiming(body: &'static [u8], claimed_len: usize, count: usize) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming().take(count) {
                let mut stream = stream.unwrap();
                let mut buf = Vec::new();
                let mut byte = [0u8; 1];
                while !buf.ends_with(b"\r\n\r\n") && stream.read(&mut byte).unwrap() == 1 {
                    buf.push(byte[0]);
                }
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    claimed_len
                );
                stream.write_all(head.as_bytes()).unwrap();
                stream.write_all(body).unwrap();
            }
        });
        format!("http://{addr}")
    }

    fn entry(id: &str, url: String, sha256: String) -> ModelEntry {
        ModelEntry {
            id: id.into(),
            name: id.into(),
            arch: "test".into(),
            scale: 2,
            license: "MIT".into(),
            commercial: true,
            size_bytes: 0,
            url,
            sha256,
            tags: vec![],
            notes: String::new(),
            format: "onnx".into(),
        }
    }

    #[test]
    fn manifest_parses_and_has_default_tag() {
        let reg = Registry::builtin();
        assert_eq!(reg.entries().len(), 8);
        let defaults: Vec<_> = reg
            .entries()
            .iter()
            .filter(|e| e.tags.iter().any(|t| t == "default"))
            .collect();
        assert_eq!(defaults.len(), 1);
        assert_eq!(defaults[0].id, "realesr-general-x4v3");
        assert!(
            reg.entries()
                .iter()
                .all(|e| e.format == "onnx"
                    && e.commercial != e.tags.iter().any(|t| t == "noncommercial"))
        );
        assert_eq!(reg.get("realesrgan-x2plus").unwrap().scale, 2);
        assert!(reg.get("nope").is_none());
    }

    #[test]
    fn local_path_layout() {
        let reg = Registry::with_dir(PathBuf::from("models-dir"));
        assert_eq!(
            reg.local_path("abc"),
            PathBuf::from("models-dir").join("abc.onnx")
        );
        assert!(!reg.is_downloaded("abc"));
    }

    #[test]
    fn scale_from_name_cases() {
        assert_eq!(scale_from_name("RealESRGAN_x2plus"), 2);
        assert_eq!(scale_from_name("4xNomos"), 4);
        assert_eq!(scale_from_name("foo-3x"), 3);
        assert_eq!(scale_from_name("model_X3.onnx"), 3);
        assert_eq!(scale_from_name("2X-thing"), 2);
        assert_eq!(scale_from_name("plain"), 4);
    }

    #[test]
    fn resolve_path_passthrough() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("custom.onnx");
        std::fs::write(&file, b"x").unwrap();
        let reg = Registry::with_dir(dir.path().join("models"));
        let resolved = reg.resolve(file.to_str().unwrap()).unwrap();
        assert_eq!(resolved.path, file);
        assert!(resolved.entry.is_none());
        assert!(matches!(
            reg.resolve("no-such-model"),
            Err(ScaleUpError::Model(_))
        ));
    }

    #[test]
    fn download_verifies_sha() {
        const BODY: &[u8] = b"pretend this is an onnx model";
        let base = serve(BODY, 2);
        let good_sha = format!("{:x}", Sha256::digest(BODY));
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::from_entries(
            vec![
                entry("good", format!("{base}/good.onnx"), good_sha.to_uppercase()),
                entry("bad", format!("{base}/bad.onnx"), "00".repeat(32)),
            ],
            dir.path().join("models"),
        );

        let mut last = (0, 0);
        let path = reg.ensure("good", &mut |d, t| last = (d, t)).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), BODY);
        assert_eq!(last, (BODY.len() as u64, BODY.len() as u64));
        assert!(reg.is_downloaded("good"));
        let resolved = reg.resolve("good").unwrap();
        assert_eq!((resolved.path, resolved.scale), (path, Some(2)));

        let err = reg.ensure("bad", &mut |_, _| {});
        assert!(matches!(err, Err(ScaleUpError::Download(_))), "{err:?}");
        assert!(!reg.local_path("bad").exists());
        assert!(!dir.path().join("models").join("bad.onnx.part").exists());
    }

    #[test]
    fn cancelled_download_is_removed() {
        const BODY: &[u8] = b"pretend this is an onnx model";
        let base = serve(BODY, 1);
        let dir = tempfile::tempdir().unwrap();
        let sha = format!("{:x}", Sha256::digest(BODY));
        let reg = Registry::from_entries(
            vec![entry("c", format!("{base}/c.onnx"), sha)],
            dir.path().to_path_buf(),
        );
        let cancel = CancelToken::new();
        cancel.cancel();
        let err = reg.ensure_cancellable("c", &mut |_, _| {}, &cancel);
        assert!(matches!(err, Err(ScaleUpError::Cancelled)), "{err:?}");
        assert!(!reg.local_path("c").exists());
        assert!(!dir.path().join("c.onnx.part").exists());
    }

    #[test]
    fn truncated_download_is_removed() {
        const BODY: &[u8] = b"short";
        let base = serve_claiming(BODY, 1000, 1);
        let dir = tempfile::tempdir().unwrap();
        let sha = format!("{:x}", Sha256::digest(BODY));
        let reg = Registry::from_entries(
            vec![entry("t", format!("{base}/t.onnx"), sha)],
            dir.path().to_path_buf(),
        );
        let err = reg.ensure("t", &mut |_, _| {});
        assert!(matches!(err, Err(ScaleUpError::Download(_))), "{err:?}");
        assert!(!reg.local_path("t").exists());
        assert!(!dir.path().join("t.onnx.part").exists());
    }

    #[test]
    fn unresponsive_server_times_out() {
        // Accepts the connection, then never answers.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let held: Vec<_> = listener.incoming().take(1).collect();
            std::thread::sleep(std::time::Duration::from_secs(30));
            drop(held);
        });
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path().to_path_buf();
        let reg = Registry::from_entries(
            vec![entry(
                "slow",
                format!("http://{addr}/slow.onnx"),
                "00".into(),
            )],
            models_dir.clone(),
        )
        .with_timeout(std::time::Duration::from_millis(300));
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || tx.send(reg.ensure("slow", &mut |_, _| {})).unwrap());
        let res = rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("download hung instead of timing out");
        assert!(matches!(res, Err(ScaleUpError::Download(_))), "{res:?}");
        assert!(!models_dir.join("slow.onnx.part").exists());
    }

    #[test]
    fn download_without_url_fails_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::from_entries(
            vec![entry("e", String::new(), String::new())],
            dir.path().to_path_buf(),
        );
        assert!(matches!(
            reg.ensure("e", &mut |_, _| {}),
            Err(ScaleUpError::Download(_))
        ));
        assert!(matches!(
            reg.ensure("unknown", &mut |_, _| {}),
            Err(ScaleUpError::Model(_))
        ));
    }
}
