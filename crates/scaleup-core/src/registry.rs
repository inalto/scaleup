//! Model catalogue (`models/manifest.json`, embedded) and verified model downloads.

use crate::backend::is_ncnn_model;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::hub::{Installed, classify_license, commercial_ok};
use crate::pipeline::CancelToken;
use crate::{Result, ScaleUpError};

const BUILTIN_MANIFEST: &str = include_str!("../../../models/manifest.json");

/// Lowercase hexadecimal spelling of a digest, as written in manifests and `SHA256SUMS` files.
pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// Sha256 of everything `reader` yields, in lowercase hexadecimal.
pub(crate) fn sha256_of(reader: &mut impl Read) -> std::io::Result<String> {
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            return Ok(hex(&hasher.finalize()));
        }
        hasher.update(&buf[..n]);
    }
}

fn default_format() -> String {
    "onnx".into()
}

fn default_source() -> String {
    "catalogue".into()
}

const NCNN_FORMAT: &str = "ncnn";
/// `ModelEntry::source` of a model installed from OpenModelDB.
pub const SOURCE_HUB: &str = "openmodeldb";
/// Models imported by the user, in the models folder, in manifest format.
const LOCAL_OVERLAY: &str = "local.json";

/// The entries of `<dir>/local.json`. A missing file is normal; an unreadable one is not fatal.
fn read_overlay(dir: &Path) -> Vec<ModelEntry> {
    let Ok(text) = fs::read_to_string(dir.join(LOCAL_OVERLAY)) else {
        return Vec::new();
    };
    parse_manifest(&text).unwrap_or_else(|e| {
        tracing::warn!("ignoring {}: {e}", dir.join(LOCAL_OVERLAY).display());
        Vec::new()
    })
}

/// What a catalogue model does.
#[derive(Deserialize, Serialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    /// Super-resolution: the models of the upscale pickers.
    #[default]
    Upscale,
    /// Inpainting, for canvas extension (`--extend-model`); never offered as an upscaler.
    Inpaint,
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
    /// File format: `"onnx"`, `"ncnn"` for `.param`/`.bin` pairs, or `"pth"` / `"safetensors"`.
    #[serde(default = "default_format")]
    pub format: String,
    /// Where the entry comes from: `"catalogue"` (built in or imported) or `"openmodeldb"`.
    #[serde(default = "default_source")]
    pub source: String,
    /// Upscaler (the default when the manifest does not say) or inpainter.
    #[serde(default)]
    pub kind: ModelKind,
}

impl ModelEntry {
    /// Tile overlap this architecture needs, when it has been measured. See `scaleup-wgpu/BENCH.md`.
    pub fn preferred_overlap(&self) -> Option<u32> {
        // Catalogue models name the PyTorch class, OpenModelDB models the architecture.
        match self.arch.to_ascii_lowercase().as_str() {
            "span" => Some(16),
            "srvggnetcompact" | "compact" => Some(32),
            "rrdbnet" | "esrgan" => Some(64),
            // Not measured: 17x17 kernels give a receptive radius above 100 pixels, and the wgpu
            // backend asks for 64 from the graph of such a model.
            "realplksr" => Some(64),
            _ => None,
        }
    }
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
    /// File of each installed OpenModelDB model, by id.
    hub_files: HashMap<String, PathBuf>,
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
        let mut entries = parse_manifest(BUILTIN_MANIFEST).expect("embedded manifest is valid");
        for local in read_overlay(&dir) {
            if !entries.iter().any(|e| e.id == local.id) {
                entries.push(local);
            }
        }
        let mut reg = Self::from_entries(entries, dir);
        reg.add_hub_models();
        reg
    }

    /// Arbitrary catalogue (tests, user manifests).
    pub fn from_entries(entries: Vec<ModelEntry>, models_dir: PathBuf) -> Self {
        Self {
            entries,
            models_dir,
            timeout: DEFAULT_TIMEOUT,
            hub_files: HashMap::new(),
        }
    }

    /// Lists the models of `<models_dir>/hub/installed.json`. A built-in or imported id wins over
    /// a hub model of the same name.
    fn add_hub_models(&mut self) {
        for m in Installed::load(&self.models_dir).models {
            if self.entries.iter().any(|e| e.id == m.id) {
                continue;
            }
            let class = classify_license(m.license.as_deref());
            self.entries.push(ModelEntry {
                id: m.id.clone(),
                name: m.name,
                arch: m.arch,
                scale: m.scale,
                license: m.license.unwrap_or_else(|| "unknown".into()),
                commercial: commercial_ok(class),
                size_bytes: fs::metadata(&m.file).map_or(0, |md| md.len()),
                url: m.source_url,
                sha256: m.sha256.unwrap_or_default(),
                tags: Vec::new(),
                notes: if m.verified {
                    "installed from OpenModelDB".into()
                } else {
                    "installed from OpenModelDB; the file could not be verified".into()
                },
                format: m.format,
                source: SOURCE_HUB.into(),
                kind: ModelKind::Upscale,
            });
            self.hub_files.insert(m.id, m.file);
        }
    }

    /// Download connect / response timeout (default 30 s).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Copies every `.param`/`.bin` pair in `dir` into the models folder and lists them in the local
    /// overlay. Returns the imported ids. The license of an imported model is not known here, so it
    /// is marked non-commercial until its owner says otherwise.
    pub fn import_ncnn(&mut self, dir: &Path) -> Result<Vec<String>> {
        fs::create_dir_all(&self.models_dir)?;
        let mut files: Vec<PathBuf> = fs::read_dir(dir)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| is_ncnn_model(p) && p.with_extension("bin").is_file())
            .collect();
        files.sort();
        let mut overlay = read_overlay(&self.models_dir);
        let mut ids = Vec::new();
        for param in files {
            let Some(id) = param.file_stem().map(|s| s.to_string_lossy().into_owned()) else {
                continue;
            };
            let builtin = self.get(&id).is_some_and(|e| e.format != NCNN_FORMAT);
            if builtin {
                continue;
            }
            let bin = param.with_extension("bin");
            fs::copy(&param, self.models_dir.join(format!("{id}.param")))?;
            fs::copy(&bin, self.models_dir.join(format!("{id}.bin")))?;
            let entry = ModelEntry {
                id: id.clone(),
                name: id.clone(),
                arch: "ncnn".into(),
                scale: scale_from_name(&id),
                license: "unknown".into(),
                commercial: false,
                size_bytes: fs::metadata(&bin)?.len(),
                url: String::new(),
                sha256: String::new(),
                tags: Vec::new(),
                notes: format!("imported from {}", dir.display()),
                format: NCNN_FORMAT.into(),
                source: default_source(),
                kind: ModelKind::Upscale,
            };
            overlay.retain(|e| e.id != id);
            overlay.push(entry.clone());
            self.entries.retain(|e| e.id != id);
            self.entries.push(entry);
            ids.push(id);
        }
        let json = serde_json::json!({ "version": 1, "models": overlay });
        fs::write(self.models_dir.join(LOCAL_OVERLAY), json.to_string())?;
        Ok(ids)
    }

    /// Lists one ONNX file in the local overlay as model `<file stem>`, copying it to
    /// `<models_dir>/<id>.onnx` unless it is already there. Like an ncnn import, its license is
    /// unknown, so it is marked non-commercial; the scale comes from the name. Fails when the id
    /// belongs to a catalogue or hub model; an earlier import of the same id is replaced.
    pub fn import_onnx(&mut self, file: &Path) -> Result<String> {
        let id = file
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ScaleUpError::Model(format!("{}: not a file name", file.display())))?;
        if self.get(&id).is_some() && !self.is_imported(&id) {
            return Err(ScaleUpError::InvalidOption(format!(
                "'{id}' is already the id of another model; rename {} and try again",
                file.display()
            )));
        }
        fs::create_dir_all(&self.models_dir)?;
        let dest = self.models_dir.join(format!("{id}.onnx"));
        let same = matches!(
            (fs::canonicalize(file), fs::canonicalize(&dest)),
            (Ok(a), Ok(b)) if a == b
        );
        if !same {
            fs::copy(file, &dest)?;
        }
        let entry = ModelEntry {
            id: id.clone(),
            name: id.clone(),
            arch: "onnx".into(),
            scale: scale_from_name(&id),
            license: "unknown".into(),
            commercial: false,
            size_bytes: fs::metadata(&dest)?.len(),
            url: String::new(),
            sha256: String::new(),
            tags: Vec::new(),
            notes: format!("imported from {}", file.display()),
            format: default_format(),
            source: default_source(),
            kind: ModelKind::Upscale,
        };
        let mut overlay = read_overlay(&self.models_dir);
        overlay.retain(|e| e.id != id);
        overlay.push(entry.clone());
        self.entries.retain(|e| e.id != id);
        self.entries.push(entry);
        let json = serde_json::json!({ "version": 1, "models": overlay });
        fs::write(self.models_dir.join(LOCAL_OVERLAY), json.to_string())?;
        Ok(id)
    }

    /// Whether `id` was imported by the user (listed in `<models_dir>/local.json`).
    pub fn is_imported(&self, id: &str) -> bool {
        read_overlay(&self.models_dir).iter().any(|e| e.id == id)
    }

    /// Removes the local files of catalogue model `id`.
    pub fn delete_local(&self, id: &str) -> Result<()> {
        let path = self.local_path(id);
        fs::remove_file(&path)?;
        if is_ncnn_model(&path) {
            match fs::remove_file(path.with_extension("bin")) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
                _ => {}
            }
        }
        Ok(())
    }

    pub fn entries(&self) -> &[ModelEntry] {
        &self.entries
    }

    /// The upscaling models: what every upscale-model listing shows.
    pub fn upscalers(&self) -> impl Iterator<Item = &ModelEntry> {
        self.entries.iter().filter(|e| e.kind == ModelKind::Upscale)
    }

    /// The inpainting models, for canvas extension.
    pub fn inpainters(&self) -> impl Iterator<Item = &ModelEntry> {
        self.entries.iter().filter(|e| e.kind == ModelKind::Inpaint)
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub fn get(&self, id: &str) -> Option<&ModelEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// `<models_dir>/<id>.onnx`, or `<id>.param` for an ncnn model (its `.bin` sits next to it);
    /// the installed file for an OpenModelDB model.
    pub fn local_path(&self, id: &str) -> PathBuf {
        if let Some(file) = self.hub_files.get(id) {
            return file.clone();
        }
        let ext = match self.get(id) {
            Some(e) if e.format == NCNN_FORMAT => "param",
            _ => "onnx",
        };
        self.models_dir.join(format!("{id}.{ext}"))
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
        if entry.source == SOURCE_HUB {
            return Err(ScaleUpError::Model(format!(
                "model '{id}' is installed from OpenModelDB but {} is missing; reinstall it with `scaleup hub install {id}`",
                path.display()
            )));
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
    let actual = crate::registry::hex(&hasher.finalize());
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

    fn write_pair(dir: &Path, name: &str) {
        std::fs::write(dir.join(format!("{name}.param")), b"7767517").unwrap();
        std::fs::write(dir.join(format!("{name}.bin")), b"weights").unwrap();
    }

    #[test]
    fn resolve_param_path() {
        let d = tempfile::tempdir().unwrap();
        write_pair(d.path(), "m");
        let reg = Registry::with_dir(d.path().join("models"));
        let path = d.path().join("m.param");
        let resolved = reg.resolve(path.to_str().unwrap()).unwrap();
        assert_eq!(resolved.path, path);
        assert_eq!(resolved.entry, None);
    }

    #[test]
    fn local_overlay_lists_imported_models() {
        let src = tempfile::tempdir().unwrap();
        write_pair(src.path(), "my-model-2x");
        write_pair(src.path(), "other-4x");
        std::fs::write(src.path().join("lonely.param"), b"no bin next to me").unwrap();
        let models = tempfile::tempdir().unwrap();
        let mut reg = Registry::with_dir(models.path().to_path_buf());
        let before = reg.entries().len();
        let mut ids = reg.import_ncnn(src.path()).unwrap();
        ids.sort();
        assert_eq!(ids, ["my-model-2x", "other-4x"]);
        assert_eq!(reg.entries().len(), before + 2);

        // A fresh registry sees them too, without any download.
        let reg = Registry::with_dir(models.path().to_path_buf());
        let e = reg.get("my-model-2x").expect("listed from the overlay");
        assert_eq!(e.format, "ncnn");
        assert_eq!(e.scale, 2);
        assert!(
            !e.commercial,
            "license unknown means not free for commercial use"
        );
        assert_eq!(
            reg.local_path("my-model-2x"),
            models.path().join("my-model-2x.param")
        );
        assert!(reg.is_downloaded("my-model-2x"));
        assert!(models.path().join("my-model-2x.bin").is_file());
        assert_eq!(
            reg.ensure("my-model-2x", &mut |_, _| {}).unwrap(),
            models.path().join("my-model-2x.param")
        );
    }

    #[test]
    fn import_twice_does_not_duplicate_and_keeps_builtin_ids() {
        let src = tempfile::tempdir().unwrap();
        write_pair(src.path(), "m");
        write_pair(src.path(), "realesr-general-x4v3");
        let models = tempfile::tempdir().unwrap();
        let mut reg = Registry::with_dir(models.path().to_path_buf());
        let ids = reg.import_ncnn(src.path()).unwrap();
        assert_eq!(ids, ["m"], "a catalogue id is not replaced");
        let count = reg.entries().len();
        reg.import_ncnn(src.path()).unwrap();
        assert_eq!(reg.entries().len(), count);
        let reg = Registry::with_dir(models.path().to_path_buf());
        assert_eq!(reg.entries().len(), count);
    }

    #[test]
    fn import_onnx_lists_one_file_as_a_local_model() {
        let src = tempfile::tempdir().unwrap();
        let file = src.path().join("conv-2x.onnx");
        std::fs::write(&file, b"onnx bytes").unwrap();
        let models = tempfile::tempdir().unwrap();
        let mut reg = Registry::with_dir(models.path().to_path_buf());
        assert!(!reg.is_imported("conv-2x"));
        assert_eq!(reg.import_onnx(&file).unwrap(), "conv-2x");
        let reg = Registry::with_dir(models.path().to_path_buf());
        assert!(reg.is_imported("conv-2x"));
        assert!(!reg.is_imported("realesr-general-x4v3"));
        let e = reg.get("conv-2x").expect("listed from the overlay");
        assert_eq!(
            (e.format.as_str(), e.scale, e.license.as_str(), e.commercial),
            ("onnx", 2, "unknown", false)
        );
        assert_eq!(
            reg.local_path("conv-2x"),
            models.path().join("conv-2x.onnx")
        );
        assert_eq!(
            std::fs::read(reg.local_path("conv-2x")).unwrap(),
            b"onnx bytes"
        );

        // A file already in place is registered without a copy; importing again replaces.
        let mut reg = Registry::with_dir(models.path().to_path_buf());
        let count = reg.entries().len();
        reg.import_onnx(&models.path().join("conv-2x.onnx"))
            .unwrap();
        assert_eq!(reg.entries().len(), count);
        assert_eq!(
            Registry::with_dir(models.path().to_path_buf())
                .entries()
                .len(),
            count
        );

        // A catalogue id is not taken over.
        let builtin = src.path().join("realesr-general-x4v3.onnx");
        std::fs::write(&builtin, b"x").unwrap();
        let err = reg.import_onnx(&builtin).unwrap_err();
        assert!(matches!(err, ScaleUpError::InvalidOption(_)), "{err:?}");
        assert!(!models.path().join("realesr-general-x4v3.onnx").exists());
    }

    #[test]
    fn delete_local_removes_both_ncnn_files() {
        let src = tempfile::tempdir().unwrap();
        write_pair(src.path(), "m");
        let models = tempfile::tempdir().unwrap();
        let mut reg = Registry::with_dir(models.path().to_path_buf());
        reg.import_ncnn(src.path()).unwrap();
        reg.delete_local("m").unwrap();
        assert!(!reg.is_downloaded("m"));
        assert!(!models.path().join("m.bin").exists());
    }

    #[test]
    fn corrupt_overlay_is_ignored() {
        let models = tempfile::tempdir().unwrap();
        std::fs::write(models.path().join("local.json"), b"{nope").unwrap();
        let reg = Registry::with_dir(models.path().to_path_buf());
        assert!(!reg.entries().is_empty());
    }
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
            source: default_source(),
            kind: ModelKind::Upscale,
        }
    }

    #[test]
    fn measured_architectures_name_their_overlap() {
        let arch = |a: &str| ModelEntry {
            arch: a.into(),
            ..entry("m", String::new(), String::new())
        };
        assert_eq!(arch("SRVGGNetCompact").preferred_overlap(), Some(32));
        assert_eq!(arch("SPAN").preferred_overlap(), Some(16));
        assert_eq!(arch("RRDBNet").preferred_overlap(), Some(64));
        assert_eq!(arch("ncnn").preferred_overlap(), None);
        // OpenModelDB names.
        assert_eq!(arch("esrgan").preferred_overlap(), Some(64));
        assert_eq!(arch("compact").preferred_overlap(), Some(32));
        assert_eq!(arch("span").preferred_overlap(), Some(16));
        assert_eq!(arch("dat").preferred_overlap(), None);
        // From the receptive radius of its 17x17 kernels, as the wgpu graph gives it.
        assert_eq!(arch("realplksr").preferred_overlap(), Some(64));
        assert_eq!(arch("RealPLKSR").preferred_overlap(), Some(64));
    }

    fn installed_model(
        id: &str,
        file: PathBuf,
        license: Option<&str>,
    ) -> crate::hub::InstalledModel {
        crate::hub::InstalledModel {
            id: id.into(),
            name: format!("{id} name"),
            arch: "compact".into(),
            scale: 2,
            license: license.map(str::to_owned),
            format: "onnx".into(),
            file,
            sha256: Some("ab".repeat(32)),
            verified: true,
            source_url: "https://github.com/a/b/m.onnx".into(),
            installed_at: "2026-11-05T10:00:00Z".into(),
        }
    }

    fn write_installed(dir: &Path, models: Vec<crate::hub::InstalledModel>) {
        Installed { models }.save(dir).unwrap();
    }

    #[test]
    fn hub_models_are_listed_and_resolve_without_download() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("hub").join("h-2x").join("weights.onnx");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"onnx").unwrap();
        write_installed(
            dir.path(),
            vec![
                installed_model("h-2x", file.clone(), Some("MIT")),
                installed_model("h-nc", dir.path().join("gone.onnx"), Some("CC-BY-NC-4.0")),
                installed_model("h-unk", dir.path().join("gone2.onnx"), None),
            ],
        );
        let reg = Registry::with_dir(dir.path().to_path_buf());
        let e = reg.get("h-2x").expect("listed");
        assert_eq!(e.source, "openmodeldb");
        assert_eq!(
            (e.format.as_str(), e.scale, e.license.as_str()),
            ("onnx", 2, "MIT")
        );
        assert!(e.commercial);
        assert_eq!(e.preferred_overlap(), Some(32));
        assert_eq!(reg.local_path("h-2x"), file);
        assert!(reg.is_downloaded("h-2x"));
        assert_eq!(reg.ensure("h-2x", &mut |_, _| {}).unwrap(), file);
        let resolved = reg.resolve("h-2x").unwrap();
        assert_eq!((resolved.path, resolved.scale), (file, Some(2)));
        assert!(!reg.get("h-nc").unwrap().commercial);
        let unk = reg.get("h-unk").unwrap();
        assert_eq!((unk.license.as_str(), unk.commercial), ("unknown", false));
        let empty = tempfile::tempdir().unwrap();
        assert!(
            Registry::with_dir(empty.path().to_path_buf())
                .entries()
                .iter()
                .all(|e| e.source == "catalogue")
        );
    }

    #[test]
    fn hub_model_with_a_missing_file_asks_for_a_reinstall() {
        let dir = tempfile::tempdir().unwrap();
        write_installed(
            dir.path(),
            vec![installed_model(
                "h-2x",
                dir.path().join("gone.onnx"),
                Some("MIT"),
            )],
        );
        let reg = Registry::with_dir(dir.path().to_path_buf());
        assert!(reg.get("h-2x").is_some());
        assert!(!reg.is_downloaded("h-2x"));
        let err = reg.resolve("h-2x").unwrap_err();
        assert!(matches!(err, ScaleUpError::Model(_)), "{err:?}");
        assert!(err.to_string().contains("reinstall"), "{err}");
    }

    #[test]
    fn builtin_and_imported_ids_win_over_hub_ids() {
        let dir = tempfile::tempdir().unwrap();
        let src = tempfile::tempdir().unwrap();
        write_pair(src.path(), "mine-2x");
        let mut reg = Registry::with_dir(dir.path().to_path_buf());
        reg.import_ncnn(src.path()).unwrap();
        write_installed(
            dir.path(),
            vec![
                installed_model(
                    "realesr-general-x4v3",
                    dir.path().join("a.onnx"),
                    Some("MIT"),
                ),
                installed_model("mine-2x", dir.path().join("b.onnx"), Some("MIT")),
            ],
        );
        let reg = Registry::with_dir(dir.path().to_path_buf());
        for id in ["realesr-general-x4v3", "mine-2x"] {
            assert_eq!(reg.entries().iter().filter(|e| e.id == id).count(), 1);
            assert_eq!(reg.get(id).unwrap().source, "catalogue");
        }
        assert_eq!(reg.local_path("mine-2x"), dir.path().join("mine-2x.param"));
    }

    #[test]
    fn every_catalogue_upscaler_names_its_overlap() {
        let dir = tempfile::tempdir().unwrap();
        for e in Registry::with_dir(dir.path().to_path_buf()).upscalers() {
            assert!(e.preferred_overlap().is_some(), "{}", e.id);
        }
    }

    #[test]
    fn kind_defaults_to_upscale_and_reads_inpaint() {
        let json = r#"{"version":1,"models":[
            {"id":"a","name":"A","arch":"x","scale":4,"license":"MIT","commercial":true,
             "size_bytes":1,"url":"","sha256":""},
            {"id":"b","name":"B","arch":"x","scale":1,"license":"MIT","commercial":true,
             "size_bytes":1,"url":"","sha256":"","kind":"inpaint"}]}"#;
        let entries = parse_manifest(json).unwrap();
        assert_eq!(entries[0].kind, ModelKind::Upscale);
        assert_eq!(entries[1].kind, ModelKind::Inpaint);
        assert_eq!(ModelKind::default(), ModelKind::Upscale);
        let back = serde_json::to_value(&entries[1]).unwrap();
        assert_eq!(back["kind"], "inpaint");
        let bad = json.replace("\"inpaint\"", "\"denoise\"");
        assert!(parse_manifest(&bad).is_err());
    }

    #[test]
    fn upscalers_and_inpainters_split_the_catalogue() {
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::with_dir(dir.path().to_path_buf());
        let ids = |it: Vec<&ModelEntry>| it.iter().map(|e| e.id.clone()).collect::<Vec<_>>();
        let inpainters = ids(reg.inpainters().collect());
        assert_eq!(inpainters, ["lama-512", "migan-512-places2"]);
        let upscalers = ids(reg.upscalers().collect());
        assert_eq!(upscalers.len() + inpainters.len(), reg.entries().len());
        assert!(upscalers.iter().all(|id| !inpainters.contains(id)));
        assert!(reg.upscalers().all(|e| e.kind == ModelKind::Upscale));
    }

    #[test]
    fn inpainting_entries_follow_the_spike() {
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::with_dir(dir.path().to_path_buf());
        let lama = reg.get("lama-512").unwrap();
        assert_eq!(
            (lama.name.as_str(), lama.license.as_str(), lama.size_bytes),
            ("LaMa (512)", "Apache-2.0", 208_044_816)
        );
        assert_eq!(
            lama.sha256,
            "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"
        );
        assert_eq!(lama.tags, ["extend", "default"]);
        assert!(lama.url.starts_with(
            "https://huggingface.co/Carve/LaMa-ONNX/resolve/c3c0c9e468934d62e79c329e35d82dd09ff8c444/"
        ));
        let migan = reg.get("migan-512-places2").unwrap();
        assert_eq!(
            (migan.license.as_str(), migan.size_bytes),
            ("MIT", 29_546_882)
        );
        assert_eq!(
            migan.sha256,
            "593eba0b7e04730f1b61c0a3cbca68d97d8d6a7ff5c6a44a7b9d7fcd880fc5ae"
        );
        assert_eq!(migan.tags, ["extend", "fast"]);
        assert!(migan.url.starts_with(
            "https://huggingface.co/andraniksargsyan/migan/resolve/406830d0fa60666da0071c342ad2fbc8f30c5c64/"
        ));
        for e in [lama, migan] {
            assert_eq!(e.kind, ModelKind::Inpaint);
            assert_eq!(
                (e.scale, e.commercial, e.format.as_str()),
                (1, true, "onnx")
            );
        }
    }

    #[test]
    fn every_catalogue_model_has_its_license_text() {
        let dir = tempfile::tempdir().unwrap();
        let licenses = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/licenses");
        for e in Registry::with_dir(dir.path().to_path_buf()).entries() {
            let text = std::fs::read_to_string(licenses.join(format!("{}.txt", e.id)))
                .unwrap_or_else(|err| panic!("{}: {err}", e.id));
            assert!(text.starts_with(&format!("Model: {} — ", e.id)), "{}", e.id);
        }
    }

    #[test]
    fn manifest_parses_and_has_default_tag() {
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::with_dir(dir.path().to_path_buf());
        assert_eq!(reg.entries().len(), 10);
        let defaults = |it: Vec<&ModelEntry>| {
            it.into_iter()
                .filter(|e| e.tags.iter().any(|t| t == "default"))
                .map(|e| e.id.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            defaults(reg.upscalers().collect()),
            ["realesr-general-x4v3"]
        );
        assert_eq!(defaults(reg.inpainters().collect()), ["lama-512"]);
        assert!(
            reg.entries().iter().all(|e| e.format == "onnx"
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
        let good_sha = crate::registry::hex(&Sha256::digest(BODY));
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
        let sha = crate::registry::hex(&Sha256::digest(BODY));
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
        let sha = crate::registry::hex(&Sha256::digest(BODY));
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
