//! The record of hub models installed on this machine (`<models_dir>/hub/installed.json`).

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::Result;

const INSTALLED_FILE: &str = "installed.json";

/// One installed model.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct InstalledModel {
    pub id: String,
    pub name: String,
    pub arch: String,
    pub scale: u32,
    pub license: Option<String>,
    /// `"onnx"`, `"pth"` or `"safetensors"`.
    pub format: String,
    pub file: PathBuf,
    pub sha256: Option<String>,
    /// The file's SHA-256 matched the catalogue.
    #[serde(default)]
    pub verified: bool,
    pub source_url: String,
    pub installed_at: String,
}

/// The installed models, in installation order.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct Installed {
    pub models: Vec<InstalledModel>,
}

fn path(models_dir: &Path) -> PathBuf {
    models_dir.join("hub").join(INSTALLED_FILE)
}

impl Installed {
    /// Reads `<models_dir>/hub/installed.json`. A missing file is an empty list; an unreadable one is
    /// ignored with a warning, like the local overlay of the registry.
    pub fn load(models_dir: &Path) -> Installed {
        let file = path(models_dir);
        let Ok(text) = fs::read_to_string(&file) else {
            return Installed::default();
        };
        serde_json::from_str(&text).unwrap_or_else(|e| {
            tracing::warn!("ignoring {}: {e}", file.display());
            Installed::default()
        })
    }

    /// Writes the list, creating the `hub` folder. The file is replaced atomically.
    pub fn save(&self, models_dir: &Path) -> Result<()> {
        let file = path(models_dir);
        fs::create_dir_all(models_dir.join("hub"))?;
        let tmp = file.with_extension("json.part");
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| crate::ScaleUpError::Model(format!("{}: {e}", file.display())))?;
        fs::write(&tmp, json)?;
        fs::rename(&tmp, &file)?;
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&InstalledModel> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Records `model`, replacing the entry with the same id in place or appending a new one.
    pub fn upsert(&mut self, model: InstalledModel) {
        match self.models.iter_mut().find(|m| m.id == model.id) {
            Some(slot) => *slot = model,
            None => self.models.push(model),
        }
    }

    /// Removes the record of `id`. Returns whether there was one.
    pub fn remove(&mut self, id: &str) -> bool {
        let before = self.models.len();
        self.models.retain(|m| m.id != id);
        self.models.len() != before
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(id: &str) -> InstalledModel {
        InstalledModel {
            id: id.into(),
            name: format!("{id} name"),
            arch: "span".into(),
            scale: 2,
            license: Some("MIT".into()),
            format: "onnx".into(),
            file: PathBuf::from(format!("hub/{id}.onnx")),
            sha256: Some("ab".repeat(32)),
            verified: true,
            source_url: "https://github.com/a/b".into(),
            installed_at: "2026-11-05T10:00:00Z".into(),
        }
    }

    #[test]
    fn installed_json_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        assert!(Installed::load(dir.path()).models.is_empty());
        let mut inst = Installed::default();
        inst.models.push(model("a"));
        inst.models.push(model("b"));
        inst.save(dir.path()).unwrap();
        assert!(dir.path().join("hub").join("installed.json").is_file());
        assert!(!dir.path().join("hub").join("installed.json.part").exists());
        assert_eq!(Installed::load(dir.path()), inst);
    }

    #[test]
    fn corrupt_installed_json_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("hub")).unwrap();
        fs::write(path(dir.path()), "{ not json").unwrap();
        assert!(Installed::load(dir.path()).models.is_empty());
    }

    #[test]
    fn upsert_replaces_in_place() {
        let mut inst = Installed::default();
        inst.upsert(model("a"));
        inst.upsert(model("b"));
        let mut a2 = model("a");
        a2.scale = 4;
        inst.upsert(a2);
        assert_eq!(inst.models.len(), 2);
        assert_eq!(inst.models[0].scale, 4);
        assert_eq!(inst.models[1].id, "b");
    }

    #[test]
    fn get_and_remove() {
        let mut inst = Installed::default();
        inst.models.push(model("a"));
        inst.models.push(model("b"));
        assert_eq!(inst.get("b").unwrap().name, "b name");
        assert!(inst.get("c").is_none());
        assert!(inst.remove("a"));
        assert!(!inst.remove("a"));
        assert_eq!(inst.models.len(), 1);
    }
}
