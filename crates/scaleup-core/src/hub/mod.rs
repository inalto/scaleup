//! OpenModelDB hub: the model catalogue, its filters and search, and the record of installed models.

pub mod catalog;
pub mod installed;
pub mod license;

pub use catalog::{
    Capabilities, Catalog, Filter, ModelCard, Readiness, Resource, cache_is_fresh, readiness,
    speed_tier,
};
pub use installed::{Installed, InstalledModel};
pub use license::{LicenseClass, classify_license, commercial_ok};
