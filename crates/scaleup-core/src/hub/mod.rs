//! OpenModelDB hub: the model catalogue, its filters and search, and the record of installed models.

pub mod catalog;
pub mod download;
pub mod installed;
pub mod license;
pub mod recommend;

pub use catalog::{
    Capabilities, Catalog, Filter, ModelCard, Readiness, Resource, cache_is_fresh, readiness,
    speed_tier,
};
pub use download::{
    Host, Http, HttpResponse, UreqHttp, classify_url, import_file, install, pick_resource,
};
pub use installed::{Installed, InstalledModel};
pub use license::{LicenseClass, classify_license, commercial_ok};
pub use recommend::{Ask, Content, Goal, Recommendation, SpeedPref, recommend};
