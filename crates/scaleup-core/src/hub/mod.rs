//! OpenModelDB hub: the model catalogue, its filters and search, and the record of installed models.

pub mod catalog;
pub mod convert;
pub mod download;
pub mod installed;
pub mod license;
pub mod recommend;

pub use catalog::{
    Capabilities, Catalog, Filter, ModelCard, Readiness, Resource, cache_is_fresh, readiness,
    speed_tier,
};
pub use convert::{
    PIP_INSTALL, Python, PytorchSource, check_checkpoint, convert_to_onnx, find_python,
    find_python_from, import_converted, import_needs_conversion, install_converted, pytorch_source,
};
pub use download::{
    Host, Http, HttpResponse, UreqHttp, classify_url, expected_file_name, import_file, install,
    pick_resource, uninstall,
};
pub use installed::{Installed, InstalledModel};
pub use license::{LicenseClass, classify_license, commercial_ok};
pub use recommend::{Ask, Content, Goal, Recommendation, SpeedPref, recommend};
