//! Licence classes of OpenModelDB models, by SPDX identifier.

/// How a model's licence limits its use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LicenseClass {
    Permissive,
    Attribution,
    ShareAlike,
    Copyleft,
    NonCommercial,
    /// No licence stated, or one this table does not know.
    Unknown,
}

/// Maps an SPDX identifier to its class. `None` and unlisted identifiers are [`LicenseClass::Unknown`].
pub fn classify_license(s: Option<&str>) -> LicenseClass {
    match s {
        Some("MIT" | "BSD-3-Clause" | "Apache-2.0" | "CC0-1.0" | "Unlicense" | "WTFPL") => {
            LicenseClass::Permissive
        }
        Some("CC-BY-4.0") => LicenseClass::Attribution,
        Some("CC-BY-SA-4.0") => LicenseClass::ShareAlike,
        Some("GPL-3.0-only") => LicenseClass::Copyleft,
        Some("CC-BY-NC-4.0" | "CC-BY-NC-SA-4.0") => LicenseClass::NonCommercial,
        _ => LicenseClass::Unknown,
    }
}

/// Whether the licence clearly allows commercial use. An unknown licence does not.
pub fn commercial_ok(c: LicenseClass) -> bool {
    matches!(
        c,
        LicenseClass::Permissive
            | LicenseClass::Attribution
            | LicenseClass::ShareAlike
            | LicenseClass::Copyleft
    )
}
