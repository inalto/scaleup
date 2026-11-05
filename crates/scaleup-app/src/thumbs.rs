//! Before/after thumbnails of OpenModelDB cards: where they are, fetching and the disk cache
//! (`<models_dir>/hub/thumbs/`). A thumbnail that cannot be had is simply absent.

use std::io::{Cursor, Read};
use std::path::Path;

use scaleup_core::hub::{Http, ModelCard};

/// Site-relative image paths of the catalogue are relative to this.
const SITE: &str = "https://openmodeldb.info";
/// The most bytes read for one image.
pub const MAX_BYTES: u64 = 2 * 1024 * 1024;
/// Long side of a cached thumbnail.
pub const MAX_SIDE: u32 = 256;

/// The `https://` URL of a catalogue image reference: an absolute `https://` URL as is, a
/// site-relative path (`/thumbs/…`) on openmodeldb.info. Anything else (plain `http`,
/// protocol-relative `//host`, relative paths, `data:`) has none.
pub fn resolve_url(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.starts_with("https://") && raw.len() > "https://".len() {
        Some(raw.to_owned())
    } else if raw.starts_with('/') && !raw.starts_with("//") {
        Some(format!("{SITE}{raw}"))
    } else {
        None
    }
}

/// The before (`LR`) and after (`SR`) images of `card`: its `thumbnail`, else its first image
/// that has either.
pub fn pair_urls(card: &ModelCard) -> [Option<String>; 2] {
    let pair = |v: &serde_json::Value| {
        let side = |key: &str| v.get(key).and_then(|u| u.as_str()).and_then(resolve_url);
        let pair = [side("LR"), side("SR")];
        pair.iter().any(Option::is_some).then_some(pair)
    };
    card.thumbnail
        .as_ref()
        .and_then(pair)
        .or_else(|| card.images.iter().find_map(pair))
        .unwrap_or_default()
}

/// The cache file name of `url`: 64-bit FNV-1a of the URL, so it is stable across builds.
pub fn cache_key(url: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in url.bytes() {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{hash:016x}.png")
}

/// Largest image side accepted from the network.
pub const MAX_DECODE_SIDE: u32 = 8192;
/// Most memory the decoder may allocate.
pub const MAX_DECODE_ALLOC: u64 = 64 * 1024 * 1024;

/// Decodes an image with limits, so a small file that declares huge dimensions is refused
/// before anything is allocated.
pub fn decode(bytes: &[u8]) -> image::ImageResult<image::RgbaImage> {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_DECODE_SIDE);
    limits.max_image_height = Some(MAX_DECODE_SIDE);
    limits.max_alloc = Some(MAX_DECODE_ALLOC);
    let mut reader = image::ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    reader.limits(limits);
    Ok(reader.decode()?.into_rgba8())
}

/// The thumbnail at `url`, from `cache_dir` or else downloaded (`https://` only, at most
/// [`MAX_BYTES`]), decoded, shrunk to [`MAX_SIDE`] and cached. `None` on any failure.
pub fn load(http: &dyn Http, url: &str, cache_dir: &Path) -> Option<image::RgbaImage> {
    let cached = cache_dir.join(cache_key(url));
    if let Ok(img) = image::open(&cached) {
        return Some(img.into_rgba8());
    }
    if !url.starts_with("https://") {
        return None;
    }
    let response = http.get(url, &[]).ok()?;
    if response.content_length.is_some_and(|n| n > MAX_BYTES) {
        return None;
    }
    let mut bytes = Vec::new();
    response
        .body
        .take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MAX_BYTES {
        return None;
    }
    let img = decode(&bytes).ok()?;
    let img = crate::worker::cap_long_side(img, MAX_SIDE);
    if std::fs::create_dir_all(cache_dir).is_ok() {
        let _ = img.save(&cached);
    }
    Some(img)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use scaleup_core::hub::{Catalog, HttpResponse};
    use scaleup_core::{Result, ScaleUpError};
    use std::sync::Mutex;

    /// Serves `body` to every request and counts them.
    pub(crate) struct Fixed {
        pub body: Vec<u8>,
        pub length: Option<u64>,
        pub seen: Mutex<Vec<String>>,
    }

    impl Fixed {
        pub fn new(body: Vec<u8>) -> Fixed {
            Fixed {
                length: Some(body.len() as u64),
                body,
                seen: Mutex::default(),
            }
        }
    }

    impl Http for Fixed {
        fn get(&self, url: &str, _: &[(String, String)]) -> Result<HttpResponse> {
            self.seen.lock().unwrap().push(url.to_owned());
            Ok(HttpResponse {
                content_type: Some("image/png".into()),
                content_disposition: None,
                content_length: self.length,
                cookies: Vec::new(),
                body: Box::new(Cursor::new(self.body.clone())),
            })
        }
    }

    struct Down;

    impl Http for Down {
        fn get(&self, url: &str, _: &[(String, String)]) -> Result<HttpResponse> {
            Err(ScaleUpError::Download(format!("{url}: offline")))
        }
    }

    pub(crate) fn png(w: u32, h: u32) -> Vec<u8> {
        let mut out = Cursor::new(Vec::new());
        image::RgbaImage::from_pixel(w, h, image::Rgba([10, 20, 30, 255]))
            .write_to(&mut out, image::ImageFormat::Png)
            .unwrap();
        out.into_inner()
    }

    #[test]
    fn resolve_url_cases() {
        assert_eq!(
            resolve_url("/thumbs/small/abc.jpg").as_deref(),
            Some("https://openmodeldb.info/thumbs/small/abc.jpg")
        );
        assert_eq!(
            resolve_url(" https://imgsli.com/i/x.jpg ").as_deref(),
            Some("https://imgsli.com/i/x.jpg")
        );
        for bad in [
            "http://imgsli.com/i/x.jpg",
            "//evil.example/x.jpg",
            "thumbs/x.jpg",
            "data:image/png;base64,AAAA",
            "https://",
            "",
        ] {
            assert_eq!(resolve_url(bad), None, "{bad}");
        }
    }

    /// A 54-byte BMP header declaring `w`×`h` 24-bit pixels, followed by no pixel data.
    fn bmp_header(w: u32, h: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"BM");
        b.extend_from_slice(&(54 + w * h * 3).to_le_bytes());
        b.extend_from_slice(&[0; 4]);
        b.extend_from_slice(&54u32.to_le_bytes());
        b.extend_from_slice(&40u32.to_le_bytes());
        b.extend_from_slice(&w.to_le_bytes());
        b.extend_from_slice(&h.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&24u16.to_le_bytes());
        b.extend_from_slice(&[0; 24]);
        b
    }

    #[test]
    fn decode_refuses_huge_declared_dimensions() {
        let err = decode(&bmp_header(30_000, 30_000)).unwrap_err();
        assert!(matches!(err, image::ImageError::Limits(_)), "{err:?}");
        // Within the side limit but over the allocation limit: 6000×6000 RGBA is 144 MB.
        let err = decode(&bmp_header(6000, 6000)).unwrap_err();
        assert!(matches!(err, image::ImageError::Limits(_)), "{err:?}");
        assert_eq!(decode(&png(3, 2)).unwrap().dimensions(), (3, 2));
    }

    #[test]
    fn cache_key_is_stable_and_distinct() {
        // FNV-1a of the empty string is the offset basis.
        assert_eq!(cache_key(""), "cbf29ce484222325.png");
        let a = cache_key("https://openmodeldb.info/thumbs/a.jpg");
        assert_eq!(a, cache_key("https://openmodeldb.info/thumbs/a.jpg"));
        assert_ne!(a, cache_key("https://openmodeldb.info/thumbs/b.jpg"));
        assert_eq!(a.len(), 20);
    }

    #[test]
    fn pair_urls_of_the_fixture_shapes() {
        let cat = Catalog::parse(include_str!(
            "../../../tests/fixtures/hub/models.sample.json"
        ))
        .unwrap();
        // A card-level `thumbnail` object with site-relative LR and SR.
        let with_thumb = cat
            .models
            .iter()
            .find(|m| m.thumbnail.is_some())
            .expect("fixture has a thumbnail");
        let [lr, sr] = pair_urls(with_thumb);
        assert!(lr.unwrap().starts_with("https://openmodeldb.info/thumbs/"));
        assert!(sr.unwrap().starts_with("https://openmodeldb.info/thumbs/"));
        // Without `thumbnail`: the first paired image, whose URLs are absolute.
        let mut images_only = with_thumb.clone();
        images_only.thumbnail = None;
        let [lr, sr] = pair_urls(&images_only);
        assert_eq!(
            lr.as_deref(),
            images_only.images[0]["LR"].as_str(),
            "{:?}",
            images_only.images
        );
        assert!(sr.unwrap().starts_with("https://"));
        let bare = cat
            .models
            .iter()
            .find(|m| m.thumbnail.is_none() && m.images.is_empty())
            .unwrap();
        assert_eq!(pair_urls(bare), [None, None]);
    }

    #[test]
    fn load_downloads_shrinks_and_caches() {
        let dir = tempfile::tempdir().unwrap();
        let url = "https://openmodeldb.info/thumbs/a.png";
        let http = Fixed::new(png(600, 300));
        let img = load(&http, url, dir.path()).expect("decoded");
        assert_eq!(img.dimensions(), (256, 128));
        assert!(dir.path().join(cache_key(url)).is_file());
        // From the cache: no network needed.
        let again = load(&Down, url, dir.path()).expect("cached");
        assert_eq!(again.dimensions(), (256, 128));
        assert_eq!(http.seen.lock().unwrap().len(), 1);
    }

    #[test]
    fn load_failures_are_just_missing_pictures() {
        let dir = tempfile::tempdir().unwrap();
        let http = Fixed::new(png(8, 8));
        assert!(load(&http, "http://openmodeldb.info/x.png", dir.path()).is_none());
        assert!(
            http.seen.lock().unwrap().is_empty(),
            "no plain http request"
        );
        assert!(load(&Down, "https://x.org/a.png", dir.path()).is_none());
        let garbage = Fixed::new(b"not an image".to_vec());
        assert!(load(&garbage, "https://x.org/b.png", dir.path()).is_none());
        let mut big = Fixed::new(vec![0u8; (MAX_BYTES + 10) as usize]);
        big.length = None;
        assert!(load(&big, "https://x.org/c.png", dir.path()).is_none());
        let mut announced = Fixed::new(png(8, 8));
        announced.length = Some(MAX_BYTES + 1);
        assert!(load(&announced, "https://x.org/d.png", dir.path()).is_none());
        assert!(!dir.path().join(cache_key("https://x.org/b.png")).exists());
    }
}
