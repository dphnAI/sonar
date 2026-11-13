// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `file://` media parts, served from `--allowed-local-media-path`.
//!
//! The media tracker fetches every URL that is not a `data:` URL over HTTP, so
//! a `file://` part is read here and handed on as a base64 `data:` URL of the
//! same part kind, which keeps the part's own options (detail, fps, long-side
//! cap, uuid).

use std::fs::{self, File};
use std::io::Read as _;
use std::path::{Component, Path};
use std::sync::Arc;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use llm_multimodal::media::{audio_max_input_bytes, image_max_input_bytes, video_max_input_bytes};
use llm_multimodal::{MediaConnectorError, MediaContentPart};
use thiserror_ext::AsReport as _;

use crate::error::{Error, Result, multimodal};

#[derive(Clone, Copy)]
enum LocalMediaKind {
    Image,
    Video,
    Audio,
}

impl LocalMediaKind {
    fn name(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
            Self::Audio => "audio",
        }
    }

    /// The per-item input cap the URL fetch path applies to this media.
    fn max_input_bytes(self) -> usize {
        match self {
            Self::Image => image_max_input_bytes(),
            Self::Video => video_max_input_bytes(),
            Self::Audio => audio_max_input_bytes(),
        }
    }

    /// The decoders sniff the content, so the subtype is not read; the
    /// modality's prefix keeps the tracker's data URL fast path.
    fn data_url_prefix(self) -> &'static str {
        match self {
            Self::Image => "data:image/unknown;base64,",
            Self::Video => "data:video/unknown;base64,",
            Self::Audio => "data:audio/unknown;base64,",
        }
    }
}

/// Whether `url` is a `file:` URL, with the scheme spelled in any case and
/// the leading whitespace a URL parser skips.
fn is_file_url(url: &str) -> bool {
    url.trim_start_matches(|c: char| c <= ' ')
        .get(..5)
        .is_some_and(|scheme| scheme.eq_ignore_ascii_case("file:"))
}

fn invalid(message: String) -> Error {
    Error::InvalidMedia(message)
}

/// Replace a `file://` URL part with a `data:` URL part read from the file,
/// confined to `allowed_root` (canonical). Other parts are left untouched.
pub(super) async fn inline_file_url(
    part: &mut MediaContentPart,
    allowed_root: Option<&Arc<Path>>,
) -> Result<()> {
    let (url, kind) = match part {
        MediaContentPart::ImageUrl { url, .. } => (url, LocalMediaKind::Image),
        MediaContentPart::VideoUrl { url, .. } => (url, LocalMediaKind::Video),
        MediaContentPart::AudioUrl { url, .. } => (url, LocalMediaKind::Audio),
        _ => return Ok(()),
    };
    if !is_file_url(url) {
        return Ok(());
    }
    let Some(root) = allowed_root else {
        return Err(invalid(
            "Cannot load local files without `--allowed-local-media-path`.".to_string(),
        ));
    };
    let path = url::Url::parse(url)
        .ok()
        .and_then(|parsed| parsed.to_file_path().ok())
        .ok_or_else(|| invalid(format!("Invalid file URL: {url}")))?;
    let root = Arc::clone(root);
    *url = tokio::task::spawn_blocking(move || read_as_data_url(&path, &root, kind))
        .await
        .map_err(|error| {
            let error = error.as_report();
            multimodal!("local media read task failed: {error}")
        })??;
    Ok(())
}

/// Read `path` as a base64 `data:` URL if it is a regular file strictly
/// under `root` once symlinks and `..` are resolved, and within the media's
/// input cap.
fn read_as_data_url(path: &Path, root: &Path, kind: LocalMediaKind) -> Result<String> {
    let outside_root = || {
        invalid(format!(
            "The file path {} must be a subpath of `--allowed-local-media-path {}`.",
            path.display(),
            root.display()
        ))
    };
    let canonical = match path.canonicalize() {
        Ok(canonical) => canonical,
        // Say whether a missing file exists only when the path is plainly
        // under the root, so the error does not probe the rest of the disk.
        Err(_) if !is_lexically_under(path, root) => return Err(outside_root()),
        Err(error) => {
            return Err(invalid(format!(
                "Cannot read local media file {}: {}",
                path.display(),
                error.as_report()
            )));
        }
    };
    if canonical == root || !canonical.starts_with(root) {
        return Err(outside_root());
    }

    let read_error = |error: std::io::Error| {
        invalid(format!(
            "Cannot read local media file {}: {}",
            path.display(),
            error.as_report()
        ))
    };
    let not_regular = || {
        invalid(format!(
            "The file path {} is not a regular file.",
            path.display()
        ))
    };
    // Check before opening: opening a FIFO for reading would block.
    if !fs::metadata(&canonical).map_err(read_error)?.is_file() {
        return Err(not_regular());
    }
    let file = File::open(&canonical).map_err(read_error)?;
    let metadata = file.metadata().map_err(read_error)?;
    if !metadata.is_file() {
        return Err(not_regular());
    }

    let limit = kind.max_input_bytes();
    let too_large = || {
        invalid(
            MediaConnectorError::PayloadTooLarge {
                media: kind.name(),
                limit,
            }
            .to_string(),
        )
    };
    let limit_u64 = u64::try_from(limit).unwrap_or(u64::MAX);
    if metadata.len() > limit_u64 {
        return Err(too_large());
    }
    // Read at most one byte past the cap, which also catches a file that
    // grew after the size check.
    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or(limit));
    file.take(limit_u64.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(read_error)?;
    if bytes.len() > limit {
        return Err(too_large());
    }

    let prefix = kind.data_url_prefix();
    let mut data_url = String::with_capacity(prefix.len() + bytes.len().div_ceil(3) * 4);
    data_url.push_str(prefix);
    STANDARD.encode_string(&bytes, &mut data_url);
    Ok(data_url)
}

/// Whether `path` names something under `root` without any `..` component.
fn is_lexically_under(path: &Path, root: &Path) -> bool {
    path.starts_with(root) && path.components().all(|component| component != Component::ParentDir)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// File content; nothing here decodes it.
    const PNG: &[u8] = b"\x89PNG\r\n\x1a\n";

    fn image_part(path: &Path) -> MediaContentPart {
        MediaContentPart::ImageUrl {
            url: url::Url::from_file_path(path).unwrap().to_string(),
            detail: None,
            uuid: Some("local".to_string()),
            max_long_side_pixel: Some(28),
        }
    }

    async fn inline(part: &mut MediaContentPart, root: Option<&Path>) -> Result<()> {
        let root = root.map(|root| Arc::from(root.canonicalize().unwrap()));
        inline_file_url(part, root.as_ref()).await
    }

    fn refusal(result: Result<()>) -> String {
        let error = result.unwrap_err();
        assert!(error.is_request_validation_error(), "{error:?}");
        error.to_string()
    }

    #[tokio::test]
    async fn file_url_is_refused_without_allowed_local_media_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.png");
        fs::write(&path, PNG).unwrap();
        let mut part = image_part(&path);

        assert_eq!(
            refusal(inline(&mut part, None).await),
            "Cannot load local files without `--allowed-local-media-path`."
        );
        // Spelled in another case, the scheme is still a file URL.
        let mut part = MediaContentPart::AudioUrl {
            url: format!("FILE://{}", path.display()),
            uuid: None,
        };
        assert!(inline(&mut part, None).await.is_err());
    }

    #[tokio::test]
    async fn file_url_must_resolve_strictly_under_the_allowed_directory() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("media");
        fs::create_dir(&root).unwrap();
        let outside = dir.path().join("secret.png");
        fs::write(&outside, PNG).unwrap();

        let escapes = [
            outside.clone(),
            // `..` out of the root.
            root.join("..").join("secret.png"),
            // A symlink inside the root to a file outside it.
            {
                let link = root.join("link.png");
                std::os::unix::fs::symlink(&outside, &link).unwrap();
                link
            },
            // The root itself is not under the root.
            root.clone(),
            // A missing file outside the root says nothing about existence.
            dir.path().join("missing.png"),
        ];
        for path in escapes {
            let message = refusal(inline(&mut image_part(&path), Some(&root)).await);
            assert!(
                message.contains("must be a subpath of `--allowed-local-media-path"),
                "{}: {message}",
                path.display()
            );
        }
    }

    #[tokio::test]
    async fn file_url_must_name_a_regular_file() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let subdir = root.join("subdir");
        fs::create_dir(&subdir).unwrap();

        let message = refusal(inline(&mut image_part(&subdir), Some(&root)).await);
        assert!(message.contains("is not a regular file"), "{message}");

        let missing = root.join("missing.png");
        let message = refusal(inline(&mut image_part(&missing), Some(&root)).await);
        assert!(
            message.contains("Cannot read local media file"),
            "{message}"
        );
    }

    #[tokio::test]
    async fn file_over_the_input_cap_is_refused() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("large.png");
        let file = File::create(&path).unwrap();
        // Sparse: the length alone is over the cap.
        file.set_len(image_max_input_bytes() as u64 + 1).unwrap();

        let message = refusal(inline(&mut image_part(&path), Some(root.path())).await);
        assert_eq!(
            message,
            format!(
                "image input payload exceeds the maximum size of {} bytes",
                image_max_input_bytes()
            )
        );
    }

    #[tokio::test]
    async fn file_url_under_the_root_becomes_a_data_url_keeping_part_options() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("nested").join("image.png");
        fs::create_dir(path.parent().unwrap()).unwrap();
        fs::write(&path, PNG).unwrap();
        let mut part = image_part(&path);

        inline(&mut part, Some(root.path())).await.unwrap();

        let MediaContentPart::ImageUrl {
            url,
            uuid,
            max_long_side_pixel,
            ..
        } = part
        else {
            panic!("expected an image URL part");
        };
        assert_eq!(
            url,
            format!("data:image/unknown;base64,{}", STANDARD.encode(PNG))
        );
        assert_eq!(uuid.as_deref(), Some("local"));
        assert_eq!(max_long_side_pixel, Some(28));
    }

    #[tokio::test]
    async fn other_urls_are_left_alone() {
        let mut part = MediaContentPart::ImageUrl {
            url: "https://example.com/file:image.png".to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        };
        let before = part.clone();
        inline(&mut part, None).await.unwrap();
        assert_eq!(format!("{part:?}"), format!("{before:?}"));
    }
}
