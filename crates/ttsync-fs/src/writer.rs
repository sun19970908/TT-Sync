//! Atomic file writer: write to temp file, then rename to final path.

use std::path::{Path, PathBuf};

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use ttsync_contract::path::SyncPath;
use ttsync_core::dataset::prune_boundary_for_path;
use ttsync_core::error::SyncError;

use crate::layout::{WorkspaceMounts, resolve_canonical_to_local, resolve_to_local};

/// Write data to a file atomically: tmp file → flush → rename.
/// Preserves mtime after write.
pub async fn write_file_atomic(
    mounts: &WorkspaceMounts,
    sync_path: &SyncPath,
    data: &mut (dyn AsyncRead + Send + Unpin),
    modified_ms: u64,
) -> Result<(), SyncError> {
    write_file_to_path(&resolve_to_local(mounts, sync_path), data, modified_ms).await
}

pub(crate) async fn write_file_to_path(
    full_path: &Path,
    data: &mut (dyn AsyncRead + Send + Unpin),
    modified_ms: u64,
) -> Result<(), SyncError> {
    if let Some(parent) = full_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| SyncError::Io(format!("create dir {}: {e}", parent.display())))?;
    }

    let tmp_path = download_tmp_path(full_path);
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tmp_path)
        .await
        .map_err(|e| SyncError::Io(format!("open {}: {e}", tmp_path.display())))?;

    copy_to_file(data, &mut file, full_path).await?;

    file.flush()
        .await
        .map_err(|e| SyncError::Io(format!("flush {}: {e}", full_path.display())))?;
    drop(file);

    rename_with_retry(&tmp_path, full_path).await?;
    set_file_modified_ms(full_path, modified_ms).await?;

    Ok(())
}

/// Read an entire sync stream into memory (used for files that must be
/// translated before being forwarded, e.g. settings and avatar bytes).
pub(crate) async fn read_all(
    data: &mut (dyn AsyncRead + Send + Unpin),
) -> Result<Vec<u8>, SyncError> {
    let mut bytes = Vec::new();
    data.read_to_end(&mut bytes)
        .await
        .map_err(|e| SyncError::Io(e.to_string()))?;
    Ok(bytes)
}

async fn copy_to_file(
    data: &mut (dyn AsyncRead + Send + Unpin),
    file: &mut tokio::fs::File,
    destination: &Path,
) -> Result<(), SyncError> {
    let mut buffer = vec![0u8; 64 * 1024];
    loop {
        let read = data.read(&mut buffer).await.map_err(|e| {
            SyncError::Io(format!("read bundle for {}: {e}", destination.display()))
        })?;
        if read == 0 {
            return Ok(());
        }
        file.write_all(&buffer[..read])
            .await
            .map_err(|e| SyncError::Io(format!("write {}: {e}", destination.display())))?;
    }
}

/// Delete a file at the given sync path.
pub async fn delete_file(mounts: &WorkspaceMounts, sync_path: &SyncPath) -> Result<(), SyncError> {
    let prune_boundary = prune_boundary_for_path(sync_path.as_str())?
        .map(|boundary| resolve_canonical_to_local(mounts, boundary));
    let full_path = resolve_to_local(mounts, sync_path);
    if let Some(boundary) = &prune_boundary {
        let parent = full_path
            .parent()
            .ok_or_else(|| SyncError::Internal("sync file has no parent directory".into()))?;
        if !parent.starts_with(boundary) {
            return Err(SyncError::Internal(format!(
                "prune boundary {} is not an ancestor of {}",
                boundary.display(),
                full_path.display()
            )));
        }
    }

    // Read-only targets (e.g. git loose objects, stored as 0444) reject
    // deletion on Windows until the attribute is cleared.
    if let Err(error) = clear_readonly_attribute(&full_path)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        return Err(SyncError::Io(format!(
            "clear attributes {}: {error}",
            full_path.display()
        )));
    }

    match tokio::fs::remove_file(&full_path).await {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(SyncError::Io(format!(
                "remove file {}: {error}",
                full_path.display()
            )));
        }
    }

    if let Some(boundary) = prune_boundary {
        prune_fileless_ancestors(&full_path, &boundary).await?;
    }

    Ok(())
}

async fn prune_fileless_ancestors(file: &Path, boundary: &Path) -> Result<(), SyncError> {
    let mut current = file
        .parent()
        .ok_or_else(|| SyncError::Internal("sync file has no parent directory".into()))?
        .to_path_buf();

    while current != boundary {
        match tokio::fs::remove_dir(&current).await {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) if error.kind() == std::io::ErrorKind::DirectoryNotEmpty => {
                let Some(directories) = collect_fileless_tree(&current).await? else {
                    return Ok(());
                };

                for directory in directories.into_iter().rev() {
                    match tokio::fs::remove_dir(&directory).await {
                        Ok(()) => {}
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                        Err(error) if error.kind() == std::io::ErrorKind::DirectoryNotEmpty => {
                            return Ok(());
                        }
                        Err(error) => {
                            return Err(SyncError::Io(format!(
                                "remove directory {}: {error}",
                                directory.display()
                            )));
                        }
                    }
                }
            }
            Err(error) => {
                return Err(SyncError::Io(format!(
                    "remove directory {}: {error}",
                    current.display()
                )));
            }
        }

        current = current
            .parent()
            .ok_or_else(|| SyncError::Internal("prune boundary is not an ancestor".into()))?
            .to_path_buf();
    }

    Ok(())
}

async fn collect_fileless_tree(root: &Path) -> Result<Option<Vec<PathBuf>>, SyncError> {
    let mut pending = vec![root.to_path_buf()];
    let mut directories = Vec::new();

    while let Some(directory) = pending.pop() {
        let mut entries = tokio::fs::read_dir(&directory).await.map_err(|error| {
            SyncError::Io(format!("read directory {}: {error}", directory.display()))
        })?;
        directories.push(directory);

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|error| SyncError::Io(format!("read directory entry: {error}")))?
        {
            let file_type = entry.file_type().await.map_err(|error| {
                SyncError::Io(format!(
                    "read file type {}: {error}",
                    entry.path().display()
                ))
            })?;
            if !file_type.is_dir() {
                return Ok(None);
            }
            pending.push(entry.path());
        }
    }

    Ok(Some(directories))
}

fn download_tmp_path(full_path: &Path) -> PathBuf {
    match full_path.extension() {
        Some(ext) if !ext.is_empty() => {
            let mut tmp_ext = ext.to_os_string();
            tmp_ext.push(".ttsync.tmp");
            full_path.with_extension(tmp_ext)
        }
        _ => full_path.with_extension("ttsync.tmp"),
    }
}

/// Whether an IO error is a transient Windows lock worth retrying:
/// ERROR_ACCESS_DENIED (5, also returned when a target handle lacks delete
/// sharing) or ERROR_SHARING_VIOLATION (32). Off Windows these raw numbers
/// mean unrelated errors (EIO/EPIPE), so the predicate is always false there.
#[cfg(windows)]
fn is_transient_windows_lock(error: &std::io::Error) -> bool {
    matches!(error.raw_os_error(), Some(5) | Some(32))
}

#[cfg(not(windows))]
fn is_transient_windows_lock(_error: &std::io::Error) -> bool {
    false
}

/// Retry an filesystem operation that hit a transient Windows lock, with
/// exponential backoff. The read-only-target recovery lives in the operation
/// closure itself; here only the retry policy is added.
async fn retry_transient_lock<F, T>(action: &str, path: &Path, mut op: F) -> Result<T, SyncError>
where
    F: FnMut() -> Result<T, std::io::Error>,
{
    const ATTEMPTS: u32 = 8;
    let mut last_error = None;
    for attempt in 0..ATTEMPTS {
        match op() {
            Ok(value) => return Ok(value),
            Err(error) if is_transient_windows_lock(&error) && attempt + 1 < ATTEMPTS => {
                last_error = Some(error);
                tokio::time::sleep(std::time::Duration::from_millis(10 * 2u64.pow(attempt))).await;
            }
            Err(error) => {
                return Err(SyncError::Io(format!(
                    "{action} {}: {error}",
                    path.display()
                )));
            }
        }
    }
    Err(SyncError::Io(format!(
        "{action} {} still locked after {ATTEMPTS} attempts: {}",
        path.display(),
        last_error
            .as_ref()
            .map(std::io::Error::to_string)
            .unwrap_or_default()
    )))
}

/// Clear the read-only attribute of an existing file. On Windows both
/// replacing and deleting a read-only file fail with ERROR_ACCESS_DENIED. A
/// missing file is not an error. No-op off Windows, where read-only mode bits
/// never block rename or unlink.
#[cfg(windows)]
fn clear_readonly_attribute(path: &Path) -> std::io::Result<()> {
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    let mut permissions = metadata.permissions();
    if permissions.readonly() {
        // Windows-only code path: toggling the DOS read-only attribute is
        // exactly what we need; the Unix "world writable" lint does not apply
        // (this function never compiles into a unix build).
        #[allow(clippy::permissions_set_readonly_false)]
        permissions.set_readonly(false);
        std::fs::set_permissions(path, permissions)?;
    }
    Ok(())
}

#[cfg(not(windows))]
fn clear_readonly_attribute(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

async fn rename_with_retry(from: &Path, to: &Path) -> Result<(), SyncError> {
    // Read-only-target recovery happens inside one attempt; a genuine sharing
    // violation stays a lock error so the outer backoff retries the rename.
    let attempt = || -> std::io::Result<()> {
        match std::fs::rename(from, to) {
            Ok(()) => Ok(()),
            // Read-only target: clear the attribute and replace it.
            Err(error) if is_transient_windows_lock(&error) => {
                clear_readonly_attribute(to)?;
                std::fs::rename(from, to)
            }
            // Platforms without replace semantics: drop the stale target first.
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                clear_readonly_attribute(to)?;
                std::fs::remove_file(to)?;
                std::fs::rename(from, to)
            }
            Err(error) => Err(error),
        }
    };

    match attempt() {
        Ok(()) => Ok(()),
        Err(error) if is_transient_windows_lock(&error) => {
            retry_transient_lock("rename", to, attempt).await
        }
        Err(error) => Err(SyncError::Io(format!(
            "rename {} -> {}: {error}",
            from.display(),
            to.display()
        ))),
    }
}

async fn set_file_modified_ms(path: &Path, modified_ms: u64) -> Result<(), SyncError> {
    let secs = (modified_ms / 1000) as i64;
    let nanos = ((modified_ms % 1000) * 1_000_000) as u32;
    let mtime = filetime::FileTime::from_unix_time(secs, nanos);
    retry_transient_lock("set mtime", path, || filetime::set_file_mtime(path, mtime)).await
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use ttsync_contract::path::SyncPath;

    use crate::layout::WorkspaceMounts;

    use super::{delete_file, retry_transient_lock, write_file_to_path};

    #[tokio::test]
    async fn deleting_last_git_file_prunes_fileless_git_tree() {
        let data_root = unique_temp_dir();
        let mounts = test_mounts(&data_root);
        let extension = mounts.extensions_root.join("example");
        let git = extension.join(".git");

        std::fs::create_dir_all(git.join("objects/info")).unwrap();
        std::fs::create_dir_all(git.join("objects/pack")).unwrap();
        std::fs::create_dir_all(git.join("refs/heads")).unwrap();
        std::fs::create_dir_all(git.join("refs/tags")).unwrap();
        std::fs::write(extension.join("manifest.json"), b"{}").unwrap();
        std::fs::write(git.join("HEAD"), b"ref: refs/heads/main\n").unwrap();
        std::fs::write(git.join("config"), b"[core]\n").unwrap();

        delete_file(
            &mounts,
            &SyncPath::new("extensions/third-party/example/.git/HEAD").unwrap(),
        )
        .await
        .unwrap();
        assert!(git.exists());
        assert!(git.join("config").exists());
        assert!(git.join("objects/info").exists());

        delete_file(
            &mounts,
            &SyncPath::new("extensions/third-party/example/.git/config").unwrap(),
        )
        .await
        .unwrap();
        assert!(!git.exists());
        assert!(extension.join("manifest.json").exists());
        assert!(mounts.extensions_root.exists());

        std::fs::remove_dir_all(data_root).unwrap();
    }

    #[tokio::test]
    async fn deleting_last_dataset_file_keeps_dataset_boundary() {
        let data_root = unique_temp_dir();
        let mounts = test_mounts(&data_root);
        let extension = mounts.extensions_root.join("example");
        std::fs::create_dir_all(&extension).unwrap();
        std::fs::write(extension.join("index.js"), b"export {};").unwrap();

        delete_file(
            &mounts,
            &SyncPath::new("extensions/third-party/example/index.js").unwrap(),
        )
        .await
        .unwrap();

        assert!(!extension.exists());
        assert!(mounts.extensions_root.exists());
        std::fs::remove_dir_all(data_root).unwrap();
    }

    #[tokio::test]
    async fn file_only_dataset_does_not_prune_parent() {
        let data_root = unique_temp_dir();
        let mounts = test_mounts(&data_root);
        std::fs::create_dir_all(&mounts.default_user_root).unwrap();
        std::fs::write(mounts.default_user_root.join("settings.json"), b"{}").unwrap();

        delete_file(
            &mounts,
            &SyncPath::new("default-user/settings.json").unwrap(),
        )
        .await
        .unwrap();

        assert!(mounts.default_user_root.exists());
        std::fs::remove_dir_all(data_root).unwrap();
    }

    #[tokio::test]
    async fn unknown_dataset_path_fails_before_removing_file() {
        let data_root = unique_temp_dir();
        let mounts = test_mounts(&data_root);
        let file = data_root.join("outside/file.txt");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"keep").unwrap();

        let result = delete_file(&mounts, &SyncPath::new("outside/file.txt").unwrap()).await;

        assert!(result.is_err());
        assert!(file.exists());
        std::fs::remove_dir_all(data_root).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn symlink_in_candidate_tree_stops_pruning_without_following_it() {
        use std::os::unix::fs::symlink;

        let data_root = unique_temp_dir();
        let mounts = test_mounts(&data_root);
        let extension = mounts.extensions_root.join("example");
        let git = extension.join(".git");
        let outside = data_root.join("outside");
        std::fs::create_dir_all(&git).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(git.join("HEAD"), b"ref: refs/heads/main\n").unwrap();
        std::fs::write(outside.join("keep.txt"), b"keep").unwrap();
        symlink(&outside, git.join("linked-directory")).unwrap();

        delete_file(
            &mounts,
            &SyncPath::new("extensions/third-party/example/.git/HEAD").unwrap(),
        )
        .await
        .unwrap();

        assert!(git.join("linked-directory").symlink_metadata().is_ok());
        assert!(outside.join("keep.txt").exists());
        std::fs::remove_dir_all(data_root).unwrap();
    }

    #[tokio::test]
    async fn delete_file_is_idempotent_for_missing_files() {
        let data_root = unique_temp_dir();
        let mounts = test_mounts(&data_root);
        let path = SyncPath::new("default-user/chats/missing.jsonl").unwrap();

        delete_file(&mounts, &path).await.expect("missing delete");

        let _ = std::fs::remove_dir_all(data_root);
    }

    fn test_mounts(data_root: &Path) -> WorkspaceMounts {
        WorkspaceMounts {
            data_root: data_root.to_path_buf(),
            default_user_root: data_root.join("default-user"),
            extensions_root: data_root.join("extensions").join("third-party"),
        }
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn replaces_and_deletes_readonly_target_files() {
        // Git stores loose objects read-only; Windows rejects replacing or
        // deleting them unless the attribute is cleared first.
        let data_root = unique_temp_dir();
        let mounts = test_mounts(&data_root);
        std::fs::create_dir_all(&mounts.default_user_root).unwrap();
        let target = mounts.default_user_root.join("readonly-object");
        std::fs::write(&target, b"old").unwrap();

        let mut perms = std::fs::metadata(&target).unwrap().permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(&target, perms).unwrap();

        write_file_to_path(&target, &mut std::io::Cursor::new(b"new".to_vec()), 100)
            .await
            .expect("readonly target is replaced");
        assert_eq!(std::fs::read(&target).unwrap(), b"new");

        // Verify deletion through the catalog-checked public path too
        // (settings.json is a file-only dataset entry).
        let settings_target = mounts.default_user_root.join("settings.json");
        std::fs::write(&settings_target, b"{}").unwrap();
        let mut perms = std::fs::metadata(&settings_target).unwrap().permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(&settings_target, perms).unwrap();
        delete_file(
            &mounts,
            &SyncPath::new("default-user/settings.json").unwrap(),
        )
        .await
        .expect("readonly target is deleted");
        assert!(!settings_target.exists());

        std::fs::remove_dir_all(data_root).unwrap();
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn retries_transient_windows_access_denied_then_succeeds() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let attempts = AtomicU32::new(0);
        let result = retry_transient_lock("probe", Path::new("locked-file"), || {
            if attempts.fetch_add(1, Ordering::SeqCst) < 2 {
                return Err(std::io::Error::from_raw_os_error(5));
            }
            Ok(42u32)
        })
        .await
        .expect("transient lock retried");
        assert_eq!(result, 42);
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    fn unique_temp_dir() -> PathBuf {
        static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        let sequence = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("ttsync-writer-test-{now}-{sequence}"))
    }
}
