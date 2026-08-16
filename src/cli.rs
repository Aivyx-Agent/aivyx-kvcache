use std::path::Path;

use crate::KvCacheError;
use crate::manifest::Manifest;

pub async fn list_slots(store_path: &Path) -> Result<String, KvCacheError> {
    let manifest = Manifest::open(&store_path.join("manifest.db"))?;
    let rows = manifest.list_all().await?;
    if rows.is_empty() {
        return Ok("no cached slots".to_string());
    }
    let mut out = String::from("MODEL           BACKEND          BYTES     HITS  LAST_USED_SECS\n");
    for row in rows {
        out.push_str(&format!(
            "{:<15} {:<16} {:<9} {:<5} {}\n",
            row.key.model_id,
            row.key.backend_id,
            row.size_bytes,
            row.hit_count,
            row.last_used_at_secs
        ));
    }
    Ok(out)
}

pub async fn stats(store_path: &Path, max_bytes: u64) -> Result<String, KvCacheError> {
    let manifest = Manifest::open(&store_path.join("manifest.db"))?;
    let total = manifest.total_bytes().await?;
    let rows = manifest.list_all().await?;
    let total_hits: u64 = rows.iter().map(|r| r.hit_count).sum();
    Ok(format!(
        "{total} / {max_bytes} bytes used, {count} slots, {total_hits} total hits",
        count = rows.len()
    ))
}

pub async fn prune(
    store_path: &Path,
    target_bytes: u64,
    dry_run: bool,
) -> Result<String, KvCacheError> {
    let manifest = Manifest::open(&store_path.join("manifest.db"))?;
    if dry_run {
        let candidates = manifest.evict_candidates(target_bytes).await?;
        let bytes: u64 = candidates.iter().map(|r| r.size_bytes).sum();
        return Ok(format!(
            "dry run: would evict {count} slots, freeing {bytes} bytes",
            count = candidates.len()
        ));
    }
    let slots_dir = store_path.join("slots");
    let removed = manifest.evict_and_remove(target_bytes).await?;
    let mut evicted = 0u64;
    let mut freed = 0u64;
    let mut skipped_unsafe = 0u64;
    for row in removed {
        // The manifest row is already gone (evict_and_remove removed it
        // inside its transaction) regardless of what happens below -- this
        // only gates the *file* deletion attempt. A handle that isn't a
        // safe single-component filename is never produced by this crate's
        // own stores, only by a hypothetical future caller of the public
        // `record()` API; skip deleting it rather than joining it onto
        // `slots_dir` and handing a path-traversal-shaped value to
        // `remove_file`.
        if !row.handle.is_safe_filename() {
            skipped_unsafe += 1;
            evicted += 1;
            freed += row.size_bytes;
            continue;
        }
        let path = slots_dir.join(row.handle.as_str());
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(KvCacheError::Backend(e.to_string())),
        }
        evicted += 1;
        freed += row.size_bytes;
    }
    if skipped_unsafe > 0 {
        Ok(format!(
            "evicted {evicted} slots, freed {freed} bytes ({skipped_unsafe} had an unsafe handle and were not deleted from disk)"
        ))
    } else {
        Ok(format!("evicted {evicted} slots, freed {freed} bytes"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CacheHandle, CacheKey, CacheMeta, KvCacheStore, LlamaServerSlotStore};

    fn a_store(dir: &std::path::Path, max_bytes: u64) -> LlamaServerSlotStore {
        LlamaServerSlotStore::open(dir, "http://127.0.0.1:0", max_bytes).unwrap()
    }

    #[tokio::test]
    async fn list_slots_reports_an_empty_store_and_then_a_recorded_one() {
        let dir = tempfile::tempdir().unwrap();
        let empty = list_slots(dir.path()).await.unwrap();
        assert!(empty.contains("no cached slots"), "got: {empty}");

        let store = a_store(dir.path(), 2_000);
        store
            .record(
                &CacheKey {
                    backend_id: "llama-server".into(),
                    model_id: "qwen3.5".into(),
                    build_hash: "b1".into(),
                    prefix_hash: "p1".into(),
                },
                CacheHandle::new("p1.slot"),
                CacheMeta {
                    size_bytes: 1234,
                    token_count: 100,
                },
            )
            .await
            .unwrap();

        let report = list_slots(dir.path()).await.unwrap();
        assert!(report.contains("qwen3.5"), "got: {report}");
        assert!(report.contains("1234"), "got: {report}");
    }

    #[tokio::test]
    async fn stats_reports_total_bytes_against_the_budget() {
        let dir = tempfile::tempdir().unwrap();
        let store = a_store(dir.path(), 1_000);
        store
            .record(
                &CacheKey {
                    backend_id: "llama-server".into(),
                    model_id: "qwen3.5".into(),
                    build_hash: "b1".into(),
                    prefix_hash: "p1".into(),
                },
                CacheHandle::new("p1.slot"),
                CacheMeta {
                    size_bytes: 300,
                    token_count: 10,
                },
            )
            .await
            .unwrap();

        let report = stats(dir.path(), 1_000).await.unwrap();
        assert!(report.contains("300"), "got: {report}");
        assert!(report.contains("1000"), "got: {report}");
    }

    #[tokio::test]
    async fn prune_dry_run_reports_without_deleting() {
        let dir = tempfile::tempdir().unwrap();
        let store = a_store(dir.path(), 1_000);
        let key = CacheKey {
            backend_id: "llama-server".into(),
            model_id: "qwen3.5".into(),
            build_hash: "b1".into(),
            prefix_hash: "p1".into(),
        };
        store
            .record(
                &key,
                CacheHandle::new("p1.slot"),
                CacheMeta {
                    size_bytes: 300,
                    token_count: 10,
                },
            )
            .await
            .unwrap();

        let report = prune(dir.path(), 0, true).await.unwrap();
        assert!(report.contains("dry run"), "got: {report}");
        // Re-open and confirm nothing was actually deleted.
        let reopened = a_store(dir.path(), 1_000);
        assert!(reopened.find(&key).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn prune_for_real_evicts_down_to_the_target() {
        let dir = tempfile::tempdir().unwrap();
        let store = a_store(dir.path(), 1_000);
        let key = CacheKey {
            backend_id: "llama-server".into(),
            model_id: "qwen3.5".into(),
            build_hash: "b1".into(),
            prefix_hash: "p1".into(),
        };
        store
            .record(
                &key,
                CacheHandle::new("p1.slot"),
                CacheMeta {
                    size_bytes: 300,
                    token_count: 10,
                },
            )
            .await
            .unwrap();

        let report = prune(dir.path(), 0, false).await.unwrap();
        assert!(report.contains("freed"), "got: {report}");
        let reopened = a_store(dir.path(), 1_000);
        assert!(reopened.find(&key).await.unwrap().is_none());
    }
}
