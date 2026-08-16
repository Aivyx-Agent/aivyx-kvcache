use std::path::PathBuf;

use async_trait::async_trait;

use crate::manifest::Manifest;
use crate::{CacheHandle, CacheKey, CacheMeta, EvictionReport, KvCacheError, KvCacheStore};

/// The real, llama-server-backed `KvCacheStore`. Drives llama-server's
/// native `/slots/{id}?action=save|restore` API (see `restore_into_slot`/
/// `save_from_slot`, added in a later commit) against files under
/// `store_path/slots/`, indexed by a `Manifest` at `store_path/manifest.db`.
pub struct LlamaServerSlotStore {
    manifest: Manifest,
    slots_dir: PathBuf,
    max_bytes: u64,
}

impl LlamaServerSlotStore {
    pub fn open(store_path: impl Into<PathBuf>, max_bytes: u64) -> Result<Self, KvCacheError> {
        let store_path = store_path.into();
        let manifest = Manifest::open(&store_path.join("manifest.db"))?;
        let slots_dir = store_path.join("slots");
        std::fs::create_dir_all(&slots_dir).map_err(|e| KvCacheError::Backend(e.to_string()))?;
        Ok(Self {
            manifest,
            slots_dir,
            max_bytes,
        })
    }

    /// Where the on-disk `.slot` file for `key` lives, by convention:
    /// `slots/<backend_id>/<model_id>/<prefix_hash>.slot`. llama-server
    /// itself is what actually writes/reads the file's contents (via its
    /// `/slots` save/restore API) — this store only tracks and indexes it.
    pub(crate) fn slot_path(&self, key: &CacheKey) -> PathBuf {
        self.slots_dir
            .join(&key.backend_id)
            .join(&key.model_id)
            .join(format!("{}.slot", key.prefix_hash))
    }
}

#[async_trait]
impl KvCacheStore for LlamaServerSlotStore {
    async fn find(&self, key: &CacheKey) -> Result<Option<CacheHandle>, KvCacheError> {
        Ok(self.manifest.find(key).await?.map(|row| row.handle))
    }

    async fn confirm_hit(&self, key: &CacheKey) -> Result<(), KvCacheError> {
        self.manifest.confirm_hit(key).await
    }

    async fn record(
        &self,
        key: &CacheKey,
        handle: CacheHandle,
        meta: CacheMeta,
    ) -> Result<(), KvCacheError> {
        if meta.size_bytes > self.max_bytes {
            return Err(KvCacheError::SlotExceedsBudget {
                size_bytes: meta.size_bytes,
                max_bytes: self.max_bytes,
            });
        }
        self.manifest.insert(key, &handle, meta).await?;
        self.evict_to_budget().await?;
        Ok(())
    }

    async fn evict_to_budget(&self) -> Result<EvictionReport, KvCacheError> {
        let candidates = self.manifest.evict_candidates(self.max_bytes).await?;
        let mut report = EvictionReport::default();
        for row in candidates {
            let path = self.slot_path(&row.key);
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(KvCacheError::Backend(e.to_string())),
            }
            self.manifest.remove(&row.key).await?;
            report.evicted_count += 1;
            report.bytes_freed += row.size_bytes;
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conformance::assert_conformance;

    #[tokio::test]
    async fn satisfies_the_kv_cache_store_contract() {
        let dir = tempfile::tempdir().unwrap();
        let store = LlamaServerSlotStore::open(dir.path(), 1_000).unwrap();
        assert_conformance(&store, 1_000).await;
    }

    #[tokio::test]
    async fn eviction_deletes_the_backing_slot_file_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let store = LlamaServerSlotStore::open(dir.path(), 100).unwrap();

        let key = CacheKey {
            backend_id: "llama-server".into(),
            model_id: "qwen3.5".into(),
            build_hash: "b1".into(),
            prefix_hash: "p1".into(),
        };
        // Simulate what llama-server itself would have written on a real save.
        let path = store.slot_path(&key);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"fake kv cache bytes").unwrap();

        store
            .record(
                &key,
                CacheHandle::new(format!("{}.slot", key.prefix_hash)),
                CacheMeta {
                    size_bytes: 50,
                    token_count: 5,
                },
            )
            .await
            .unwrap();
        assert!(path.exists());

        // Push total over budget so eviction actually removes p1.
        let key2 = CacheKey {
            prefix_hash: "p2".into(),
            ..key.clone()
        };
        store
            .record(
                &key2,
                CacheHandle::new("p2.slot"),
                CacheMeta {
                    size_bytes: 100,
                    token_count: 5,
                },
            )
            .await
            .unwrap();

        assert!(store.find(&key).await.unwrap().is_none());
        assert!(!path.exists(), "evicted slot's file must be deleted from disk");
    }
}
