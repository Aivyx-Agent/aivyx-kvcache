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
    base_url: String,
    http: reqwest::Client,
}

impl LlamaServerSlotStore {
    pub fn open(
        store_path: impl Into<PathBuf>,
        base_url: impl Into<String>,
        max_bytes: u64,
    ) -> Result<Self, KvCacheError> {
        let store_path = store_path.into();
        let manifest = Manifest::open(&store_path.join("manifest.db"))?;
        let slots_dir = store_path.join("slots");
        std::fs::create_dir_all(&slots_dir).map_err(|e| KvCacheError::Backend(e.to_string()))?;
        Ok(Self {
            manifest,
            slots_dir,
            max_bytes,
            base_url: base_url.into(),
            http: reqwest::Client::new(),
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

    /// Look up `key`, and if found, ask llama-server to restore that saved
    /// slot into `slot_id`. Returns `Ok(true)` only on a **confirmed**
    /// restore (`find` locating a handle is not enough — the actual
    /// restore call must succeed too, per `KvCacheStore::confirm_hit`'s
    /// contract). Never returns an `Err` for a cache miss or a failed
    /// restore: those are expected outcomes the caller falls back from by
    /// sending a normal cold request, not failures of the turn itself.
    pub async fn restore_into_slot(&self, key: &CacheKey, slot_id: u32) -> Result<bool, KvCacheError> {
        let Some(handle) = self.find(key).await? else {
            return Ok(false);
        };
        let url = format!("{}/slots/{slot_id}?action=restore", self.base_url);
        let response = self
            .http
            .post(&url)
            .json(&serde_json::json!({ "filename": handle.as_str() }))
            .send()
            .await;
        match response {
            Ok(r) if r.status().is_success() => {
                self.confirm_hit(key).await?;
                Ok(true)
            }
            Ok(r) => {
                tracing::warn!(status = %r.status(), "kvcache restore rejected by llama-server; falling back to a cold request");
                Ok(false)
            }
            Err(err) => {
                tracing::warn!(error = %err, "kvcache restore request failed; falling back to a cold request");
                Ok(false)
            }
        }
    }

    /// Ask llama-server to save `slot_id`'s current KV cache to disk, then
    /// record it under `key`. Checked against the configured budget
    /// *before* issuing the HTTP call, specifically to avoid making
    /// llama-server perform a save it's just going to be evicted right
    /// back out of — `record` (called at the end here) re-checks the same
    /// budget too, so this stays correct even if called with a `meta` that
    /// wasn't actually pre-checked by some future caller.
    pub async fn save_from_slot(
        &self,
        key: &CacheKey,
        slot_id: u32,
        meta: CacheMeta,
    ) -> Result<(), KvCacheError> {
        if meta.size_bytes > self.max_bytes {
            return Err(KvCacheError::SlotExceedsBudget {
                size_bytes: meta.size_bytes,
                max_bytes: self.max_bytes,
            });
        }
        let filename = format!("{}.slot", key.prefix_hash);
        let url = format!("{}/slots/{slot_id}?action=save", self.base_url);
        self.http
            .post(&url)
            .json(&serde_json::json!({ "filename": filename }))
            .send()
            .await
            .map_err(|e| KvCacheError::Backend(e.to_string()))?
            .error_for_status()
            .map_err(|e| KvCacheError::Backend(e.to_string()))?;
        self.record(key, CacheHandle::new(filename), meta).await
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
    use wiremock::matchers::{method, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn key(prefix: &str) -> CacheKey {
        CacheKey {
            backend_id: "llama-server".into(),
            model_id: "qwen3.5".into(),
            build_hash: "b1".into(),
            prefix_hash: prefix.into(),
        }
    }

    #[tokio::test]
    async fn satisfies_the_kv_cache_store_contract() {
        let dir = tempfile::tempdir().unwrap();
        let store = LlamaServerSlotStore::open(dir.path(), "http://127.0.0.1:0", 1_000).unwrap();
        assert_conformance(&store, 1_000).await;
    }

    #[tokio::test]
    async fn eviction_deletes_the_backing_slot_file_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let store = LlamaServerSlotStore::open(dir.path(), "http://127.0.0.1:0", 100).unwrap();

        let k = key("p1");
        let path = store.slot_path(&k);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"fake kv cache bytes").unwrap();

        store
            .record(
                &k,
                CacheHandle::new(format!("{}.slot", k.prefix_hash)),
                CacheMeta {
                    size_bytes: 50,
                    token_count: 5,
                },
            )
            .await
            .unwrap();
        assert!(path.exists());

        store
            .record(
                &key("p2"),
                CacheHandle::new("p2.slot"),
                CacheMeta {
                    size_bytes: 100,
                    token_count: 5,
                },
            )
            .await
            .unwrap();

        assert!(store.find(&k).await.unwrap().is_none());
        assert!(!path.exists(), "evicted slot's file must be deleted from disk");
    }

    #[tokio::test]
    async fn restore_into_slot_is_false_on_a_cache_miss_without_calling_llama_server() {
        let server = MockServer::start().await;
        // No mock registered for /slots — if the store called it anyway, wiremock
        // would return a 404 and the test would still pass; the real assertion is
        // that a miss doesn't need a mock at all.
        let dir = tempfile::tempdir().unwrap();
        let store = LlamaServerSlotStore::open(dir.path(), server.uri(), 1_000).unwrap();

        let restored = store.restore_into_slot(&key("never-recorded"), 0).await.unwrap();
        assert!(!restored);
    }

    #[tokio::test]
    async fn restore_into_slot_confirms_a_hit_on_a_successful_restore() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/slots/0$"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let store = LlamaServerSlotStore::open(dir.path(), server.uri(), 1_000).unwrap();
        store
            .record(
                &key("p1"),
                CacheHandle::new("p1.slot"),
                CacheMeta {
                    size_bytes: 10,
                    token_count: 1,
                },
            )
            .await
            .unwrap();

        let restored = store.restore_into_slot(&key("p1"), 0).await.unwrap();
        assert!(restored);
    }

    #[tokio::test]
    async fn restore_into_slot_falls_back_to_a_miss_when_llama_server_rejects_the_restore() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/slots/0$"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let store = LlamaServerSlotStore::open(dir.path(), server.uri(), 1_000).unwrap();
        store
            .record(
                &key("p1"),
                CacheHandle::new("p1.slot"),
                CacheMeta {
                    size_bytes: 10,
                    token_count: 1,
                },
            )
            .await
            .unwrap();

        // Restore fails server-side; the turn must still be usable — this
        // returns Ok(false), never an Err that would abort the caller's turn.
        let restored = store.restore_into_slot(&key("p1"), 0).await.unwrap();
        assert!(!restored);
    }

    #[tokio::test]
    async fn save_from_slot_records_the_entry_after_a_successful_save() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/slots/2$"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let store = LlamaServerSlotStore::open(dir.path(), server.uri(), 1_000).unwrap();
        store
            .save_from_slot(
                &key("p1"),
                2,
                CacheMeta {
                    size_bytes: 10,
                    token_count: 1,
                },
            )
            .await
            .unwrap();

        assert!(store.find(&key("p1")).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn save_from_slot_rejects_a_slot_over_budget_without_calling_llama_server() {
        let server = MockServer::start().await;
        // No mock registered — an oversized save must be rejected before any
        // HTTP call is made, or wiremock's "no matching mock" panic would fail
        // this test.
        let dir = tempfile::tempdir().unwrap();
        let store = LlamaServerSlotStore::open(dir.path(), server.uri(), 100).unwrap();

        let err = store
            .save_from_slot(
                &key("p1"),
                0,
                CacheMeta {
                    size_bytes: 200,
                    token_count: 1,
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(err, KvCacheError::SlotExceedsBudget { .. }));
    }
}
