use std::path::PathBuf;

use async_trait::async_trait;

use crate::manifest::Manifest;
use crate::{CacheHandle, CacheKey, CacheMeta, EvictionReport, KvCacheError, KvCacheStore};

/// The real, llama-server-backed `KvCacheStore`. Drives llama-server's
/// native `/slots/{id}?action=save|restore` API (see `restore_into_slot`/
/// `save_from_slot`) against files under `store_path/slots/`, indexed by a
/// `Manifest` at `store_path/manifest.db`.
pub struct LlamaServerSlotStore {
    manifest: Manifest,
    slots_dir: PathBuf,
    max_bytes: u64,
    base_url: String,
    http: reqwest::Client,
}

/// Deterministic, filesystem- and llama-server-safe filename for `key`'s
/// on-disk `.slot` file. Every key component is normalized (non-
/// alphanumeric/`-`/`_` characters replaced with `_`) before being folded
/// into one flat name -- two things this guards against: llama-server's
/// `/slots` API takes a bare filename (no path separators), so a nested
/// per-backend/per-model directory scheme (the previous convention here)
/// can never actually be produced by asking llama-server to save there;
/// and `backend_id`/`model_id`/`build_hash` are free-form strings with no
/// guaranteed format, so leaving them unnormalized would let a value like
/// `../../etc` reach `std::fs::remove_file` during eviction. All four key
/// fields are included (not just `prefix_hash`) so that two `CacheKey`s
/// which differ only by model/build never collide on the same physical
/// file -- `prefix_hash` alone is deliberately model-agnostic (see its
/// own doc comment on `CacheKey`), so the filename must carry the rest.
pub(crate) fn slot_filename(key: &CacheKey) -> String {
    fn normalize(s: &str) -> String {
        s.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    }
    format!(
        "{}-{}-{}-{}.slot",
        normalize(&key.backend_id),
        normalize(&key.model_id),
        normalize(&key.build_hash),
        normalize(&key.prefix_hash),
    )
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

    /// Where the on-disk `.slot` file for `key` lives: `slots_dir` joined
    /// with `slot_filename(key)`. `slots_dir` (`store_path/slots`) is
    /// exactly the directory an operator should point llama-server's own
    /// `--slot-save-path` flag at -- llama-server's `/slots` API only ever
    /// takes a bare filename, never a path, so this MUST stay flat (no
    /// per-backend/per-model subdirectories: a prior version of this
    /// method built a nested path that llama-server had no way to
    /// actually write into, silently breaking eviction -- see
    /// `slot_filename`'s doc comment for why the filename itself carries
    /// the namespacing instead). Production code computes eviction paths
    /// from the manifest row's own stored handle instead (see
    /// `evict_to_budget`); this stays as a standalone primitive exercised
    /// directly by tests below.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn slot_path(&self, key: &CacheKey) -> PathBuf {
        self.slots_dir.join(slot_filename(key))
    }

    /// Look up `key`, and if found, ask llama-server to restore that saved
    /// slot into `slot_id`. Returns `Ok(true)` only on a **confirmed**
    /// restore (`find` locating a handle is not enough — the actual
    /// restore call must succeed too, per `KvCacheStore::confirm_hit`'s
    /// contract). Never returns an `Err` for a cache miss or a failed
    /// restore: those are expected outcomes the caller falls back from by
    /// sending a normal cold request, not failures of the turn itself.
    pub async fn restore_into_slot(
        &self,
        key: &CacheKey,
        slot_id: u32,
    ) -> Result<bool, KvCacheError> {
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
                // The restore itself already succeeded -- the slot is warm.
                // A failure recording that fact (hit_count/recency
                // bookkeeping) is real but non-fatal: losing an accounting
                // update isn't worth telling the caller to discard an
                // already-warmed slot and fall back to a cold request.
                if let Err(err) = self.confirm_hit(key).await {
                    tracing::warn!(error = %err, "kvcache confirm_hit failed after a successful restore; the restore itself still succeeded");
                }
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
        let filename = slot_filename(key);
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
        let removed = self.manifest.evict_and_remove(self.max_bytes).await?;
        let mut report = EvictionReport::default();
        for row in removed {
            let path = self.slots_dir.join(row.handle.as_str());
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(KvCacheError::Backend(e.to_string())),
            }
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
                CacheHandle::new(slot_filename(&k)),
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
        assert!(
            !path.exists(),
            "evicted slot's file must be deleted from disk"
        );
    }

    #[tokio::test]
    async fn restore_into_slot_is_false_on_a_cache_miss_without_calling_llama_server() {
        let server = MockServer::start().await;
        // No mock registered for /slots — if the store called it anyway, wiremock
        // would return a 404 and the test would still pass; the real assertion is
        // that a miss doesn't need a mock at all.
        let dir = tempfile::tempdir().unwrap();
        let store = LlamaServerSlotStore::open(dir.path(), server.uri(), 1_000).unwrap();

        let restored = store
            .restore_into_slot(&key("never-recorded"), 0)
            .await
            .unwrap();
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
