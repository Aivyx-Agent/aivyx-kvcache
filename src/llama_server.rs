use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;

use crate::manifest::Manifest;
use crate::{CacheHandle, CacheKey, CacheMeta, EvictionReport, KvCacheError, KvCacheStore};

/// Per-request timeout for every call this store's `http` client makes
/// (`restore_into_slot`/`save_from_slot`, both driven through the same
/// client). Deliberately much larger than a lightweight JSON-probe
/// timeout (e.g. aivyx-coder's own 3s `/props` probe) -- a save/restore
/// transfers the actual KV-cache slot contents, which this crate's own
/// README documents as "headroom for multi-GB caches at long context
/// windows", so a bound tuned for a tiny status check would spuriously
/// fail large, otherwise-healthy transfers. Still finite: without this,
/// `ensure_kv_slot_checked_out` (aivyx-coder) runs this on the hot path
/// of every turn, before the turn's own cancellation check, so a wedged
/// llama-server connection previously hung every subsequent turn
/// indefinitely rather than falling back to an unpinned/cold session.
const SLOT_HTTP_TIMEOUT: Duration = Duration::from_secs(60);

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
/// alphanumeric/`_` characters, including `-`, replaced with `_`) before
/// being folded into one flat name -- two things this guards against:
/// llama-server's `/slots` API takes a bare filename (no path separators),
/// so a nested per-backend/per-model directory scheme (the previous
/// convention here) can never actually be produced by asking llama-server
/// to save there; and `backend_id`/`model_id`/`build_hash` are free-form
/// strings with no guaranteed format, so leaving them unnormalized would
/// let a value like `../../etc` reach `std::fs::remove_file` during
/// eviction. All four key fields are included (not just `prefix_hash`) so
/// that two `CacheKey`s which differ only by model/build never collide on
/// the same physical file -- `prefix_hash` alone is deliberately
/// model-agnostic (see its own doc comment on `CacheKey`), so the filename
/// must carry the rest.
///
/// `-` is normalized away (mapped to `_`) specifically because it's also
/// used below as the field delimiter -- otherwise a normalized field could
/// itself contain the delimiter, letting two different `CacheKey`s (e.g.
/// `{backend:"llama", model:"server-qwen"}` vs.
/// `{backend:"llama-server", model:"qwen"}`) produce the identical joined
/// text. And because normalization is inherently lossy (`.`/`:`/`/` all
/// collapse to the same `_`, so e.g. `"qwen3.5:32b"` and `"qwen3_5_32b"`
/// also collide), an unambiguous delimiter alone isn't enough -- the
/// trailing hash of the raw, un-normalized four-tuple is what actually
/// guarantees two distinct `CacheKey`s produce two distinct filenames; the
/// normalized fields are kept only for human readability.
pub(crate) fn slot_filename(key: &CacheKey) -> String {
    fn normalize(s: &str) -> String {
        s.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    }
    let hash = fnv1a(
        format!(
            "{}\u{0}{}\u{0}{}\u{0}{}",
            key.backend_id, key.model_id, key.build_hash, key.prefix_hash
        )
        .as_bytes(),
    );
    format!(
        "{}-{}-{}-{}-{:016x}.slot",
        normalize(&key.backend_id),
        normalize(&key.model_id),
        normalize(&key.build_hash),
        normalize(&key.prefix_hash),
        hash,
    )
}

/// Plain FNV-1a, used only to disambiguate `slot_filename`'s otherwise-
/// lossy normalization -- not required to match any other implementation
/// byte-for-byte (unlike e.g. aivyx-recall's own `fnv1a`, which has a
/// documented cross-repo stability requirement this one does not share).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

impl LlamaServerSlotStore {
    pub fn open(
        store_path: impl Into<PathBuf>,
        base_url: impl Into<String>,
        max_bytes: u64,
    ) -> Result<Self, KvCacheError> {
        Self::open_with_http_timeout(store_path, base_url, max_bytes, SLOT_HTTP_TIMEOUT)
    }

    /// Same as `open`, but with the `http` client's per-request timeout
    /// as an explicit parameter rather than the fixed `SLOT_HTTP_TIMEOUT`
    /// constant -- exists so tests can exercise the exact same
    /// client-construction path with a short timeout instead of waiting
    /// out the real (deliberately generous, for multi-GB transfers)
    /// production value. Not part of the public API: `open` is the only
    /// real caller outside this module's own tests.
    fn open_with_http_timeout(
        store_path: impl Into<PathBuf>,
        base_url: impl Into<String>,
        max_bytes: u64,
        http_timeout: Duration,
    ) -> Result<Self, KvCacheError> {
        let store_path = store_path.into();
        let manifest = Manifest::open(&store_path.join("manifest.db"))?;
        let slots_dir = store_path.join("slots");
        std::fs::create_dir_all(&slots_dir).map_err(|e| KvCacheError::Backend(e.to_string()))?;
        // `.build()` only fails on TLS/resolver init issues -- surfaced
        // as a real `Err` here (unlike the previous infallible
        // `reqwest::Client::new()`, itself just `.build().expect(..)`)
        // since a caller opening a store deserves to see that rather
        // than a panic.
        let http = reqwest::Client::builder()
            .timeout(http_timeout)
            .build()
            .map_err(|e| KvCacheError::Backend(e.to_string()))?;
        Ok(Self {
            manifest,
            slots_dir,
            max_bytes,
            base_url: base_url.into(),
            http,
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

        // llama-server just wrote the real file to disk -- measure it
        // rather than trust the caller's meta.size_bytes, which is often a
        // rough estimate or placeholder (see docs/superpowers/specs/
        // 2026-08-16-real-llama-server-e2e-test-design.md's own "known
        // limitation" note: a caller that under-reports size_bytes
        // silently defeats evict_to_budget's whole accounting). A stat
        // failure (e.g. a test harness that mocks the HTTP call without
        // writing a real file) falls back to the caller-supplied size
        // rather than failing the save outright.
        let path = self.slots_dir.join(&filename);
        let real_meta = match std::fs::metadata(&path) {
            Ok(fs_meta) => CacheMeta { size_bytes: fs_meta.len(), token_count: meta.token_count },
            Err(_) => meta,
        };

        let handle = CacheHandle::new(filename);
        if let Err(err) = self.record(key, handle, real_meta).await {
            // record() rejected the *real* size as over budget even though
            // the caller's own estimate passed the pre-check above --
            // llama-server already wrote the file, so clean up the orphan
            // rather than leave a file on disk the manifest never learns
            // about.
            let _ = std::fs::remove_file(&path);
            return Err(err);
        }
        Ok(())
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
            if !row.handle.is_safe_filename() {
                tracing::warn!(
                    handle = row.handle.as_str(),
                    "kvcache manifest row had an unsafe handle; skipping file deletion (row is already removed from the manifest)"
                );
                report.evicted_count += 1;
                report.bytes_freed += row.size_bytes;
                continue;
            }
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

    #[test]
    fn slot_filename_normalizes_path_traversal_attempts() {
        let key = CacheKey {
            backend_id: "../../etc".into(),
            model_id: "../../etc".into(),
            build_hash: "b1".into(),
            prefix_hash: "p1".into(),
        };
        let filename = slot_filename(&key);
        assert!(!filename.contains('/'));
        assert!(!filename.contains(".."));
        let path = std::path::Path::new(&filename);
        assert_eq!(path.components().count(), 1);
    }

    #[test]
    fn slot_filename_differs_for_keys_that_differ_only_in_model_id() {
        let a = CacheKey {
            backend_id: "llama-server".into(),
            model_id: "model-a".into(),
            build_hash: "b1".into(),
            prefix_hash: "p1".into(),
        };
        let b = CacheKey {
            model_id: "model-b".into(),
            ..a.clone()
        };
        assert_ne!(slot_filename(&a), slot_filename(&b));
    }

    #[test]
    fn slot_filename_differs_even_when_normalization_would_otherwise_collide() {
        // Normalization is lossy: '.' and '_' both map to '_', so
        // "qwen3.5" and "qwen3_5" produce byte-identical *normalized* text
        // even though they're distinct raw model ids. Without the raw-field
        // hash suffix, these two keys' filenames would be indistinguishable
        // -- confirmed by checking the normalized portion (everything before
        // the final `-<hash>.slot` segment) really is identical here, so
        // the hash is what's actually doing the disambiguating work below,
        // not an accidental difference elsewhere in the string.
        let a = CacheKey {
            backend_id: "llama-server".into(),
            model_id: "qwen3.5".into(),
            build_hash: "b1".into(),
            prefix_hash: "p1".into(),
        };
        let b = CacheKey {
            model_id: "qwen3_5".into(),
            ..a.clone()
        };
        let (filename_a, filename_b) = (slot_filename(&a), slot_filename(&b));
        let normalized_prefix = |f: &str| f.rsplit_once('-').unwrap().0.to_string();
        assert_eq!(
            normalized_prefix(&filename_a),
            normalized_prefix(&filename_b),
            "test setup is invalid: these two raw model ids must normalize \
             to identical text for this test to actually exercise the hash"
        );
        assert_ne!(filename_a, filename_b);
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

    #[tokio::test]
    async fn save_from_slot_uses_the_real_file_size_not_the_caller_supplied_placeholder() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/slots/0$"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        // Budget is exactly key1's real size -- key1 alone fits, but
        // key1 + key2 together don't, forcing eviction once key2 lands.
        let store = LlamaServerSlotStore::open(dir.path(), server.uri(), 5_000).unwrap();

        let k1 = key("p1");
        // Simulate llama-server having written the real file BEFORE our
        // mock HTTP response returns -- wiremock itself never writes real
        // files, so pre-stage it at the exact path save_from_slot expects.
        std::fs::write(store.slot_path(&k1), vec![0u8; 5_000]).unwrap();

        // Caller passes a wildly wrong placeholder (1 byte, matching the
        // real-world caller this bug was found in). If the fix works, the
        // manifest records the real 5,000-byte size instead.
        store
            .save_from_slot(&k1, 0, CacheMeta { size_bytes: 1, token_count: 1 })
            .await
            .unwrap();

        let k2 = key("p2");
        std::fs::write(store.slot_path(&k2), vec![0u8; 100]).unwrap();
        store
            .record(
                &k2,
                CacheHandle::new(slot_filename(&k2)),
                CacheMeta { size_bytes: 100, token_count: 1 },
            )
            .await
            .unwrap();

        assert!(
            store.find(&k1).await.unwrap().is_none(),
            "key1 must have been evicted once its REAL ~5,000-byte size pushed the store over \
             its 5,000-byte budget alongside key2's 100 bytes -- if the placeholder \
             size_bytes=1 were still being used, key1+key2 would total only 101 bytes and \
             neither would evict"
        );
        assert!(store.find(&k2).await.unwrap().is_some(), "key2 must still be present");
    }

    #[tokio::test]
    async fn save_from_slot_falls_back_to_the_caller_supplied_size_when_no_real_file_exists() {
        // No file written at the slot path -- exactly what every OTHER
        // wiremock-based test in this module already does (they assert
        // success without ever writing a real file). This proves the fix
        // doesn't break every pre-existing test's own implicit assumption.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/slots/0$"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let store = LlamaServerSlotStore::open(dir.path(), server.uri(), 1_000).unwrap();

        store
            .save_from_slot(&key("p1"), 0, CacheMeta { size_bytes: 10, token_count: 1 })
            .await
            .unwrap();

        assert!(store.find(&key("p1")).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn save_from_slot_deletes_the_orphaned_file_when_the_real_size_exceeds_budget() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/slots/0$"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        // max_bytes is large enough that the caller's placeholder (1 byte)
        // passes the pre-check, but smaller than the real file that turns
        // out to exist on disk once the save "completes".
        let store = LlamaServerSlotStore::open(dir.path(), server.uri(), 100).unwrap();

        let k1 = key("p1");
        let path = store.slot_path(&k1);
        std::fs::write(&path, vec![0u8; 5_000]).unwrap();

        let err = store
            .save_from_slot(&k1, 0, CacheMeta { size_bytes: 1, token_count: 1 })
            .await
            .unwrap_err();
        assert!(matches!(err, KvCacheError::SlotExceedsBudget { .. }));
        assert!(
            !path.exists(),
            "the orphaned file llama-server already wrote must be cleaned up when the real \
             size turns out to exceed budget"
        );
        assert!(
            store.find(&k1).await.unwrap().is_none(),
            "no manifest entry should exist for a save that was ultimately rejected"
        );
    }

    /// Regression test for the backlog item closing this store's own
    /// unbounded-hang class: `http` previously had no per-request timeout
    /// at all (`reqwest::Client::new()`), so a wedged llama-server
    /// connection hung `restore_into_slot`/`save_from_slot` forever --
    /// and since aivyx-coder's `ensure_kv_slot_checked_out` runs this on
    /// the hot path of every turn, before that turn's own cancellation
    /// check, a single hang there wedged every subsequent turn too. Uses
    /// `open_with_http_timeout` (not the real `SLOT_HTTP_TIMEOUT`, which
    /// is deliberately generous for multi-GB real transfers) so this
    /// stays fast: the mocked restore response delays for 2s, well past
    /// the short 150ms timeout under test, and the outer 1.5s
    /// `tokio::time::timeout` exists only as a safety bound so a
    /// regression here fails this test loudly instead of hanging the
    /// suite for the full 2s mock delay.
    #[tokio::test]
    async fn restore_into_slot_does_not_hang_against_an_unresponsive_server() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/slots/0$"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(2)))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let store = LlamaServerSlotStore::open_with_http_timeout(
            dir.path(),
            server.uri(),
            1_000,
            Duration::from_millis(150),
        )
        .unwrap();
        store
            .record(&key("p1"), CacheHandle::new("p1.slot"), CacheMeta { size_bytes: 10, token_count: 1 })
            .await
            .unwrap();

        let started = std::time::Instant::now();
        let outcome =
            tokio::time::timeout(Duration::from_millis(1_500), store.restore_into_slot(&key("p1"), 0))
                .await;

        match outcome {
            Ok(inner) => {
                // A timed-out HTTP call is a caught, expected failure mode
                // inside restore_into_slot (same as any other backend
                // error) -- it falls back to `Ok(false)`, not a raw `Err`;
                // see its own doc comment. The real signal that the
                // client's own timeout (not this test's outer safety net)
                // is what fired is the elapsed time below.
                assert_eq!(
                    inner.unwrap(),
                    false,
                    "a timed-out restore must fall back to a miss, not a confirmed hit"
                );
                assert!(
                    started.elapsed() < Duration::from_millis(1_000),
                    "the client's own 150ms timeout should have fired well before this outer \
                     safety bound; took {:?}",
                    started.elapsed()
                );
            }
            Err(_) => panic!(
                "the client had no working timeout of its own -- the outer 1.5s safety bound \
                 fired instead of the configured 150ms timeout"
            ),
        }
    }
}
