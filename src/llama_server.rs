use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
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
///
/// `max_bytes` is per-instance, not stored in the (shared, cross-process)
/// manifest -- when several processes `open` the same `store_path` with
/// different budgets, each one's own eviction only ever enforces its own
/// `max_bytes` against the shared total, so in steady state the smallest
/// configured budget wins. See the README's own note on this.
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

/// Process-local counter folded into `unique_slot_filename`'s nonce --
/// only needs to be unique across concurrent saves *within this process*;
/// `std::process::id()` (also folded in) is what disambiguates it from
/// every other process sharing the same store.
static SAVE_NONCE: AtomicU64 = AtomicU64::new(0);

/// A fresh filename for a **new** save of `key`'s slot -- unlike
/// `slot_filename` (deterministic, used to locate a key's *current* file
/// for reads and eviction), every call here returns a different name, even
/// for the same key. This is what closes two related failure modes a
/// purely deterministic per-key filename used to have: two concurrent
/// savers of the same key racing to write the exact same physical file
/// (whoever finishes last silently wins, corrupting or truncating
/// whichever save "lost"), and an eviction of the *existing* row for this
/// key -- still pointing at that same deterministic name -- deleting the
/// file a fresh save just finished writing to it, or a failed fresh save's
/// own error-cleanup deleting the existing, still-valid row's file because
/// they happened to share a name. See `save_from_slot`'s doc comment for
/// how the resulting unique file is then swapped in and the previous one
/// cleaned up only after the new row has committed.
///
/// Combines this process's pid with a per-process monotonic counter: unique
/// within a single process's lifetime (the counter) and, assuming no pid
/// reuse, across every process concurrently sharing one store (the pid).
/// This is collision avoidance, not a security boundary, so that's enough
/// without pulling in a `rand` dependency.
fn unique_slot_filename(key: &CacheKey) -> String {
    let base = slot_filename(key);
    let base = base.strip_suffix(".slot").unwrap_or(&base);
    let nonce = SAVE_NONCE.fetch_add(1, Ordering::Relaxed);
    format!("{base}-{}-{nonce:x}.slot", std::process::id())
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
    /// back out of — `record_returning_previous` (called below) re-checks
    /// the same budget too, so this stays correct even if called with a
    /// `meta` that wasn't actually pre-checked by some future caller.
    ///
    /// Every call writes to its own fresh filename (`unique_slot_filename`,
    /// never `key`'s deterministic `slot_filename`) specifically so this
    /// never collides with whatever file `key`'s *existing* row (if any)
    /// still points at -- on success, that existing file is only removed
    /// *after* the new row has committed below, never before (a concurrent
    /// restore could still be reading it) and never by the error path (see
    /// below), which only ever touches the file *this* attempt just wrote.
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
        let filename = unique_slot_filename(key);
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
            Ok(fs_meta) => CacheMeta {
                size_bytes: fs_meta.len(),
                token_count: meta.token_count,
            },
            Err(_) => meta,
        };

        let handle = CacheHandle::new(filename.clone());
        let previous_handle = match self.record_returning_previous(key, handle, real_meta).await {
            Ok(previous) => previous,
            Err(err) => {
                // record() rejected the *real* size as over budget even
                // though the caller's own estimate passed the pre-check
                // above -- llama-server already wrote the file, so clean
                // up the orphan rather than leave a file on disk the
                // manifest never learns about. This is always exactly the
                // file *this* attempt just wrote (`filename` is unique per
                // call), so it can never reach out and delete a different,
                // still-valid row's file.
                let _ = std::fs::remove_file(&path);
                return Err(err);
            }
        };

        // The new row is committed under `key` now -- only now is it safe
        // to remove whatever file the row we just superseded pointed at.
        // Doing this any earlier would risk deleting a file a concurrent
        // restore is still reading; doing it via the error path above
        // would risk deleting it on a failed *subsequent* save instead of
        // this one's own file. A `NotFound` here just means something else
        // (another process's own save of this key, or a prior eviction)
        // already cleaned it up.
        if let Some(previous_handle) = previous_handle
            && previous_handle.as_str() != filename
            && previous_handle.is_safe_filename()
        {
            let previous_path = self.slots_dir.join(previous_handle.as_str());
            match std::fs::remove_file(&previous_path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        path = %previous_path.display(),
                        "kvcache could not remove a superseded slot file after a successful \
                         save; the new row has already committed, so this is now an orphan"
                    );
                }
            }
        }
        Ok(())
    }

    /// Same as the `KvCacheStore::record` trait method, but also returns
    /// whatever handle `key` pointed at immediately before this call (see
    /// `Manifest::insert_returning_previous`). Kept as an inherent method
    /// (not part of the public trait) since only `save_from_slot` needs
    /// the previous handle -- every other caller of `record` just needs
    /// the plain `Result<(), KvCacheError>`.
    async fn record_returning_previous(
        &self,
        key: &CacheKey,
        handle: CacheHandle,
        meta: CacheMeta,
    ) -> Result<Option<CacheHandle>, KvCacheError> {
        if meta.size_bytes > self.max_bytes {
            return Err(KvCacheError::SlotExceedsBudget {
                size_bytes: meta.size_bytes,
                max_bytes: self.max_bytes,
            });
        }
        let previous = self
            .manifest
            .insert_returning_previous(key, &handle, meta)
            .await?;
        self.evict_to_budget().await?;
        Ok(previous)
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
        self.record_returning_previous(key, handle, meta).await?;
        Ok(())
    }

    /// Note on multi-process deployments: `evict_and_remove` (via the
    /// shared sqlite manifest) atomically removes manifest rows *before*
    /// this loop deletes their backing files, so once a row is selected
    /// for eviction here it is gone from the index regardless of whether
    /// its file deletion below actually succeeds. A `remove_file` error
    /// that isn't a plain `NotFound` is therefore logged and skipped
    /// (its file becomes an orphan on disk, not counted as freed) rather
    /// than aborting the loop -- an early return here used to leave every
    /// *later* row's file un-deleted too (even when perfectly removable),
    /// and surfaced as an `Err` out of whatever unrelated `record()` call
    /// happened to trigger this eviction pass.
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
                Ok(()) => {
                    report.evicted_count += 1;
                    report.bytes_freed += row.size_bytes;
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    report.evicted_count += 1;
                    report.bytes_freed += row.size_bytes;
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        path = %path.display(),
                        "kvcache eviction could not remove a slot file after its manifest row \
                         was already removed; continuing with any remaining evictions instead \
                         of aborting (the row no longer counts toward budget, but its file is \
                         now orphaned on disk)"
                    );
                }
            }
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conformance::assert_conformance;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use wiremock::matchers::{method, path_regex};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    /// Test-only `Respond` that mimics what the real llama-server does on a
    /// successful `/slots?action=save`: it actually writes `size_bytes` of
    /// real content to disk, under whatever filename the request asked it
    /// to save as. Needed because `save_from_slot` now picks a fresh,
    /// unique filename on every call (see `unique_slot_filename`) -- a test
    /// can no longer pre-compute the exact path it will use and pre-stage a
    /// file there ahead of time; it has to discover the real filename from
    /// the request, exactly as the real server would receive it. Records
    /// every filename it was asked to save as into `observed`, in call
    /// order, so the test can look up the real on-disk path after the fact.
    struct WriteRealSlotFile {
        dir: PathBuf,
        /// Size (in bytes) to write for the Nth call (0-indexed); the last
        /// entry is reused for any call beyond the list's length.
        sizes: Vec<usize>,
        call_index: AtomicUsize,
        observed: Arc<Mutex<Vec<String>>>,
    }

    impl WriteRealSlotFile {
        fn new(dir: PathBuf, size_bytes: usize) -> (Self, Arc<Mutex<Vec<String>>>) {
            Self::with_sizes(dir, vec![size_bytes])
        }

        fn with_sizes(dir: PathBuf, sizes: Vec<usize>) -> (Self, Arc<Mutex<Vec<String>>>) {
            let observed = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    dir,
                    sizes,
                    call_index: AtomicUsize::new(0),
                    observed: observed.clone(),
                },
                observed,
            )
        }
    }

    impl Respond for WriteRealSlotFile {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let body: serde_json::Value = request.body_json().unwrap();
            let filename = body["filename"].as_str().unwrap().to_string();
            let idx = self.call_index.fetch_add(1, Ordering::SeqCst);
            let size = *self
                .sizes
                .get(idx)
                .unwrap_or_else(|| self.sizes.last().unwrap());
            std::fs::write(self.dir.join(&filename), vec![0u8; size]).unwrap();
            self.observed.lock().unwrap().push(filename);
            ResponseTemplate::new(200)
        }
    }

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
        let dir = tempfile::tempdir().unwrap();
        let slots_dir = dir.path().join("slots");
        // The mock stands in for llama-server actually writing the real
        // 5,000-byte file to disk, under whatever filename the request
        // asks for (save_from_slot no longer uses a predictable,
        // precomputable filename -- see unique_slot_filename).
        let (responder, _observed) = WriteRealSlotFile::new(slots_dir, 5_000);
        Mock::given(method("POST"))
            .and(path_regex(r"^/slots/0$"))
            .respond_with(responder)
            .mount(&server)
            .await;

        // Budget is exactly key1's real size -- key1 alone fits, but
        // key1 + key2 together don't, forcing eviction once key2 lands.
        let store = LlamaServerSlotStore::open(dir.path(), server.uri(), 5_000).unwrap();

        let k1 = key("p1");

        // Caller passes a wildly wrong placeholder (1 byte, matching the
        // real-world caller this bug was found in). If the fix works, the
        // manifest records the real 5,000-byte size instead.
        store
            .save_from_slot(
                &k1,
                0,
                CacheMeta {
                    size_bytes: 1,
                    token_count: 1,
                },
            )
            .await
            .unwrap();

        let k2 = key("p2");
        std::fs::write(store.slot_path(&k2), vec![0u8; 100]).unwrap();
        store
            .record(
                &k2,
                CacheHandle::new(slot_filename(&k2)),
                CacheMeta {
                    size_bytes: 100,
                    token_count: 1,
                },
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
        assert!(
            store.find(&k2).await.unwrap().is_some(),
            "key2 must still be present"
        );
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
            .save_from_slot(
                &key("p1"),
                0,
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
    async fn save_from_slot_deletes_the_orphaned_file_when_the_real_size_exceeds_budget() {
        let server = MockServer::start().await;
        let dir = tempfile::tempdir().unwrap();
        let slots_dir = dir.path().join("slots");
        // The mock writes a real 5,000-byte file under whatever filename
        // the request names -- standing in for llama-server already
        // having written it by the time the save "completes".
        let (responder, observed) = WriteRealSlotFile::new(slots_dir.clone(), 5_000);
        Mock::given(method("POST"))
            .and(path_regex(r"^/slots/0$"))
            .respond_with(responder)
            .mount(&server)
            .await;

        // max_bytes is large enough that the caller's placeholder (1 byte)
        // passes the pre-check, but smaller than the real file that turns
        // out to exist on disk once the save "completes".
        let store = LlamaServerSlotStore::open(dir.path(), server.uri(), 100).unwrap();

        let k1 = key("p1");

        let err = store
            .save_from_slot(
                &k1,
                0,
                CacheMeta {
                    size_bytes: 1,
                    token_count: 1,
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(err, KvCacheError::SlotExceedsBudget { .. }));

        let filename = observed.lock().unwrap()[0].clone();
        let path = slots_dir.join(&filename);
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

        let started = std::time::Instant::now();
        let outcome = tokio::time::timeout(
            Duration::from_millis(1_500),
            store.restore_into_slot(&key("p1"), 0),
        )
        .await;

        match outcome {
            Ok(inner) => {
                // A timed-out HTTP call is a caught, expected failure mode
                // inside restore_into_slot (same as any other backend
                // error) -- it falls back to `Ok(false)`, not a raw `Err`;
                // see its own doc comment. The real signal that the
                // client's own timeout (not this test's outer safety net)
                // is what fired is the elapsed time below.
                assert!(
                    !inner.unwrap(),
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

    /// Regression test for a real reliability bug: `slot_filename` is
    /// deterministic per `CacheKey`, so a naive `save_from_slot` that used
    /// it directly would pick the exact same on-disk filename for every
    /// save of the same key -- re-saving a key overwrites, in place, the
    /// very file the manifest's *existing* row for that key still points
    /// at while the save is in flight, and (see the next test) a
    /// concurrent eviction of that pre-existing row can delete the file
    /// out from under a save that just finished writing to it. The fix is
    /// `unique_slot_filename`: every save gets its own filename, and the
    /// previous file is only removed after the new row has committed.
    #[tokio::test]
    async fn resaving_the_same_key_never_reuses_the_previous_physical_filename() {
        let server = MockServer::start().await;
        let dir = tempfile::tempdir().unwrap();
        let slots_dir = dir.path().join("slots");
        let (responder, observed) = WriteRealSlotFile::new(slots_dir.clone(), 10);
        Mock::given(method("POST"))
            .and(path_regex(r"^/slots/0$"))
            .respond_with(responder)
            .mount(&server)
            .await;

        let store = LlamaServerSlotStore::open(dir.path(), server.uri(), 1_000).unwrap();
        let k = key("p1");

        store
            .save_from_slot(
                &k,
                0,
                CacheMeta {
                    size_bytes: 10,
                    token_count: 1,
                },
            )
            .await
            .unwrap();
        let first_filename = observed.lock().unwrap()[0].clone();
        let first_path = slots_dir.join(&first_filename);
        assert!(first_path.exists(), "the first save's file must exist");

        store
            .save_from_slot(
                &k,
                0,
                CacheMeta {
                    size_bytes: 10,
                    token_count: 1,
                },
            )
            .await
            .unwrap();
        let second_filename = observed.lock().unwrap()[1].clone();
        assert_ne!(
            first_filename, second_filename,
            "two saves of the same key must never reuse the same physical filename"
        );
        let second_path = slots_dir.join(&second_filename);
        assert!(second_path.exists(), "the new file must exist");
        assert!(
            !first_path.exists(),
            "the superseded file must be cleaned up once the new row has committed"
        );

        // The manifest's current row for this key must always point at a
        // file that actually exists -- never the deleted, superseded one.
        let handle = store.find(&k).await.unwrap().unwrap();
        assert_eq!(handle.as_str(), second_filename);
        assert!(slots_dir.join(handle.as_str()).exists());
    }

    /// Regression test: the save error path must only ever remove the
    /// file it *just* wrote for *this* attempt, never a different,
    /// already-committed row's file. Under the old deterministic-filename
    /// scheme, a second (failing) save of the same key wrote to the exact
    /// same path as the first (still-valid) save, so the error path's
    /// cleanup deleted the first row's file out from under it -- leaving a
    /// manifest row pointing at nothing.
    #[tokio::test]
    async fn save_from_slot_error_path_never_deletes_an_earlier_committed_rows_file() {
        let server = MockServer::start().await;
        let dir = tempfile::tempdir().unwrap();
        let slots_dir = dir.path().join("slots");
        // First call writes 10 real bytes (fits the budget below); the
        // second writes 500, blowing it.
        let (responder, observed) = WriteRealSlotFile::with_sizes(slots_dir.clone(), vec![10, 500]);
        Mock::given(method("POST"))
            .and(path_regex(r"^/slots/0$"))
            .respond_with(responder)
            .mount(&server)
            .await;

        let store = LlamaServerSlotStore::open(dir.path(), server.uri(), 10).unwrap();
        let k = key("p1");

        store
            .save_from_slot(
                &k,
                0,
                CacheMeta {
                    size_bytes: 10,
                    token_count: 1,
                },
            )
            .await
            .unwrap();
        let first_filename = observed.lock().unwrap()[0].clone();
        let first_path = slots_dir.join(&first_filename);
        assert!(first_path.exists());

        let err = store
            .save_from_slot(
                &k,
                0,
                CacheMeta {
                    size_bytes: 1,
                    token_count: 1,
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(err, KvCacheError::SlotExceedsBudget { .. }));

        let second_filename = observed.lock().unwrap()[1].clone();
        assert_ne!(
            first_filename, second_filename,
            "each save attempt must use its own filename"
        );
        let second_path = slots_dir.join(&second_filename);
        assert!(
            !second_path.exists(),
            "the rejected attempt's own oversized file must be cleaned up"
        );
        assert!(
            first_path.exists(),
            "an earlier, still-recorded row's file must never be deleted by a later failed \
             save's error path"
        );
        assert_eq!(
            store.find(&k).await.unwrap().unwrap().as_str(),
            first_filename,
            "the manifest must still point at the earlier, still-valid save"
        );
    }

    /// Regression test: `evict_to_budget` must keep going even if it can't
    /// remove one row's backing file -- `evict_and_remove` has already
    /// atomically deleted the manifest rows by the time file deletion
    /// runs, so aborting partway through used to leave later rows'
    /// (perfectly removable) files un-deleted and orphaned, and also
    /// surfaced as an `Err` out of the `record()` call that triggered the
    /// eviction in the first place -- failing an unrelated save over a
    /// single bad file elsewhere in the store. A directory stands in for
    /// "a file `remove_file` can't delete" (fails with a real, non-
    /// `NotFound` error on Linux).
    #[tokio::test]
    async fn evict_to_budget_continues_past_a_file_it_cannot_remove() {
        let dir = tempfile::tempdir().unwrap();
        let store = LlamaServerSlotStore::open(dir.path(), "http://127.0.0.1:0", 100).unwrap();

        let k1 = key("p1");
        let k2 = key("p2");
        let k3 = key("p3");

        // k1's "file" is actually a directory -- remove_file on it fails
        // with something other than NotFound.
        std::fs::create_dir_all(store.slot_path(&k1)).unwrap();
        store
            .record(
                &k1,
                CacheHandle::new(slot_filename(&k1)),
                CacheMeta {
                    size_bytes: 100,
                    token_count: 1,
                },
            )
            .await
            .unwrap();

        let p2_path = store.slot_path(&k2);
        std::fs::write(&p2_path, b"p2 bytes").unwrap();
        store
            .record(
                &k2,
                CacheHandle::new(slot_filename(&k2)),
                CacheMeta {
                    size_bytes: 100,
                    token_count: 1,
                },
            )
            .await
            .unwrap();

        // Recording k3 forces eviction of both k1 and k2 (LRU order).
        // Before the fix, failing to remove k1's (directory) file aborted
        // eviction entirely and this call itself returned Err.
        store
            .record(
                &k3,
                CacheHandle::new(slot_filename(&k3)),
                CacheMeta {
                    size_bytes: 100,
                    token_count: 1,
                },
            )
            .await
            .unwrap();

        assert!(store.find(&k1).await.unwrap().is_none());
        assert!(store.find(&k2).await.unwrap().is_none());
        assert!(store.find(&k3).await.unwrap().is_some());

        // k2 comes after k1 in eviction order -- a return-early bug would
        // have aborted before ever attempting k2's (perfectly removable)
        // file.
        assert!(
            !p2_path.exists(),
            "k2's file must still be deleted even though k1 (evicted first) couldn't be"
        );
    }
}
