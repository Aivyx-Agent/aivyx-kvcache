//! Backend-agnostic KV-cache persistence/sharing layer for local LLM
//! serving. See `KvCacheStore`'s own doc comment (added in a later
//! commit) for the contract every implementation must satisfy.

use async_trait::async_trait;
use thiserror::Error;

/// Identifies one cached prefix. Two turns produce the same key only when
/// they'd prefill identically: same backend, same model/build, and an
/// identical **stable** prompt prefix — system prompt + tool defs + repo
/// map. Never per-turn conversation content: including that would thrash
/// the key on every message, defeating the point of caching the prefix at
/// all.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub backend_id: String,
    pub model_id: String,
    pub build_hash: String,
    pub prefix_hash: String,
}

/// Opaque handle to a saved cache entry. Implementations choose their own
/// internal shape (a slot filename, for `LlamaServerSlotStore`);
/// consumers must treat the inner value as opaque and round-trip it
/// through `new`/`as_str` only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheHandle(String);

impl CacheHandle {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Size/shape metadata recorded alongside a saved entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheMeta {
    pub size_bytes: u64,
    pub token_count: u64,
}

/// Result of an `evict_to_budget` pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EvictionReport {
    pub evicted_count: u64,
    pub bytes_freed: u64,
}

/// Errors a `KvCacheStore` can return.
#[derive(Debug, Error)]
pub enum KvCacheError {
    #[error("kvcache slot of {size_bytes} bytes exceeds the configured budget of {max_bytes} bytes")]
    SlotExceedsBudget { size_bytes: u64, max_bytes: u64 },
    #[error("kvcache backend error: {0}")]
    Backend(String),
}

/// The minimum KV-cache persistence surface. Implementations are expected
/// to be called from multiple tasks — and, for a shared on-disk store,
/// multiple separate OS processes — concurrently (`Send + Sync`).
///
/// Contract every implementation must satisfy (exercised by
/// `conformance::assert_conformance`, run against every implementation in
/// this crate):
/// - `find` never has side effects — it must not change `hit_count` or
///   recency, even when it locates a handle.
/// - `confirm_hit` on a key with no recorded entry is a no-op, not an
///   error.
/// - `record` followed immediately by `find` on the same key returns
///   `Some` with the handle just recorded.
/// - `record` of a slot whose `size_bytes` alone exceeds the store's
///   configured budget fails with `KvCacheError::SlotExceedsBudget`,
///   without partially recording it.
/// - `evict_to_budget` removes entries in least-recently-used order
///   (`confirm_hit` counts as use; `find` alone does not) until total
///   recorded bytes are at or under budget, and reports how many entries
///   and bytes it freed.
/// - A store with nothing recorded yet: `find` returns `None` for any
///   key, `evict_to_budget` is a no-op returning a zeroed report.
#[async_trait]
pub trait KvCacheStore: Send + Sync {
    async fn find(&self, key: &CacheKey) -> Result<Option<CacheHandle>, KvCacheError>;
    async fn confirm_hit(&self, key: &CacheKey) -> Result<(), KvCacheError>;
    async fn record(
        &self,
        key: &CacheKey,
        handle: CacheHandle,
        meta: CacheMeta,
    ) -> Result<(), KvCacheError>;
    async fn evict_to_budget(&self) -> Result<EvictionReport, KvCacheError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_equality_is_field_wise() {
        let a = CacheKey {
            backend_id: "llama-server".into(),
            model_id: "qwen3.5".into(),
            build_hash: "b1".into(),
            prefix_hash: "p1".into(),
        };
        let b = a.clone();
        assert_eq!(a, b);
        let mut c = a.clone();
        c.prefix_hash = "p2".into();
        assert_ne!(a, c);
    }

    #[test]
    fn cache_handle_round_trips_its_id() {
        let h = CacheHandle::new("some-filename.slot");
        assert_eq!(h.as_str(), "some-filename.slot");
    }

    #[test]
    fn slot_exceeds_budget_error_message_includes_both_sizes() {
        let err = KvCacheError::SlotExceedsBudget {
            size_bytes: 500,
            max_bytes: 100,
        };
        assert_eq!(
            err.to_string(),
            "kvcache slot of 500 bytes exceeds the configured budget of 100 bytes"
        );
    }
}
