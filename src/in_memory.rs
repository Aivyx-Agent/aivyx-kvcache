use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::{CacheHandle, CacheKey, CacheMeta, EvictionReport, KvCacheError, KvCacheStore};

struct Entry {
    handle: CacheHandle,
    size_bytes: u64,
    last_used_at: u64,
}

/// Deterministic in-process fake, no persistence. For this crate's own
/// tests and for consumers that don't want real filesystem/sqlite I/O in
/// theirs.
pub struct InMemoryKvCacheStore {
    max_bytes: u64,
    entries: Mutex<HashMap<CacheKey, Entry>>,
    clock: Mutex<u64>,
}

impl InMemoryKvCacheStore {
    pub fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            entries: Mutex::new(HashMap::new()),
            clock: Mutex::new(0),
        }
    }

    /// A monotonically increasing logical clock, used instead of wall-clock
    /// time so tests get deterministic LRU ordering regardless of how fast
    /// they run.
    fn tick(&self) -> u64 {
        let mut clock = self.clock.lock().unwrap();
        *clock += 1;
        *clock
    }
}

#[async_trait]
impl KvCacheStore for InMemoryKvCacheStore {
    async fn find(&self, key: &CacheKey) -> Result<Option<CacheHandle>, KvCacheError> {
        Ok(self.entries.lock().unwrap().get(key).map(|e| e.handle.clone()))
    }

    async fn confirm_hit(&self, key: &CacheKey) -> Result<(), KvCacheError> {
        let now = self.tick();
        if let Some(entry) = self.entries.lock().unwrap().get_mut(key) {
            entry.last_used_at = now;
        }
        Ok(())
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
        let now = self.tick();
        self.entries.lock().unwrap().insert(
            key.clone(),
            Entry {
                handle,
                size_bytes: meta.size_bytes,
                last_used_at: now,
            },
        );
        self.evict_to_budget().await?;
        Ok(())
    }

    async fn evict_to_budget(&self) -> Result<EvictionReport, KvCacheError> {
        let mut entries = self.entries.lock().unwrap();
        let mut total: u64 = entries.values().map(|e| e.size_bytes).sum();
        if total <= self.max_bytes {
            return Ok(EvictionReport::default());
        }
        let mut ordered: Vec<CacheKey> = entries.keys().cloned().collect();
        ordered.sort_by_key(|k| entries[k].last_used_at);

        let mut report = EvictionReport::default();
        for key in ordered {
            if total <= self.max_bytes {
                break;
            }
            if let Some(entry) = entries.remove(&key) {
                total = total.saturating_sub(entry.size_bytes);
                report.evicted_count += 1;
                report.bytes_freed += entry.size_bytes;
            }
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
        let store = InMemoryKvCacheStore::new(1_000);
        assert_conformance(&store, 1_000).await;
    }
}
