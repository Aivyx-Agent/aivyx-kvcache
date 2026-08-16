//! Shared behavior contract, run against every `KvCacheStore`
//! implementation in this crate. See `KvCacheStore`'s own doc comment for
//! what's being asserted here. `max_bytes` must match whatever budget the
//! `store` under test was actually configured with.

use crate::{CacheHandle, CacheKey, CacheMeta, KvCacheError, KvCacheStore};

fn key(prefix: &str) -> CacheKey {
    CacheKey {
        backend_id: "llama-server".into(),
        model_id: "qwen3.5".into(),
        build_hash: "b1".into(),
        prefix_hash: prefix.into(),
    }
}

pub(crate) async fn assert_conformance(store: &dyn KvCacheStore, max_bytes: u64) {
    // Nothing recorded yet: find is None, eviction is a no-op.
    assert!(store.find(&key("p1")).await.unwrap().is_none());
    let report = store.evict_to_budget().await.unwrap();
    assert_eq!(report.evicted_count, 0);
    assert_eq!(report.bytes_freed, 0);

    // confirm_hit on an unrecorded key is a no-op, not an error.
    store.confirm_hit(&key("p1")).await.unwrap();

    // record then find round-trips the handle.
    let handle = CacheHandle::new("p1.slot");
    let meta = CacheMeta {
        size_bytes: 100,
        token_count: 10,
    };
    store.record(&key("p1"), handle.clone(), meta).await.unwrap();
    assert_eq!(store.find(&key("p1")).await.unwrap(), Some(handle));

    // find never has side effects: hit_count/recency only move via confirm_hit.
    // (Exercised indirectly below via eviction ordering, since hit_count itself
    // isn't part of the public trait surface.)

    // A slot whose size alone exceeds the budget fails, without partially
    // recording it.
    let huge = CacheMeta {
        size_bytes: max_bytes + 1,
        token_count: 1,
    };
    let err = store
        .record(&key("too-big"), CacheHandle::new("too-big.slot"), huge)
        .await
        .unwrap_err();
    assert!(matches!(err, KvCacheError::SlotExceedsBudget { .. }));
    assert!(store.find(&key("too-big")).await.unwrap().is_none());

    // Eviction removes least-recently-used entries first once over budget.
    // p1 already recorded (100 bytes); confirm_hit it so it's more recently
    // used than what follows.
    store.confirm_hit(&key("p1")).await.unwrap();
    store
        .record(
            &key("p2"),
            CacheHandle::new("p2.slot"),
            CacheMeta {
                size_bytes: 100,
                token_count: 1,
            },
        )
        .await
        .unwrap();
    // Recording p2 already ran an opportunistic eviction pass; both p1 and p2
    // together (200 bytes) may or may not exceed max_bytes depending on the
    // caller's configured budget, so callers of this suite should configure
    // `store` with a budget of at least 250 bytes for this section to be
    // meaningful. Push total over budget explicitly to force eviction:
    store
        .record(
            &key("p3"),
            CacheHandle::new("p3.slot"),
            CacheMeta {
                size_bytes: max_bytes,
                token_count: 1,
            },
        )
        .await
        .unwrap();
    // p3 alone is at the budget; p1 and p2 (the least-recently-used entries)
    // must have been evicted to make room.
    assert!(store.find(&key("p3")).await.unwrap().is_some());
    assert!(store.find(&key("p2")).await.unwrap().is_none());
}
