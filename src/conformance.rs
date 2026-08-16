//! Shared behavior contract, run against every `KvCacheStore`
//! implementation in this crate. See `KvCacheStore`'s own doc comment for
//! what's being asserted here. `max_bytes` must match whatever budget the
//! `store` under test was actually configured with, and must be at least
//! 200 (see `assert_conformance`'s doc comment for why).

use crate::{CacheHandle, CacheKey, CacheMeta, KvCacheError, KvCacheStore};

fn key(prefix: &str) -> CacheKey {
    CacheKey {
        backend_id: "llama-server".into(),
        model_id: "qwen3.5".into(),
        build_hash: "b1".into(),
        prefix_hash: prefix.into(),
    }
}

/// `max_bytes` must be at least 200: the LRU-eviction section below records
/// two 100-byte entries that must both fit before a third entry forces
/// eviction of exactly one of them.
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
    store
        .record(&key("p1"), handle.clone(), meta)
        .await
        .unwrap();
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

    // LRU eviction: confirm_hit moves recency, find() alone does not.
    // p1 already exists (recorded above, 100 bytes). Record p2, another
    // 100-byte entry -- naturally more recently used than p1 since it's
    // recorded after.
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

    // Reverse the natural order: confirm_hit(p1) makes p1 the more recently
    // used of the two, even though p2 was recorded after it. If this were a
    // no-op, p1 would remain the older entry and the eviction below would
    // pick it instead of p2.
    store.confirm_hit(&key("p1")).await.unwrap();

    // find() alone must not affect recency -- call it repeatedly on p2 and
    // confirm below that it still gets evicted, proving find() didn't
    // protect it. If find() incorrectly bumped recency, p2 would survive
    // the eviction below instead of p1.
    for _ in 0..3 {
        store.find(&key("p2")).await.unwrap();
    }

    // Force eviction of exactly one 100-byte entry: recording an entry
    // sized (max_bytes - 100) brings the total to max_bytes + 100 -- 100
    // bytes over budget, just enough to require evicting exactly one of
    // p1/p2, so the eviction decision is fully determined by which of the
    // two is actually least-recently-used.
    store
        .record(
            &key("p3"),
            CacheHandle::new("p3.slot"),
            CacheMeta {
                size_bytes: max_bytes - 100,
                token_count: 1,
            },
        )
        .await
        .unwrap();

    // If confirm_hit and find behaved correctly, p2 (the true LRU entry)
    // was evicted; p1 (confirm_hit'd) and p3 (just recorded) both survive.
    // A confirm_hit that silently no-ops, or a find() that wrongly bumps
    // recency, would instead evict p1 here -- these assertions actually
    // distinguish correct from broken behavior, unlike the previous
    // version of this section.
    assert!(store.find(&key("p1")).await.unwrap().is_some());
    assert!(store.find(&key("p3")).await.unwrap().is_some());
    assert!(store.find(&key("p2")).await.unwrap().is_none());
}
