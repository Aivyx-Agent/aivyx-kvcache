use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension, Row, TransactionBehavior, params};
use tokio::sync::Mutex;

use crate::{CacheHandle, CacheKey, CacheMeta, KvCacheError};

/// sqlite-backed index of saved cache entries. WAL mode is what makes
/// this safe across separate OS processes writing concurrently —
/// `aivyx-coder` and `aivyx` each run their own llama-server-backed
/// process today (confirmed, not just assumed), so cross-process safety
/// is a real requirement here. This is the one deliberate departure from
/// `aivyx-recall`'s `FileRecall`, whose single in-process
/// `tokio::sync::Mutex` never had to cover true cross-process writers.
pub(crate) struct Manifest {
    conn: Mutex<Connection>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ManifestRow {
    pub(crate) key: CacheKey,
    pub(crate) handle: CacheHandle,
    pub(crate) size_bytes: u64,
    pub(crate) token_count: u64,
    pub(crate) last_used_at_secs: u64,
    pub(crate) hit_count: u64,
}

impl Manifest {
    pub(crate) fn open(db_path: &Path) -> Result<Self, KvCacheError> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| KvCacheError::Backend(e.to_string()))?;
        }
        let conn = Connection::open(db_path).map_err(|e| KvCacheError::Backend(e.to_string()))?;
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| KvCacheError::Backend(e.to_string()))?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS slots (
                backend_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                build_hash TEXT NOT NULL,
                prefix_hash TEXT NOT NULL,
                handle TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                created_at_secs INTEGER NOT NULL,
                last_used_at_nanos INTEGER NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (backend_id, model_id, build_hash, prefix_hash)
            )",
            [],
        )
        .map_err(|e| KvCacheError::Backend(e.to_string()))?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub(crate) async fn find(&self, key: &CacheKey) -> Result<Option<ManifestRow>, KvCacheError> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT handle, size_bytes, token_count, last_used_at_nanos, hit_count
             FROM slots WHERE backend_id = ?1 AND model_id = ?2 AND build_hash = ?3 AND prefix_hash = ?4",
            params![key.backend_id, key.model_id, key.build_hash, key.prefix_hash],
            |row| {
                Ok(ManifestRow {
                    key: key.clone(),
                    handle: CacheHandle::new(row.get::<_, String>(0)?),
                    size_bytes: row.get::<_, i64>(1)? as u64,
                    token_count: row.get::<_, i64>(2)? as u64,
                    last_used_at_secs: nanos_to_secs(row.get::<_, i64>(3)?),
                    hit_count: row.get::<_, i64>(4)? as u64,
                })
            },
        )
        .optional()
        .map_err(|e| KvCacheError::Backend(e.to_string()))
    }

    pub(crate) async fn confirm_hit(&self, key: &CacheKey) -> Result<(), KvCacheError> {
        let now = now_nanos();
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE slots SET hit_count = hit_count + 1, last_used_at_nanos = ?5
             WHERE backend_id = ?1 AND model_id = ?2 AND build_hash = ?3 AND prefix_hash = ?4",
            params![
                key.backend_id,
                key.model_id,
                key.build_hash,
                key.prefix_hash,
                now
            ],
        )
        .map_err(|e| KvCacheError::Backend(e.to_string()))?;
        Ok(())
    }

    pub(crate) async fn insert(
        &self,
        key: &CacheKey,
        handle: &CacheHandle,
        meta: CacheMeta,
    ) -> Result<(), KvCacheError> {
        let now_secs = now_secs();
        let now_nanos = now_nanos();
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO slots
                (backend_id, model_id, build_hash, prefix_hash, handle, size_bytes, token_count, created_at_secs, last_used_at_nanos, hit_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 0)
             ON CONFLICT (backend_id, model_id, build_hash, prefix_hash) DO UPDATE SET
                handle = excluded.handle,
                size_bytes = excluded.size_bytes,
                token_count = excluded.token_count,
                last_used_at_nanos = excluded.last_used_at_nanos,
                hit_count = 0",
            params![
                key.backend_id,
                key.model_id,
                key.build_hash,
                key.prefix_hash,
                handle.as_str(),
                meta.size_bytes as i64,
                meta.token_count as i64,
                now_secs,
                now_nanos,
            ],
        )
        .map_err(|e| KvCacheError::Backend(e.to_string()))?;
        Ok(())
    }

    /// Removes a single row by key. Production eviction paths use the
    /// atomic `evict_and_remove` instead (see its doc comment for why a
    /// separate select-then-delete isn't safe across processes); this
    /// stays as a standalone primitive exercised directly by tests below.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) async fn remove(&self, key: &CacheKey) -> Result<(), KvCacheError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM slots WHERE backend_id = ?1 AND model_id = ?2 AND build_hash = ?3 AND prefix_hash = ?4",
            params![key.backend_id, key.model_id, key.build_hash, key.prefix_hash],
        )
        .map_err(|e| KvCacheError::Backend(e.to_string()))?;
        Ok(())
    }

    /// Atomically selects least-recently-used rows to evict (until total
    /// recorded bytes are at or under `target_bytes`) and removes them
    /// from the manifest, all inside a single `BEGIN IMMEDIATE`
    /// transaction -- unlike a separate select-then-delete, this closes
    /// the cross-process race where two separate OS processes' `record()`
    /// calls could otherwise both select overlapping candidates and
    /// double-evict, or one process could delete a row a moment after
    /// another process refreshed it via `confirm_hit`/`insert`. Returns
    /// the removed rows so the caller can delete their backing files
    /// afterward (outside this transaction -- a crash between commit and
    /// file deletion leaves only a harmless orphaned file, never a
    /// dangling manifest row, since the row is already gone by then).
    pub(crate) async fn evict_and_remove(
        &self,
        target_bytes: u64,
    ) -> Result<Vec<ManifestRow>, KvCacheError> {
        let mut conn = self.conn.lock().await;
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| KvCacheError::Backend(e.to_string()))?;

        let rows: Vec<ManifestRow> = {
            let mut stmt = tx
                .prepare(
                    "SELECT backend_id, model_id, build_hash, prefix_hash, handle, size_bytes, token_count, last_used_at_nanos, hit_count
                     FROM slots ORDER BY last_used_at_nanos ASC, rowid ASC",
                )
                .map_err(|e| KvCacheError::Backend(e.to_string()))?;
            stmt.query_map([], row_from_sql)
                .map_err(|e| KvCacheError::Backend(e.to_string()))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| KvCacheError::Backend(e.to_string()))?
        };

        let mut total: u64 = rows.iter().map(|r| r.size_bytes).sum();
        let mut removed = Vec::new();
        for row in rows {
            if total <= target_bytes {
                break;
            }
            total = total.saturating_sub(row.size_bytes);
            tx.execute(
                "DELETE FROM slots WHERE backend_id = ?1 AND model_id = ?2 AND build_hash = ?3 AND prefix_hash = ?4",
                params![
                    row.key.backend_id,
                    row.key.model_id,
                    row.key.build_hash,
                    row.key.prefix_hash
                ],
            )
            .map_err(|e| KvCacheError::Backend(e.to_string()))?;
            removed.push(row);
        }

        tx.commit()
            .map_err(|e| KvCacheError::Backend(e.to_string()))?;
        Ok(removed)
    }

    pub(crate) async fn total_bytes(&self) -> Result<u64, KvCacheError> {
        let conn = self.conn.lock().await;
        let total: i64 = conn
            .query_row(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM slots",
                [],
                |row| row.get(0),
            )
            .map_err(|e| KvCacheError::Backend(e.to_string()))?;
        Ok(total as u64)
    }

    /// Rows to remove, oldest-`last_used_at`-first, stopping once
    /// removing them would bring the running total at or under
    /// `target_bytes`. Read-only — callers are responsible for actually
    /// deleting the returned rows (and their backing files) via `remove`.
    pub(crate) async fn evict_candidates(
        &self,
        target_bytes: u64,
    ) -> Result<Vec<ManifestRow>, KvCacheError> {
        let rows = self.all_rows("ASC").await?;
        let mut total: u64 = rows.iter().map(|r| r.size_bytes).sum();
        let mut candidates = Vec::new();
        for row in rows {
            if total <= target_bytes {
                break;
            }
            total = total.saturating_sub(row.size_bytes);
            candidates.push(row);
        }
        Ok(candidates)
    }

    /// Every row, most-recently-used first — for the CLI's `list`.
    pub(crate) async fn list_all(&self) -> Result<Vec<ManifestRow>, KvCacheError> {
        self.all_rows("DESC").await
    }

    async fn all_rows(&self, order: &'static str) -> Result<Vec<ManifestRow>, KvCacheError> {
        let conn = self.conn.lock().await;
        let sql = format!(
            "SELECT backend_id, model_id, build_hash, prefix_hash, handle, size_bytes, token_count, last_used_at_nanos, hit_count
             FROM slots ORDER BY last_used_at_nanos {order}, rowid {order}"
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| KvCacheError::Backend(e.to_string()))?;
        let rows = stmt
            .query_map([], row_from_sql)
            .map_err(|e| KvCacheError::Backend(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| KvCacheError::Backend(e.to_string()))?;
        Ok(rows)
    }
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Nanosecond-resolution timestamp used for `last_used_at` ordering.
///
/// `evict_candidates`/`list_all` order by recency, and `confirm_hit` on an
/// entry inserted moments earlier must sort strictly after it. Whole-second
/// resolution (`now_secs`) isn't fine enough for that in practice — several
/// inserts plus a `confirm_hit` can land in the same wall-clock second
/// (this is exactly what the eviction-order test exercises), which makes
/// `ORDER BY` ties resolve arbitrarily instead of by recency. Nanosecond
/// resolution keeps `ManifestRow::last_used_at_secs` (derived via
/// `nanos_to_secs`) as a plain seconds-since-epoch value for display, while
/// the DB orders on the finer-grained column internally.
fn now_nanos() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as i64
}

fn nanos_to_secs(nanos: i64) -> u64 {
    (nanos as u64) / 1_000_000_000
}

fn row_from_sql(row: &Row) -> rusqlite::Result<ManifestRow> {
    Ok(ManifestRow {
        key: CacheKey {
            backend_id: row.get(0)?,
            model_id: row.get(1)?,
            build_hash: row.get(2)?,
            prefix_hash: row.get(3)?,
        },
        handle: CacheHandle::new(row.get::<_, String>(4)?),
        size_bytes: row.get::<_, i64>(5)? as u64,
        token_count: row.get::<_, i64>(6)? as u64,
        last_used_at_secs: nanos_to_secs(row.get::<_, i64>(7)?),
        hit_count: row.get::<_, i64>(8)? as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(prefix: &str) -> CacheKey {
        CacheKey {
            backend_id: "llama-server".into(),
            model_id: "qwen3.5".into(),
            build_hash: "b1".into(),
            prefix_hash: prefix.into(),
        }
    }

    #[tokio::test]
    async fn insert_find_confirm_hit_and_remove_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let m = Manifest::open(&dir.path().join("manifest.db")).unwrap();

        // Unwritten key: find returns None, no side effects.
        assert!(m.find(&key("p1")).await.unwrap().is_none());

        let handle = CacheHandle::new("p1.slot");
        let meta = CacheMeta {
            size_bytes: 1000,
            token_count: 50,
        };
        m.insert(&key("p1"), &handle, meta).await.unwrap();

        let row = m.find(&key("p1")).await.unwrap().unwrap();
        assert_eq!(row.handle, handle);
        assert_eq!(row.size_bytes, 1000);
        assert_eq!(row.token_count, 50);
        assert_eq!(row.hit_count, 0);

        // confirm_hit on a real key bumps hit_count and recency.
        m.confirm_hit(&key("p1")).await.unwrap();
        let row = m.find(&key("p1")).await.unwrap().unwrap();
        assert_eq!(row.hit_count, 1);

        // confirm_hit on a missing key is a no-op, not an error.
        m.confirm_hit(&key("does-not-exist")).await.unwrap();

        // total_bytes reflects what's recorded.
        assert_eq!(m.total_bytes().await.unwrap(), 1000);

        // remove deletes it; find afterward is None again.
        m.remove(&key("p1")).await.unwrap();
        assert!(m.find(&key("p1")).await.unwrap().is_none());
        assert_eq!(m.total_bytes().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn re_inserting_an_existing_key_resets_hit_count() {
        let dir = tempfile::tempdir().unwrap();
        let m = Manifest::open(&dir.path().join("manifest.db")).unwrap();

        let meta = CacheMeta {
            size_bytes: 100,
            token_count: 10,
        };
        m.insert(&key("p1"), &CacheHandle::new("v1.slot"), meta)
            .await
            .unwrap();
        m.confirm_hit(&key("p1")).await.unwrap();
        assert_eq!(m.find(&key("p1")).await.unwrap().unwrap().hit_count, 1);

        // Re-inserting (a fresh save superseding the old one) resets hit_count
        // and swaps in the new handle/size.
        m.insert(
            &key("p1"),
            &CacheHandle::new("v2.slot"),
            CacheMeta {
                size_bytes: 200,
                token_count: 20,
            },
        )
        .await
        .unwrap();
        let row = m.find(&key("p1")).await.unwrap().unwrap();
        assert_eq!(row.hit_count, 0);
        assert_eq!(row.handle, CacheHandle::new("v2.slot"));
        assert_eq!(row.size_bytes, 200);
    }

    #[tokio::test]
    async fn evict_candidates_returns_least_recently_used_first_until_under_budget() {
        let dir = tempfile::tempdir().unwrap();
        let m = Manifest::open(&dir.path().join("manifest.db")).unwrap();

        // Three 100-byte entries, inserted in order p1, p2, p3.
        for p in ["p1", "p2", "p3"] {
            m.insert(
                &key(p),
                &CacheHandle::new(format!("{p}.slot")),
                CacheMeta {
                    size_bytes: 100,
                    token_count: 1,
                },
            )
            .await
            .unwrap();
        }
        // Touch p1 so it's most-recently-used despite being inserted first;
        // p2 is now the least-recently-used entry.
        m.confirm_hit(&key("p1")).await.unwrap();

        // Total is 300; budget of 150 must evict until <= 150, i.e. evict
        // p2 then p3 (p2 is LRU, p3 is next), leaving only p1 (100 <= 150).
        let candidates = m.evict_candidates(150).await.unwrap();
        let evicted: Vec<&str> = candidates
            .iter()
            .map(|r| r.key.prefix_hash.as_str())
            .collect();
        assert_eq!(evicted, vec!["p2", "p3"]);

        // evict_candidates itself doesn't remove anything.
        assert_eq!(m.total_bytes().await.unwrap(), 300);
    }

    #[tokio::test]
    async fn evict_and_remove_atomically_removes_lru_rows_until_under_budget() {
        let dir = tempfile::tempdir().unwrap();
        let m = Manifest::open(&dir.path().join("manifest.db")).unwrap();

        for p in ["p1", "p2", "p3"] {
            m.insert(
                &key(p),
                &CacheHandle::new(format!("{p}.slot")),
                CacheMeta {
                    size_bytes: 100,
                    token_count: 1,
                },
            )
            .await
            .unwrap();
        }
        m.confirm_hit(&key("p1")).await.unwrap();

        let removed = m.evict_and_remove(150).await.unwrap();
        let removed_hashes: Vec<&str> =
            removed.iter().map(|r| r.key.prefix_hash.as_str()).collect();
        assert_eq!(removed_hashes, vec!["p2", "p3"]);

        // Unlike evict_candidates, this actually removed the rows.
        assert_eq!(m.total_bytes().await.unwrap(), 100);
        assert!(m.find(&key("p1")).await.unwrap().is_some());
        assert!(m.find(&key("p2")).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn list_all_returns_every_row_without_removing_any() {
        let dir = tempfile::tempdir().unwrap();
        let m = Manifest::open(&dir.path().join("manifest.db")).unwrap();
        assert_eq!(m.list_all().await.unwrap().len(), 0);

        m.insert(
            &key("p1"),
            &CacheHandle::new("p1.slot"),
            CacheMeta {
                size_bytes: 100,
                token_count: 1,
            },
        )
        .await
        .unwrap();
        let rows = m.list_all().await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].key.prefix_hash, "p1");
    }

    #[tokio::test]
    async fn manifest_persists_across_a_fresh_instance_pointed_at_the_same_file() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("manifest.db");
        {
            let m = Manifest::open(&db_path).unwrap();
            m.insert(
                &key("p1"),
                &CacheHandle::new("p1.slot"),
                CacheMeta {
                    size_bytes: 100,
                    token_count: 1,
                },
            )
            .await
            .unwrap();
        }
        let reopened = Manifest::open(&db_path).unwrap();
        assert!(reopened.find(&key("p1")).await.unwrap().is_some());
    }
}
