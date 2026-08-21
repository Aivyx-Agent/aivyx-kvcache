//! Real llama-server end-to-end test -- gated behind `--ignored`, needs a
//! real `llama-server` binary and a real GGUF model on disk. See this
//! crate's own `docs/superpowers/specs/2026-08-16-real-llama-server-e2e-test-design.md`
//! for the design this implements, and `README.md`'s "Running the real
//! llama-server test" section for how to run it.

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use aivyx_kvcache::{CacheKey, CacheMeta, KvCacheStore, LlamaServerSlotStore};

/// Kills the wrapped `llama-server` child on drop -- `std::process::Child`
/// does NOT do this by default, so without this guard a panicking
/// assertion mid-test would leak the server process.
struct ServerGuard(Child);

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Binds an ephemeral port and immediately drops the listener, handing the
/// OS-assigned port to the caller. A small, accepted TOCTOU window between
/// the drop and llama-server actually binding it -- fine for a local,
/// single-purpose test.
fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind an ephemeral port");
    listener
        .local_addr()
        .expect("failed to read back the bound port")
        .port()
}

/// Polls `GET /health` every ~500ms for up to 120s. Model loading is the
/// dominant, highly variable cost (CPU vs. GPU, quant size, cold vs. warm
/// disk cache) -- timeout panics with a clear message instead of hanging
/// the test suite indefinitely.
async fn wait_for_health(base_url: &str) {
    let client = reqwest::Client::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    loop {
        if std::time::Instant::now() > deadline {
            panic!(
                "llama-server did not report healthy within 120s at {base_url}/health -- \
                 check the model path and GPU offload settings"
            );
        }
        match client.get(format!("{base_url}/health")).send().await {
            Ok(resp) if resp.status().is_success() => return,
            _ => tokio::time::sleep(Duration::from_millis(500)).await,
        }
    }
}

fn key(model_path: &str, prefix_hash: &str) -> CacheKey {
    CacheKey {
        backend_id: "llama-server".to_string(),
        model_id: model_path.to_string(),
        build_hash: "e2e-test".to_string(),
        prefix_hash: prefix_hash.to_string(),
    }
}

#[tokio::test]
#[ignore]
async fn real_llama_server_save_restore_evict_round_trip() {
    let model_path = std::env::var("AIVYX_KVCACHE_E2E_MODEL_PATH").unwrap_or_else(|_| {
        panic!(
            "AIVYX_KVCACHE_E2E_MODEL_PATH is required for this test -- point it at a real GGUF \
             model file, then run:\n\
             AIVYX_KVCACHE_E2E_MODEL_PATH=/path/to/model.gguf \
             cargo test --test real_llama_server_e2e -- --ignored"
        )
    });
    let llama_server_bin = std::env::var("AIVYX_KVCACHE_E2E_LLAMA_SERVER_BIN")
        .unwrap_or_else(|_| "llama-server".to_string());

    let store_dir = tempfile::tempdir().expect("failed to create a temp store dir");
    let store_path = store_dir.path();
    let port = free_port();
    let base_url = format!("http://127.0.0.1:{port}");

    // Construct the store BEFORE spawning llama-server -- this creates
    // store_path/slots/, which is then passed to --slot-save-path, avoiding
    // a race where llama-server might otherwise start before the directory
    // exists.
    let store = LlamaServerSlotStore::open(store_path, &base_url, 1_000_000_000)
        .expect("failed to open LlamaServerSlotStore");
    let slots_dir = store_path.join("slots");

    let child = Command::new(&llama_server_bin)
        .arg("--model")
        .arg(&model_path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--slot-save-path")
        .arg(&slots_dir)
        .arg("--parallel")
        .arg("1")
        .arg("--ctx-size")
        .arg("512")
        .arg("-ngl")
        .arg("99")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| {
            panic!(
                "failed to spawn `{llama_server_bin}`: {e} -- is it on PATH, or set \
                 AIVYX_KVCACHE_E2E_LLAMA_SERVER_BIN?"
            )
        });
    let _guard = ServerGuard(child);

    wait_for_health(&base_url).await;

    let http = reqwest::Client::new();
    let key1 = key(&model_path, "e2e-prefix-1");

    // Step 1: give slot 0 real KV state via a native /completion call --
    // no chat template needed. --parallel 1 guarantees exactly one slot
    // (id 0) exists, so every save/restore call below targets it
    // unambiguously.
    http.post(format!("{base_url}/completion"))
        .json(&serde_json::json!({"prompt": "The quick brown fox", "n_predict": 1}))
        .send()
        .await
        .expect("completion request failed")
        .error_for_status()
        .expect("completion request returned an error status");

    // Step 2: real save.
    store
        .save_from_slot(&key1, 0, CacheMeta { size_bytes: 100, token_count: 5 })
        .await
        .expect("save_from_slot failed");
    let entries: Vec<_> = std::fs::read_dir(&slots_dir)
        .expect("failed to read slots dir")
        .collect::<Result<_, _>>()
        .expect("failed to list slots dir entries");
    assert_eq!(
        entries.len(),
        1,
        "expected exactly one file in slots/ after one save"
    );

    // Step 3: real restore -- a genuine successful round-trip against real
    // llama-server, not a mocked 200.
    let restored = store
        .restore_into_slot(&key1, 0)
        .await
        .expect("restore_into_slot returned an error");
    assert!(
        restored,
        "restore_into_slot must return true on a real successful restore"
    );

    // Step 4: real eviction deletes the real file. Re-issue the
    // completion (slot 0's content doesn't need to differ), then save a
    // second key against a store opened with a tighter budget -- key1
    // alone fits, key1+key2 together don't -- forcing eviction of key1.
    // This is the step that directly re-proves the crate's original
    // Critical bug is fixed: the same save -> evict -> verify-the-
    // real-file-is-gone sequence, against a real server instead of a mock.
    http.post(format!("{base_url}/completion"))
        .json(&serde_json::json!({"prompt": "The quick brown fox", "n_predict": 1}))
        .send()
        .await
        .expect("second completion request failed")
        .error_for_status()
        .expect("second completion request returned an error status");

    let evicting_store = LlamaServerSlotStore::open(store_path, &base_url, 150)
        .expect("failed to open a second LlamaServerSlotStore with a tighter budget");
    let key2 = key(&model_path, "e2e-prefix-2");
    evicting_store
        .save_from_slot(&key2, 0, CacheMeta { size_bytes: 100, token_count: 5 })
        .await
        .expect("save_from_slot for key2 failed");

    assert!(
        evicting_store
            .find(&key1)
            .await
            .expect("find(key1) failed")
            .is_none(),
        "key1 must have been evicted once key1+key2 exceeded the 150-byte budget"
    );
    assert!(
        evicting_store
            .find(&key2)
            .await
            .expect("find(key2) failed")
            .is_some(),
        "key2 must still be present"
    );
    let entries_after_eviction: Vec<_> = std::fs::read_dir(&slots_dir)
        .expect("failed to read slots dir after eviction")
        .collect::<Result<_, _>>()
        .expect("failed to list slots dir entries after eviction");
    assert_eq!(
        entries_after_eviction.len(),
        1,
        "expected exactly one file in slots/ after eviction -- key1's file (the one llama-server \
         itself actually wrote) must be gone from disk, not just removed from the manifest"
    );
}
