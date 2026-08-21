# Real llama-server E2E Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and actually run, against a real `llama-server`, the gated end-to-end test designed 2026-08-16 (`docs/superpowers/specs/2026-08-16-real-llama-server-e2e-test-design.md`) but never written — the one test shape proven this session to catch what mocked tests structurally can't.

**Architecture:** One new `#[ignore]`d `#[tokio::test]` integration test (`tests/real_llama_server_e2e.rs`) that spawns and tears down its own real `llama-server` process, then drives `LlamaServerSlotStore` against it through a save → restore → evict-and-verify-the-real-file-is-gone sequence. Verified for real on a GPU test rig (`10.80.80.148`) reached over SSH, since this machine has no `llama-server` binary; the rig needs a Rust toolchain installed first, since it has none today.

**Tech Stack:** Rust (`aivyx-kvcache` crate), `tokio`, `reqwest`, `tempfile` (all already crate dependencies — no new ones), real `llama-server` (already installed on the rig) + a real GGUF model (already present on the rig).

## Global Constraints

- The test must be `#[ignore]`d and never run in CI — this crate has no CI config to change anyway, and the design's own "Explicitly out of scope" section rules this out.
- `AIVYX_KVCACHE_E2E_MODEL_PATH` is **required, no default** — missing it must panic immediately with a message naming the env var and an example invocation. `AIVYX_KVCACHE_E2E_LLAMA_SERVER_BIN` is optional, defaults to `"llama-server"` resolved via `$PATH`.
- The test must be black-box: read `store_path/slots/`'s real directory listing via `std::fs::read_dir`, never reach into `pub(crate)` internals like `slot_filename`/`slot_path` (an integration test in `tests/` can't see them anyway).
- `std::process::Child` does not kill its child on drop — the spawned `llama-server` must be wrapped in a guard type whose `Drop` impl calls `kill()` + `wait()`, so a panicking assertion still cleans up the process.
- One combined test function, not several `#[ignore]`d tests — avoids paying llama-server's model-load startup cost more than once.

---

### Task 1: Write the real llama-server E2E test + README section

**Files:**
- Create: `tests/real_llama_server_e2e.rs`
- Modify: `README.md` (append a new section documenting how to run it)

**Interfaces:**
- Consumes: `aivyx_kvcache::{CacheKey, CacheMeta, LlamaServerSlotStore, KvCacheStore}` — all already `pub` from `src/lib.rs`. Exact real signatures (verified against `src/llama_server.rs` and `src/lib.rs`):
  - `LlamaServerSlotStore::open(store_path: impl Into<PathBuf>, base_url: impl Into<String>, max_bytes: u64) -> Result<Self, KvCacheError>`
  - `LlamaServerSlotStore::save_from_slot(&self, key: &CacheKey, slot_id: u32, meta: CacheMeta) -> Result<(), KvCacheError>`
  - `LlamaServerSlotStore::restore_into_slot(&self, key: &CacheKey, slot_id: u32) -> Result<bool, KvCacheError>`
  - `KvCacheStore::find(&self, key: &CacheKey) -> Result<Option<CacheHandle>, KvCacheError>` (trait method — needs `use aivyx_kvcache::KvCacheStore;` in scope to call `.find()`)
  - `CacheKey { backend_id: String, model_id: String, build_hash: String, prefix_hash: String }` (all fields `pub`, struct is `Clone`)
  - `CacheMeta { size_bytes: u64, token_count: u64 }` (both fields `pub`)
- Produces: nothing later tasks import — Task 2 runs this exact file's compiled test binary, unmodified.

- [ ] **Step 1: Write the test file**

Create `tests/real_llama_server_e2e.rs`:

```rust
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
```

Note the one deliberate deviation from the 2026-08-16 design doc's literal spawn command: this adds `-ngl 99` (full GPU offload) to the `llama-server` invocation, which the original design omitted. Without it, `llama-server` defaults to CPU-only inference, which would make this test far slower and less representative of the real target hardware (both `aivyx-coder`'s README and this session's own real-rig verification use `-ngl 99` for exactly this reason). Harmless on a CPU-only machine — llama-server ignores `-ngl` if there's no GPU to offload to.

- [ ] **Step 2: Verify the test compiles and is discovered as ignored**

Run: `cargo test --test real_llama_server_e2e -- --ignored --list`
Expected: compiles cleanly, and the output lists
`real_llama_server_save_restore_evict_round_trip: test` (confirming the test exists and is discoverable — it will not actually run here, since no `AIVYX_KVCACHE_E2E_MODEL_PATH` is set and no real infra exists on this machine; that's Task 2).

Also run the crate's full existing suite to confirm nothing else broke:
Run: `cargo test --workspace`
Expected: PASS, same count as before this task (this new test is `#[ignore]`d, so it doesn't add to the normal run's pass count, only to what `--list` reports).

- [ ] **Step 3: Add the README section**

Append to `README.md` (after the existing content, as a new final section):

```markdown

## Running the real llama-server test

`tests/real_llama_server_e2e.rs` drives a real `llama-server` process
through a save → restore → evict round trip — the one test shape that
can catch a path-convention mismatch between this crate's own filename
scheme and what llama-server can actually read/write (every mocked test
in `src/`'s unit tests agrees with the store's own convention by
construction; only a real server, writing files on its own terms, can
disagree). Gated behind `--ignored` — it needs a real `llama-server`
binary and a real GGUF model on disk, and is never run in CI.

```sh
AIVYX_KVCACHE_E2E_MODEL_PATH=/path/to/model.gguf \
  cargo test --test real_llama_server_e2e -- --ignored
```

`AIVYX_KVCACHE_E2E_LLAMA_SERVER_BIN` optionally overrides the
`llama-server` binary used (default: resolved via `$PATH`).

**The one load-bearing operational fact this test's own setup
demonstrates:** point `--slot-save-path` at exactly the directory this
crate's `LlamaServerSlotStore::open`'s `store_path` argument resolves to
(`store_path/slots`, not `store_path` itself) — llama-server's `/slots`
API only ever takes a bare filename, never a path, so this directory
must be the one you told llama-server to save into. Get this wrong and
every save/restore call 501s with "This server does not support slots
action" (if `--slot-save-path` is missing entirely) or silently can't
find its own files (if the paths don't match).
```

- [ ] **Step 4: Commit**

```bash
git add tests/real_llama_server_e2e.rs README.md
git commit -m "test: add real llama-server e2e test (save/restore/evict round trip)"
```

---

### Task 2: Run the test for real, on the GPU test rig

**Files:** none modified in this repo — this task installs a Rust toolchain on a remote machine and runs Task 1's test there. If the test fails for a reason that points to a bug in `src/`, fix it here and re-run this task; don't mark this task done on a failing run.

**Interfaces:**
- Consumes: Task 1's `tests/real_llama_server_e2e.rs`, unmodified, plus the crate's own `Cargo.toml`/`Cargo.lock`/`src/`.
- Produces: a PASS/FAIL report with the real test output — no interface other tasks depend on.

**Context:** this machine has no `llama-server` binary; the rig (`10.80.80.148`, reachable via `ssh 10.80.80.148`, already has `llama-server` at `/usr/bin/llama-server` and a GGUF model at `/home/julian/models/Qwen3.5-9B-Q4_K_M.gguf`) has no Rust toolchain. This task installs one there, then builds and runs Task 1's test against the rig's real GPU (confirmed present: an RTX 3090) and real model.

**A note on SSH reliability, from this session's own real experience running commands against this exact rig:** multi-line heredocs and heavily-quoted inline one-liners over `ssh host '...'` are fragile — nested shell-escaping across the SSH layer repeatedly produced corrupted commands or silent no-ops in earlier work this session. Prefer writing any nontrivial command sequence to a local file first and `scp`-ing it over, then running it with a single short `ssh host 'bash /path/to/script.sh'` (or `python3 /path/to/script.py`) call. Every step below follows that pattern.

- [ ] **Step 1: Install Rust on the rig**

```bash
ssh 10.80.80.148 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'
```

Expected: installs to `~/.cargo` and `~/.rustup` on the rig, printing a success message ending in something like `Rust is installed now. Great!`.

Verify it's on `PATH` for future non-interactive SSH commands (rustup's installer appends to `~/.profile`/`~/.bashrc`, which a non-interactive `ssh host 'cmd'` shell does not source):

```bash
ssh 10.80.80.148 '~/.cargo/bin/cargo --version'
```

Expected: prints a real cargo version (e.g. `cargo 1.9x.x`). Use the explicit `~/.cargo/bin/cargo` path (not bare `cargo`) in every subsequent step in this task, for the same non-interactive-shell reason.

- [ ] **Step 2: Copy this crate's source to the rig**

From this repo's root (the worktree this task is running in):

```bash
ssh 10.80.80.148 'mkdir -p ~/aivyx-kvcache-e2e'
rsync -az --exclude target --exclude .git ./ 10.80.80.148:~/aivyx-kvcache-e2e/
```

Expected: `rsync` completes with no errors. `--exclude target` skips this machine's own build artifacts (irrelevant and large — the rig will build fresh); `--exclude .git` skips history (not needed to build or test).

Verify Task 1's test file made it over:

```bash
ssh 10.80.80.148 'test -f ~/aivyx-kvcache-e2e/tests/real_llama_server_e2e.rs && echo PRESENT'
```

Expected: `PRESENT`.

- [ ] **Step 3: Confirm no stray llama-server is already running, then run the test**

A stray `llama-server` from earlier manual work on this rig could hold the port this test's `free_port()` happens to pick, or (more likely) just be a leftover process worth cleaning up before a fresh run:

```bash
ssh 10.80.80.148 'pkill -9 -f llama-server; sleep 2; pgrep -f llama-server; echo "pgrep_exit=$?"'
```

Expected: `pgrep_exit=1` (no matching process — confirms nothing is running). If it prints a PID instead, something didn't die; investigate before proceeding rather than starting the test against a machine with an unknown llama-server already holding GPU memory.

Now run the test itself. This will take a few minutes — model loading onto the GPU is the dominant cost, and the test's own 120s health-check timeout is generous for that:

```bash
ssh 10.80.80.148 'cd ~/aivyx-kvcache-e2e && AIVYX_KVCACHE_E2E_MODEL_PATH=/home/julian/models/Qwen3.5-9B-Q4_K_M.gguf ~/.cargo/bin/cargo test --test real_llama_server_e2e -- --ignored --nocapture'
```

Expected: `test real_llama_server_save_restore_evict_round_trip ... ok`, `test result: ok. 1 passed; 0 failed`. If it fails, read the actual assertion message — per this task's own file list note above, a real failure here means a real bug in `src/llama_server.rs`, not a flaw in the test; fix the source, re-run Task 1's Step 2 (`cargo test --workspace` locally) to confirm the fix doesn't break anything else, then re-run this step.

- [ ] **Step 4: Confirm the spawned llama-server was actually cleaned up**

The test's own `ServerGuard` should have killed its child process on completion:

```bash
ssh 10.80.80.148 'pgrep -f llama-server; echo "pgrep_exit=$?"'
```

Expected: `pgrep_exit=1` — no llama-server process left running. If one is still present, `ServerGuard`'s `Drop` impl didn't fire as expected; this is worth investigating (a real resource-leak bug in the test itself), not something to wave off given this task's own note above.

- [ ] **Step 5: Record the result**

No commit for this task (nothing in this repo was modified — Task 1 already committed the test file and README). Report PASS with the full real test output (the `cargo test ... --nocapture` transcript), or FAIL with the exact assertion message and what was done to investigate — for inclusion in the project's closing summary and the `aivyx-ecosystem/ROADMAP.md` update this whole `aivyx-kvcache` adoption project will need once all three plans (this one, `aivyx-coder`'s, `aivyx`'s) are complete.
