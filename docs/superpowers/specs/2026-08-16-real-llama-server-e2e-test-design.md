# Real llama-server E2E test — design

_2026-08-16._ Adds the real-llama-server end-to-end test the original
design (`aivyx-ecosystem/docs/superpowers/specs/2026-08-16-aivyx-kvcache-design.md`)
called for but the implementation plan never tasked — a gap the plan's own
final review flagged, since it's the one test shape that would have caught
the Critical slot-path/filename bug that review found (every mocked test
agrees with the store's own path convention by construction; only a real
llama-server, writing files on its own terms, can disagree).

## Problem

Every existing test in this crate either exercises `Manifest`/
`InMemoryKvCacheStore` directly (no HTTP at all) or drives `LlamaServerSlotStore`
against a `wiremock` mock that only asserts request shape (method, path) —
never against a real llama-server actually reading/writing files on disk.
That gap is structural, not incidental: a wiremock test can only ever
confirm the store's HTTP calls look right, never that llama-server can
actually *do* what the store asked. The original Critical bug (a nested
`slots/<backend>/<model>/<hash>.slot` path llama-server's flat-filename
API could never populate) is exactly the class of bug this leaves
unguarded against for good.

## Scope

One new gated integration test, plus a short README update. Not part of
CI, not run by default — `#[ignore]`d, run manually via
`cargo test -- --ignored` once `llama-server` and a GGUF model are
available locally. Nothing about the crate's public API, trait, or
existing test suite changes.

## Location and gating

`tests/real_llama_server_e2e.rs` — a Rust integration test file, separate
from `src/`'s unit tests (which stay `wiremock`-based; this file is
additive, not a replacement). Cargo automatically discovers and compiles
`tests/*.rs` as its own binary with access to the crate's public API and
its existing `[dependencies]` (`reqwest`, `serde_json`, `tokio`) — no new
Cargo.toml dependencies needed.

One `#[ignore]`d `#[tokio::test]` function. Two env vars, read at the top:

- `AIVYX_KVCACHE_E2E_MODEL_PATH` — **required**, no default (any default
  would hardcode a path specific to one machine). Missing → the test
  panics immediately with a message naming the env var and giving an
  example `cargo test --test real_llama_server_e2e -- --ignored`
  invocation.
- `AIVYX_KVCACHE_E2E_LLAMA_SERVER_BIN` — optional, defaults to
  `"llama-server"` (resolved via `$PATH`).

## Process management

Fully self-contained — the test spawns and tears down its own
`llama-server` instance, no manual setup required beyond having the
binary and a model on disk:

1. Open a `tempfile::tempdir()` as `store_path`.
2. Pick a free TCP port: bind a throwaway `std::net::TcpListener` to
   `127.0.0.1:0`, read back the OS-assigned port, drop the listener before
   handing that port to llama-server (a small, accepted TOCTOU window —
   fine for a local, single-purpose test).
3. Construct `LlamaServerSlotStore::open(store_path, base_url, max_bytes)`
   **before** spawning llama-server — this creates `store_path/slots/`,
   which is then passed to llama-server's `--slot-save-path`, avoiding a
   race where llama-server might otherwise start before the directory
   exists.
4. Spawn:
   ```
   llama-server --model <MODEL_PATH> --host 127.0.0.1 --port <PORT>
                --slot-save-path <store_path>/slots --parallel 1 --ctx-size 512
   ```
   `--parallel 1` guarantees exactly one slot (id `0`) exists, so every
   save/restore call in the test targets it unambiguously — no need to
   discover or track which slot a completion landed on.
5. Wrap the `std::process::Child` in a guard type that calls `kill()` +
   `wait()` in its `Drop` impl, so the process is cleaned up even if a
   later assertion panics. `std::process::Child` does **not** kill on drop
   by default — this guard is required, not decorative.
6. Poll `GET http://127.0.0.1:<port>/health` every ~500ms for up to 120s
   (model loading is the dominant, highly variable cost — CPU vs. GPU,
   quant size, cold vs. warm disk cache) before proceeding. Timeout →
   panic with a clear message (don't hang the test suite indefinitely).

## What the test proves, and how

Deliberately black-box: the test never reaches into `pub(crate)` internals
like `slot_filename`/`slot_path` (an integration test in `tests/` couldn't
see them anyway — it only has the crate's public API), and instead reads
`store_path/slots/`'s actual directory listing via `std::fs::read_dir`.
This is a more honest proxy for what a real operator would observe than
asserting an exact predicted filename, and it's what makes this test
capable of catching a path-convention mismatch the store's own internals
might otherwise agree with by construction.

Sequence, one combined test function (not split across several
`#[ignore]`d tests, to avoid paying llama-server's startup cost more than
once):

1. **Give slot 0 real KV state.** POST `/completion` (llama-server's
   native endpoint — no chat template needed) with a short prompt and
   `"n_predict": 1`, so the request is fast but slot 0 now holds
   non-trivial state (not an empty/virgin slot).
2. **Real save.** `store.save_from_slot(&key1, 0, meta)` → assert `Ok(())`.
   Assert `store_path/slots/` now contains exactly one file.
3. **Real restore.** `store.restore_into_slot(&key1, 0)` → assert
   `Ok(true)` — a genuine successful round-trip against real llama-server,
   not a mocked 200.
4. **Real eviction deletes the real file.** Re-issue the completion (slot
   0's content doesn't need to differ), then
   `store.save_from_slot(&key2, 0, meta)` against a store opened with a
   `max_bytes` sized so key1 alone fits but key1+key2 together don't —
   forcing `record`'s opportunistic eviction to remove key1. Assert
   `store.find(&key1)` is now `None`, `store.find(&key2)` is `Some`, and
   `store_path/slots/` again contains exactly one file (key2's) — key1's
   file, the one llama-server itself actually wrote, is gone from disk.
   This step is the one that directly re-proves the original Critical bug
   is fixed: it's the same save → evict → verify-the-real-file-is-gone
   sequence, against a real server instead of a mock.

`CacheMeta.size_bytes` values used for `save_from_slot` in this test are
small placeholders (e.g. `100`), not measured from the real saved file —
matching a known, already-logged limitation (`save_from_slot` trusts
caller-supplied size and doesn't stat the file itself, tracked as a
follow-up in `aivyx-ecosystem/ROADMAP.md`). Not this test's concern to
fix; `max_bytes` is chosen relative to these placeholder values purely to
make the eviction step in this test deterministic.

## Documentation

Add a short section to `README.md`: how to run this test manually
(the exact env vars, an example command), and the `--slot-save-path` ↔
`store_path/slots` coupling this test's own setup demonstrates — the
review's Recommendation #3 ("document the single most load-bearing
operational fact for a consumer"), still open until now.

## Explicitly out of scope

- Installing `llama-server` or running this test in this session — the
  test is written and reviewed now; running/verifying it happens whenever
  `llama-server` is available.
- CI integration — this test will never run in CI (needs a real model and
  meaningful startup time); no CI config exists in this crate to change
  anyway.
- Fixing `save_from_slot`'s trust-the-caller-supplied-size limitation —
  logged separately, unrelated to what this test needs to prove.
- Testing generation *correctness* (that the restored KV state actually
  produces the same continuation as before saving) — out of scope; this
  test proves the save/restore/evict *mechanism* works against a real
  server, not model output fidelity.
