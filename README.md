# aivyx-kvcache

Backend-agnostic local KV-cache persistence/sharing layer for local LLM
serving.

A `KvCacheStore` trait (`find`/`confirm_hit`/`record`/`evict_to_budget`,
keyed on backend + model/build + a hash of the *stable* prompt prefix)
plus two implementations: `InMemoryKvCacheStore` (a deterministic fake,
for tests) and `LlamaServerSlotStore` (the real backend — drives
llama-server's native `/slots/{id}?action=save|restore` API against
files on disk, indexed by an sqlite manifest in WAL mode so it stays
correct across separate OS processes writing concurrently).

Deliberately minimal — no daemon, no background eviction thread
(`evict_to_budget` runs opportunistically inside `record`), and only one
real backend today. vLLM (via something LMCache-shaped) and Ollama (if
it ever grows a save/restore hook) can implement the same trait later
without a rewrite.

A `aivyx-kvcache` CLI (`list`/`stats`/`prune`) inspects and reclaims
space from the store directly.

Point the store directory (`--store-path` / `AIVYX_KVCACHE_DIR`) at a
dedicated volume — physical I/O isolation from your OS/model-weights
disk, headroom for multi-GB caches at long context windows, and
survival across an OS reinstall are all deliberate goals, not just
config flexibility.

Not yet consumed by either `aivyx-coder` or `aivyx` — integrating this
crate into each app's own `aivyx-llm` is separate, explicit follow-on
work, not an automatic consequence of this crate existing (the sibling
`aivyx-recall` crate has direct history here: it was built the same way
and `aivyx`'s own memory crate never actually migrated onto it).

See `docs/superpowers/specs/2026-08-16-aivyx-kvcache-design.md` in the
`aivyx-ecosystem` repo for the full design rationale.

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
