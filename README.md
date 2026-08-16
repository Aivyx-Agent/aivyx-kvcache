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
