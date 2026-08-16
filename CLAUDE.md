# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## What this is

`aivyx-kvcache` is a small, backend-agnostic KV-cache persistence layer:
a `KvCacheStore` trait plus `InMemoryKvCacheStore` (test fake) and
`LlamaServerSlotStore` (real — llama-server's `/slots` API + an sqlite
manifest). It exists to let `aivyx-coder` and `aivyx` (the flagship
Personal Assistant, in its local-backend mode) persist and share the
expensive prefill work of a long, mostly-stable prompt prefix (system
prompt + tool defs + repo map) across session resumes and across their
otherwise-separate llama-server processes. See `README.md` and
`aivyx-ecosystem/docs/superpowers/specs/2026-08-16-aivyx-kvcache-design.md`
for the full rationale — this file only covers what's specific to
working in this repo's code.

Neither `aivyx-coder` nor `aivyx` actually depends on this crate yet;
that integration is separate follow-on work, tracked outside this repo.

## Build, test, lint

```sh
cargo build
cargo test
cargo clippy --all-targets
cargo fmt
```

Single crate (a library plus one `aivyx-kvcache` CLI binary under
`src/bin/`), no workspace — no `-p` flag needed. Single test:
`cargo test <test_name>`.

## Architecture

- `lib.rs` — the trait (`KvCacheStore`), the key/handle/meta/report
  types, `KvCacheError`, and the contract every implementation must
  satisfy (documented on `KvCacheStore`'s own doc comment).
- `manifest.rs` — `Manifest`, the sqlite-backed (WAL mode) index behind
  `LlamaServerSlotStore`. `pub(crate)` — not part of the public API.
- `in_memory.rs` — `InMemoryKvCacheStore`, a deterministic in-process
  fake with no persistence.
- `llama_server.rs` — `LlamaServerSlotStore`, the real implementation.
  Its `KvCacheStore` trait methods (`find`/`confirm_hit`/`record`/
  `evict_to_budget`) are pure manifest bookkeeping and file-deletion, no
  HTTP; `restore_into_slot`/`save_from_slot` are the llama-server-specific
  methods that actually call its `/slots` API, built on top of the trait
  methods.
- `conformance.rs` — `assert_conformance`, run against every
  `KvCacheStore` implementation as a `&dyn KvCacheStore` trait object.
  Any new implementor should be tested the same way.
- `cli.rs` / `bin/aivyx-kvcache.rs` — the `list`/`stats`/`prune` CLI. The
  logic lives in `cli.rs` as plain functions returning `String` reports
  (testable without capturing stdout); `bin/aivyx-kvcache.rs` is a thin
  `clap` wrapper around them.

### Cache key invariant

`CacheKey.prefix_hash` must only ever hash the **stable** prompt prefix
(system prompt + tool defs + repo map) — never per-turn conversation
content. This is load-bearing, not incidental: including per-turn
content would produce a fresh cache key on every message, defeating the
entire point of persisting the expensive, mostly-static prefill work.
Any code that constructs a `CacheKey` should be checked against this
before merging.

## Where to look next

- `README.md` — quick orientation and the design-doc pointer.
- `aivyx-ecosystem/docs/superpowers/specs/2026-08-16-aivyx-kvcache-design.md`
  — the full design: why a backend-agnostic trait with only one real
  implementation, the storage layout, eviction policy, and the explicit
  out-of-scope list for v1.
