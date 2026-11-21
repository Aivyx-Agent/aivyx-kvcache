# Changelog

## 0.1.1

- Security: redirects are never followed by the HTTP client `LlamaServerSlotStore::open` builds. Model servers are local-only, so a 307/308 can no longer make slot save/restore requests go to another host.
- Fix: `save_from_slot` now requires a 2xx response. With redirects off, a 3xx previously passed `error_for_status`, recorded a phantom manifest row and deleted the previous file for that key.
- New `LlamaServerSlotStore::open_with_client`, to supply your own `reqwest::Client` (for example one with a local-only resolver). `SLOT_HTTP_TIMEOUT` is now public.

## 0.1.0

- Initial release.
