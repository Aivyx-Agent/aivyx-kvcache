# Security Policy

aivyx-kvcache is a library, not a standalone application — it
provides backend-agnostic local KV-cache persistence/sharing for local
LLM serving, consumed by `aivyx` and `aivyx-coder` as a dependency. A
vulnerability here is usually most consequential in how it affects one
of those two consuming projects, and should still be reported.

This is not a bug bounty program.

## Reporting a vulnerability

Email **jccorbett67@gmail.com** with details. This repo is currently
private, so GitHub Security Advisories' private vulnerability
reporting isn't available yet (GitHub only offers it on public
repositories) — it will be added as a second channel if this repo
goes public. We aim to resolve or provide a remediation plan for a
confirmed vulnerability within 90 days of the report, or coordinate a
later disclosure date directly with the reporter if a fix genuinely
needs longer. Credit is offered in release notes at the reporter's
preference.
