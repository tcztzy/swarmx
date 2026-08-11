# Direct Harness release acceptance

The release acceptance gate proves that the ordinary SwarmX direct Harness can
complete its smallest useful Provider-backed path without real credentials or
external network access. Run it from the repository root:

```shell
pnpm test:e2e:release
```

`pnpm run ci:node` includes this command. The primary GitHub CI quality job runs
the same named gate, and the release quality job must pass `pnpm run ci` before
any packaging or publication job can start.

## Acceptance contract

The test starts from a newly created temporary home and gives settings,
Provider auth, Model catalog cache, and Session storage explicit paths beneath
that home. It places credential-shaped values in the ambient environment as
tripwires, then requires an empty configuration to expose no Desktop Provider.

The only usable connection is saved through `ModelCatalogService` and
`FileProviderAuthStore`. Its credential is a fixed non-secret placeholder and
its endpoint is a deterministic HTTP server bound to `127.0.0.1`. Catalog
refresh must call that server's OpenAI-compatible Models endpoint, persist the
returned Model and ModelSupply, and resolve a ready `swarmx:<model>` Agent
composition.

Execution crosses the real package boundary used by Desktop Main:

```text
Desktop settings/auth/catalog services
  -> Core composition resolver
  -> Core Swarm and direct Agent
  -> OpenAI-compatible streaming client
  -> Core append-only Session JSONL
```

The local endpoint must observe the placeholder credential and a streaming
request for the discovered runtime Model. The test requires more than one live
assistant chunk, the combined final assistant result, token usage, and the same
final result in the persisted Session.

Finally, the endpoint is stopped, the loaded module graph is reset, and fresh
settings, auth, and catalog services are constructed against the same isolated
files. A read-only catalog load and a fresh Session load must recover the
explicit Provider, cached Model/ModelSupply, ready composition, credential, and
conversation without calling Provider discovery. Recovery fails if it depends
on ambient credentials, a live endpoint, an in-memory Session object, or an
automatic refresh.

## Deliberate boundary

This is a cross-package Main/Core release E2E, not an Electron UI test. It uses
the production services and native HTTP client but does not start Electron or
exercise Renderer, Preload, IPC sender authorization, Project tools, TLS, DNS,
rate limits, or a real external Provider. The separate macOS Desktop smoke gate
covers the built Electron/Renderer boundary. Real Provider compatibility still
depends on an operator's endpoint, credential, account policy, availability,
and wire-protocol conformance.
