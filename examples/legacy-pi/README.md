# Retired builtin Pi reference

This directory preserves the builtin adapter and its historical test sources byte for byte.
It is evidence for review, not a runnable package, supported adapter or production fallback.
The files retain their original relative imports and are excluded from production compilation
and the default test suite. `manifest.json` identifies the originals and hashes.

SwarmX now orchestrates external Agents. Codex is the default external integration. Configure
an independent ACP endpoint and explicit `acp` Host policy to use Pi or another framework;
see `../../docs/acp.md`. SwarmX no longer directly imports or declares Pi SDKs, creates Pi sessions,
loads Pi skills, manages model context or supplies a direct Pi-to-ProductServices tool bridge.
The separate DSH upstream SDK still transitively depends on `pi-ai`; this archive does not
claim that the complete dependency graph contains no Pi libraries.

Existing native files and historical `pi:` Host records remain untouched. The retired `pi`
selection fails explicitly; it does not start another Agent or reconstruct native history.
An external agent owns its native storage and must explicitly validate compatible session
ownership. This archive makes no claim that changing an endpoint automatically resumes old
Host conversation IDs. External ACP currently receives no Host MCP credential; other native
integrations keep their existing revocable, execution-bound product-tool bridges.

Historical test results apply to their recorded candidate only. New core and ACP acceptance
must be obtained from the current tests and exact working-tree candidate.
