# Harness icon provenance

Checked 2026-07-21. These files are bundled so the desktop does not depend on a runtime CDN.

| File | Upstream source | Revision | License / note |
|---|---|---|---|
| `claude_code.svg` | `agentclientprotocol/registry:claude-acp/icon.svg` | `c47300d575354b69c348bd0ed77265bb9a698336` | Apache-2.0 registry asset for the Claude Agent ACP adapter; it is not a claim that Claude Code itself has native ACP. |
| `codex.svg` | `agentclientprotocol/registry:codex-acp/icon.svg` | `c47300d575354b69c348bd0ed77265bb9a698336` | Apache-2.0 registry asset for Codex ACP. |
| `opencode.svg` | `agentclientprotocol/registry:opencode/icon.svg` | `c47300d575354b69c348bd0ed77265bb9a698336` | Apache-2.0 registry copy of the OpenCode mark. |
| `hermes.svg` | `NousResearch/hermes-agent:acp_registry/icon.svg` | `7acaff5ef2bcbaa22bd23b72efe60906123a4f55` | MIT; verified from the local upstream checkout at `~/.hermes/hermes-agent`. |
| `pi.svg` | `agentclientprotocol/registry:pi-acp/icon.svg` | `01aca4c6e97dc0f84bf8e9b3529d4261cfdd2c84` | Apache-2.0 registry asset for the `pi-acp` adapter. |
| `kimi.svg` | `MoonshotAI/kimi-code:apps/vscode/resources/kimi-icon.svg` | `188c0fcbf7c884d4a86bd4eebd012b0ab7aeb5da` | MIT; whitespace-normalized official Kimi Code icon markup. |

The same SVG markup is embedded in `src/harness-icon-data.ts` so reusable renderer consumers do not need to copy this public directory. SwarmX uses its own application glyph. OpenClaw intentionally uses the deterministic fallback because a revisioned, redistributable ACP/product icon was not verified; the official `openclaw acp` command itself is verified separately. The OpenClaw and Kimi Code projects declare MIT, while the registry-hosted assets above are Apache-2.0. All names and marks remain the property of their respective owners. See `packages/desktop/THIRD_PARTY_NOTICES.md` for complete notices and licenses.
