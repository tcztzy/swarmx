# Third-party notices

The SwarmX desktop renderer embeds selected upstream SVG markup as data URLs so
the reusable `@swarmx/desktop/renderer` export does not depend on a host public
directory. Product names and marks remain the property of their respective
owners; inclusion here does not imply endorsement.

## Lucide UI glyphs

The SwarmX application glyph and deterministic fallbacks use icons from the
declared `lucide-react` 1.17.0 dependency. They are interface glyphs, not
official product marks. Lucide is licensed under ISC, with the package's
Feather-derived subset under MIT; the complete bundled notice is at
`third_party_licenses/LUCIDE_ISC.txt`.

## Agent Client Protocol registry icons

The Claude Agent ACP, Codex, OpenCode, and Pi ACP SVG markup is copied verbatim
from `agentclientprotocol/registry` at the listed commits:

- `claude-acp/icon.svg` at commit `c47300d575354b69c348bd0ed77265bb9a698336`
- `codex-acp/icon.svg` at commit `c47300d575354b69c348bd0ed77265bb9a698336`
- `opencode/icon.svg` at commit `c47300d575354b69c348bd0ed77265bb9a698336`
- `pi-acp/icon.svg` at commit `01aca4c6e97dc0f84bf8e9b3529d4261cfdd2c84`

Sources:

- <https://github.com/agentclientprotocol/registry/tree/c47300d575354b69c348bd0ed77265bb9a698336>
- <https://github.com/agentclientprotocol/registry/tree/01aca4c6e97dc0f84bf8e9b3529d4261cfdd2c84/pi-acp>

License: Apache License 2.0. The complete upstream license is included at
`third_party_licenses/ACP_REGISTRY_APACHE-2.0.txt`.

## Hermes Agent icon

The Hermes Agent SVG markup is copied verbatim from
`NousResearch/hermes-agent` commit
`7acaff5ef2bcbaa22bd23b72efe60906123a4f55`, path
`acp_registry/icon.svg`.

Source: <https://github.com/NousResearch/hermes-agent/tree/7acaff5ef2bcbaa22bd23b72efe60906123a4f55>

Copyright (c) 2025 Nous Research. License: MIT. The complete upstream license,
verified from the local upstream checkout at `~/.hermes/hermes-agent/LICENSE`,
is included at `third_party_licenses/HERMES_MIT.txt`.

## OpenClaw interoperability

SwarmX interoperates with the official `openclaw acp` command but does not
embed an OpenClaw project icon. The UI intentionally uses its bundled generic
fallback glyph because no revisioned, redistributable official product/ACP icon
was verified for this change.

The official OpenClaw project at commit
`fab69517b3d4ab4e94ec16f7744f2966769ee611` declares the MIT License, not the
Apache License 2.0. Its license is included for transparency at
`third_party_licenses/OPENCLAW_MIT.txt`; no OpenClaw source or brand asset is
copied into this renderer.

Source: <https://github.com/openclaw/openclaw/tree/fab69517b3d4ab4e94ec16f7744f2966769ee611>

## OpenAI Codex apply_patch grammar

The Project `apply_patch` custom tool includes the Lark grammar from OpenAI
Codex CLI tag `rust-v0.144.4`, path
`codex-rs/core/src/tools/handlers/apply_patch.lark`.

Source: <https://github.com/openai/codex/blob/rust-v0.144.4/codex-rs/core/src/tools/handlers/apply_patch.lark>

Copyright OpenAI. License: Apache License 2.0. The complete Apache License 2.0
text is already included at
`third_party_licenses/ACP_REGISTRY_APACHE-2.0.txt`.

## Provider product marks

The Provider matrix bundles the following official product marks so runtime
rendering does not contact third-party sites:

- DeepSeek API documentation favicon, copied verbatim from
  <https://api-docs.deepseek.com/img/favicon.svg> on 2026-07-13.
- Packy logo, copied verbatim from <https://www.packyapi.com/logo.svg> on
  2026-07-13.
- New API logo, copied verbatim from `web/default/public/logo.png` in
  `QuantumNous/new-api` commit
  `7c28993f6bd9e92616f3f578212577f8b7c40b45`. The upstream repository is
  licensed under GNU AGPL-3.0; source and license are available at
  <https://github.com/QuantumNous/new-api/tree/7c28993f6bd9e92616f3f578212577f8b7c40b45>.

DeepSeek, Packy, and New API names and marks remain the property of their
respective owners. Inclusion identifies configured interoperable services and
does not imply endorsement.
