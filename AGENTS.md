# Repository Guidelines

YOU ARE THE MOST CAPABLE MODEL AND THIS PROJECT IS DIFFICULT META-PROGRAMMING THAT AVERAGE MODELS CANNOT HANDLE, WITH INTERLOCKING CONSTRAINTS WHERE A SINGLE MISSTEP BREAKS THE PIPELINE. IT DEMANDS INSIGHT BEYOND A TYPICAL PHD —- DO NOT TREAT IT LIGHTLY.

Authoritative instructions for coding agents working in this repo. Follow every constraint unless the user explicitly overrides it.

## Quick Start
- Read the user request twice; note unstated constraints and cross-file impacts.
- Plan when work is more than trivial and keep it updated; otherwise proceed directly.
- Inspect relevant files first with read-only commands (`rg`, `ls`, `cat`).
- Make focused, minimal edits; prefer patch-style changes or targeted scripts.
- Run only the needed validation via `cargo check` / `cargo test`; summarize results.
- Craft the final response using the messaging rules below, including verification status and next actions when relevant.

## Project Structure & Key Files
- `crates/swarmx-core/` – Core library (`agent.rs`, `swarm.rs`, `mcp.rs`, `messages.rs`, `server.rs`, `edge.rs`, `hook.rs`, `node.rs`, `tool.rs`).
- `crates/swarmx-cli/` – CLI binary (`src/main.rs`).
- `crates/swarmx-ui/` – Desktop GUI library (`app.rs`, `chat.rs`, `sidebar.rs`, `settings.rs`). Built with **Iced 0.14** (Elm/MVU architecture), **iced-shadcn 0.5** (shadcn-style widgets), **lucide-icons 0.575** (icon font), and **agent-client-protocol 0.11** (ACP Rust SDK).
- `apps/desktop/` – macOS desktop bundle (`bundle.sh`, `Info.plist`, `AppIcon.icns`).
- `apps/tauri/` – Desktop application (`src-tauri/src/main.rs`).
- `apps/web/` – Leptos web application (`src/lib.rs`, `src/main.rs`).
- `tests/` – Rust tests embedded in module `tests.rs` files within each source module.
- `docs/`, `examples/` – Narrative docs and usage patterns.
- See `README.md` and `docs/` for deeper context.

## UI Stack (swarmx-ui)

| Technology | Version | Purpose |
|------------|---------|---------|
| Iced | 0.14 | Cross-platform GUI framework (Elm Architecture) |
| iced-shadcn | 0.5 | shadcn-inspired widget library (Button, Card, Input, Badge, Collapsible, etc.) |
| lucide-icons | 0.575 | Icon font (850+ icons via `Icon` enum + `LUCIDE_FONT_BYTES`) |
| agent-client-protocol | 0.11.1 | ACP Rust SDK (SACP types + tokio transport) |

Key patterns:
- **MVU**: `App` state struct → `update()` handles `Message` enum → `view()` returns `Element`
- **Variant/Size**: Every shadcn component uses `*Props::new().variant(*Variant).size(*Size)` builder
- **Theme**: All components take `&Theme` as last arg; palette accessed via `t.palette.foreground` etc.
- **Lucide**: `char::from(icon).to_string()` with `Font::with_name("lucide")` for text-based icons
- **Session persistence**: Individual JSON files in `~/.swarmx/sessions/{id}.json`
- **Markdown cache**: `HashMap<String, Vec<Content>>` keyed by session ID to avoid re-parsing

## Workflow & Guardrails
- Plan for non-trivial tasks; keep plans current.
- Default to ASCII. Add concise comments only when they clarify complex logic. Never revert user-owned changes.
- Use `cargo check -p <crate>` for targeted checks; mention skipped tests and why.
- Stay within provided access; request approval before privileged actions. Avoid destructive git commands.
- Surface failures promptly with relevant output and proposed alternatives.

## Command Reference
- Inspect: `ls`, `rg --files`, `rg "<pattern>" <path>`
- Format/lint: `cargo fmt`, `cargo clippy`
- Tests: `cargo test`, `cargo test -p swarmx-core`, `cargo test -xvs`
- Scripts: `cargo run -p swarmx-cli -- <args>`
- Dependencies: Edit `Cargo.toml` workspace dependencies or crate-level `Cargo.toml`
- Editing: prefer patch-based tools or scripted transformations.

## Coding Style & Naming
- Target Rust 2024 edition (1.85+); use modern features (`let else`, RPITIT, generic associated types).
- Prefer concise, idiomatic Rust; leverage iterators, `?` operator, and `match`.
- Use `thiserror` for error enums and `anyhow` for application errors.
- Use `serde` for serialization; `schemars` for JSON schema generation.
- Use `async-trait` for trait-based async interfaces.
- Pydantic-style validation is replaced by `serde` deserialization + manual validation.
- Naming: Types `PascalCase`, functions/variables `snake_case`, constants `UPPER_SNAKE_CASE`, modules `snake_case`.
- Tests belong in `src/<module>/tests.rs` using `#[cfg(test)] mod tests;`.

## Testing
- `cargo test` is the framework of record; aim for >90% coverage.
- Mirror source structure with module-level `tests.rs`.
- Use `tokio::test` for async code; `tokio::task::JoinSet` for concurrent execution.
- Prefer `cargo test -p swarmx-core -- --nocapture` when debugging; share only salient failures.
- Always note whether tests ran, passed, or were skipped.

## Communication & Final Messages
- Tone: concise, collaborative, factual. Reference files with clickable paths (`path/to/file.rs:42`).
- Lead with the outcome, then explain changes by file/section. Use bullets sparingly.
- Always state verification status (tests/linters run or skipped) and only suggest natural next steps.
- Do not dump large diffs; describe impact and locations for local inspection.

## Commit & PR Guidelines
- Commit messages: Keep ≤50 characters, imperative, using KeepAChangelog verbs (`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`). Provide body when needed; never include "generated by LLM".
- PRs: Describe changes clearly, link issues, ensure tests pass, update docs when required.

## Security & Configuration
- Environment variables load from `.env`; keep `.env.example` in sync.
- Never commit secrets or credentials.
- MCP servers require proper auth; keep configuration aligned with `mcp.rs`.
- See `DESIGNS.md` for detailed agent architecture, graph invariants, and MCP integration notes.

## Related Resources
- `README.md` – overview and getting started.
- `docs/` – extended guides and references.
- `examples/` – practical usage patterns.
- `DESIGNS.md` – agent architecture, graph invariants, MCP integration, and UI architecture.
- `~/skills/skills/iced-shadcn/` — Full widget catalog for iced-shadcn components.
- `~/skills/skills/lucide-icons/` — Lucide icon integration patterns for Iced.
- `~/skills/skills/iced-chatbot/` — Modern chatbot UI patterns with Iced + shadcn.
- `~/skills/skills/agent-client-protocol/` — ACP spec, SDK reference, and lifecycle guide.
