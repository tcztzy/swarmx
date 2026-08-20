# SPEC

## §G GOAL

SwarmX presents the published DeepSeek Harness Web profile as one safe, local Electron desktop surface.

## §C CONSTRAINTS

- Harness owns sessions, tools, model access, persistence, permissions, and browser UI.
- SwarmX owns only in-process boot, Electron lifecycle, navigation, and desktop security policy.
- Renderer remains sandboxed remote content with no preload or Node integration.
- Server binds `127.0.0.1` on an OS-assigned port.
- Workspace uses `pnpm@11.7.0`; build tooling follows DSH TypeScript 6 + tsdown client conventions.

## §I INTERFACES

- cmd: `pnpm start` → build, boot Harness, open one Electron window.
- cmd: `pnpm build` → emit desktop host + DSH-compatible client plugin artifacts.
- file: `$DSH_HOME/profiles/web/cordis.patch.yml` → user-owned profile overrides.
- package: `@swarmx/dsh-ui-conversation` → Retry/Edit client plugin composed after DSH bundles and before user patches.

## §V INVARIANTS

V1: ∀ desktop boot → Harness browser handoff disabled; exactly one SwarmX-owned surface opens.
V2: ∀ renderer navigation → same Harness origin stays in-window; foreign `http:`/`https:` may open externally; every other protocol is denied.
V3: ∀ renderer browser permission → denied unless a later documented capability adds a focused allow rule and regression test.
V4: ∀ generated build output → excluded from source lint; authored source remains formatted and checked.
V5: ∀ initial renderer load failure → error is reported and the window is shown instead of remaining invisibly blocked.
V6: ∀ re-run lookup → use real DSH conversation nodes and turn boundaries; missing or out-of-window evidence declines safely.
V7: ∀ Edit action → prepare a separate session before the selected turn, open it, seed its draft; source session remains unchanged.
V8: ∀ first-turn Retry/Edit → create a fresh session in the source Workspace (adopting its cwd when ungrouped) because DSH `session.fork` cannot represent an empty completed-turn prefix.
V9: ∀ closed user-originated message with a safe preparation strategy → Edit renders as an icon beside that user message; assistant action rows own no SwarmX Edit control.
V10: ∀ turn ending in error → visible failure message and Retry icon render even when no finalized assistant message exists.
V11: ∀ Edit/Retry → source log remains append-only and unchanged; active child starts before the superseded turn and appends only the revised/retried user input; superseded messages, `llm/retry`, and `turn/end(error)` remain in source history but stay outside child Chat/model projections and LLM prompt payloads.
V12: ∀ eligible user-message action row → the clock, Copy, and Edit render in that order; Copy and Edit each reserve their own 28px lane with the native 10px action gap, so neither icon overlaps the clock.
V13: ∀ SwarmX conversation action icon → its semantic component resolves through one local `conversationIcons` map; changing icon artwork requires no action/controller/layout or client-bundle configuration change.
V14: ∀ reusable package → `packages/<group>/<package>`; browser plugin at `packages/client/ui-conversation`; executable assembly at `apps/desktop`.
V15: ∀ clean root build → host/client TypeScript references + tsdown emit desktop `dist/` and plugin `lib/{index.js,client.js,types/**}`.
V16: ∀ shared build tool → root ownership; client bundle keeps DSH module-table identities external.

## §T TASKS

id|status|task|cites
T1|x|suppress duplicate system-browser handoff|V1
T2|x|restrict external navigation and permissions|V2,V3
T3|x|restore reliable source lint and code map check|V4
T4|x|compose and integration-test Retry/Edit plugin|I.package
T5|.|add packaging, signing, updates, and release smoke tests|I.cmd
T6|x|surface initial renderer load failures|V5
T7|x|render and execute Retry/Edit against real DSH conversation nodes|V6,V7,V8
T8|x|place Edit on user messages and Retry on failed turns|V9,V10
T9|x|pin append-only branch and model-projection isolation|V11
T10|x|keep user clock, Copy, and Edit in one non-overlapping visual sequence|V12
T11|x|centralize conversation icon selection and consolidate retained legacy icon assets|V13
T12|x|align workspace layout and build system with DSH|V14,V15,V16,I.cmd,I.package

## §B BUGS

id|date|cause|fix
B1|2026-08-20|Electron launcher passed empty Web args, preserving upstream auto-open default|V1
B2|2026-08-20|navigation fence delegated unvalidated protocols to `shell.openExternal`|V2
B3|2026-08-20|renderer session installed no explicit permission policy|V3
B4|2026-08-20|Biome scanned generated `lib/` output and used a stale schema URL|V4
B5|2026-08-20|the launcher ignored `loadURL` rejection while waiting indefinitely for `ready-to-show`|V5
B6|2026-08-20|turn lookup read nonexistent `location.turn.turn`; real assistant nodes expose top-level `turn`, so both actions always rendered null|V6
B7|2026-08-20|Edit staged private controller state but no composer submission path consumed it|V7
B8|2026-08-20|the extension treated omitted `atSeq` as session start, but DSH forks through the last completed turn when the anchor is omitted|V8
B9|2026-08-20|Edit registered in the finalized-assistant-only action slot, so it appeared on the wrong author row|V9
B10|2026-08-20|Retry registered in the finalized-assistant-only action slot, so error turns without an assistant message had no action or visible terminal failure|V10
B11|2026-08-20|source-log preservation and model-projection isolation were implicit, leaving future Edit/Retry changes free to overwrite history or leak failure metadata into prompts|V11
B12|2026-08-20|the derived Edit row was pulled left of Copy without reserving space in the native clock-and-Copy row, so longer clock labels overlapped Edit and the icon order was reversed|V12
B13|2026-08-20|the new icon map used a multiline import that differed from Biome's canonical formatting|V4
B14|2026-08-20|README required Corepack although supported Node 26 omits it|§C package-manager requirement only
B15|2026-08-20|pnpm script policy allowed Electron only, so DSH native/build dependencies stayed unbuilt|§C DSH-aligned `allowBuilds`
