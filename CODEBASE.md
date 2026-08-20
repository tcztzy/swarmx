# SwarmX codebase map

SwarmX is a thin Electron host around the published DeepSeek Harness Web profile, plus one client plugin composed after the published bundles and before user patches.

## Runtime flow

`apps/desktop/src/main.ts` boots `apps/desktop/src/harness.ts`, then loads its loopback URL through the security boundary in `apps/desktop/src/window.ts`.

The Harness supplies the complete browser UI and `/api` transport. SwarmX has no renderer, preload, alternate session store, or model client.

## Authored source and tests

| Path | Ownership |
| --- | --- |
| `apps/desktop/src/harness.ts` | Profile creation, patch composition, in-process Harness boot, loopback URL |
| `apps/desktop/src/main.ts` | Electron startup, window recreation, Harness shutdown |
| `apps/desktop/src/window.ts` | BrowserWindow construction, navigation and permission boundary |
| `apps/desktop/tests/harness.test.ts` | Real profile boot, browser-handoff and HTTP integration |
| `apps/desktop/tests/window.test.ts` | Window security regression tests |
| `packages/client/tsdown.client.ts` | DSH-compatible client-plugin build preset and module-table external policy |
| `packages/client/ui-conversation/tsdown.config.ts` | Retry/Edit host/client artifact configuration |
| `packages/client/ui-conversation/src/index.ts` | Retry/Edit plugin host half |
| `packages/client/ui-conversation/src/css.d.ts` | CSS Module type boundary |
| `packages/client/ui-conversation/src/fork-boundary.ts` | Safe rerun preparation strategy resolution |
| `packages/client/ui-conversation/src/rerun.ts` | Fork, open, and prompt orchestration |
| `packages/client/ui-conversation/src/turn-origin.ts` | User text lookup within a conversation turn |
| `packages/client/ui-conversation/src/error-turn.ts` | Terminal failure selection for the turn-tail row |
| `packages/client/ui-conversation/src/user-edit-node.ts` | Derived Edit node for user-authored messages |
| `packages/client/ui-conversation/src/client/actions.tsx` | User Edit and failed-turn Retry icon controls |
| `packages/client/ui-conversation/src/client/controller.ts` | Per-session Retry/Edit orchestration |
| `packages/client/ui-conversation/src/client/index.ts` | Client slot registration |
| `packages/client/ui-conversation/src/client/icons.ts` | Single semantic Edit/Retry icon mapping |
| `packages/client/ui-conversation/src/client/slots.ts` | Injected action contract |
| `packages/client/ui-conversation/tests/client-build.test.ts` | Client build face, entry, external, and loader wrapper contract |
| `packages/client/ui-conversation/tests/controller.test.ts` | Controller behavior |
| `packages/client/ui-conversation/tests/error-turn.test.ts` | Terminal failure selection behavior |
| `packages/client/ui-conversation/tests/fork-boundary.test.ts` | Boundary behavior |
| `packages/client/ui-conversation/tests/history-projection.test.ts` | Append-only branch and model-projection isolation |
| `packages/client/ui-conversation/tests/rerun.test.ts` | Re-run orchestration behavior |
| `packages/client/ui-conversation/tests/turn-origin.test.ts` | Turn lookup behavior |
| `packages/client/ui-conversation/tests/user-edit-node.test.ts` | User-message Edit node behavior |
| `packages/client/ui-conversation/tests/user-edit-layout.test.ts` | User-message clock, Copy, and Edit layout contract |
| `packages/client/ui-conversation/tests/conversation-icons.test.ts` | Conversation icon mapping contract |
| `scripts/clean.ts` | Removes only known desktop and client-plugin build outputs |
| `scripts/check-codebase-docs.mjs` | Checks every authored source/test path is mapped here |
| `scripts/workspace-layout.test.ts` | Grouped workspace, package-manager, and root build-tool ownership contract |

Generated `dist/` and `lib/` artifacts are not source. DSH does not publish its client tsdown preset; the local minimal preset is release-coupled to the DSH module table and guarded by `client-build.test.ts`.
