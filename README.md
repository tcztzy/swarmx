# SwarmX

An Electron desktop application for the [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness).

The harness owns the agent: sessions, tools, the agent loop, permissions,
credentials, persistence, and telemetry. SwarmX contributes the desktop surface
and one small client extension for non-destructive Retry/Edit actions.

## How it works

DSH's browser surface is a profile bundle that serves a complete UI over HTTP.
SwarmX composes its conversation extension after DSH's bundles, boots the result
**inside the Electron main process**, suppresses the profile's normal
system-browser handoff, and points one Electron window at the port it bound to:

```
Electron main ──boot()──> DSH web profile + SwarmX actions ──HTTP/127.0.0.1──> BrowserWindow
                          (dsh-base + dsh-web-app + one patch)
```

Consequences of that shape:

- **No replacement renderer.** All published `dsh-client-ui-*` packages are
  reused; the local extension enters through the public conversation slot.
- **No IPC bridge.** The renderer already reaches the harness through the
  `/api` transport its client plugins speak.
- **No second process.** `boot()` returns the root context, so the harness lives
  and dies with the window.

The desktop host remains three small source files:

| File | Role |
| --- | --- |
| `src/harness.ts` | Compose the profile's patch layers, boot, report the bound URL |
| `src/window.ts` | Create the window, fence navigation to the harness origin |
| `src/main.ts` | Sequence startup and shutdown |

`packages/client/ui-conversation` owns the Retry/Edit extension. Edit is an icon on
each eligible user message, following its timestamp and Copy action in one
non-overlapping visual sequence. A failed turn keeps its error text visible and adds
a Retry icon even when no final assistant message exists. Retry prepares a
separate session before the selected turn and sends the original text; Edit
opens that session with the original prompt as a draft. Later turns use DSH's
fork primitive; the first turn uses a fresh session in the same Workspace
(adopting the source cwd when it was ungrouped) because DSH cannot fork an empty
completed-turn prefix. Neither action mutates the source history.

Retry/Edit are append-only branch operations. The superseded user message,
automatic-retry records, and terminal error remain durable in the source
session. The active child begins before that turn and appends only the revised
or retried user text, so the superseded turn is not rendered there and failure
metadata is not part of the messages sent to the model provider.

## Requirements

Node `^22.19.0 || >=24.0.0`.

## Running

```shell
pnpm install
pnpm start
```

First run creates `~/.dsh/profiles/web/` from DSH's shipped template. A model
provider must be configured before the harness can answer; do that in the app's
own settings, or in `~/.dsh/settings.yaml`.

## Customizing the UI

### Conversation action icons

`packages/client/ui-conversation/src/client/icons.ts` is the only source-level
mapping for SwarmX conversation icons. It maps the semantic `edit` and `retry`
actions to 16px React icon components. The defaults reuse DSH primitives so
size, `currentColor`, hover, and accessibility behavior stay native; replacing
either mapping value does not require changing the action components, CSS, or
client bundler.

### Harness profile

Every adjustment happens in the profile's patch layer —
`~/.dsh/profiles/web/cordis.patch.yml` — without touching this repository. A
patch replaces the targeted row's whole `config`, so restate the fields you
keep.

Remove a UI domain:

```yaml
- id: ui-workflow-run
  disabled: true
```

Add your own plugin row:

```yaml
- insert:
    - id: my-panel
      name: '@me/dsh-client-ui-my-panel'
```

Within a client plugin, `ctx.slots.register` contributes a component into a
declared slot, and a lower `priority` shadows an existing occupant without
modifying the package that registered it. Recolor by overriding the `--dsw-*`
alias tokens `dsh-client-ui-theme` publishes.

See DSH's own documentation for the slot contract and the cookbooks.

## Security boundary

The renderer is remote content: no preload, no Node integration, context
isolation and sandbox on. The server binds loopback on an OS-assigned port.
Navigation away from that origin is cancelled. Only `http:` and `https:` links
are handed to the OS browser; local files, scripts, and custom protocols are
blocked. Chromium permission requests are denied unless SwarmX explicitly adds
and tests a capability, so model-rendered content cannot acquire camera,
microphone, location, notification, or similar browser authority.

## License

MIT
