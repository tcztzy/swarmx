# Independent domain applications

SwarmX owns native Agent access, delegation, Host authority, Work, Memory and observed execution
evidence. Domain applications such as GEEPilot own scientific entities, methods, artifact storage,
notebooks, runtimes and interpretation. Their existing data, revisions and hashes remain theirs;
SwarmX does not delete or rewrite old scientific stores.

## Native execution and ACP

Set `SWARMX_CWD` to an existing project directory and optionally `SWARMX_HOME` to a dedicated
private product home, then run `pnpm start` from the SwarmX checkout. Independent programs and
native skills stay with that project. Native runtimes retain their own authentication, configuration
and transcripts. See [runtime setup](runtime-platform.md) and [ACP](acp.md).

External applications can launch the existing `pnpm --silent acp` entry point after building,
using official ACP initialization/session/prompt and native permission flows. ACP stdout is reserved
for protocol traffic. ACP execution access does not register a domain evidence resolver or grant
additional Host authority. Saving or resuming an ACP session is separate from scientific data access.

## Trusted in-process reference provider

`ProductServicesOptions.referenceProvider` and `startDesktopPlatform({referenceProvider, ...})`
accept this structural interface from trusted host configuration:

```ts
interface ReferenceProvider {
  readonly scheme: string;
  readonly requiredPermissions: readonly ToolGrant[];
  resolve(id: string): { id: string; exactId: string; revision: string };
  checkResource(id: string): undefined | {
    ruleId: "source.invalid" | "source.unresolved";
    severity: "error" | "warning";
    message: string;
  };
}
```

The provider is borrowed, synchronous and already bound to the canonical workspace. Its creator
owns shutdown. Provider configuration is never accepted from tool arguments, renderer IPC or ACP
messages. A provider must retain its domain's canonical IDs, current revisions and failure behavior;
SwarmX does not implement the domain model or fall back to historical content.

Work checks the retained `science.read` permission and every declared provider grant before
calling `resolve`. Submitted IDs must use that provider's scheme; both returned `exactId` and
`revision` must equal the submitted values. Missing provider, unsupported scheme, failed lookup or
mismatch rejects submission without recording accepted artifacts. Submission never grants acceptance.

Memory validates execution URNs through the Host journal. Matching domain references invoke
`checkResource` only with the required read authority; its invalid/unresolved diagnostics are kept.
An unconfigured legacy `sx:` reference produces an unresolved warning. Ordinary URLs and local
Memory links keep their own validation. This preserves stored citations without pretending that an
unavailable source was verified. Provider exceptions are not converted into successful resolution.

GEEPilot's workspace-bound provider can be supplied by an embedding host. The existing external
ACP launcher does **not** automatically register it: cross-process domain resolution is not
implemented here. An in-process seam and independently passing tests do not establish a complete
external ACP scientific workflow.
