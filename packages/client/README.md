# Client packages

Browser-side packages extend the published DSH Web surface and keep shared runtime identities external from their client bundles.

| Package | Purpose |
| --- | --- |
| [`ui-conversation/`](ui-conversation) | Non-destructive Retry/Edit actions and generic Side View |
| [`ui-science/`](ui-science) | Additive Science Workspace with artifact Side View integration |

`tsdown.client.ts` owns the DSH-compatible loader wrapper, CSS Module compilation, and external-module policy shared by this group.
