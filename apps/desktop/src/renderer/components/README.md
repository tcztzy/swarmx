# Official UI sources

These MIT-licensed files are copied from [assistant-ui](https://github.com/assistant-ui/assistant-ui/tree/1a5da0f272668cf313e5213e49aa70e0f987de6d/packages/ui/src).
The source revision is pinned by that link. Keep its formatting so the diff remains reviewable.
Each source may have at most ten added/deleted lines relative to upstream. Supply project data,
labels and layout through existing props/slots; delete replaced implementations and styles.
Do not move generic interactions into project wrappers to evade this limit.

Paths below are relative to upstream `packages/ui/src/`. Unlisted adaptation counts are zero.

| Local source | Upstream source | Changed lines | Reason |
| --- | --- | ---: | --- |
| `assistant-ui/elements/model-selector.tsx` | `components/react/assistant-ui/elements/model-selector.radix.tsx` | 4 | Localized, associated cmdk navigation label |
| `assistant-ui/elements/model-selector.aui.tsx` | `components/react/assistant-ui/elements/model-selector.aui.tsx` | 2 | Export the existing model-context registration for composition |
| `assistant-ui/elements/reasoning.tsx` | `components/react/assistant-ui/elements/reasoning.tsx` | 3 | Trigger children supply the native Worked label |
| `assistant-ui/elements/reasoning.aui.tsx` | `components/react/assistant-ui/elements/reasoning.aui.tsx` | 0 | Official scroll-lock integration |
| `assistant-ui/elements/tool-group.tsx` | `components/react/assistant-ui/elements/tool-group.aui.tsx` | 3 | Trigger children supply native activity summaries |
| `assistant-ui/elements/tool-fallback.tsx` | `components/react/assistant-ui/elements/tool-fallback.aui.tsx` | 8 | Trigger children and translated result/error labels |
| `assistant-ui/elements/terminal-block.tsx` | `components/react/assistant-ui/elements/terminal-block.tsx` | 8 | Status slot displays the actual native outcome |
| `assistant-ui/elements/markdown-text.tsx` | `components/react/assistant-ui/elements/markdown-text.tsx` | 3 | Translated code-copy label |
| `assistant-ui/elements/tooltip-icon-button.tsx` | `components/react/assistant-ui/elements/tooltip-icon-button.radix.tsx` | 0 | Official tooltip/copy button |
| `assistant-ui/elements/surfaces.tsx` | `components/react/assistant-ui/elements/surfaces.tsx` | 0 | Terminal styles |
| `assistant-ui/utils/range.ts` | `components/react/assistant-ui/utils/range.ts` | 0 | Terminal's upstream dependency |
| `ui/radix/code-block.tsx` | `components/react/ui/radix/code-block.tsx` | 5 | Translated copy feedback and region label |
| Other `ui/radix/*.tsx` | Matching `components/react/ui/radix/*.tsx` | 0 | Button, Badge, Input, Textarea, NativeSelect, Tabs, Collapsible, Popover, Command, Dialog, Tooltip |
| `../hooks/use-copy-to-clipboard.ts` | `hooks/use-copy-to-clipboard.ts` | 0 | Markdown code-copy feedback |
| `../lib/utils.ts` | `lib/utils.ts` | 0 | Official `cn` export |

The renderer resolves both registry UI import forms to this single Radix directory.
React provides ref-as-prop support. Upstream code is excluded from local formatting/lint rewriting;
TypeScript, renderer interaction tests and production bundling still check its integration.

Native catalogs/defaults, run modes, turn grouping/timing, approvals, source provenance and
execution outcomes remain project responsibilities. Native reasoning is hidden. Commentary
uses Reasoning and has no message or code-copy action; final answers remain outside the disclosure.

The conversation composition and shared light theme retain the [homepage Demo](https://github.com/assistant-ui/assistant-ui/tree/1a5da0f272668cf313e5213e49aa70e0f987de6d/apps/docs/components/pages/home/demo)
layout: avatar-free messages, neutral user bubbles, unboxed assistant text, a shared 42rem
column and bottom composer controls. Fonts are bundled in `../fonts/` with their own licenses.
Do not add actions without a working native contract.

See [LICENSE](LICENSE) for the source license.
