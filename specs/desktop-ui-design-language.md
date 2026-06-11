# Desktop UI Design Language Spec

## Status

Accepted baseline for the Electron desktop renderer.

## Goal

Define the SwarmX desktop UI as a Codex-like agent runtime surface: dense, calm, operational, and optimized for inspecting long-running agent sessions.

## Scope

This spec governs `packages/desktop/src/renderer`.

It covers visual language, layout, interaction states, message rendering, responsive behavior, and verification rules. It does not define core agent behavior, ACP protocol behavior, persistence, or harness execution.

## Design References

- Primary product metaphor: Codex app runtime screen.
- Component language: shadcn/ui-style composition, neutral tokens, small radii, visible focus states, semantic variants.
- Icon language: lucide-style line icons with consistent optical weight.

## Core Principles

1. The UI is an agent runtime, not a marketing page.
2. The first screen must be usable immediately: session rail, runtime transcript, and composer are all visible.
3. Preserve information density without visual noise.
4. Prefer structural hierarchy over decorative treatment.
5. Use neutral color tokens; avoid one-note accent palettes and decorative gradients.
6. Keep controls code-native and interactive.
7. Do not hide agent/tool execution behind generic chat bubbles.

## Visual Tokens

### Color

Use dark neutral tokens as the default desktop theme:

| Token | Value | Use |
| --- | --- | --- |
| `--background` | `#09090b` | App root and runtime canvas |
| `--foreground` | `#f4f4f5` | Primary text |
| `--card` | `#101013` | Panels, composer, neutral event blocks |
| `--card-hover` | `#1b1b20` | Hover and selected surfaces |
| `--muted` | `#a1a1aa` | Secondary text and icons |
| `--muted-foreground` | `#71717a` | Tertiary text and metadata |
| `--border` | `#27272a` | Primary borders |
| `--border-subtle` | `#1f1f23` | Dividers and low-emphasis borders |
| `--input` | `#151519` | Inputs and select controls |
| `--primary` | `#f4f4f5` | Primary button surface |
| `--primary-foreground` | `#09090b` | Primary button text |
| `--danger` | `#f87171` | Errors and destructive status |
| `--success` | `#34d399` | Active/running/success status |

Accent colors are semantic only. Do not introduce purple, blue, beige, orange, or decorative color systems unless a future spec explicitly changes the theme.

### Shape

- Default radius: `8px`.
- Compact inner radius: `6px` or `7px`.
- Large surface radius: `10px` to `12px`.
- Avoid pill shapes except for status badges.

### Typography

- Use system sans stack with `Inter` first when available.
- Use SF Mono-style stack for tool names, command output, and code-like text.
- Keep app chrome small and deliberate:
  - Header title: 14px, semibold.
  - Sidebar item title: 13px, medium.
  - Metadata: 11px to 12px.
  - Transcript body: 14px.
  - Tool output: 12.5px monospace.
- Letter spacing stays `0`.
- Do not scale font size with viewport width.

### Motion

- Motion is functional only.
- Sidebar collapse: short opacity/translate transition.
- Loading icons may spin.
- Avoid animated backgrounds, decorative pulses, or ornamental motion.

## Layout

### App Shell

The desktop app is a two-column grid:

1. Left sidebar: fixed session rail, default width `288px`.
2. Runtime: flexible main surface with header, transcript, and composer.

The app shell must set `height: 100vh`, `min-height: 0`, and `overflow: hidden` so the composer remains pinned inside the viewport.

### Sidebar

The sidebar contains:

1. Brand block.
2. Harness selector and New action.
3. Harness/Project segmented grouping control.
4. Grouped session list.
5. Inline session discovery errors.

Session rows use icon, title, and metadata. Local sessions are selectable; external ACP sessions may be visible but disabled until load behavior is implemented.

### Runtime Header

The header is a compact command/status bar:

- Left: sidebar toggle, current run title, run subtitle.
- Right: running/ready badge, event count, alerts, destructive action when applicable.
- Height target: `58px` on desktop.
- It should feel like app chrome, not a page hero.

### Transcript

The transcript is centered with a maximum width near `900px`.

Messages render as an event stream:

- User message: bright foreground card, readable contrast.
- Assistant message: neutral event card.
- Thinking: subdued event card with loading affordance.
- Tool call/result: monospace-friendly card with tool label.
- System/error: danger semantic card.

Do not render all message kinds as identical chat bubbles.

### Composer

The composer is docked at the bottom of the runtime:

- Width matches transcript max width.
- Contains textarea, harness context, and Send button.
- Uses a bordered elevated surface.
- Must remain fully visible on desktop and mobile viewports.

## Components

### Buttons

Button variants:

- `default`: primary action, light surface on dark background.
- `secondary`: neutral bordered action.
- `ghost`: icon or low-emphasis chrome action.
- `destructive`: danger action.

Button sizes:

- `sm`: sidebar and compact controls.
- `md`: normal action.
- `icon`: square icon-only actions.

Buttons with icons use icon+text for clear commands and icon-only for chrome actions. Icon-only buttons must have accessible labels or titles.

### Badges

Use badges for runtime status and compact counters only:

- `neutral`: ready/counts.
- `active`: running/success.
- `danger`: alerts/errors.

Do not use badges as decorative labels.

### Selects

Harness selection is a compact select shell with leading icon. It must not look like a browser-default select.

### Segmented Controls

Use segmented controls for 2-option mode switches such as Harness/Project grouping. Active state is a contained dark surface, not a colorful accent.

### Empty State

The empty state is centered in the transcript area and includes:

- Harness icon mark.
- Short title.
- One-line context text.
- Primary action.

It must not become a landing page or explanatory marketing block.

## Icons

- Prefer lucide-react icons.
- Icons inside controls use consistent 14px to 17px sizing.
- Session row icons sit inside small bordered square containers.
- Tool, user, assistant, system, and thinking events must have distinct icons.
- Do not use emoji or plain text glyphs as UI icons.

## Responsive Behavior

### Desktop

- Sidebar visible by default.
- Runtime header remains single row when space allows.
- Transcript and composer max width stay aligned.

### Narrow Viewports

- Sidebar collapses to zero width.
- Header may wrap into two rows.
- Composer padding and textarea height reduce.
- Transcript event rail shrinks from `34px` to `28px`.
- No text or controls may overflow the viewport.

## Accessibility

- Interactive elements are native buttons/selects/textareas.
- Icon-only buttons require `aria-label` or `title`.
- Tabs/segmented controls expose tab roles and selected state.
- Focus styles use tokenized ring/border treatment.
- Text contrast must remain readable in dark mode.

## Forbidden Patterns

- Marketing hero sections.
- Decorative blobs, orbs, bokeh, or ornamental gradients.
- Large rounded cards inside other cards.
- Browser-default controls.
- Identical rendering for messages, tool calls, and thinking states.
- Purple/blue dominant palettes.
- Beige/cream/brown/orange dominant palettes.
- Viewport-width font scaling.
- Hidden composer or clipped primary controls.
- Unlabeled icon-only controls.
- Fake static screenshots as app UI.

## Implementation Rules

1. Keep renderer code componentized enough that app shell, sidebar, runtime header, transcript event, button, and badge concepts are visible in the JSX.
2. Keep design tokens centralized in the renderer stylesheet.
3. Prefer existing React/Electron/Vite conventions over adding a new styling runtime.
4. Full Tailwind/shadcn initialization is optional; shadcn-style source composition and tokens are sufficient unless a future task explicitly migrates the build pipeline.
5. Do not change IPC, preload API, session persistence, or agent execution as part of visual-only changes.

## Verification Checklist

For any future desktop UI change:

1. Run renderer typecheck: `pnpm --filter @swarmx/desktop exec tsc --noEmit -p tsconfig.web.json`.
2. Run lint: `pnpm lint`.
3. Run desktop build: `pnpm --filter @swarmx/desktop build`.
4. Open the Electron app or renderer preview with a preload mock.
5. Verify desktop viewport:
   - Sidebar visible.
   - Header status visible.
   - Transcript readable.
   - Composer fully visible.
6. Verify narrow viewport:
   - Sidebar collapsed.
   - Header not overlapping.
   - Transcript content scrolls.
   - Composer and Send button remain inside viewport.
7. Check browser/dev console for warnings and errors.

## Current Baseline Files

- `packages/desktop/src/renderer/src/App.tsx`
- `packages/desktop/src/renderer/src/assets/styles.css`
- `packages/desktop/src/renderer/index.html`
- `packages/desktop/package.json`
