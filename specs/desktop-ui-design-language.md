# Desktop UI Design Language Spec

## Status

Accepted baseline for the Electron desktop renderer. Updated 2026-06-17 with the
Liquid Runtime visual language.

## Goal

Define the SwarmX desktop UI as a Codex-like agent runtime surface: dense, calm,
operational, and optimized for inspecting long-running agent sessions. The current
visual direction is Liquid Runtime: a modern dark command center with restrained
frosted-glass material, rounded controls, subtle depth, and natural transitions.

## Scope

This spec governs `packages/desktop/src/renderer`.

It covers visual language, layout, interaction states, message rendering, responsive behavior, and verification rules. It does not define core agent behavior, ACP protocol behavior, persistence, or harness execution.

## Design References

- Primary product metaphor: Codex app runtime screen.
- Component language: shadcn/ui-style composition, neutral tokens, small radii, visible focus states, semantic variants.
- Icon language: lucide-style line icons with consistent optical weight.
- Material direction: Apple Liquid Glass uses translucent, adaptive layers that
  bring focus to content and navigation without breaking familiarity
  (https://www.apple.com/newsroom/2025/06/apple-introduces-a-delightful-and-elegant-new-software-design/).
- Practical desktop material: Fluent Acrylic uses translucency, blur, tint, and
  texture for depth, but should avoid stacked acrylic panes and protect legibility
  (https://learn.microsoft.com/en-us/windows/apps/design/style/acrylic).
- Motion reference: Material 3 Expressive validates motion, shape, color, and
  containment when they clarify hierarchy and user journeys, not when they replace
  familiar patterns (https://design.google/library/expressive-material-design-google-research).

## Core Principles

1. The UI is an agent runtime, not a marketing page.
2. The first screen must be usable immediately: session rail, runtime transcript, and composer are all visible.
3. Preserve information density without visual noise.
4. Prefer structural hierarchy over decorative treatment.
5. Use neutral color tokens with limited cool accents; avoid one-note accent
   palettes and decorative gradients.
6. Keep controls code-native and interactive.
7. Do not hide agent/tool execution behind generic chat bubbles.
8. Glass is a functional layer for chrome, controls, and event surfaces. It must
   increase depth and focus without reducing transcript readability.
9. Motion should make state changes feel responsive and continuous. It must stay
   short, interruptible, and compatible with `prefers-reduced-motion`.

## Behavior Invariants

| id | Rule |
| --- | --- |
| V1 | Every visible session row is actionable. Clicking a session loads its latest available conversation into the runtime transcript. |
| V2 | Session detail loading uses one cache path for local and ACP sessions. Local sessions may auto-preload; ACP sessions preload only on user intent such as hover, focus, or click. |
| V3 | A preloaded session detail must be reused on click without an immediate duplicate load request. Failed preloads must not poison the cache. |
| V4 | Conversational message content (`message` and `thinking`) renders through safe Markdown with raw HTML escaped, and inline/fenced code uses monospace code styling. Tool call/result content remains literal text. |
| V5 | Glass material must preserve text contrast and must degrade to solid neutral surfaces when blur, transparency, or motion are unavailable or reduced. |
| V6 | Visual transitions must be functional, short, and natural: hover, selection, sidebar collapse, loading, composer focus, and event entry may animate; continuous decorative motion is forbidden. |

## Bug Log

| id | date | cause | fix |
| --- | --- | --- | --- |
| B1 | 2026-06-11 | ACP session rows were disabled and the renderer only loaded local session files by id. | V1, V2, V3 |
| B2 | 2026-06-11 | Conversational messages were rendered as plain text, so Markdown code spans stayed visible as backtick text in the sans body font. | V4 |

## Visual Tokens

### Color

Use dark neutral tokens as the default desktop theme. Liquid Runtime adds
translucent material tokens on top of the existing graphite base:

| Token | Value | Use |
| --- | --- | --- |
| `--background` | `#07080b` | App root and runtime canvas |
| `--foreground` | `#f6f7fb` | Primary text |
| `--card` | `rgba(18, 20, 26, 0.78)` | Panels, composer, neutral event blocks |
| `--card-strong` | `rgba(24, 27, 35, 0.9)` | Higher-emphasis glass surfaces |
| `--card-hover` | `rgba(38, 42, 54, 0.72)` | Hover and selected surfaces |
| `--muted` | `#b1b7c3` | Secondary text and icons |
| `--muted-foreground` | `#77808f` | Tertiary text and metadata |
| `--border` | `rgba(180, 193, 214, 0.18)` | Primary glass borders |
| `--border-subtle` | `rgba(180, 193, 214, 0.11)` | Dividers and low-emphasis borders |
| `--input` | `rgba(12, 14, 19, 0.72)` | Inputs and select controls |
| `--primary` | `#f4f4f5` | Primary button surface |
| `--primary-foreground` | `#09090b` | Primary button text |
| `--accent` | `#95e9ff` | Sparse focus/highlight edge only |
| `--danger` | `#f87171` | Errors and destructive status |
| `--success` | `#34d399` | Active/running/success status |
| `--glass-blur` | `22px` | Standard backdrop blur |
| `--glass-noise-opacity` | `0.035` | Subtle material grain |
| `--glass-highlight` | `rgba(255, 255, 255, 0.12)` | Inset edge highlight |

Accent colors are semantic or edge-only. Do not introduce purple, blue, beige,
orange, or decorative color systems unless a future spec explicitly changes the
theme. A small cyan highlight is allowed for focus rings, active edges, and
selected glass surfaces, but it must not dominate the screen.

### Shape

- Default radius: `10px`.
- Compact inner radius: `8px`.
- Large glass surface radius: `14px`.
- Docked composer radius: `16px`.
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
- Default transition: `180ms` to `220ms` using `cubic-bezier(0.2, 0.8, 0.2, 1)`.
- Hover and press states may use `translateY(-1px)`, border/highlight changes,
  and soft shadow changes.
- Sidebar collapse: opacity, grid width, and translate transition.
- Composer focus: border, highlight, and shadow transition.
- Event entry: subtle opacity/translate transition only when new content appears.
- Loading icons may spin.
- Honor `prefers-reduced-motion: reduce` by removing transforms, smooth scroll,
  and non-essential animation.
- Avoid animated backgrounds, decorative pulses, or ornamental motion.

## Layout

### App Shell

The desktop app is a two-column grid with a deep graphite runtime base and glass
chrome layered above it:

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

Session rows use icon, title, and metadata. All visible local and ACP sessions are
selectable; hover/focus may preload detail so click-to-open feels immediate.
The sidebar uses a translucent in-app material with solid fallback. It should
feel like persistent app chrome, not a floating card.

### Runtime Header

The header is a compact command/status bar:

- Left: sidebar toggle, current run title, run subtitle.
- Right: running/ready badge, event count, alerts, destructive action when applicable.
- Height target: `58px` on desktop.
- It should feel like app chrome, not a page hero.
- It uses frosted material and a low-contrast bottom edge so the transcript can
  scroll beneath the chrome without losing context.

### Transcript

The transcript is centered with a maximum width near `900px`.

Messages render as an event stream:

- User message: bright foreground card, readable contrast.
- Assistant message: neutral event card.
- Thinking: subdued event card with loading affordance.
- Tool call/result: monospace-friendly card with tool label.
- System/error: danger semantic card.

Do not render all message kinds as identical chat bubbles.
Cards may use glass material, but message text sits on a sufficiently tinted
surface. User messages remain the brightest event surface for fast scanning.

Conversational message content supports safe Markdown for common authoring syntax:

- Inline code and fenced code blocks render with the mono stack.
- Lists, links, blockquotes, and GFM tables render inside the event card without changing the app shell layout.
- Raw HTML is not interpreted as HTML.
- Tool call/result content remains literal output text and is not Markdown-rendered.

### Composer

The composer is docked at the bottom of the runtime:

- Width matches transcript max width.
- Contains textarea, harness context, and Send button.
- Uses a bordered elevated glass surface with visible focus treatment.
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

Buttons with icons use icon+text for clear commands and icon-only for chrome
actions. Icon-only buttons must have accessible labels or titles. Hover and
press states use natural material changes rather than large color shifts.

### Badges

Use badges for runtime status and compact counters only:

- `neutral`: ready/counts.
- `active`: running/success.
- `danger`: alerts/errors.

Do not use badges as decorative labels.

### Selects

Harness selection is a compact select shell with leading icon. It must not look like a browser-default select.

### Segmented Controls

Use segmented controls for 2-option mode switches such as Harness/Project
grouping. Active state is a contained glass surface with a subtle active edge,
not a colorful accent fill.

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
- Glass and blur effects must have solid-color fallbacks.
- Motion-heavy effects must be reduced under `prefers-reduced-motion`.

## Forbidden Patterns

- Marketing hero sections.
- Decorative blobs, orbs, bokeh, or ornamental gradients.
- Full-screen novelty glass that makes content harder to read.
- Multiple adjacent transparent panes that create noisy seams.
- Continuous decorative motion or animated backgrounds.
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
6. Implement glass as CSS material tokens and reusable surface patterns; do not
   add image assets or screenshot-based UI to simulate the effect.
7. Keep all new animation CSS behind tokenized timing/easing and reduced-motion
   fallbacks.

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
   - Glass surfaces preserve contrast.
   - Hover, selected, and focus states transition naturally.
6. Verify narrow viewport:
   - Sidebar collapsed.
   - Header not overlapping.
   - Transcript content scrolls.
   - Composer and Send button remain inside viewport.
   - Rounded controls do not clip or overflow.
7. Check browser/dev console for warnings and errors.
8. Check reduced-motion behavior by confirming non-essential transforms and
   smooth scrolling are disabled.

## Current Baseline Files

- `packages/desktop/src/renderer/src/App.tsx`
- `packages/desktop/src/renderer/src/assets/styles.css`
- `packages/desktop/src/renderer/index.html`
- `packages/desktop/package.json`
