# Harness selector QA

Final result: **passed**.

## Visual evidence

Source truth: the two user-supplied Codex composer screenshots:

- `/var/folders/dc/cbvy15k16vz7s8mls82v1ng80000gn/T/codex-clipboard-cbe99690-d0af-4e24-9f09-11d1b4f93c18.png` — 772 × 382 pixels, model menu.
- `/var/folders/dc/cbvy15k16vz7s8mls82v1ng80000gn/T/codex-clipboard-d5d5679a-71f2-426a-bd79-5779d9aedd4a.png` — 765 × 205 pixels, effort popover.

The implementation is the production renderer served by the authenticated local Host in the
Codex in-app browser. Desktop captures use a 1040 × 760 CSS viewport; narrow captures use
390 × 720. Captured pixels match CSS dimensions, so no density scaling was applied. Source
images are cropped composer regions, not full app windows; comparison focuses on those regions.

- `artifacts/harness-qa/model-desktop.jpg`: initial model menu, native Astra selection.
- `artifacts/harness-qa/effort-desktop.jpg`: native Max effort, slider, reset and supported levels.
- `artifacts/harness-qa/model-narrow.jpg`: narrow model menu and visible send control.
- `artifacts/harness-qa/harness-narrow.jpg`: four native Harness choices.
- `artifacts/harness-qa/model-desktop-final.jpg`: final build after the native Sol/Medium run.
- `artifacts/harness-qa/controls.jpg`: 784 × 420 focused crop from the final desktop capture,
  at x=256, y=340, with no resizing or reconstruction.

Each source and its corresponding implementation were opened together in the same comparison
input. The final focused crop was also compared with the model-menu source in a shared input.
Native Electron accessibility checks verified the same controls; browser captures provide the
visual evidence because native window screenshot capture was unavailable.

## Findings

No open P0, P1 or P2 visual findings.

| Surface | Assessment |
| --- | --- |
| Typography | System sans-serif, compact 12–13 px controls, clear menu hierarchy and readable model names. Long names truncate on the trigger, while menu labels remain readable. |
| Layout | Rounded composer and 256 px popovers, right-aligned model/effort controls, selected checkmark and compact spacing follow the references. Narrow controls wrap without hiding Send or overflowing the viewport. |
| Colors | White surfaces, neutral text/borders and subtle shadows match the existing product. Indigo is reserved for the effort control. |
| Assets | Controls reuse the project's existing icon component. No decorative or raster assets are needed for this scope. |
| Content | Chinese product labels, native model names and an explicit next-message hint. Harness is the requested additional control. |

Intentional differences: effort levels have clickable labels for direct selection and keyboard
access; this makes the popover taller than the reference. Native catalogs determine available
models and levels, so the UI does not invent the reference's recommendation group or choices.
The preview contains a short verification conversation rather than the source's file-change
and permission content. These differences are accepted for the requested selector scope.

## Comparison and verification history

The first desktop/narrow comparisons found no actionable visual mismatch. Functional review
then found two P2 recovery/race issues: a history failure locked Harness selection, and a
pending new-session request did not lock it. Regression tests reproduced both. The final build
permits switching after history failure and locks Harness during creation; the 13 renderer tests
pass. Final desktop evidence confirms the visual treatment remains intact after these fixes.

Verified interactions: Harness switching between Codex and Claude; native model loading;
model selection preserving draft/history; slider arrow keys; reset to native default; disabled
controls during generation; native Sol/Medium response and history persistence; narrow menus.
The preview console reported no errors during the live run. Native catalog reads succeeded for
Codex and Claude. Hermes/OpenClaw setting mappings are covered by simulated protocol tests;
authenticated live runs were not performed for them. Hermes does not enumerate reliable effort
levels, so its effort control stays disabled.

Full suite before the final UI guards: 229 passed, 2 opt-in tests skipped. Final guard regression
suite: 13 passed. Host/renderer type checks, production build, lint and documentation coverage
passed. The renderer retains the existing large-chunk build warning.

Implementation checklist complete; no further visual polish is required for this scope.
