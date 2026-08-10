# SwarmX application icon

`icon.png` and `icon.icns` are static packaging assets generated from
`../src/renderer/public/app-icon.svg`. The SVG remains the canonical source
used by the renderer and favicon.

Matching PNG exports for the transparent, grayscale, white, and black SVG
variants are stored in `variants/`. See
`../src/renderer/public/ICON_VARIANTS.md` for intended usage.

Regenerate the SVG and its embedded Renderer data URL with
`scripts/rebuild-icon.py`, then render a 1024 px PNG with `rsvg-convert` and
assemble the standard macOS icon sizes with `iconutil`.

`mem-runtime/` is generated separately by
`../scripts/build-mem-runtime.mjs`. It contains the target-specific private
`swarmx-mem` MCP executable and a SHA-256 manifest consumed by Desktop Main.
The directory is generated and ignored; release packaging rebuilds it from the
locked repository Cargo crate rather than accepting a locally installed
binary.
