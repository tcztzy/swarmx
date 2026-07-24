# SwarmX icon variants

| Asset | Background | Color | Suggested use |
|---|---|---|---|
| `app-icon.svg` | Dark rounded panel | Cyan, blue, violet | Default application icon and favicon |
| `app-icon-transparent.svg` | Transparent | Cyan, blue, violet | Documents and layouts that provide their own background |
| `app-icon-grayscale.svg` | Dark grayscale panel | White through charcoal | Neutral UI, print previews, accessibility review |
| `app-icon-monochrome-light.svg` | Transparent | White | One-color marks on dark backgrounds |
| `app-icon-monochrome-dark.svg` | Transparent | Near-black | One-color marks on light backgrounds |

The color and grayscale SVGs retain the circulating color animation and honor
`prefers-reduced-motion`. The monochrome SVGs are static. Matching 1024 px PNG
exports live in `../../../build/variants`.
