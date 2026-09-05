# Git and DVC

The Host-owned `ProductServices` instance owns one `DvcService` and its Git status projection for
the authorized workspace. Renderer REST reads Git/DVC state and may request bounded DVC pull or
reproduce operations. Native Agents reach the same owner through product MCP carriers.

The package is vendor-neutral and receives no Agent transcript, A2A state, AG-UI event, or renderer
type. Workspace authorization and subprocess cancellation are enforced at the Host boundary.
