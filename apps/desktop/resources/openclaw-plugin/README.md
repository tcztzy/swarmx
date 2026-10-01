# SwarmX Host tools for OpenClaw

Exposes the SwarmX Host product tools (shared Memory first) to OpenClaw sessions.

The plugin reads a bridge descriptor that the SwarmX Host publishes under
`$SWARMX_HOME/openclaw/bridges/*.json` while it runs, and forwards each tool call
over the Host's private Unix socket. The Host authorizes every call against the
execution that owns the calling OpenClaw session; sessions without an active
execution are rejected.

Install on the machine that runs the OpenClaw Gateway, with the same user and
filesystem as the SwarmX Host:

```sh
openclaw plugins install <path-to-this-directory> --force --accept-capabilities
```

Set `SWARMX_OPENCLAW_BRIDGE` to a descriptor path to bypass bridge discovery, or
`SWARMX_HOME` when the Host uses a non-default product directory.
