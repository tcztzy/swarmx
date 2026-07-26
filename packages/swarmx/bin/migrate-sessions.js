#!/usr/bin/env node

process.argv.splice(2, 0, "sessions", "migrate");
await import("@swarmx/cli");
