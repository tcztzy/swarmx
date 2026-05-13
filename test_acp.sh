#!/usr/bin/env bash
# Test claude-agent-acp connectivity end-to-end.
# Runs the ACP client, sends initialize + session/list, prints result.
set -euo pipefail

ACP_BIN="claude-agent-acp"
echo "=== ACP Diagnostic ==="

# 1. Check binary
echo -n "1. Binary: "
if command -v "$ACP_BIN" &>/dev/null; then
    echo "FOUND at $(command -v "$ACP_BIN")"
elif command -v bun &>/dev/null && bun x "@agentclientprotocol/claude-agent-acp" --version &>/dev/null; then
    echo "FOUND via bun x"
    ACP_BIN="bun x @agentclientprotocol/claude-agent-acp"
else
    echo "NOT FOUND — install with: bun install -g @agentclientprotocol/claude-agent-acp"
    exit 1
fi

# 2. Version
echo -n "2. Version: "
$ACP_BIN --version 2>&1 || echo "(no --version flag)"

# 3. Start ACP process and send initialize + session/list via stdin
echo "3. ACP protocol test (initialize + session/list):"
TMPDIR="${TMPDIR:-/tmp}"
ACP_OUT="$TMPDIR/acp_test_output.$$.json"

# Send ACP JSON-RPC messages to the process
# Initialize request
INIT_REQ='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientInfo":{"name":"swarmx-test","version":"0.1"}}}'
# Session list request
LIST_REQ='{"jsonrpc":"2.0","id":2,"method":"session/list","params":{}}'

# Run ACP, feed requests with delay so server processes both
(echo "$INIT_REQ"; sleep 1; echo "$LIST_REQ"; sleep 5) | $ACP_BIN 2>"$ACP_OUT.stderr" | head -20 > "$ACP_OUT.stdout" || true

echo "--- stdout ---"
cat "$ACP_OUT.stdout" 2>/dev/null || echo "(empty)"
echo "--- stderr ---"
cat "$ACP_OUT.stderr" 2>/dev/null || echo "(empty)"

# Check if we got valid JSON responses
INIT_RESP=$(grep -o '"id":1[^}]*}' "$ACP_OUT.stdout" 2>/dev/null || echo "")
LIST_RESP=$(grep -o '"id":2[^}]*}' "$ACP_OUT.stdout" 2>/dev/null || echo "")

if [ -n "$INIT_RESP" ]; then
    echo "4. Initialize: OK (got response)"
else
    echo "4. Initialize: FAILED (no response)"
fi

if echo "$LIST_RESP" | grep -q "sessions"; then
    echo "5. Session list: OK (got sessions)"
elif [ -n "$LIST_RESP" ]; then
    echo "5. Session list: GOT RESPONSE but no sessions field"
else
    echo "5. Session list: FAILED (no response)"
fi

# Cleanup
rm -f "$ACP_OUT" "$ACP_OUT.stdout" "$ACP_OUT.stderr"
echo "=== Done ==="
