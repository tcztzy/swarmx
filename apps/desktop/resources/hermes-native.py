"""Start Hermes' own gateway with the execution's process-local MCP tools."""

import contextlib
import json
import os
import sys

import hermes_bootstrap

hermes_bootstrap.harden_import_path()

from tui_gateway import entry, server


@server.method("swarmx.session.wait")
def wait_for_execution(rid, params):
    session, error = server._sess_nowait(params, rid)
    if error:
        return error
    previous = None
    while True:
        with server._sessions_lock:
            thread = session.get("_run_thread")
            if thread is None or thread is previous:
                return server._ok(rid, {"running": bool(session.get("running"))})
        # Native idle precedes queued follow-ups; join outside the lock they need to start.
        thread.join()
        previous = thread


server._LONG_HANDLERS |= {"swarmx.session.wait"}

servers = json.loads(os.environ.pop("SWARMX_HERMES_MCP", "{}"))
if servers:
    with contextlib.redirect_stdout(sys.stderr):
        from tools.mcp_tool import get_registered_mcp_server_names, register_mcp_servers

        enabled = server._load_enabled_toolsets()
        register_mcp_servers(servers)
        if not set(servers).issubset(get_registered_mcp_server_names()):
            raise RuntimeError("Hermes could not register the Host MCP tools")
        if enabled is not None:
            os.environ["HERMES_TUI_TOOLSETS"] = ",".join(
                dict.fromkeys([*enabled, *(f"mcp-{name}" for name in servers)])
            )

entry.main()
