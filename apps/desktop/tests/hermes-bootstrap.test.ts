import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { expect, it } from "vitest";

const launcher = fileURLToPath(new URL("../resources/hermes-native.py", import.meta.url));
const fixture = `
import os, runpy, sys, threading, types
os.environ["SWARMX_HERMES_MCP"] = '{"swarmx":{"url":"http://fixture"}}'
bootstrap = types.ModuleType("hermes_bootstrap")
bootstrap.harden_import_path = lambda: None
sys.modules["hermes_bootstrap"] = bootstrap
gateway = types.ModuleType("tui_gateway")
methods, sessions = {}, {}
def method(name):
    def register(handler):
        methods[name] = handler
        return handler
    return register
def session_nowait(params, rid):
    session = sessions.get(params.get("session_id"))
    return (session, None) if session else (None, {
        "jsonrpc": "2.0", "id": rid,
        "error": {"code": 4001, "message": "session not found"},
    })
gateway.server = types.SimpleNamespace(
    _load_enabled_toolsets=lambda: ["terminal"],
    method=method,
    _LONG_HANDLERS=frozenset({"prompt.submit"}),
    _sessions_lock=threading.Lock(),
    _sess_nowait=session_nowait,
    _ok=lambda rid, result: {"jsonrpc": "2.0", "id": rid, "result": result},
)
def main():
    assert "SWARMX_HERMES_MCP" not in os.environ
    assert os.environ["HERMES_TUI_TOOLSETS"] == "terminal,mcp-swarmx"
    print("native gateway")
gateway.entry = types.SimpleNamespace(main=main)
sys.modules["tui_gateway"] = gateway
sys.modules["tools"] = types.ModuleType("tools")
mcp = types.ModuleType("tools.mcp_tool")
def register(servers):
    assert servers == {"swarmx": {"url": "http://fixture"}}
    print("native registration diagnostic")
mcp.register_mcp_servers = register
mcp.get_registered_mcp_server_names = lambda: {"swarmx"} if sys.argv[2] == "registered" else set()
sys.modules["tools.mcp_tool"] = mcp
`;

function runFixture(registration: string, checks = "", scenario = "") {
  return promisify(execFile)("python3", [
    "-c",
    `${fixture}\n${checks}\nrunpy.run_path(sys.argv[1], run_name="__main__")`,
    launcher,
    registration,
    scenario,
  ]);
}

it("enables the registered native MCP toolset and keeps diagnostics off protocol stdout", async () => {
  const output = await runFixture("registered");
  expect(output.stdout).toBe("native gateway\n");
  expect(output.stderr).toBe("native registration diagnostic\n");
});

it("fails startup when Hermes could not register the required Host tools", async () => {
  await expect(runFixture("missing")).rejects.toMatchObject({
    stdout: "",
    stderr: expect.stringContaining("Hermes could not register the Host MCP tools"),
  });
});

it.each(["consumed", "queued"])(
  "waits for native prompt thread cleanup when pending input is %s",
  async (scenario) => {
    const output = await runFixture(
      "registered",
      `
def verify_wait():
    server = gateway.server
    assert server._LONG_HANDLERS == {"prompt.submit", "swarmx.session.wait"}
    wait = methods["swarmx.session.wait"]
    release_current, release_followup = threading.Event(), threading.Event()
    followup_started, finished = threading.Event(), threading.Event()
    responses = []
    session = sessions["live"] = {"running": True}
    class RunThread(threading.Thread):
        def __init__(self, target):
            super().__init__(target=target, daemon=True)
            self.joining = threading.Event()
        def join(self, timeout=None):
            self.joining.set()
            super().join(timeout)
    def followup():
        assert release_followup.wait(2), "follow-up was not released"
        session["running"] = False
    next_thread = RunThread(followup)
    def current():
        assert release_current.wait(2), "current run was not released"
        if sys.argv[3] == "queued":
            with server._sessions_lock:
                session["running"] = True
                session["_run_thread"] = next_thread
                next_thread.start()
                followup_started.set()
        else:
            session["running"] = False
    current_thread = RunThread(current)
    with server._sessions_lock:
        session["_run_thread"] = current_thread
        current_thread.start()
    # Native idle can precede pending-input draining at the end of the run.
    session["running"] = False
    def call_wait():
        responses.append(wait(7, {"session_id": "live"}))
        finished.set()
    threading.Thread(target=call_wait, daemon=True).start()
    assert current_thread.joining.wait(1), "wait must join the run during idle"
    assert not finished.is_set(), "wait returned before native cleanup"
    release_current.set()
    if sys.argv[3] == "queued":
        assert followup_started.wait(1), "joining while locked blocks the follow-up"
        assert next_thread.joining.wait(1), "wait did not follow the replacement run"
        assert not finished.is_set(), "wait returned before the follow-up ended"
        release_followup.set()
    assert finished.wait(1), "wait did not settle after the last run ended"
    assert responses == [{"jsonrpc": "2.0", "id": 7, "result": {"running": False}}]
    print("native wait complete")
gateway.entry.main = verify_wait
`,
      scenario,
    );
    expect(output.stdout).toBe("native wait complete\n");
    expect(output.stderr).toBe("native registration diagnostic\n");
  },
);

it.each(["not-started", "stale-running", "missing"])(
  "preserves the native session result for %s sessions",
  async (scenario) => {
    const output = await runFixture(
      "registered",
      `
def verify_wait():
    wait = methods["swarmx.session.wait"]
    scenario = sys.argv[3]
    if scenario == "missing":
        expected = {"error": {"code": 4001, "message": "session not found"}}
    else:
        running = scenario == "stale-running"
        session = sessions["live"] = {"running": running}
        if running:
            thread = threading.Thread(target=lambda: None)
            thread.start()
            thread.join()
            session["_run_thread"] = thread
        expected = {"result": {"running": running}}
    assert wait(8, {"session_id": "live"}) == {"jsonrpc": "2.0", "id": 8, **expected}
    print("native session result")
gateway.entry.main = verify_wait
`,
      scenario,
    );
    expect(output.stdout).toBe("native session result\n");
    expect(output.stderr).toBe("native registration diagnostic\n");
  },
);
