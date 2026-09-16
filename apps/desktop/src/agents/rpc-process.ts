import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import {
  createJSONRPCSuccessResponse,
  isJSONRPCRequest,
  isJSONRPCResponse,
  JSONRPCClient,
  type JSONRPCRequest,
  JSONRPCServer,
  JSONRPCServerAndClient,
} from "json-rpc-2.0";

/** Native Codex and Hermes NDJSON transport; the library owns JSON-RPC validation and dispatch. */
export function rpcProcess(
  command: string,
  args: string[],
  cwd: string,
  receive: (request: JSONRPCRequest) => Promise<unknown>,
  failed: (error: Error) => void,
  env: NodeJS.ProcessEnv = process.env,
) {
  const lifetime = new AbortController();
  // A native runtime may leave descendants holding inherited stdout; dispose must reach them.
  const detached = process.platform !== "win32";
  const child = spawn(command, args, {
    cwd,
    env: { ...env, ELECTRON_RUN_AS_NODE: "1" },
    stdio: ["pipe", "pipe", "inherit"],
    detached,
  });
  const terminate = (signal: NodeJS.Signals) => {
    try {
      if (detached && child.pid !== undefined) process.kill(-child.pid, signal);
      else child.kill(signal);
    } catch {
      // The owned process group is already gone.
    }
  };
  const closed = new Promise<void>((resolve) => child.once("close", () => resolve()));
  const rpc = new JSONRPCServerAndClient(
    new JSONRPCServer(),
    new JSONRPCClient((message) => {
      child.stdin.write(`${JSON.stringify(message)}\n`);
    }),
  );
  const fail = (error: Error) => {
    if (lifetime.signal.aborted) return;
    lifetime.abort(error);
    rpc.rejectAllPendingRequests(error.message);
    failed(error);
  };
  rpc.server.applyMiddleware(async (_next, request) => {
    try {
      const result = await receive(request);
      return request.id === undefined ? null : createJSONRPCSuccessResponse(request.id, result);
    } catch (error) {
      if (request.id === undefined) fail(error instanceof Error ? error : new Error(String(error)));
      throw error;
    }
  });
  child.once("error", fail);
  child.stdin.on("error", fail);
  child.once("exit", (code, signal) => fail(new Error(`Agent exited (${signal ?? code}).`)));
  const lines = createInterface({ input: child.stdout });
  lines.on("line", (line) => {
    void Promise.resolve()
      .then(async () => {
        // Codex's native envelope omits jsonrpc; it otherwise uses the library's standard framing.
        const message = { jsonrpc: "2.0", ...JSON.parse(line) };
        if (!isJSONRPCRequest(message) && !isJSONRPCResponse(message))
          throw new Error("Invalid native JSON-RPC frame.");
        await rpc.receiveAndSend(message);
      })
      .catch(fail);
  });
  return {
    signal: lifetime.signal as AbortSignal,
    async request(method: string, params: object): Promise<unknown> {
      lifetime.signal.throwIfAborted();
      return rpc.request(method, params);
    },
    notify: (method: string, params: object) => {
      lifetime.signal.throwIfAborted();
      return rpc.notify(method, params);
    },
    async dispose() {
      lifetime.abort(new Error("Native connection closed."));
      rpc.rejectAllPendingRequests("Native connection closed.");
      lines.close();
      terminate("SIGTERM");
      const force = setTimeout(() => terminate("SIGKILL"), 3_000);
      force.unref();
      try {
        await closed;
      } finally {
        clearTimeout(force);
      }
    },
  };
}
