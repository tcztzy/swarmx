import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import {
  createJSONRPCSuccessResponse,
  JSONRPCClient,
  type JSONRPCRequest,
  JSONRPCServer,
  JSONRPCServerAndClient,
} from "json-rpc-2.0";

/** Shared by the two native NDJSON APIs; the library owns JSON-RPC requests and errors. */
export function rpcProcess(
  command: string,
  args: string[],
  cwd: string,
  receive: (request: JSONRPCRequest) => Promise<unknown>,
  failed: (error: Error) => void,
) {
  const child = spawn(command, args, {
    cwd,
    env: { ...process.env, ELECTRON_RUN_AS_NODE: "1" },
    stdio: ["pipe", "pipe", "inherit"],
  });
  const rpc = new JSONRPCServerAndClient(
    new JSONRPCServer(),
    new JSONRPCClient((message) => {
      child.stdin.write(`${JSON.stringify(message)}\n`);
    }),
  );
  rpc.server.applyMiddleware(async (_next, request) => {
    const result = await receive(request);
    return request.id === undefined ? null : createJSONRPCSuccessResponse(request.id, result);
  });
  const fail = (error: Error) => {
    rpc.rejectAllPendingRequests(error.message);
    failed(error);
  };
  child.once("error", fail);
  child.once("exit", (code, signal) => fail(new Error(`Agent exited (${signal ?? code}).`)));
  const lines = createInterface({ input: child.stdout });
  lines.on("line", (line) => {
    // Codex intentionally omits the jsonrpc field in its native envelopes.
    void Promise.resolve()
      .then(() => rpc.receiveAndSend({ jsonrpc: "2.0", ...JSON.parse(line) }))
      .catch(fail);
  });
  return {
    request: (method: string, params: object): Promise<unknown> =>
      Promise.resolve(rpc.request(method, params)),
    notify: (method: string, params: object) => rpc.notify(method, params),
    async dispose() {
      lines.close();
      child.kill();
    },
  };
}
