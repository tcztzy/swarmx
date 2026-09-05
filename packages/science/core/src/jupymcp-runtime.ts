import { createHash } from "node:crypto";
import { chmodSync, existsSync } from "node:fs";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import type { CallToolResult, ReadResourceResult } from "@modelcontextprotocol/sdk/types.js";
import { CallToolResultSchema } from "@modelcontextprotocol/sdk/types.js";
import type { NotebookOutputBlock } from "./contracts.js";
import { ScienceError } from "./errors.js";
import { scrubbedScienceEnvironment } from "./subprocess.js";

const NOTEBOOK_PREFIX = ".swarmx-science-notebook-";
const NOTEBOOK_ID = /^[A-Za-z0-9_-]{1,200}$/u;
const DISABLED_ENVIRONMENT_KEYS = [
  "ALL_PROXY",
  "HTTP_PROXY",
  "HTTPS_PROXY",
  "PYTHONHOME",
  "PYTHONINSPECT",
  "PYTHONPATH",
  "PYTHONSTARTUP",
  "all_proxy",
  "http_proxy",
  "https_proxy",
] as const;

export type JupyMcpToolResult = Pick<CallToolResult, "content" | "isError">;
export type JupyMcpResourceResult = Pick<ReadResourceResult, "contents">;

export interface JupyMcpPeer {
  callTool(
    name: string,
    args: Readonly<Record<string, unknown>>,
    signal?: AbortSignal,
  ): Promise<JupyMcpToolResult>;
  readResource(uri: string, signal?: AbortSignal): Promise<JupyMcpResourceResult>;
  close(): Promise<void>;
}

export interface JupyMcpConnectionOptions {
  readonly args: readonly string[];
  readonly command: string;
  readonly cwd: string;
  readonly env: Readonly<Record<string, string>>;
  readonly requestTimeoutMs: number;
  readonly signal?: AbortSignal;
}

export type JupyMcpPeerFactory = (options: JupyMcpConnectionOptions) => Promise<JupyMcpPeer>;

export interface JupyMcpRuntimeConfig {
  readonly args: readonly string[];
  readonly command: string;
  readonly env?: Readonly<Record<string, string>>;
  readonly maxNotebookBytes: number;
  readonly maxOutputBytes: number;
  readonly requestTimeoutMs: number;
}

export interface JupyMcpExecutionRequest {
  readonly inputEnvironment?: Readonly<Record<string, string>>;
  readonly notebookId: string;
  readonly source: string;
  readonly workspaceKey: string;
  readonly workspaceRoot: string;
}

export interface JupyMcpExecutionResult {
  readonly durationMs: number;
  readonly environment: Record<string, string>;
  readonly exitCode: number | null;
  readonly outputs: NotebookOutputBlock[];
  readonly signal: string | null;
  readonly status: "succeeded" | "failed";
  readonly stderr: { readonly text: string; readonly truncated: boolean };
  readonly stdout: { readonly text: string; readonly truncated: boolean };
}

interface BoundedText {
  readonly text: string;
  readonly truncated: boolean;
}

function childEnvironment(explicit: Readonly<Record<string, string>> = {}): Record<string, string> {
  const environment = { ...scrubbedScienceEnvironment(), ...explicit };
  for (const key of DISABLED_ENVIRONMENT_KEYS) delete environment[key];
  environment.NO_PROXY = "*";
  environment.no_proxy = "*";
  return environment;
}

function requestOptions(signal: AbortSignal | undefined, timeout: number) {
  return signal ? { signal, timeout } : { timeout };
}

async function connectJupyMcp(options: JupyMcpConnectionOptions): Promise<JupyMcpPeer> {
  const client = new Client({ name: "swarmx-science", version: "0.1.0" }, { capabilities: {} });
  const transport = new StdioClientTransport({
    args: [...options.args],
    command: options.command,
    cwd: options.cwd,
    env: { ...options.env },
    maxBufferSize: 10 * 1024 * 1024,
    stderr: "pipe",
  });
  try {
    await client.connect(transport, requestOptions(options.signal, options.requestTimeoutMs));
  } catch (error) {
    await client.close().catch(() => undefined);
    throw error;
  }
  return {
    async callTool(name, args, signal) {
      const result = await client.request(
        {
          method: "tools/call",
          params: { arguments: { ...args }, name },
        },
        CallToolResultSchema,
        requestOptions(signal, options.requestTimeoutMs),
      );
      return {
        content: result.content,
        ...(result.isError === undefined ? {} : { isError: result.isError }),
      };
    },
    async close() {
      await client.close();
    },
    async readResource(uri, signal) {
      const result = await client.readResource(
        { uri },
        requestOptions(signal, options.requestTimeoutMs),
      );
      return { contents: result.contents };
    },
  };
}

class OutputBudget {
  private remaining: number;

  constructor(maxBytes: number) {
    this.remaining = maxBytes;
  }

  base64(data: string): { readonly data: string; readonly truncated: boolean } {
    const decoded = Buffer.from(data, "base64");
    if (decoded.toString("base64") !== data) return { data: "", truncated: true };
    if (decoded.byteLength <= this.remaining) {
      this.remaining -= decoded.byteLength;
      return { data, truncated: false };
    }
    this.remaining = 0;
    return { data: "", truncated: true };
  }

  text(value: string): BoundedText {
    const bytes = Buffer.from(value, "utf8");
    if (bytes.byteLength <= this.remaining) {
      this.remaining -= bytes.byteLength;
      return { text: value, truncated: false };
    }
    const text = bytes
      .subarray(0, this.remaining)
      .toString("utf8")
      .replace(/\uFFFD$/u, "");
    this.remaining = 0;
    return { text, truncated: true };
  }
}

function blockMetaName(block: CallToolResult["content"][number]): unknown {
  return block._meta?.name;
}

function diagnosticOutput(message: string, budget: OutputBudget): NotebookOutputBlock {
  const bounded = budget.text(message);
  return {
    type: "execute_result",
    data: [
      {
        mime: "text/plain",
        data: bounded.text,
        encoding: "utf8",
        truncated: bounded.truncated,
      },
    ],
  };
}

function streamOutput(
  outputs: readonly NotebookOutputBlock[],
  name: "stdout" | "stderr",
): BoundedText {
  const blocks = outputs.filter(
    (output): output is Extract<NotebookOutputBlock, { type: "stream" }> =>
      output.type === "stream" && output.name === name,
  );
  return {
    text: blocks.map((output) => output.text).join(""),
    truncated: blocks.some((output) => output.truncated),
  };
}

function normalizeContent(
  result: JupyMcpToolResult,
  maxOutputBytes: number,
): {
  readonly outputs: NotebookOutputBlock[];
  readonly status: "succeeded" | "failed";
  readonly stderr: BoundedText;
  readonly stdout: BoundedText;
} {
  const budget = new OutputBudget(maxOutputBytes);
  if (result.isError) {
    const message =
      result.content
        .filter((block): block is Extract<typeof block, { type: "text" }> => block.type === "text")
        .map((block) => block.text)
        .join("\n") || "JupyMCP reported an execution error";
    const bounded = budget.text(message);
    return {
      outputs: [
        {
          type: "error",
          name: "JupyMCPError",
          message: bounded.text,
          truncated: bounded.truncated,
        },
      ],
      status: "failed",
      stderr: bounded,
      stdout: { text: "", truncated: false },
    };
  }

  const outputs: NotebookOutputBlock[] = [];
  for (const block of result.content) {
    if (block.type === "text") {
      const name = blockMetaName(block);
      if (name === "stdout" || name === "stderr") {
        const bounded = budget.text(block.text);
        outputs.push({ type: "stream", name, text: bounded.text, truncated: bounded.truncated });
      } else {
        const bounded = budget.text(block.text);
        outputs.push({
          type: "execute_result",
          data: [
            {
              mime: "text/plain",
              data: bounded.text,
              encoding: "utf8",
              truncated: bounded.truncated,
            },
          ],
        });
      }
      continue;
    }
    if (block.type === "image" || block.type === "audio") {
      const bounded = budget.base64(block.data);
      outputs.push({
        type: "display_data",
        data: [
          {
            mime: block.mimeType,
            data: bounded.data,
            encoding: "base64",
            truncated: bounded.truncated,
          },
        ],
      });
      continue;
    }
    if (block.type === "resource") {
      const mime = block.resource.mimeType ?? "application/octet-stream";
      if ("text" in block.resource) {
        const bounded = budget.text(block.resource.text);
        outputs.push({
          type: "display_data",
          data: [
            {
              mime,
              data: bounded.text,
              encoding: "utf8",
              truncated: bounded.truncated,
            },
          ],
        });
      } else {
        const bounded = budget.base64(block.resource.blob);
        outputs.push({
          type: "display_data",
          data: [
            {
              mime,
              data: bounded.data,
              encoding: "base64",
              truncated: bounded.truncated,
            },
          ],
        });
      }
      continue;
    }
    if (block.type === "resource_link") {
      outputs.push(diagnosticOutput(`${block.name}: ${block.uri}`, budget));
      continue;
    }
    outputs.push(diagnosticOutput("Unsupported JupyMCP content block", budget));
  }

  return {
    outputs,
    status: "succeeded",
    stderr: streamOutput(outputs, "stderr"),
    stdout: streamOutput(outputs, "stdout"),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function multilineText(value: unknown): string | undefined {
  if (typeof value === "string") return value;
  if (Array.isArray(value) && value.every((item) => typeof item === "string")) {
    return value.join("");
  }
  return undefined;
}

function canonicalMimeData(
  value: unknown,
  budget: OutputBudget,
): Extract<NotebookOutputBlock, { type: "display_data" | "execute_result" }>["data"] {
  if (!isRecord(value)) return [];
  return Object.entries(value).map(([mime, item]) => {
    const data = multilineText(item) ?? JSON.stringify(item);
    const binary =
      mime.startsWith("image/") || mime.startsWith("audio/") || mime === "application/pdf";
    if (binary) {
      const bounded = budget.base64(data);
      return {
        mime,
        data: bounded.data,
        encoding: "base64" as const,
        truncated: bounded.truncated,
      };
    }
    const bounded = budget.text(data);
    return {
      mime,
      data: bounded.text,
      encoding: "utf8" as const,
      truncated: bounded.truncated,
    };
  });
}

function normalizeCanonicalNotebook(
  text: string,
  maxOutputBytes: number,
): ReturnType<typeof normalizeContent> | undefined {
  let notebook: unknown;
  try {
    notebook = JSON.parse(text);
  } catch {
    return undefined;
  }
  if (!isRecord(notebook) || !Array.isArray(notebook.cells)) return undefined;
  const cell = notebook.cells.findLast(
    (candidate) =>
      isRecord(candidate) && candidate.cell_type === "code" && Array.isArray(candidate.outputs),
  );
  if (!isRecord(cell) || !Array.isArray(cell.outputs)) return undefined;
  const budget = new OutputBudget(maxOutputBytes);
  const outputs: NotebookOutputBlock[] = [];
  for (const value of cell.outputs) {
    if (!isRecord(value)) continue;
    if (value.output_type === "stream") {
      const textValue = multilineText(value.text);
      const name = value.name === "stderr" ? "stderr" : "stdout";
      if (textValue === undefined) continue;
      const bounded = budget.text(textValue);
      outputs.push({ type: "stream", name, text: bounded.text, truncated: bounded.truncated });
      continue;
    }
    if (value.output_type === "error") {
      const bounded = budget.text(typeof value.evalue === "string" ? value.evalue : "Kernel error");
      outputs.push({
        type: "error",
        name: typeof value.ename === "string" ? value.ename : "KernelError",
        message: bounded.text,
        truncated: bounded.truncated,
      });
      continue;
    }
    if (value.output_type === "display_data" || value.output_type === "execute_result") {
      outputs.push({
        type: value.output_type,
        data: canonicalMimeData(value.data, budget),
      });
    }
  }
  return {
    outputs,
    status: outputs.some((output) => output.type === "error") ? "failed" : "succeeded",
    stdout: streamOutput(outputs, "stdout"),
    stderr: streamOutput(outputs, "stderr"),
  };
}

function notebookPath(notebookId: string): string {
  if (!NOTEBOOK_ID.test(notebookId)) {
    throw new ScienceError("Notebook id is unsafe for JupyMCP storage", "INVALID_REQUEST");
  }
  return `${NOTEBOOK_PREFIX}${notebookId}.ipynb`;
}

function environmentSetup(input: Readonly<Record<string, string>>): string {
  const keys = JSON.stringify(Object.keys(input));
  return [
    "import os as _swarmx_os",
    "from IPython import get_ipython as _swarmx_get_ipython",
    `_swarmx_os.environ.update(${JSON.stringify(input)})`,
    `def _swarmx_install_cleanup(_swarmx_os=_swarmx_os, _swarmx_ip=_swarmx_get_ipython(), _swarmx_keys=${keys}):`,
    "    def _swarmx_cleanup(_swarmx_result):",
    "        for _swarmx_key in _swarmx_keys:",
    "            _swarmx_os.environ.pop(_swarmx_key, None)",
    '        _swarmx_ip.events.unregister("post_run_cell", _swarmx_cleanup)',
    '    _swarmx_ip.events.register("post_run_cell", _swarmx_cleanup)',
    "_swarmx_install_cleanup()",
    "del _swarmx_install_cleanup, _swarmx_get_ipython, _swarmx_os",
  ].join("\n");
}

function waitForTurn(previous: Promise<void>, signal?: AbortSignal): Promise<void> {
  if (!signal) return previous;
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const abort = () => reject(signal.reason);
    signal.addEventListener("abort", abort, { once: true });
    previous.then(
      () => {
        signal.removeEventListener("abort", abort);
        resolve();
      },
      (error) => {
        signal.removeEventListener("abort", abort);
        reject(error);
      },
    );
  });
}

function assertToolSuccess(result: JupyMcpToolResult, action: string): void {
  if (result.isError) {
    throw new ScienceError(`JupyMCP ${action} failed`, "JUPYMCP_UNAVAILABLE");
  }
}

function startedKernelId(result: JupyMcpToolResult): string {
  assertToolSuccess(result, "kernel start");
  const id = result.content.find(
    (block): block is Extract<(typeof result.content)[number], { type: "text" }> =>
      block.type === "text",
  )?.text;
  if (id === undefined || !NOTEBOOK_ID.test(id)) {
    throw new ScienceError("JupyMCP returned an invalid kernel id", "JUPYMCP_UNAVAILABLE");
  }
  return id;
}

class JupyMcpWorkspace {
  private closePromise: Promise<void> | undefined;
  private environmentTail = Promise.resolve();

  constructor(readonly peer: JupyMcpPeer) {}

  serializeInputEnvironment<T>(operation: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    const previous = this.environmentTail.catch(() => undefined);
    let release: () => void = () => {};
    const current = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.environmentTail = previous.then(() => current);
    return (async () => {
      try {
        await waitForTurn(previous, signal);
        return await operation();
      } finally {
        release();
      }
    })();
  }

  close(): Promise<void> {
    if (this.closePromise) return this.closePromise;
    this.closePromise = (async () => {
      let failure: unknown;
      try {
        const result = await this.peer.callTool("shutdown_all", { now: true });
        assertToolSuccess(result, "workspace shutdown");
      } catch (error) {
        failure = error;
      }
      try {
        await this.peer.close();
      } catch (error) {
        failure ??= error;
      }
      if (failure) {
        throw new ScienceError("JupyMCP workspace did not close cleanly", "JUPYMCP_UNAVAILABLE", {
          cause: failure,
        });
      }
    })();
    return this.closePromise;
  }
}

class JupyMcpController {
  private closePromise: Promise<void> | undefined;
  private kernelPromise: Promise<string> | undefined;
  private open = true;
  private tail = Promise.resolve();

  constructor(
    private readonly workspace: JupyMcpWorkspace,
    private readonly path: string,
    private readonly maxNotebookBytes: number,
    private readonly maxOutputBytes: number,
    private readonly workspaceRoot: string,
  ) {}

  execute(
    source: string,
    inputEnvironment: Readonly<Record<string, string>>,
    signal?: AbortSignal,
  ): Promise<JupyMcpExecutionResult> {
    const hasInputEnvironment = Object.keys(inputEnvironment).length > 0;
    return this.serial(() => {
      const operation = async () => {
        this.ensureOpen();
        signal?.throwIfAborted();
        const kernelId = await this.kernel(signal);
        if (hasInputEnvironment) {
          const setup = await this.peer.callTool(
            "execute",
            {
              code: environmentSetup(inputEnvironment),
              kernel_id: kernelId,
              path: ":memory:",
              silent: true,
              store_history: false,
              type: "ipynb",
            },
            signal,
          );
          assertToolSuccess(setup, "input setup");
        }
        const startedAt = Date.now();
        const result = await this.peer.callTool(
          "execute",
          {
            code: source,
            kernel_id: kernelId,
            path: this.path,
            silent: false,
            store_history: true,
            type: "ipynb",
          },
          signal,
        );
        signal?.throwIfAborted();
        const absoluteNotebookPath = `${this.workspaceRoot}/${this.path}`;
        if (existsSync(absoluteNotebookPath)) chmodSync(absoluteNotebookPath, 0o600);
        let normalized = normalizeContent(result, this.maxOutputBytes);
        if (!result.isError) {
          const canonicalText = await this.readNotebookText(signal).catch((error: unknown) => {
            if (error instanceof ScienceError) return undefined;
            throw error;
          });
          if (canonicalText !== undefined) {
            normalized =
              normalizeCanonicalNotebook(canonicalText, this.maxOutputBytes) ?? normalized;
          }
        }
        return {
          ...normalized,
          durationMs: Date.now() - startedAt,
          environment: {
            kernelName: "python3",
            notebookController: "jupymcp",
            transport: "stdio",
          },
          exitCode: normalized.status === "succeeded" ? 0 : null,
          signal: null,
        };
      };
      return hasInputEnvironment
        ? this.workspace.serializeInputEnvironment(operation, signal)
        : operation();
    }, signal);
  }

  private get peer(): JupyMcpPeer {
    return this.workspace.peer;
  }

  readNotebook(signal?: AbortSignal): Promise<string> {
    return this.serial(() => this.readNotebookText(signal), signal);
  }

  close(): Promise<void> {
    if (this.closePromise) return this.closePromise;
    this.open = false;
    this.closePromise = (async () => {
      await this.tail.catch(() => undefined);
      const pendingKernel = this.kernelPromise;
      if (!pendingKernel) return;
      const kernelId = await pendingKernel.catch(() => undefined);
      if (kernelId === undefined) return;
      try {
        const result = await this.peer.callTool("shutdown_kernel", {
          kernel_id: kernelId,
          now: true,
        });
        assertToolSuccess(result, "kernel shutdown");
      } catch (error) {
        throw new ScienceError("JupyMCP controller did not close cleanly", "JUPYMCP_UNAVAILABLE", {
          cause: error,
        });
      }
    })();
    return this.closePromise;
  }

  private ensureOpen(): void {
    if (!this.open) throw new ScienceError("JupyMCP controller is closed", "SCIENCE_CLOSED");
  }

  private async kernel(signal?: AbortSignal): Promise<string> {
    const existing = this.kernelPromise;
    if (existing) return existing;
    const pending = this.peer
      .callTool("start_kernel", { kernel_name: "python3" }, signal)
      .then(startedKernelId);
    this.kernelPromise = pending;
    try {
      return await pending;
    } catch (error) {
      if (this.kernelPromise === pending) this.kernelPromise = undefined;
      throw error;
    }
  }

  private async readNotebookText(signal?: AbortSignal): Promise<string> {
    const result = await this.peer.readResource(`notebook://${this.path}`, signal);
    const text = result.contents.find((content) => "text" in content)?.text;
    if (text === undefined || Buffer.byteLength(text, "utf8") > this.maxNotebookBytes) {
      throw new ScienceError(
        "JupyMCP Notebook resource is unavailable or too large",
        "JUPYMCP_UNAVAILABLE",
      );
    }
    return text;
  }

  private serial<T>(operation: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    const previous = this.tail.catch(() => undefined);
    let release: () => void = () => {};
    const current = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.tail = previous.then(() => current);
    return (async () => {
      try {
        await waitForTurn(previous, signal);
        return await operation();
      } finally {
        release();
      }
    })();
  }
}

/** Workspace-scoped JupyMCP peers with isolated persistent per-Notebook controllers. */
export class JupyMcpRuntime {
  private readonly controllers = new Map<string, Promise<JupyMcpController>>();
  private readonly lifecycle = new AbortController();
  private readonly workspaces = new Map<string, Promise<JupyMcpWorkspace>>();
  private open = true;

  constructor(
    private readonly config: JupyMcpRuntimeConfig,
    private readonly peerFactory: JupyMcpPeerFactory = connectJupyMcp,
  ) {}

  async execute(
    request: JupyMcpExecutionRequest,
    signal?: AbortSignal,
  ): Promise<JupyMcpExecutionResult> {
    this.ensureOpen();
    signal?.throwIfAborted();
    const effectiveSignal = signal
      ? AbortSignal.any([signal, this.lifecycle.signal])
      : this.lifecycle.signal;
    const key = this.controllerKey(request.workspaceKey, request.workspaceRoot, request.notebookId);
    const controller = await this.controller(key, request, effectiveSignal);
    try {
      return await controller.execute(
        request.source,
        request.inputEnvironment ?? {},
        effectiveSignal,
      );
    } catch (error) {
      this.controllers.delete(key);
      await controller.close();
      if (effectiveSignal.aborted) effectiveSignal.throwIfAborted();
      if (error instanceof ScienceError) throw error;
      throw new ScienceError("JupyMCP execution failed", "JUPYMCP_UNAVAILABLE", { cause: error });
    }
  }

  async readNotebook(
    request: Omit<JupyMcpExecutionRequest, "inputEnvironment" | "source">,
    signal?: AbortSignal,
  ): Promise<string> {
    this.ensureOpen();
    signal?.throwIfAborted();
    const effectiveSignal = signal
      ? AbortSignal.any([signal, this.lifecycle.signal])
      : this.lifecycle.signal;
    const key = this.controllerKey(request.workspaceKey, request.workspaceRoot, request.notebookId);
    const controller = await this.controller(key, request, effectiveSignal);
    return controller.readNotebook(effectiveSignal);
  }

  async close(): Promise<void> {
    if (!this.open) return;
    this.open = false;
    this.lifecycle.abort(new ScienceError("JupyMCP runtime is closed", "SCIENCE_CLOSED"));
    const controllers = await Promise.allSettled(this.controllers.values());
    this.controllers.clear();
    const settled = await Promise.allSettled(
      controllers
        .filter(
          (result): result is PromiseFulfilledResult<JupyMcpController> =>
            result.status === "fulfilled",
        )
        .map((result) => result.value.close()),
    );
    const workspaces = await Promise.allSettled(this.workspaces.values());
    this.workspaces.clear();
    const closedWorkspaces = await Promise.allSettled(
      workspaces
        .filter(
          (result): result is PromiseFulfilledResult<JupyMcpWorkspace> =>
            result.status === "fulfilled",
        )
        .map((result) => result.value.close()),
    );
    const failure = [...settled, ...closedWorkspaces].find(
      (result): result is PromiseRejectedResult => result.status === "rejected",
    );
    if (failure) throw failure.reason;
  }

  private async controller(
    key: string,
    request: Pick<JupyMcpExecutionRequest, "notebookId" | "workspaceKey" | "workspaceRoot">,
    signal?: AbortSignal,
  ): Promise<JupyMcpController> {
    const existing = this.controllers.get(key);
    if (existing) return existing;
    const path = notebookPath(request.notebookId);
    const workspaceKey = this.workspaceKey(request.workspaceKey, request.workspaceRoot);
    const pending = this.workspace(workspaceKey, request.workspaceRoot, signal).then(
      (workspace) =>
        new JupyMcpController(
          workspace,
          path,
          this.config.maxNotebookBytes,
          this.config.maxOutputBytes,
          request.workspaceRoot,
        ),
    );
    this.controllers.set(key, pending);
    try {
      return await pending;
    } catch (error) {
      this.controllers.delete(key);
      if (signal?.aborted) signal.throwIfAborted();
      throw new ScienceError("Configured JupyMCP server is unavailable", "JUPYMCP_UNAVAILABLE", {
        cause: error,
      });
    }
  }

  private async workspace(
    key: string,
    workspaceRoot: string,
    signal?: AbortSignal,
  ): Promise<JupyMcpWorkspace> {
    const existing = this.workspaces.get(key);
    if (existing) return existing;
    const options: JupyMcpConnectionOptions = {
      args: this.config.args,
      command: this.config.command,
      cwd: workspaceRoot,
      env: childEnvironment(this.config.env),
      requestTimeoutMs: this.config.requestTimeoutMs,
      ...(signal ? { signal } : {}),
    };
    const pending = this.peerFactory(options).then((peer) => new JupyMcpWorkspace(peer));
    this.workspaces.set(key, pending);
    try {
      return await pending;
    } catch (error) {
      this.workspaces.delete(key);
      if (signal?.aborted) signal.throwIfAborted();
      throw new ScienceError("Configured JupyMCP server is unavailable", "JUPYMCP_UNAVAILABLE", {
        cause: error,
      });
    }
  }

  private controllerKey(workspaceKey: string, workspaceRoot: string, notebookId: string): string {
    return createHash("sha256")
      .update(`${workspaceKey}\0${workspaceRoot}\0${notebookId}`)
      .digest("hex");
  }

  private workspaceKey(workspaceKey: string, workspaceRoot: string): string {
    return createHash("sha256").update(`${workspaceKey}\0${workspaceRoot}`).digest("hex");
  }

  private ensureOpen(): void {
    if (!this.open) throw new ScienceError("JupyMCP runtime is closed", "SCIENCE_CLOSED");
  }
}
