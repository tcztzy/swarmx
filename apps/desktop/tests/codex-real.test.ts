import { spawn, spawnSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { SwarmJournal } from "@swarmx/dsh-swarm";
import { describe, expect, it } from "vitest";
import { startRuntimeBridge } from "../src/runtime/bridge.js";
import { CodexJsonRpcConnection } from "../src/runtime/codex/connection.js";
import {
  codexAppServerArgs,
  codexAppServerEnvironment,
  startCodexRuntime,
} from "../src/runtime/codex/index.js";
import { CodexMemberBindingStore } from "../src/runtime/codex/member-bindings.js";
import { CodexConversationRuntime, type CodexRpcClient } from "../src/runtime/codex/runtime.js";
import {
  CODEX_PROVISIONING_INTERRUPTED_ERROR,
  reconcileCodexSwarmBindings,
} from "../src/runtime/codex/swarm-recovery.js";
import type {
  ApprovalRequestedEvent,
  RuntimeEvent,
  UserMessageItem,
  WorkspaceScope,
} from "../src/runtime/contracts.js";
import { ConversationController } from "../src/runtime/controller.js";
import {
  type SwarmRecoveryOwner,
  startSwarmRecoveryOwner,
} from "../src/runtime/swarm-recovery-owner.js";
import { WorkspaceAuthority } from "../src/runtime/workspace.js";

const command = process.env.SWARMX_CODEX_COMMAND ?? "codex";
const available = spawnSync(command, ["--version"], { encoding: "utf8" }).status === 0;
const fullAcceptance = process.env.SWARMX_CODEX_FULL_ACCEPTANCE === "1";

describe("Codex production recovery wiring", () => {
  it("reconciles a persisted native member through startCodexRuntime", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-production-recovery-"));
    const workspaceRoot = join(root, "workspace");
    const productHome = join(root, "product");
    const serverPath = join(root, "fake-app-server.mjs");
    mkdirSync(workspaceRoot);
    const workspace = new WorkspaceAuthority().mint(workspaceRoot);
    const journalRoot = join(productHome, "swarm");
    const journal = new SwarmJournal(journalRoot);
    const teamId = `codex-mcp-thread:${"d".repeat(64)}`;
    const memberId = randomUUID();
    const conversationId = "codex:persisted-member";
    journal.append(teamId, {
      type: "team/created",
      data: {
        createdAt: 1,
        lead: {
          createdAt: 1,
          description: "Lead",
          id: teamId,
          name: "lead",
          phase: "active",
          role: "lead",
          modelPolicy: { source: "observed" },
        },
        name: "Production recovery team",
        workspaceKey: journal.workspaceKey(workspace.root),
      },
    });
    journal.append(teamId, {
      type: "member/updated",
      data: {
        createdAt: 2,
        description: "Interrupted native member",
        error: CODEX_PROVISIONING_INTERRUPTED_ERROR,
        id: memberId,
        name: "worker",
        phase: "failed",
        role: "legacy",
        modelPolicy: { source: "legacy-default" },
      },
    });
    new CodexMemberBindingStore(journal, journal.workspaceKey(workspace.root)).claim({
      id: memberId,
      conversationId,
    });
    journal.close();
    writeFileSync(
      serverPath,
      `import { createInterface } from "node:readline";
const lines = createInterface({ input: process.stdin });
const respond = (value) => process.stdout.write(JSON.stringify(value) + "\\n");
lines.on("line", (line) => {
  const request = JSON.parse(line);
  if (request.method === "initialize") {
    respond({ id: request.id, result: { userAgent: "fake-codex" } });
    return;
  }
  if (request.method === "thread/read") {
    respond({
      id: request.id,
      result: {
        thread: {
          id: "persisted-member",
          cwd: ${JSON.stringify(workspace.root)},
          name: "Persisted member",
          preview: "joined",
          archived: false,
          createdAt: 1,
          updatedAt: 2,
          historyMode: "legacy",
          turns: [{
            id: "turn-1",
            status: "completed",
            startedAt: 1,
            completedAt: 2,
            items: [{
              id: "item-1",
              type: "userMessage",
              content: [{ type: "input_text", text: "Join the Team." }]
            }]
          }]
        }
      }
    });
    return;
  }
  if (request.method === "thread/list") {
    respond({ id: request.id, result: { data: [], nextCursor: null } });
  }
});
`,
    );
    let runtime: Awaited<ReturnType<typeof startCodexRuntime>> | undefined;
    try {
      runtime = await startCodexRuntime({
        command: process.execPath,
        args: [serverPath],
        productHome,
        workspace,
        bridgeUrl: "http://127.0.0.1:1/",
        bridgeToken: "production-recovery-test",
        mcpServerPath: join(root, "unused-mcp-server.js"),
      });

      const recovered = new SwarmJournal(journalRoot, { mode: "client" });
      expect(recovered.get(teamId)?.members).toEqual(
        expect.arrayContaining([expect.objectContaining({ id: memberId, phase: "active" })]),
      );
      expect(
        new CodexMemberBindingStore(recovered, recovered.workspaceKey(workspace.root)).get(
          memberId,
        ),
      ).toEqual({ id: memberId, conversationId });
      recovered.close();
    } finally {
      await runtime?.dispose();
      rmSync(root, { recursive: true, force: true });
    }
  }, 30_000);
});

describe.skipIf(!available)("real Codex App Server", () => {
  it("ships the allowlisted experimental revert and exact fork-boundary protocol", () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-schema-"));
    try {
      const generated = spawnSync(
        command,
        ["app-server", "generate-json-schema", "--experimental", "--out", root],
        { encoding: "utf8" },
      );
      expect(generated.status, generated.stderr).toBe(0);
      const schema = JSON.parse(
        readFileSync(join(root, "v2", "ThreadRevertParams.json"), "utf8"),
      ) as { required?: string[]; properties?: Record<string, unknown> };
      expect(schema.required).toEqual(expect.arrayContaining(["threadId", "beforeTurnId"]));
      expect(schema.properties).toHaveProperty("beforeTurnId");
      const forkSchema = JSON.parse(
        readFileSync(join(root, "v2", "ThreadForkParams.json"), "utf8"),
      ) as { properties?: Record<string, unknown> };
      expect(forkSchema.properties).toHaveProperty("beforeTurnId");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("initializes over stdio, validates a native thread list, and disposes the child", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-list-"));
    const codexHome = join(root, "codex");
    mkdirSync(codexHome);
    let runtime: Awaited<ReturnType<typeof startCodexRuntime>> | undefined;
    try {
      runtime = await startCodexRuntime({
        command,
        env: { CODEX_HOME: codexHome, CODEX_APP_SERVER_DISABLE_MANAGED_CONFIG: "1" },
      });
      const conversations = await runtime.list();
      expect(conversations.every((conversation) => conversation.runtime === "codex")).toBe(true);
      expect(
        conversations.every((conversation) => conversation.conversationId.startsWith("codex:")),
      ).toBe(true);
    } finally {
      await runtime?.dispose();
      rmSync(root, { recursive: true, force: true });
    }
  }, 30_000);

  it("creates and hydrates an isolated legacy thread by default", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-paginated-"));
    const workspaceRoot = join(root, "workspace");
    const codexHome = join(root, "codex");
    mkdirSync(workspaceRoot);
    mkdirSync(codexHome);
    const workspace = new WorkspaceAuthority().mint(workspaceRoot);
    let runtime: Awaited<ReturnType<typeof startCodexRuntime>> | undefined;
    try {
      runtime = await startCodexRuntime({ command, env: { CODEX_HOME: codexHome } });
      const created = await runtime.create({ workspace });
      await expect(runtime.read(created.conversationId)).resolves.toMatchObject({
        conversationId: created.conversationId,
        turns: [],
      });
    } finally {
      await runtime?.dispose();
      rmSync(root, { recursive: true, force: true });
    }
  }, 30_000);

  it("resumes a persisted stored thread after App Server restart", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-resume-"));
    const workspaceRoot = join(root, "workspace");
    const codexHome = join(root, "codex");
    mkdirSync(workspaceRoot);
    mkdirSync(codexHome);
    const launch = async () => {
      const child = spawn(command, ["app-server"], {
        cwd: workspaceRoot,
        env: { ...process.env, CODEX_HOME: codexHome },
        stdio: ["pipe", "pipe", "pipe"],
      });
      const connection = new CodexJsonRpcConnection(child);
      await connection.initialize();
      return connection;
    };
    let first: CodexJsonRpcConnection | undefined;
    let second: CodexJsonRpcConnection | undefined;
    try {
      first = await launch();
      const started = (await first.request("thread/start", {
        cwd: workspaceRoot,
        approvalPolicy: "never",
      })) as { thread: { id: string } };
      await first.request("thread/inject_items", {
        threadId: started.thread.id,
        items: [
          {
            type: "message",
            role: "user",
            content: [{ type: "input_text", text: "persisted probe" }],
          },
        ],
      });
      await first.dispose();
      first = undefined;

      second = await launch();
      await expect(
        second.request("thread/read", { threadId: started.thread.id, includeTurns: false }),
      ).resolves.toMatchObject({ thread: { status: { type: "notLoaded" } } });
      await expect(
        second.request("thread/resume", { threadId: started.thread.id }),
      ).resolves.toMatchObject({ thread: { id: started.thread.id } });
      await expect(second.request("thread/loaded/list")).resolves.toMatchObject({
        data: expect.arrayContaining([started.thread.id]),
      });
    } finally {
      await first?.dispose();
      await second?.dispose();
      rmSync(root, { recursive: true, force: true });
    }
  }, 30_000);

  it("initializes with the required SwarmX stdio MCP server", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-mcp-"));
    const codexHome = join(root, "codex");
    mkdirSync(codexHome);
    const workspace = new WorkspaceAuthority().mint(root);
    const recoveryOwner = startSwarmRecoveryOwner(join(root, "home", "swarm"));
    const bridge = await startRuntimeBridge(workspace, recoveryOwner);
    let runtime: Awaited<ReturnType<typeof startCodexRuntime>> | undefined;
    try {
      runtime = await startCodexRuntime({
        command,
        productHome: join(root, "home"),
        workspace,
        bridgeUrl: bridge.url,
        bridgeToken: bridge.token,
        mcpServerPath: join(
          process.cwd(),
          "apps",
          "desktop",
          "dist",
          "runtime",
          "codex",
          "mcp-server.js",
        ),
        env: { CODEX_HOME: codexHome, CODEX_APP_SERVER_DISABLE_MANAGED_CONFIG: "1" },
      });
      bridge.attach(runtime);
      await expect(runtime.list()).resolves.toBeInstanceOf(Array);
    } finally {
      await runtime?.dispose();
      await bridge.dispose();
      await recoveryOwner.dispose();
      rmSync(root, { recursive: true, force: true });
    }
  }, 30_000);
});

describe.skipIf(!available || !fullAcceptance)("real Codex App Server full acceptance", () => {
  it("uses exact blank-Thread deletion and observes external native archive authority", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-native-retirement-"));
    const workspaceRoot = join(root, "workspace");
    const codexHome = join(root, "codex");
    mkdirSync(workspaceRoot);
    mkdirSync(codexHome);
    const workspace = new WorkspaceAuthority().mint(workspaceRoot);
    const responses = new MockResponsesServer();
    await responses.listen();
    writeFileSync(
      join(codexHome, "config.toml"),
      [
        'model = "mock-model"',
        'model_provider = "swarmx_acceptance"',
        'sandbox_mode = "read-only"',
        "[model_providers.swarmx_acceptance]",
        'name = "SwarmX acceptance provider"',
        `base_url = "${responses.baseUrl}/v1"`,
        'wire_api = "responses"',
        "supports_websockets = false",
        "request_max_retries = 0",
        "stream_max_retries = 0",
        "",
      ].join("\n"),
      { mode: 0o600 },
    );
    const child = spawn(command, ["app-server"], {
      cwd: workspaceRoot,
      env: {
        ...process.env,
        CODEX_HOME: codexHome,
        CODEX_APP_SERVER_DISABLE_MANAGED_CONFIG: "1",
      },
      stdio: ["pipe", "pipe", "pipe"],
    });
    const connection = new CodexJsonRpcConnection(child);
    let runtime: CodexConversationRuntime | undefined;
    try {
      await connection.initialize();
      runtime = new CodexConversationRuntime(connection);
      const memberId = randomUUID();
      const blank = await runtime.createProvisionedMember({ workspace }, memberId);
      await runtime.retireProvisionedMember(blank.conversationId, memberId);
      await expect(
        connection.request("thread/read", {
          threadId: blank.conversationId.replace(/^codex:/u, ""),
          includeTurns: false,
        }),
      ).rejects.toMatchObject({ code: -32600 });

      responses.enqueueFinal("EXTERNAL_ARCHIVE_READY");
      const events: RuntimeEvent[] = [];
      runtime.subscribe((event) => events.push(event));
      const created = await runtime.create({ workspace });
      const threadId = created.conversationId.replace(/^codex:/u, "");
      const turnMark = events.length;
      await runtime.start({
        conversationId: created.conversationId,
        text: "EXTERNAL_ARCHIVE_PROBE",
      });
      await waitForTerminalTurn(events, turnMark);
      await connection.request("thread/archive", { threadId });
      let archived = await runtime.read(created.conversationId);
      for (let attempt = 0; !archived.archived && attempt < 100; attempt += 1) {
        await delay(25);
        archived = await runtime.read(created.conversationId);
      }
      expect(archived).toMatchObject({
        archived: true,
        conversationId: created.conversationId,
      });
      await expect(
        runtime.start({ conversationId: created.conversationId, text: "must not resume" }),
      ).rejects.toThrow();
    } finally {
      await runCleanupSteps([
        () => (runtime === undefined ? connection.dispose() : runtime.dispose()),
        () => responses.close(),
        async () => rmSync(root, { recursive: true, force: true }),
      ]);
    }
  }, 60_000);

  it("settles a pending typed MCP approval while disposing every owned process", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-pending-approval-"));
    const workspaceRoot = join(root, "workspace");
    const codexHome = join(root, "codex");
    const productHome = join(root, "product");
    mkdirSync(workspaceRoot);
    mkdirSync(codexHome);
    const workspace = new WorkspaceAuthority().mint(workspaceRoot);
    const recoveryOwner = startSwarmRecoveryOwner(join(productHome, "swarm"));
    const responses = new MockResponsesServer();
    let target: AcceptanceRuntime | undefined;
    try {
      await responses.listen();
      writeFileSync(
        join(codexHome, "config.toml"),
        [
          'model = "mock-model"',
          'model_provider = "swarmx_acceptance"',
          'sandbox_mode = "read-only"',
          "[model_providers.swarmx_acceptance]",
          'name = "SwarmX acceptance provider"',
          `base_url = "${responses.baseUrl}/v1"`,
          'wire_api = "responses"',
          "supports_websockets = false",
          "request_max_retries = 0",
          "stream_max_retries = 0",
          "",
        ].join("\n"),
        { mode: 0o600 },
      );
      target = await launchAcceptanceRuntime({
        codexHome,
        productHome,
        serverPath: join(
          process.cwd(),
          "apps",
          "desktop",
          "dist",
          "runtime",
          "codex",
          "mcp-server.js",
        ),
        workspace,
        swarmRecoveryOwner: recoveryOwner,
      });
      responses.enqueueFinal("APPROVAL_DISPOSAL_PROBE_READY");
      const created = await target.runtime.create({ workspace });
      const turnMark = target.events.length;
      await target.runtime.start({
        conversationId: created.conversationId,
        text: "APPROVAL_DISPOSAL_PROBE",
      });
      await waitForTerminalTurn(target.events, turnMark);
      const interrupted = responses.enqueueInterrupted("APPROVAL_DISPOSAL_INTERRUPT");
      const interruptMark = target.events.length;
      const interruptedTurn = await target.runtime.start({
        conversationId: created.conversationId,
        text: "APPROVAL_DISPOSAL_INTERRUPT",
      });
      await withTimeout(interrupted.reached, "pending-approval interrupt delta");
      await target.runtime.interrupt({
        conversationId: created.conversationId,
        turnId: interruptedTurn.turnId,
      });
      await waitForTerminalTurn(target.events, interruptMark);
      await target.rpc.request("mcpServerStatus/list", {
        threadId: nativeCodexId(created.conversationId),
        detail: "full",
      });
      await target.rpc.request("mcpServer/tool/call", {
        threadId: nativeCodexId(created.conversationId),
        server: "swarmx",
        tool: "pkb",
        arguments: {
          action: "create_knowledge",
          request: { requestId: randomUUID(), body: "invalid" },
        },
      });
      const mark = target.events.length;
      const pending = target.rpc.request("mcpServer/tool/call", {
        threadId: nativeCodexId(created.conversationId),
        server: "swarmx",
        tool: "pkb",
        arguments: {
          action: "create_knowledge",
          request: {
            requestId: randomUUID(),
            body: "# Pending disposal",
            description: "Must not survive runtime disposal.",
            title: "Pending disposal",
            type: "Finding",
          },
        },
      });
      const settlement = withTimeout(pending, "pending MCP disposal").then(
        (value) => ({ status: "fulfilled" as const, value }),
        (error: unknown) => ({ status: "rejected" as const, error }),
      );
      const pendingState = await Promise.race([
        waitForApproval(target.events, mark).then(() => "approval" as const),
        settlement.then((result) => ({ result })),
      ]);
      if (pendingState !== "approval") {
        throw new Error(
          `Pending MCP call settled before approval: ${JSON.stringify(pendingState.result)}`,
        );
      }
      recordOwnedProcesses(target);
      await withTimeout(target.runtime.dispose(), "pending-approval runtime disposal");
      target.disposed = true;
      const result = await settlement;
      if (result.status === "fulfilled") {
        expect(result.value).toMatchObject({
          isError: true,
          content: [
            expect.objectContaining({ text: expect.stringMatching(/disposed|exited|closed/iu) }),
          ],
        });
      } else {
        expect(result.error).toEqual(
          expect.objectContaining({ message: expect.stringMatching(/disposed|exited|closed/iu) }),
        );
      }
      await withTimeout(childExit(target.child), "pending-approval App Server exit");
      expect(await liveOwnedProcesses(target.ownedProcesses)).toEqual([]);
      await withTimeout(target.bridge.dispose(), "pending-approval bridge disposal");
      target.bridgeDisposed = true;
    } finally {
      await runCleanupSteps([
        () => disposeAcceptanceRuntime(target),
        () => responses.close(),
        () => recoveryOwner.dispose(),
        () => removeTreeAfterChildrenExit(root),
      ]);
    }
  }, 60_000);

  it("runs the complete native lifecycle and shared product matrix without owned orphans", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-full-"));
    const workspaceRoot = join(root, "workspace");
    const codexHome = join(root, "codex");
    const productHome = join(root, "product");
    mkdirSync(workspaceRoot);
    mkdirSync(codexHome);
    const workspace = new WorkspaceAuthority().mint(workspaceRoot);
    const responses = new MockResponsesServer();
    let first: AcceptanceRuntime | undefined;
    let second: AcceptanceRuntime | undefined;
    let recoveryOwner: SwarmRecoveryOwner | undefined;
    let conversationId: string | undefined;
    let memberConversationId: string | undefined;
    let phase = "setup";
    let failure: unknown;
    let failed = false;
    try {
      recoveryOwner = startSwarmRecoveryOwner(join(productHome, "swarm"));
      await responses.listen();
      writeFileSync(
        join(codexHome, "config.toml"),
        [
          'model = "mock-model"',
          'model_provider = "swarmx_acceptance"',
          'sandbox_mode = "read-only"',
          "[model_providers.swarmx_acceptance]",
          'name = "SwarmX acceptance provider"',
          `base_url = "${responses.baseUrl}/v1"`,
          'wire_api = "responses"',
          "supports_websockets = false",
          "request_max_retries = 0",
          "stream_max_retries = 0",
          "",
        ].join("\n"),
        { mode: 0o600 },
      );
      const serverPath = join(
        process.cwd(),
        "apps",
        "desktop",
        "dist",
        "runtime",
        "codex",
        "mcp-server.js",
      );
      const launchInput = {
        codexHome,
        productHome,
        serverPath,
        workspace,
        swarmRecoveryOwner: recoveryOwner,
      };
      const steerLead = responses.enqueuePaused("BEFORE_STEER");
      responses.enqueueFinal("AFTER_STEER");
      const interrupted = responses.enqueueInterrupted("INTERRUPT_STREAM");
      first = await launchAcceptanceRuntime(launchInput);
      const created = await first.runtime.create({ workspace });
      conversationId = created.conversationId;

      const firstMark = first.events.length;
      const firstTurn = await first.runtime.start({
        conversationId,
        text: "INITIAL_MARKER",
      });
      await withTimeout(steerLead.reached, "initial streamed delta");
      await waitForRuntimeEvent(
        first.events,
        firstMark,
        (event) => event.type === "item_delta" && event.delta === "BEFORE_STEER",
      );
      await first.runtime.steer({
        conversationId,
        turnId: firstTurn.turnId,
        text: "STEER_MARKER",
      });
      steerLead.release();
      await waitForRuntimeEvent(
        first.events,
        firstMark,
        (event) =>
          event.type === "turn_status" &&
          event.turnId === firstTurn.turnId &&
          event.status === "completed",
      );
      await waitForResponseBody(responses, 2, "STEER_MARKER");

      const afterSteer = await first.runtime.read(conversationId);
      expect(
        afterSteer.turns
          .flatMap((turn) => turn.items)
          .filter((item) => item.type === "assistant_message"),
      ).toContainEqual(expect.objectContaining({ text: "AFTER_STEER" }));

      const interruptMark = first.events.length;
      const interruptedTurn = await first.runtime.start({
        conversationId,
        text: "INTERRUPT_MARKER",
      });
      await withTimeout(interrupted.reached, "interrupt streamed delta");
      await waitForRuntimeEvent(
        first.events,
        interruptMark,
        (event) => event.type === "item_delta" && event.delta === "INTERRUPT_STREAM",
      );
      await first.runtime.interrupt({
        conversationId,
        turnId: interruptedTurn.turnId,
      });
      await waitForRuntimeEvent(
        first.events,
        interruptMark,
        (event) =>
          event.type === "turn_status" &&
          event.turnId === interruptedTurn.turnId &&
          event.status === "interrupted",
      );

      const status = (await withTimeout(
        first.rpc.request("mcpServerStatus/list", {
          threadId: nativeCodexId(conversationId),
          detail: "full",
        }),
        "MCP server status",
      )) as {
        data: Array<{
          name: string;
          runtimeStatus?: string | null;
          tools: Record<string, { inputSchema: unknown }>;
        }>;
      };
      const swarmx = status.data.find((server) => server.name === "swarmx");
      expect(swarmx).toMatchObject({ runtimeStatus: "connected" });
      const pkbInputSchema = swarmx?.tools.pkb?.inputSchema;
      expect(pkbInputSchema).toMatchObject({
        type: "object",
        additionalProperties: false,
        required: ["action", "request"],
        properties: {
          action: { type: "string", enum: expect.arrayContaining(["create_knowledge"]) },
          request: { type: "object" },
        },
      });
      const requestAdditionalProperties = (
        pkbInputSchema as {
          properties?: { request?: { additionalProperties?: unknown } };
        }
      )?.properties?.request?.additionalProperties;
      expect(isUnconstrainedJsonSchema(requestAdditionalProperties)).toBe(true);
      recordOwnedProcesses(first);

      const firstRpc = first.rpc;
      const nativeThreadId = nativeCodexId(conversationId);
      const callPkb = (request: Record<string, unknown>) =>
        firstRpc.request("mcpServer/tool/call", {
          threadId: nativeThreadId,
          server: "swarmx",
          tool: "pkb",
          arguments: { action: "create_knowledge", request },
        });
      const invalidMark = first.events.length;
      const invalid = (await withTimeout(
        callPkb({
          requestId: randomUUID(),
          body: "invalid",
          description: "missing title and type",
        }),
        "invalid PKB call",
      )) as { isError?: boolean | null };
      expect(invalid.isError).toBe(true);
      expect(
        first.events.slice(invalidMark).some((event) => event.type === "approval_requested"),
      ).toBe(false);

      const rejectedRequest = {
        requestId: randomUUID(),
        body: "# Rejected",
        description: "Must remain uncommitted.",
        title: "Rejected real MCP change",
        type: "Finding",
      };
      const rejected = (await callMcpWithApproval(
        first,
        () => callPkb(rejectedRequest),
        "decline",
      )) as { isError?: boolean | null };
      expect(rejected.isError).toBe(true);

      const requestId = randomUUID();
      const workspaceRequest = {
        requestId,
        body: "# Real MCP default scope",
        description: "Created through the real Codex App Server.",
        title: "Real MCP default scope",
        type: "Finding",
      };
      const omitted = mcpTextValue(
        await callMcpWithApproval(first, () => callPkb(workspaceRequest), "accept", true),
      );
      const explicitWorkspace = mcpTextValue(
        await callMcpWithApproval(
          first,
          () => callPkb({ ...workspaceRequest, scope: "workspace" }),
          "accept",
        ),
      );
      const global = mcpTextValue(
        await callMcpWithApproval(
          first,
          () =>
            callPkb({
              ...workspaceRequest,
              requestId: randomUUID(),
              scope: "global",
              title: "Real MCP global scope",
            }),
          "accept",
        ),
      );
      expect(omitted).toMatchObject({ data: { id: expect.stringMatching(/^workspaces\//u) } });
      expect(explicitWorkspace).toMatchObject({ data: { id: omitted.data.id } });
      expect(global).toMatchObject({ data: { id: expect.stringMatching(/^global\/concepts\//u) } });

      const science = mcpTextValue(
        await first.rpc.request("mcpServer/tool/call", {
          threadId: nativeCodexId(conversationId),
          server: "swarmx",
          tool: "science_notebook",
          arguments: {
            action: "create_project",
            request: { requestId: randomUUID(), title: "Full acceptance project" },
          },
        }),
      );
      expect(science.data.id).toEqual(expect.any(String));
      const createdSwarm = mcpJsonValue<{ action: string; data: { kind: string; name: string } }>(
        await first.rpc.request("mcpServer/tool/call", {
          threadId: nativeCodexId(conversationId),
          server: "swarmx",
          tool: "swarm",
          arguments: { action: "create", request: { name: "Real Codex team" } },
        }),
      );
      expect(createdSwarm).toMatchObject({
        action: "create",
        data: { kind: "active", name: "Real Codex team" },
      });
      const swarmStatus = mcpJsonValue<{
        action: string;
        data: { kind: string; name: string };
      }>(
        await first.rpc.request("mcpServer/tool/call", {
          threadId: nativeCodexId(conversationId),
          server: "swarmx",
          tool: "swarm",
          arguments: { action: "status", request: {} },
        }),
      );
      expect(swarmStatus).toMatchObject({
        action: "status",
        data: { kind: "active", name: "Real Codex team" },
      });

      const memberMark = first.events.length;
      responses.enqueueFinal("MEMBER_READY");
      const addedMember = mcpJsonValue<{
        action: string;
        data: {
          kind: string;
          memberName: string;
          members: Array<{ name: string; role: string; status: string }>;
          name: string;
          role: string;
        };
      }>(
        await first.rpc.request("mcpServer/tool/call", {
          threadId: nativeCodexId(conversationId),
          server: "swarmx",
          tool: "swarm",
          arguments: {
            action: "add_member",
            request: {
              description: "Exercises durable native member identity.",
              name: "worker",
              prompt: "Join the Team and wait.",
            },
          },
        }),
      );
      expect(addedMember).toMatchObject({
        action: "add_member",
        data: {
          kind: "active",
          memberName: "lead",
          members: expect.arrayContaining([
            expect.objectContaining({ name: "worker", role: "legacy" }),
          ]),
          name: "Real Codex team",
          role: "lead",
        },
      });
      await waitForTerminalTurn(first.events, memberMark);
      const firstRuntime = first.runtime;
      memberConversationId = await waitForConversation(firstRuntime, async (candidate) => {
        if (candidate.conversationId === conversationId) return false;
        const snapshot = await firstRuntime.read(candidate.conversationId);
        return snapshot.turns
          .flatMap((turn) => turn.items)
          .some((item) => item.type === "user_message" && item.text === "Join the Team and wait.");
      });
      const memberStatus = mcpJsonValue<{
        action: string;
        data: { kind: string; memberName: string; role: string };
      }>(
        await first.rpc.request("mcpServer/tool/call", {
          threadId: nativeCodexId(memberConversationId),
          server: "swarmx",
          tool: "swarm",
          arguments: { action: "status", request: {} },
        }),
      );
      expect(memberStatus).toMatchObject({
        action: "status",
        data: { kind: "active", memberName: "worker", role: "legacy" },
      });

      phase = "disposing the first runtime after native member materialization";
      recordOwnedProcesses(first);
      await withTimeout(first.runtime.dispose(), "first runtime disposal");
      first.disposed = true;
      await withTimeout(childExit(first.child), "first App Server exit");
      await withTimeout(interrupted.closed, "disposed Responses stream closure");
      expect(await liveOwnedProcesses(first.ownedProcesses)).toEqual([]);
      await withTimeout(first.bridge.dispose(), "first bridge disposal");
      first.bridgeDisposed = true;

      responses.enqueueFinal("RETRY_OK");
      responses.enqueueFinal("EDIT_OK");
      phase = "restarting the App Server";
      second = await launchAcceptanceRuntime(launchInput);
      phase = "reading the persisted lead Thread";
      const nativeRead = (await second.rpc.request("thread/read", {
        threadId: nativeCodexId(conversationId),
        includeTurns: true,
      })) as { thread: { historyMode: string } };
      expect(nativeRead.thread.historyMode).toBe("paginated");
      expect(second.rpc.calls.filter((call) => call.method === "thread/resume")).toHaveLength(0);
      const restartedConversationIds = (await second.runtime.list()).map(
        (conversation) => conversation.conversationId,
      );
      expect(restartedConversationIds, JSON.stringify(restartedConversationIds)).toContain(
        memberConversationId,
      );
      await expect(second.runtime.read(memberConversationId)).resolves.toMatchObject({
        conversationId: memberConversationId,
        workspace: { id: workspace.id },
      });
      phase = "resuming the persisted member Thread";
      await expect(
        second.rpc.request("thread/resume", {
          threadId: nativeCodexId(memberConversationId),
        }),
      ).resolves.toMatchObject({
        thread: { id: nativeCodexId(memberConversationId) },
      });

      phase = "restoring the member Thread authority";
      const beforeMemberHydration = codexSwarmDurableState(productHome, workspace.root);
      const restoredMemberStatus = mcpJsonValue<{
        action: string;
        data: { kind: string; memberName: string; role: string };
      }>(
        await second.rpc.request("mcpServer/tool/call", {
          threadId: nativeCodexId(memberConversationId),
          server: "swarmx",
          tool: "swarm",
          arguments: { action: "status", request: {} },
        }),
      );
      expect(restoredMemberStatus).toMatchObject({
        action: "status",
        data: { kind: "active", memberName: "worker", role: "legacy" },
      });
      expect(codexSwarmDurableState(productHome, workspace.root)).toEqual(beforeMemberHydration);
      expect(
        second.rpc.calls.filter(
          (call) =>
            call.method === "thread/resume" &&
            call.params?.threadId === nativeCodexId(memberConversationId),
        ),
      ).toHaveLength(1);
      expect(
        second.rpc.calls.filter(
          (call) =>
            call.method === "thread/resume" &&
            call.params?.threadId === nativeCodexId(conversationId),
        ),
      ).toHaveLength(0);

      phase = "retrying the persisted lead Thread";
      const beforeRetry = await second.runtime.read(conversationId);
      const interruptedUser = lastUserItem(beforeRetry);
      const controller = new ConversationController(second.runtime);
      let eventMark = second.events.length;
      const retried = await controller.retry(conversationId, interruptedUser.id);
      expect(retried.conversationId).toBe(conversationId);
      await waitForTerminalTurn(second.events, eventMark);
      expect(second.rpc.calls.filter((call) => call.method === "thread/resume")).toHaveLength(2);
      expect(second.rpc.calls.filter((call) => call.method === "thread/revert")).toHaveLength(1);
      phase = "restoring the resumed lead Thread authority";
      const restoredLeadStatus = mcpJsonValue<{
        action: string;
        data: { kind: string; memberName: string; role: string };
      }>(
        await second.rpc.request("mcpServer/tool/call", {
          threadId: nativeCodexId(conversationId),
          server: "swarmx",
          tool: "swarm",
          arguments: { action: "status", request: {} },
        }),
      );
      expect(restoredLeadStatus).toMatchObject({
        action: "status",
        data: { kind: "active", memberName: "lead", role: "lead" },
      });

      const afterRetry = await second.runtime.read(conversationId);
      const retryUser = lastUserItem(afterRetry);
      phase = "editing the persisted lead Thread";
      eventMark = second.events.length;
      const edited = await controller.edit(conversationId, retryUser.id, "EDIT_MARKER");
      expect(edited.conversationId).toBe(conversationId);
      await waitForTerminalTurn(second.events, eventMark);
      expect(second.rpc.calls.filter((call) => call.method === "thread/resume")).toHaveLength(2);
      expect(second.rpc.calls.filter((call) => call.method === "thread/revert")).toHaveLength(2);

      const finalSnapshot = await second.runtime.read(conversationId);
      expect(lastUserItem(finalSnapshot).text).toBe("EDIT_MARKER");
      expect(
        finalSnapshot.turns.flatMap((turn) => turn.items).filter((item) => item.type === "tool"),
      ).toEqual([]);
      const firstTurnBoundary = finalSnapshot.turns[0]?.id;
      const laterTurnBoundary = finalSnapshot.turns[1]?.id;
      if (firstTurnBoundary === undefined || laterTurnBoundary === undefined) {
        throw new Error("Full acceptance requires two completed native turns before forking.");
      }
      phase = "forking and archiving native Threads";
      const firstFork = await second.runtime.fork({
        conversationId,
        beforeTurnId: firstTurnBoundary,
      });
      const laterFork = await second.runtime.fork({
        conversationId,
        beforeTurnId: laterTurnBoundary,
      });
      expect((await second.runtime.read(firstFork.conversationId)).turns).toHaveLength(0);
      expect((await second.runtime.read(laterFork.conversationId)).turns).toHaveLength(1);
      await second.runtime.archive(firstFork.conversationId);
      await second.runtime.archive(laterFork.conversationId);
      phase = "archiving the Swarm and its native member Thread";
      const archivedSwarm = mcpJsonValue<{ action: string; data: { kind: string } }>(
        await second.rpc.request("mcpServer/tool/call", {
          threadId: nativeCodexId(conversationId),
          server: "swarmx",
          tool: "swarm",
          arguments: { action: "archive", request: {} },
        }),
      );
      expect(archivedSwarm).toMatchObject({ action: "archive", data: { kind: "archived" } });
      expect(second.rpc.calls).toContainEqual({
        method: "thread/archive",
        params: { threadId: nativeCodexId(memberConversationId) },
      });
      await second.runtime.archive(conversationId);

      phase = "disposing the second runtime";
      recordOwnedProcesses(second);
      await withTimeout(second.runtime.dispose(), "second runtime disposal");
      second.disposed = true;
      await withTimeout(childExit(second.child), "second App Server exit");
      expect(await liveOwnedProcesses(second.ownedProcesses)).toEqual([]);
      await withTimeout(second.bridge.dispose(), "second bridge disposal");
      second.bridgeDisposed = true;
    } catch (error) {
      failure = new Error(
        `Full Codex acceptance failed while ${phase} (lead=${conversationId ?? "unset"}, member=${memberConversationId ?? "unset"}): ${error instanceof Error ? error.message : String(error)}`,
        { cause: error },
      );
      failed = true;
    }
    try {
      await runCleanupSteps([
        () => disposeAcceptanceRuntime(first),
        () => disposeAcceptanceRuntime(second),
        () => withTimeout(responses.close(), "mock Responses server disposal"),
        () => recoveryOwner?.dispose() ?? Promise.resolve(),
        () => removeTreeAfterChildrenExit(root),
      ]);
    } catch (error) {
      if (!failed) failure = error;
      failed = true;
    }
    if (failed) throw failure;
  }, 120_000);
});

describe("Codex App Server launch configuration", () => {
  it("attempts every acceptance cleanup step and preserves the first failure", async () => {
    const calls: string[] = [];
    const first = new Error("first cleanup failed");
    await expect(
      runCleanupSteps([
        async () => {
          calls.push("first");
          throw first;
        },
        async () => {
          calls.push("second");
        },
        async () => {
          calls.push("third");
          throw new Error("later cleanup failed");
        },
      ]),
    ).rejects.toBe(first);
    expect(calls).toEqual(["first", "second", "third"]);
  });

  it("rechecks process fingerprints and attempts every matching PID", async () => {
    const firstOwned = { pid: 101, startedAt: "start-a", command: "codex app-server" };
    const reused = { pid: 102, startedAt: "start-b", command: "swarmx mcp" };
    const lastOwned = { pid: 103, startedAt: "start-c", command: "codex helper" };
    const owned = [firstOwned, reused, lastOwned];
    const current = new Map([
      [101, firstOwned],
      [102, { ...reused, startedAt: "reused-pid" }],
      [103, lastOwned],
    ]);
    const calls: number[] = [];
    const first = new Error("first kill failed");

    await expect(
      killOwnedProcesses(
        owned,
        (pid) => current.get(pid),
        (pid) => {
          calls.push(pid);
          if (pid === 101) throw first;
          throw new Error("later kill failed");
        },
      ),
    ).rejects.toBe(first);
    expect(calls).toEqual([101, 103]);
  });

  it("passes MCP secrets through inherited environment names, not process arguments", () => {
    const options = {
      productHome: "/private/swarmx",
      scienceConfig: { embedArtifactMetadata: false, maxArtifactBytes: 4_096 },
      workspace: {
        id: "workspace-1",
        label: "workspace",
        root: "/private/workspace",
        token: "scope-token",
      },
      bridgeUrl: "http://127.0.0.1:1234/",
      bridgeToken: "bridge-secret",
      mcpServerPath: "/app/mcp-server.js",
    } as const;
    const args = codexAppServerArgs(options);
    const environment = codexAppServerEnvironment(options);
    expect(args.at(-1)).toBe("app-server");
    expect(args.join(" ")).toContain("SWARMX_BRIDGE_TOKEN");
    expect(args.join(" ")).not.toContain("bridge-secret");
    expect(args.join(" ")).not.toContain("/private/workspace");
    expect(JSON.parse(environment.SWARMX_SCIENCE_CONFIG as string)).toEqual(options.scienceConfig);
  });
});

function mcpTextValue(value: unknown): { data: { id: string } } {
  return mcpJsonValue(value);
}

function isUnconstrainedJsonSchema(value: unknown): boolean {
  return (
    value === true ||
    (value !== null &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      Object.keys(value).length === 0)
  );
}

function mcpJsonValue<Value>(value: unknown): Value {
  const response = value as { content?: unknown[] };
  const item = response.content?.find(
    (candidate): candidate is { type: "text"; text: string } =>
      candidate !== null &&
      typeof candidate === "object" &&
      "type" in candidate &&
      candidate.type === "text" &&
      "text" in candidate &&
      typeof candidate.text === "string",
  );
  if (item === undefined) throw new Error("MCP tool returned no text content.");
  try {
    return JSON.parse(item.text) as Value;
  } catch (error) {
    throw new Error(`MCP tool returned non-JSON text: ${item.text}`, { cause: error });
  }
}

interface Deferred {
  readonly promise: Promise<void>;
  resolve(): void;
}

function deferred(): Deferred {
  let resolve: (() => void) | undefined;
  return {
    promise: new Promise<void>((current) => {
      resolve = current;
    }),
    resolve: () => resolve?.(),
  };
}

interface ResponseGate {
  readonly reached: Promise<void>;
  readonly closed: Promise<void>;
  release(): void;
}

interface ResponsePlan {
  readonly events: ReadonlyArray<Record<string, unknown>>;
  readonly pauseAfter?: number;
  readonly holdUntilClosed: boolean;
  readonly reached: Deferred;
  readonly release: Deferred;
  readonly closed: Deferred;
}

function responseEvents(id: string, text: string): Array<Record<string, unknown>> {
  const itemId = `message-${id}`;
  return [
    { type: "response.created", response: { id } },
    {
      type: "response.output_item.added",
      item: {
        type: "message",
        role: "assistant",
        id: itemId,
        content: [{ type: "output_text", text: "" }],
      },
    },
    { type: "response.output_text.delta", delta: text },
    {
      type: "response.output_item.done",
      item: {
        type: "message",
        role: "assistant",
        id: itemId,
        content: [{ type: "output_text", text }],
      },
    },
    {
      type: "response.completed",
      response: {
        id,
        usage: {
          input_tokens: 1,
          input_tokens_details: null,
          output_tokens: 1,
          output_tokens_details: null,
          total_tokens: 2,
        },
      },
    },
  ];
}

class MockResponsesServer {
  readonly requests: Array<{ path: string; body: string }> = [];
  private readonly plans: ResponsePlan[] = [];
  private sequence = 0;
  private readonly server = createServer((request, response) => {
    void this.handle(request, response);
  });

  async listen(): Promise<void> {
    await new Promise<void>((resolve, reject) => {
      this.server.once("error", reject);
      this.server.listen(0, "127.0.0.1", resolve);
    });
  }

  get baseUrl(): string {
    const address = this.server.address() as AddressInfo;
    return `http://127.0.0.1:${String(address.port)}`;
  }

  enqueueFinal(text: string): ResponseGate {
    return this.enqueue(responseEvents(`response-${String(++this.sequence)}`, text));
  }

  enqueuePaused(text: string): ResponseGate {
    return this.enqueue(responseEvents(`response-${String(++this.sequence)}`, text), 2);
  }

  enqueueInterrupted(delta: string): ResponseGate {
    return this.enqueue(
      responseEvents(`response-${String(++this.sequence)}`, delta).slice(0, 3),
      2,
      true,
    );
  }

  async close(): Promise<void> {
    for (const plan of this.plans) plan.release.resolve();
    this.server.closeAllConnections();
    await new Promise<void>((resolve) => this.server.close(() => resolve()));
  }

  private enqueue(
    events: ReadonlyArray<Record<string, unknown>>,
    pauseAfter?: number,
    holdUntilClosed = false,
  ): ResponseGate {
    const plan: ResponsePlan = {
      events,
      ...(pauseAfter === undefined ? {} : { pauseAfter }),
      holdUntilClosed,
      reached: deferred(),
      release: deferred(),
      closed: deferred(),
    };
    this.plans.push(plan);
    return {
      reached: plan.reached.promise,
      closed: plan.closed.promise,
      release: plan.release.resolve,
    };
  }

  private async handle(request: IncomingMessage, response: ServerResponse): Promise<void> {
    const chunks: Buffer[] = [];
    for await (const chunk of request) {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    }
    this.requests.push({
      path: request.url ?? "",
      body: Buffer.concat(chunks).toString("utf8"),
    });
    const plan = this.plans.shift();
    if (plan === undefined) {
      response.writeHead(500);
      response.end("No queued response.");
      return;
    }

    response.once("close", plan.closed.resolve);
    response.writeHead(200, { "content-type": "text/event-stream" });
    for (const [index, event] of plan.events.entries()) {
      response.write(`event: ${String(event.type)}\ndata: ${JSON.stringify(event)}\n\n`);
      if (index !== plan.pauseAfter) continue;
      plan.reached.resolve();
      if (plan.holdUntilClosed) {
        await plan.closed.promise;
        return;
      }
      await Promise.race([plan.release.promise, plan.closed.promise]);
      if (response.destroyed) return;
    }
    response.end();
  }
}

class RecordingRpc implements CodexRpcClient {
  readonly calls: Array<{ method: string; params?: Record<string, unknown> }> = [];

  constructor(readonly connection: CodexJsonRpcConnection) {}

  request(method: string, params?: Record<string, unknown>): Promise<unknown> {
    this.calls.push({ method, ...(params === undefined ? {} : { params }) });
    return this.connection.request(method, params);
  }

  onNotification(...args: Parameters<CodexRpcClient["onNotification"]>): () => void {
    return this.connection.onNotification(...args);
  }

  onRequest(...args: Parameters<CodexRpcClient["onRequest"]>): () => void {
    return this.connection.onRequest(...args);
  }

  dispose(): Promise<void> {
    return this.connection.dispose();
  }
}

interface ProcessFingerprint {
  readonly pid: number;
  readonly startedAt: string;
  readonly command: string;
}

interface ProcessSnapshot extends ProcessFingerprint {
  readonly parentPid: number;
}

interface AcceptanceRuntime {
  readonly runtime: CodexConversationRuntime;
  readonly rpc: RecordingRpc;
  readonly child: ReturnType<typeof spawn>;
  readonly bridge: Awaited<ReturnType<typeof startRuntimeBridge>>;
  readonly events: RuntimeEvent[];
  ownedProcesses: ProcessFingerprint[];
  disposed: boolean;
  bridgeDisposed: boolean;
}

async function launchAcceptanceRuntime(input: {
  readonly codexHome: string;
  readonly productHome: string;
  readonly serverPath: string;
  readonly workspace: WorkspaceScope;
  readonly swarmRecoveryOwner?: Parameters<typeof startRuntimeBridge>[1];
}): Promise<AcceptanceRuntime> {
  const bridge = await startRuntimeBridge(input.workspace, input.swarmRecoveryOwner);
  const child = spawn(
    command,
    codexAppServerArgs({
      productHome: input.productHome,
      workspace: input.workspace,
      bridgeUrl: bridge.url,
      bridgeToken: bridge.token,
      mcpServerPath: input.serverPath,
    }),
    {
      cwd: input.workspace.root,
      env: {
        ...process.env,
        CODEX_HOME: input.codexHome,
        CODEX_APP_SERVER_DISABLE_MANAGED_CONFIG: "1",
        SWARMX_BRIDGE_TOKEN: bridge.token,
        SWARMX_BRIDGE_URL: bridge.url,
        SWARMX_HOME: input.productHome,
        SWARMX_WORKSPACE_ID: input.workspace.id,
        SWARMX_WORKSPACE_LABEL: input.workspace.label,
        SWARMX_WORKSPACE_ROOT: input.workspace.root,
      },
      stdio: ["pipe", "pipe", "pipe"],
    },
  );
  const connection = new CodexJsonRpcConnection(child);
  const rpc = new RecordingRpc(connection);
  const runtime = new CodexConversationRuntime(rpc, { paginatedHistory: true });
  const events: RuntimeEvent[] = [];
  runtime.subscribe((event) => events.push(event));
  const result: AcceptanceRuntime = {
    runtime,
    rpc,
    child,
    bridge,
    events,
    ownedProcesses: [],
    disposed: false,
    bridgeDisposed: false,
  };
  try {
    await connection.initialize();
    await reconcileCodexSwarmBindings({
      journalRoot: join(input.productHome, "swarm"),
      runtime,
      workspace: input.workspace,
    });
    bridge.attach(runtime);
    recordOwnedProcesses(result);
  } catch (error) {
    try {
      await disposeAcceptanceRuntime(result);
    } catch {}
    throw error;
  }
  return result;
}

function codexSwarmDurableState(productHome: string, workspaceRoot: string): unknown {
  const journal = new SwarmJournal(join(productHome, "swarm"), { mode: "client" });
  try {
    const workspaceKey = journal.workspaceKey(workspaceRoot);
    return {
      bindings: new CodexMemberBindingStore(journal, workspaceKey).list(),
      teams: journal
        .list()
        .filter((team) => team.workspaceKey === workspaceKey)
        .map((team) => ({
          id: team.id,
          members: team.members.map((member) => ({ id: member.id, phase: member.phase })),
          revision: team.revision,
        })),
    };
  } finally {
    journal.close();
  }
}

async function disposeAcceptanceRuntime(value: AcceptanceRuntime | undefined): Promise<void> {
  if (value === undefined) return;
  await runCleanupSteps([
    async () => {
      recordOwnedProcesses(value);
    },
    async () => {
      if (value.disposed) return;
      try {
        await withTimeout(value.runtime.dispose(), "fallback runtime disposal");
      } finally {
        value.disposed = true;
      }
    },
    async () => {
      try {
        await withTimeout(childExit(value.child), "fallback App Server exit");
      } catch (error) {
        await runCleanupSteps([
          async () => {
            throw error;
          },
          async () => {
            value.child.kill("SIGKILL");
          },
          async () => {
            await withTimeout(childExit(value.child), "forced App Server exit");
          },
        ]);
      }
    },
    async () => {
      const live = await liveOwnedProcesses(value.ownedProcesses);
      await runCleanupSteps([
        () => killOwnedProcesses([...live].reverse()),
        async () => {
          const remaining = await liveOwnedProcesses(live);
          if (remaining.length > 0) {
            throw new Error(
              `Owned Codex processes survived cleanup: ${remaining.map(describeProcess).join(", ")}.`,
            );
          }
        },
      ]);
    },
    async () => {
      if (value.bridgeDisposed) return;
      try {
        await withTimeout(value.bridge.dispose(), "fallback bridge disposal");
      } finally {
        value.bridgeDisposed = true;
      }
    },
  ]);
}

async function runCleanupSteps(steps: readonly (() => Promise<void>)[]): Promise<void> {
  let firstError: unknown;
  let failed = false;
  for (const step of steps) {
    try {
      await step();
    } catch (error) {
      if (!failed) firstError = error;
      failed = true;
    }
  }
  if (failed) throw firstError;
}

async function callMcpWithApproval(
  target: AcceptanceRuntime,
  invoke: () => Promise<unknown>,
  decision: "accept" | "decline",
  rejectInvalidForm = false,
): Promise<unknown> {
  const mark = target.events.length;
  const pending = invoke();
  const approval = await waitForApproval(target.events, mark);
  expect(approval).toMatchObject({
    kind: "elicitation",
    questions: [{ id: "confirm", type: "boolean", required: true }],
  });
  const identity = {
    runtime: approval.runtime,
    conversationId: approval.conversationId,
    turnId: approval.turnId,
    itemId: approval.itemId,
    approvalId: approval.approvalId,
  };
  if (rejectInvalidForm) {
    await expect(
      target.runtime.respondToApproval({
        ...identity,
        decision: "accept",
        form: { confirm: "yes" },
      }),
    ).rejects.toThrow();
  }
  await target.runtime.respondToApproval({
    ...identity,
    decision,
    ...(decision === "accept" ? { form: { confirm: true } } : {}),
  });
  return withTimeout(pending, "approved MCP tool response");
}

async function waitForApproval(
  events: readonly RuntimeEvent[],
  start: number,
): Promise<ApprovalRequestedEvent> {
  const event = await waitForRuntimeEvent(
    events,
    start,
    (candidate) => candidate.type === "approval_requested",
  );
  if (event.type !== "approval_requested") throw new Error("Approval event invariant failed.");
  return event;
}

async function waitForTerminalTurn(events: readonly RuntimeEvent[], start: number): Promise<void> {
  await waitForRuntimeEvent(
    events,
    start,
    (event) => event.type === "turn_status" && event.status !== "running",
  );
}

async function waitForConversation(
  runtime: CodexConversationRuntime,
  predicate: (
    conversation: Awaited<ReturnType<CodexConversationRuntime["list"]>>[number],
  ) => boolean | Promise<boolean>,
): Promise<string> {
  for (let attempt = 0; attempt < 400; attempt += 1) {
    for (const conversation of await runtime.list()) {
      if (await predicate(conversation)) return conversation.conversationId;
    }
    await delay(25);
  }
  throw new Error("Timed out waiting for a native Codex conversation.");
}

async function waitForRuntimeEvent(
  events: readonly RuntimeEvent[],
  start: number,
  predicate: (event: RuntimeEvent) => boolean,
): Promise<RuntimeEvent> {
  for (let attempt = 0; attempt < 400; attempt += 1) {
    const event = events.slice(start).find(predicate);
    if (event !== undefined) return event;
    await delay(25);
  }
  throw new Error("Timed out waiting for a Codex runtime event.");
}

async function waitForResponseBody(
  responses: MockResponsesServer,
  minimumRequests: number,
  marker: string,
): Promise<void> {
  for (let attempt = 0; attempt < 400; attempt += 1) {
    if (
      responses.requests.length >= minimumRequests &&
      responses.requests.slice(0, minimumRequests).some((request) => request.body.includes(marker))
    ) {
      return;
    }
    await delay(25);
  }
  throw new Error(`Timed out waiting for Responses request marker "${marker}".`);
}

function lastUserItem(
  snapshot: Awaited<ReturnType<CodexConversationRuntime["read"]>>,
): UserMessageItem {
  for (const turn of [...snapshot.turns].reverse()) {
    const item = [...turn.items]
      .reverse()
      .find((candidate): candidate is UserMessageItem => candidate.type === "user_message");
    if (item !== undefined) return item;
  }
  throw new Error("Codex snapshot contains no user message.");
}

function nativeCodexId(value: string): string {
  if (!value.startsWith("codex:") || value.length === "codex:".length) {
    throw new Error(`Expected Codex-qualified id, received "${value}".`);
  }
  return value.slice("codex:".length);
}

function descendantProcessFingerprints(rootPid: number | undefined): ProcessFingerprint[] {
  if (rootPid === undefined) return [];
  const snapshots = processSnapshots();
  const byPid = new Map(snapshots.map((snapshot) => [snapshot.pid, snapshot]));
  const root = byPid.get(rootPid);
  if (root === undefined) return [];
  const children = new Map<number, ProcessSnapshot[]>();
  for (const snapshot of snapshots) {
    children.set(snapshot.parentPid, [...(children.get(snapshot.parentPid) ?? []), snapshot]);
  }
  const result = new Map<number, ProcessFingerprint>([[rootPid, root]]);
  const pending = [root];
  for (const parent of pending) {
    for (const child of children.get(parent.pid) ?? []) {
      if (result.has(child.pid)) continue;
      result.set(child.pid, child);
      pending.push(child);
    }
  }
  return [...result.values()];
}

function recordOwnedProcesses(value: AcceptanceRuntime): void {
  const captured = descendantProcessFingerprints(value.child.pid);
  const merged = new Map(
    value.ownedProcesses.map((fingerprint) => [processFingerprintKey(fingerprint), fingerprint]),
  );
  for (const fingerprint of captured) {
    merged.set(processFingerprintKey(fingerprint), fingerprint);
  }
  value.ownedProcesses = [...merged.values()];
}

async function liveOwnedProcesses(
  ownedProcesses: readonly ProcessFingerprint[],
): Promise<ProcessFingerprint[]> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const current = new Map(processSnapshots().map((snapshot) => [snapshot.pid, snapshot]));
    const live = ownedProcesses.filter((owned) => sameProcess(owned, current.get(owned.pid)));
    if (live.length === 0) return [];
    await delay(50);
  }
  const current = new Map(processSnapshots().map((snapshot) => [snapshot.pid, snapshot]));
  return ownedProcesses.filter((owned) => sameProcess(owned, current.get(owned.pid)));
}

async function killOwnedProcesses(
  ownedProcesses: readonly ProcessFingerprint[],
  inspect: (pid: number) => ProcessFingerprint | undefined = currentProcessFingerprint,
  kill: (pid: number) => void = killProcess,
): Promise<void> {
  await runCleanupSteps(
    ownedProcesses.map((owned) => async () => {
      if (!sameProcess(owned, inspect(owned.pid))) return;
      kill(owned.pid);
    }),
  );
}

function processSnapshots(): ProcessSnapshot[] {
  const listed = spawnSync("ps", ["-axo", "pid=,ppid=,lstart=,command="], {
    encoding: "utf8",
  });
  if (listed.status !== 0) {
    throw new Error(`Unable to inspect owned Codex processes: ${listed.stderr}`);
  }
  return listed.stdout
    .split("\n")
    .map(parseProcessSnapshot)
    .filter((snapshot): snapshot is ProcessSnapshot => snapshot !== undefined);
}

function currentProcessFingerprint(pid: number): ProcessFingerprint | undefined {
  const listed = spawnSync("ps", ["-p", String(pid), "-o", "pid=,ppid=,lstart=,command="], {
    encoding: "utf8",
  });
  if (listed.status === 1 && listed.stdout.trim() === "") return undefined;
  if (listed.status !== 0) {
    throw new Error(`Unable to verify owned Codex process ${String(pid)}: ${listed.stderr}`);
  }
  return listed.stdout
    .split("\n")
    .map(parseProcessSnapshot)
    .find((snapshot) => snapshot?.pid === pid);
}

function parseProcessSnapshot(line: string): ProcessSnapshot | undefined {
  const match = /^\s*(\d+)\s+(\d+)\s+(.{24})\s+(.+)$/u.exec(line);
  if (match === null) return undefined;
  const pid = Number(match[1]);
  const parentPid = Number(match[2]);
  const startedAt = match[3];
  const command = match[4];
  if (
    !Number.isSafeInteger(pid) ||
    !Number.isSafeInteger(parentPid) ||
    startedAt === undefined ||
    command === undefined
  ) {
    return undefined;
  }
  return { pid, parentPid, startedAt, command };
}

function sameProcess(
  expected: ProcessFingerprint,
  current: ProcessFingerprint | undefined,
): boolean {
  return (
    current !== undefined &&
    current.pid === expected.pid &&
    current.startedAt === expected.startedAt &&
    current.command === expected.command
  );
}

function processFingerprintKey(value: ProcessFingerprint): string {
  return JSON.stringify([value.pid, value.startedAt, value.command]);
}

function describeProcess(value: ProcessFingerprint): string {
  return `${String(value.pid)} (${value.command})`;
}

function killProcess(pid: number): void {
  try {
    process.kill(pid, "SIGKILL");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
  }
}

async function withTimeout<Value>(
  promise: Promise<Value>,
  label: string,
  timeoutMs = 15_000,
): Promise<Value> {
  const controller = new AbortController();
  const timeout = delay(timeoutMs, undefined, { signal: controller.signal }).then(() => {
    throw new Error(`Timed out waiting for ${label}.`);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    controller.abort();
  }
}

function childExit(child: ReturnType<typeof spawn>): Promise<void> {
  if (child.pid === undefined || child.exitCode !== null || child.signalCode !== null) {
    return Promise.resolve();
  }
  return new Promise((resolve) => child.once("exit", () => resolve()));
}

async function removeTreeAfterChildrenExit(path: string): Promise<void> {
  for (let attempt = 0; ; attempt += 1) {
    try {
      rmSync(path, { recursive: true, force: true });
      return;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOTEMPTY" || attempt === 19) throw error;
      await delay(50);
    }
  }
}
