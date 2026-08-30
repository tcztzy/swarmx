import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { ElicitRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { createPkbToolDefinition } from "@swarmx/dsh-pkb";
import { createScienceToolDefinitions } from "@swarmx/dsh-science/tools";
import { SwarmJournal } from "@swarmx/dsh-swarm";
import { createSwarmToolDefinition } from "@swarmx/dsh-swarm/tools";
import { afterEach, describe, expect, it, vi } from "vitest";
import { startRuntimeBridge } from "../src/runtime/bridge.js";
import {
  connectProductMcpTransport,
  createProductMcpHost,
  disposeInOrder,
} from "../src/runtime/codex/mcp-server.js";
import { CodexMemberBindingStore } from "../src/runtime/codex/member-bindings.js";
import type {
  ConversationRuntime,
  ConversationSnapshot,
  ConversationSummary,
  WorkspaceScope,
} from "../src/runtime/contracts.js";
import { WorkspaceAuthority } from "../src/runtime/workspace.js";

const roots: string[] = [];

function memberBindings(swarmRoot: string, workspaceRoot: string) {
  const journal = new SwarmJournal(swarmRoot, { mode: "client" });
  try {
    return new CodexMemberBindingStore(journal, journal.workspaceKey(workspaceRoot)).list();
  } finally {
    journal.close();
  }
}

function initializeSwarmStorage(productHome: string): void {
  const swarmRoot = join(productHome, "swarm");
  if (existsSync(join(swarmRoot, "swarm.sqlite"))) return;
  const owner = new SwarmJournal(swarmRoot);
  owner.close();
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("Codex product MCP", () => {
  it("attempts every product-host disposer and preserves the first failure", async () => {
    const calls: string[] = [];
    const first = new Error("first disposer failed");
    await expect(
      disposeInOrder([
        async () => {
          calls.push("first");
          throw first;
        },
        async () => {
          calls.push("second");
        },
        async () => {
          calls.push("third");
          throw new Error("later disposer failed");
        },
      ]),
    ).rejects.toBe(first);
    expect(calls).toEqual(["first", "second", "third"]);
  });

  it("disposes the product host when stdio transport connection fails", async () => {
    const connectionError = new Error("stdio connection failed");
    const cleanupError = new Error("cleanup failed");
    const connect = vi.fn(async () => {
      throw connectionError;
    });
    const dispose = vi.fn(async () => undefined);

    await expect(
      connectProductMcpTransport({ connect } as never, {} as never, dispose),
    ).rejects.toBe(connectionError);
    expect(dispose).toHaveBeenCalledOnce();

    dispose.mockRejectedValueOnce(cleanupError);
    await expect(
      connectProductMcpTransport({ connect } as never, {} as never, dispose),
    ).rejects.toEqual(
      expect.objectContaining({
        errors: [connectionError, cleanupError],
        message: "Codex product MCP transport startup and cleanup failed",
      }),
    );
  });

  it("rolls back partial product-host startup when persisted Swarm state is invalid", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-startup-rollback-"));
    roots.push(root);
    const environment = {
      SWARMX_BRIDGE_TOKEN: "bridge-token",
      SWARMX_BRIDGE_URL: "http://127.0.0.1:1/",
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: "workspace-1",
      SWARMX_WORKSPACE_LABEL: "workspace",
      SWARMX_WORKSPACE_ROOT: root,
    };
    initializeSwarmStorage(environment.SWARMX_HOME);
    const initialized = await createProductMcpHost(environment);
    await initialized.dispose();
    const journal = new SwarmJournal(join(root, "home", "swarm"));
    const databasePath = journal.databasePath;
    const workspaceKey = journal.workspaceKey(root);
    journal.close();
    const database = new DatabaseSync(databasePath);
    database
      .prepare(
        `INSERT INTO swarm_member_bindings(workspace_key, runtime, member_id, handle)
         VALUES (?, 'codex', ?, '')`,
      )
      .run(workspaceKey, randomUUID());
    database.close();

    await expect(createProductMcpHost(environment)).rejects.toThrow("binding store is invalid");
    const repaired = new DatabaseSync(databasePath);
    repaired.prepare("DELETE FROM swarm_member_bindings WHERE handle = ''").run();
    repaired.close();
    const retried = await createProductMcpHost(environment);
    await retried.dispose();
  });

  it("publishes the shared Science and PKB tool definitions over MCP", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-"));
    roots.push(root);
    initializeSwarmStorage(join(root, "home"));
    const host = await createProductMcpHost({
      SWARMX_BRIDGE_TOKEN: "bridge-token",
      SWARMX_BRIDGE_URL: "http://127.0.0.1:1/",
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: "workspace-1",
      SWARMX_WORKSPACE_LABEL: "workspace",
      SWARMX_WORKSPACE_ROOT: root,
    });
    const client = new Client(
      { name: "swarmx-test", version: "0.1.0" },
      { capabilities: { elicitation: { form: {} } } },
    );
    client.setRequestHandler(ElicitRequestSchema, async () => ({
      action: "accept",
      content: { confirm: true },
    }));
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    serverTransport.sessionId = "published-tools-session";
    try {
      await Promise.all([client.connect(clientTransport), host.server.connect(serverTransport)]);
      const tools = await client.listTools();
      expect(tools.tools.map((tool) => tool.name)).toEqual([
        "science_notebook",
        "science_write",
        "science_figure",
        "science_experiment",
        "science_record",
        "science_query",
        "literature_search",
        "science_export",
        "pkb",
        "swarm",
      ]);
      const ownedDefinitions = [
        ...createScienceToolDefinitions({} as never),
        createPkbToolDefinition({} as never),
        createSwarmToolDefinition({} as never),
      ];
      for (const definition of ownedDefinitions) {
        const inputSchema = tools.tools.find((tool) => tool.name === definition.name)?.inputSchema;
        const mcpParameters = "mcpParameters" in definition ? definition.mcpParameters : undefined;
        expect(inputSchema).toEqual(mcpParameters ?? definition.parameters);
      }
      expect(tools.tools.find((tool) => tool.name === "pkb")?.inputSchema).toMatchObject({
        type: "object",
        required: ["action", "request"],
        properties: {
          action: { enum: expect.arrayContaining(["create_knowledge"]) },
          request: { type: "object" },
        },
      });
      const pkbResult = await client.callTool({
        name: "pkb",
        arguments: {
          action: "create_knowledge",
          request: {
            requestId: randomUUID(),
            body: "# Codex default scope\n\nCreated through MCP.",
            description: "Created through MCP with an omitted scope.",
            title: "Codex default scope",
            type: "Finding",
          },
        },
      });
      expect(textValue(pkbResult)).toMatchObject({
        action: "create_knowledge",
        data: { id: expect.stringMatching(/^workspaces\//u) },
      });
      const projectResult = await client.callTool({
        name: "science_notebook",
        arguments: {
          action: "create_project",
          request: { requestId: randomUUID(), title: "Codex project" },
        },
      });
      const project = textValue(projectResult).data as { id: string };
      const image = Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAADCAYAAAC56t6BAAAAEUlEQVR4nGP4z8DwH4QZMBgAoXkL9U3EmgcAAAAASUVORK5CYII=",
        "base64",
      );
      writeFileSync(join(root, "point.png"), image);
      const artifactResult = await client.callTool({
        name: "science_record",
        arguments: {
          action: "register_artifact",
          request: {
            requestId: randomUUID(),
            projectId: project.id,
            relativePath: "point.png",
            kind: "figure",
            title: "point.png",
            mime: "image/png",
            runId: null,
            environment: {},
            license: null,
            sourceEntityIds: [],
            reproducibilityMetadata: false,
          },
        },
      });
      const artifact = textValue(artifactResult).data as {
        digest: string;
        id: string;
        mime: string;
        title: string;
      };
      const inspected = await client.callTool({
        name: "science_query",
        arguments: {
          action: "inspect_annotation",
          request: {
            type: "comment",
            id: "annotation-1",
            comment: "Inspect this pixel",
            created_at: 1_787_371_200_000,
            target: {
              type: "image_point",
              artifact_id: artifact.id,
              project_id: project.id,
              title: artifact.title,
              digest: artifact.digest,
              mime: artifact.mime,
              point: { x: 0.25, y: 0.75 },
            },
          },
        },
      });
      expect(textValue(inspected)).toMatchObject({
        data: {
          attachment: { bytes: image.byteLength, height: 3, mediaType: "image/png", width: 2 },
        },
      });
      expect(inspected.content).toEqual([
        expect.objectContaining({
          type: "text",
          text: expect.stringContaining("Inspect this pixel"),
        }),
        expect.objectContaining({
          type: "image",
          data: image.toString("base64"),
          mimeType: "image/png",
        }),
      ]);
      const malformed = Buffer.concat([
        Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
        Buffer.from([0, 0, 0, 13]),
        Buffer.from("IHDR", "ascii"),
        Buffer.from([0, 0, 0, 2, 0, 0, 0, 3]),
      ]);
      writeFileSync(join(root, "malformed.png"), malformed);
      const malformedArtifactResult = await client.callTool({
        name: "science_record",
        arguments: {
          action: "register_artifact",
          request: {
            requestId: randomUUID(),
            projectId: project.id,
            relativePath: "malformed.png",
            kind: "figure",
            title: "malformed.png",
            mime: "image/png",
            runId: null,
            environment: {},
            license: null,
            sourceEntityIds: [],
            reproducibilityMetadata: false,
          },
        },
      });
      const malformedArtifact = textValue(malformedArtifactResult).data as {
        digest: string;
        id: string;
        mime: string;
        title: string;
      };
      const malformedInspection = await client.callTool({
        name: "science_query",
        arguments: {
          action: "inspect_annotation",
          request: {
            type: "comment",
            id: "annotation-malformed",
            comment: "Reject the header-only image",
            created_at: 1_787_371_200_000,
            target: {
              type: "image_point",
              artifact_id: malformedArtifact.id,
              project_id: project.id,
              title: malformedArtifact.title,
              digest: malformedArtifact.digest,
              mime: malformedArtifact.mime,
              point: { x: 0.25, y: 0.75 },
            },
          },
        },
      });
      expect(malformedInspection.isError).toBe(true);
      expect(malformedInspection.content).toEqual([
        expect.objectContaining({ type: "text", text: expect.stringContaining("malformed") }),
      ]);
      const created = await client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Codex team" } },
      });
      expect(created.isError, JSON.stringify(created)).not.toBe(true);
      expect(created.content).toEqual([
        expect.objectContaining({ type: "text", text: expect.stringContaining("Codex team") }),
      ]);
    } finally {
      await client.close();
      await host.dispose();
    }
  });

  it("applies the Host-projected Science limits inside the Codex MCP carrier", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-science-config-"));
    roots.push(root);
    const connection = await connectHost({
      SWARMX_BRIDGE_TOKEN: "bridge-token",
      SWARMX_BRIDGE_URL: "http://127.0.0.1:1/",
      SWARMX_HOME: join(root, "home"),
      SWARMX_SCIENCE_CONFIG: JSON.stringify({ maxArtifactBytes: 1 }),
      SWARMX_WORKSPACE_ID: "workspace-1",
      SWARMX_WORKSPACE_LABEL: "workspace",
      SWARMX_WORKSPACE_ROOT: root,
    });
    try {
      const projectResult = await connection.client.callTool({
        name: "science_notebook",
        arguments: {
          action: "create_project",
          request: { requestId: randomUUID(), title: "Bounded Codex project" },
        },
      });
      const project = textValue(projectResult).data as { id: string };
      writeFileSync(join(root, "too-large.txt"), "xx");
      const rejected = await connection.client.callTool({
        name: "science_record",
        arguments: {
          action: "register_artifact",
          request: {
            requestId: randomUUID(),
            projectId: project.id,
            relativePath: "too-large.txt",
            kind: "dataset",
            title: "too-large.txt",
            mime: "text/plain",
            runId: null,
            environment: {},
            license: null,
            sourceEntityIds: [],
            reproducibilityMetadata: false,
          },
        },
      });
      expect(rejected.isError).toBe(true);
      expect(rejected.content).toEqual([
        expect.objectContaining({ text: expect.stringMatching(/exceeds.*configured.*1/iu) }),
      ]);
    } finally {
      await connection.client.close();
      await connection.host.dispose();
    }
  });

  it("binds Swarm lead authority to the exact native Codex thread across restarts", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-authority-"));
    roots.push(root);
    const environment = {
      SWARMX_BRIDGE_TOKEN: "bridge-token",
      SWARMX_BRIDGE_URL: "http://127.0.0.1:1/",
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: "workspace-1",
      SWARMX_WORKSPACE_LABEL: "workspace",
      SWARMX_WORKSPACE_ROOT: root,
    };
    const first = await connectHost(environment);
    try {
      const created = await first.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Transport-owned team" } },
        _meta: { threadId: "native-thread-1" },
      });
      expect(textValue(created)).toMatchObject({
        action: "create",
        data: { kind: "active", name: "Transport-owned team" },
      });
      const status = await first.client.callTool({
        name: "swarm",
        arguments: { action: "status", request: {} },
        _meta: { threadId: "native-thread-1" },
      });
      expect(textValue(status)).toMatchObject({
        action: "status",
        data: { kind: "active", name: "Transport-owned team" },
      });
      const archive = await first.client.callTool({
        name: "swarm",
        arguments: { action: "archive", request: {} },
        _meta: { threadId: "native-thread-2" },
      });
      expect(archive.isError).toBe(true);
    } finally {
      await first.client.close();
      await first.host.dispose();
    }

    const second = await connectHost(environment);
    try {
      const status = await second.client.callTool({
        name: "swarm",
        arguments: { action: "status", request: {} },
        _meta: { threadId: "native-thread-1" },
      });
      expect(textValue(status)).toMatchObject({
        action: "status",
        data: { kind: "active", name: "Transport-owned team" },
      });
      const archived = await second.client.callTool({
        name: "swarm",
        arguments: { action: "archive", request: {} },
        _meta: { threadId: "native-thread-1" },
      });
      expect(archived.isError).not.toBe(true);
    } finally {
      await second.client.close();
      await second.host.dispose();
    }
  });

  it("applies Team role, write-attempt, and effect guards to Codex product tools", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-product-guards-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtime = fakeBridgeRuntime(workspace);
    const bridge = await startRuntimeBridge(workspace);
    bridge.attach(runtime.value);
    const environment = {
      SWARMX_BRIDGE_TOKEN: bridge.token,
      SWARMX_BRIDGE_URL: bridge.url,
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: workspace.id,
      SWARMX_WORKSPACE_LABEL: workspace.label,
      SWARMX_WORKSPACE_ROOT: workspace.root,
    };
    const connection = await connectHost(environment);
    try {
      await connection.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Guarded products" } },
        _meta: { threadId: "lead-thread" },
      });

      const deniedLeadMutation = await connection.client.callTool({
        name: "science_notebook",
        arguments: {
          action: "create_project",
          request: { requestId: randomUUID(), title: "Denied lead project" },
        },
        _meta: { threadId: "lead-thread" },
      });
      expect(deniedLeadMutation.isError).toBe(true);
      expect(deniedLeadMutation.content).toEqual([
        expect.objectContaining({
          type: "text",
          text: expect.stringContaining("active write task attempt"),
        }),
      ]);
      const deniedLeadPkbMutation = await connection.client.callTool({
        name: "pkb",
        arguments: {
          action: "create_knowledge",
          request: {
            body: "# Denied\n\nNo write attempt.",
            description: "Must be denied before approval.",
            title: "Denied Team PKB write",
            type: "Decision",
          },
        },
        _meta: { threadId: "lead-thread" },
      });
      expect(deniedLeadPkbMutation.isError).toBe(true);
      expect(deniedLeadPkbMutation.content).toEqual([
        expect.objectContaining({
          type: "text",
          text: expect.stringContaining("active write task attempt"),
        }),
      ]);

      await connection.client.callTool({
        name: "swarm",
        arguments: {
          action: "add_member",
          request: {
            description: "Owns one guarded mutation",
            name: "worker",
            prompt: "Join the Team.",
            role: "implementer",
          },
        },
        _meta: { threadId: "lead-thread" },
      });
      const deniedMemberPkb = await connection.client.callTool({
        name: "pkb",
        arguments: { action: "search_knowledge", request: { query: "forbidden" } },
        _meta: { threadId: "member-thread-1" },
      });
      expect(deniedMemberPkb.isError).toBe(true);
      expect(deniedMemberPkb.content).toEqual([
        expect.objectContaining({
          type: "text",
          text: expect.stringContaining("cannot delegate or access PKB"),
        }),
      ]);

      const createdTask = await connection.client.callTool({
        name: "swarm",
        arguments: {
          action: "create_task",
          request: {
            assignedTo: "worker",
            blockedBy: [],
            description: "Create one Science project",
            kind: "write",
            subject: "Guarded Science mutation",
            writeScopes: ["science"],
          },
        },
        _meta: { threadId: "lead-thread" },
      });
      expect(textValue(createdTask)).toMatchObject({
        data: { tasks: [expect.objectContaining({ ownerName: "worker", status: "in_progress" })] },
      });

      const allowed = await connection.client.callTool({
        name: "science_notebook",
        arguments: {
          action: "create_project",
          request: { requestId: randomUUID(), title: "Allowed member project" },
        },
        _meta: { threadId: "member-thread-1" },
      });
      expect(allowed.isError).not.toBe(true);
      const journal = new SwarmJournal(join(root, "home", "swarm"), { mode: "client" });
      expect(journal.list()[0]?.effects).toEqual([
        expect.objectContaining({
          ownerId: expect.any(String),
          status: "succeeded",
          toolName: "science_notebook",
        }),
      ]);
      journal.close();

      const archived = await connection.client.callTool({
        name: "swarm",
        arguments: { action: "archive", request: {} },
        _meta: { threadId: "lead-thread" },
      });
      expect(archived.isError).not.toBe(true);
      const archivedRead = await connection.client.callTool({
        name: "pkb",
        arguments: { action: "search_knowledge", request: { query: "nothing" } },
        _meta: { threadId: "lead-thread" },
      });
      expect(archivedRead.isError).not.toBe(true);
      const archivedMutation = await connection.client.callTool({
        name: "science_notebook",
        arguments: {
          action: "create_project",
          request: { requestId: randomUUID(), title: "Archived mutation" },
        },
        _meta: { threadId: "lead-thread" },
      });
      expect(archivedMutation.isError).toBe(true);
      expect(archivedMutation.content).toEqual([
        expect.objectContaining({
          type: "text",
          text: expect.stringContaining("active write task attempt"),
        }),
      ]);
    } finally {
      await connection.client.close();
      await connection.host.dispose();
      await bridge.dispose();
    }
  });

  it("uses an exact MCP transport session as continuation authority", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-session-authority-"));
    roots.push(root);
    const environment = {
      SWARMX_BRIDGE_TOKEN: "bridge-token",
      SWARMX_BRIDGE_URL: "http://127.0.0.1:1/",
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: "workspace-1",
      SWARMX_WORKSPACE_LABEL: "workspace",
      SWARMX_WORKSPACE_ROOT: root,
    };
    const first = await connectHost(environment, "transport-session-1");
    try {
      const created = await first.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Session-owned team" } },
      });
      expect(created.isError).not.toBe(true);
    } finally {
      await first.client.close();
      await first.host.dispose();
    }

    const foreign = await connectHost(environment, "transport-session-2");
    try {
      const status = await foreign.client.callTool({
        name: "swarm",
        arguments: { action: "status", request: {} },
      });
      expect(textValue(status)).toMatchObject({ data: { kind: "inactive" } });
      const archive = await foreign.client.callTool({
        name: "swarm",
        arguments: { action: "archive", request: {} },
      });
      expect(archive.isError).toBe(true);
    } finally {
      await foreign.client.close();
      await foreign.host.dispose();
    }

    const resumed = await connectHost(environment, "transport-session-1");
    try {
      const status = await resumed.client.callTool({
        name: "swarm",
        arguments: { action: "status", request: {} },
      });
      expect(textValue(status)).toMatchObject({
        data: { kind: "active", name: "Session-owned team" },
      });
      const archived = await resumed.client.callTool({
        name: "swarm",
        arguments: { action: "archive", request: {} },
      });
      expect(archived.isError).not.toBe(true);
    } finally {
      await resumed.client.close();
      await resumed.host.dispose();
    }
  });

  it("scopes native Codex Thread authority to the canonical workspace", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-workspace-authority-"));
    roots.push(root);
    const workspaceA = join(root, "workspace-a");
    const workspaceB = join(root, "workspace-b");
    mkdirSync(workspaceA);
    mkdirSync(workspaceB);
    const shared = {
      SWARMX_BRIDGE_TOKEN: "bridge-token",
      SWARMX_BRIDGE_URL: "http://127.0.0.1:1/",
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_LABEL: "workspace",
    };
    const first = await connectHost({
      ...shared,
      SWARMX_WORKSPACE_ID: "workspace-a",
      SWARMX_WORKSPACE_ROOT: workspaceA,
    });
    try {
      const created = await first.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Workspace A team" } },
        _meta: { threadId: "same-native-thread" },
      });
      expect(created.isError).not.toBe(true);
    } finally {
      await first.client.close();
      await first.host.dispose();
    }

    const second = await connectHost({
      ...shared,
      SWARMX_WORKSPACE_ID: "workspace-b",
      SWARMX_WORKSPACE_ROOT: workspaceB,
    });
    try {
      const status = await second.client.callTool({
        name: "swarm",
        arguments: { action: "status", request: {} },
        _meta: { threadId: "same-native-thread" },
      });
      expect(textValue(status)).toMatchObject({
        action: "status",
        data: { kind: "inactive", revision: 0 },
      });
      const archive = await second.client.callTool({
        name: "swarm",
        arguments: { action: "archive", request: {} },
        _meta: { threadId: "same-native-thread" },
      });
      expect(archive.isError).toBe(true);
      const created = await second.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Workspace B team" } },
        _meta: { threadId: "same-native-thread" },
      });
      expect(created.isError).not.toBe(true);
      const ownArchive = await second.client.callTool({
        name: "swarm",
        arguments: { action: "archive", request: {} },
        _meta: { threadId: "same-native-thread" },
      });
      expect(ownArchive.isError).not.toBe(true);
    } finally {
      await second.client.close();
      await second.host.dispose();
    }
  });

  it("never recovers or rehydrates active Swarm state from another workspace", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-workspace-recovery-"));
    roots.push(root);
    const workspaceA = join(root, "workspace-a");
    const workspaceB = join(root, "workspace-b");
    mkdirSync(workspaceA);
    mkdirSync(workspaceB);
    const shared = {
      SWARMX_BRIDGE_TOKEN: "bridge-token",
      SWARMX_BRIDGE_URL: "http://127.0.0.1:1/",
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_LABEL: "workspace",
    };
    const first = await connectHost({
      ...shared,
      SWARMX_WORKSPACE_ID: "workspace-a",
      SWARMX_WORKSPACE_ROOT: workspaceA,
    });
    try {
      await first.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Live workspace A team" } },
        _meta: { threadId: "workspace-a-lead" },
      });
      const added = await first.client.callTool({
        name: "swarm",
        arguments: {
          action: "add_member",
          request: {
            description: "Remains active while another workspace starts",
            name: "worker",
            prompt: "Join and wait.",
          },
        },
        _meta: { threadId: "workspace-a-lead" },
      });
      expect(added.isError).not.toBe(true);
      const task = await first.client.callTool({
        name: "swarm",
        arguments: {
          action: "create_task",
          request: {
            assignedTo: "worker",
            blockedBy: [],
            description: "Remain in progress",
            kind: "read",
            subject: "Cross-workspace recovery guard",
            writeScopes: [],
          },
        },
        _meta: { threadId: "workspace-a-lead" },
      });
      expect(task.isError).not.toBe(true);
      const beforeSecond = textValue(
        await first.client.callTool({
          name: "swarm",
          arguments: { action: "status", request: {} },
          _meta: { threadId: "workspace-a-lead" },
        }),
      );
      expect(beforeSecond).toMatchObject({
        data: { tasks: [expect.objectContaining({ status: "in_progress" })] },
      });

      const second = await connectHost({
        ...shared,
        SWARMX_WORKSPACE_ID: "workspace-b",
        SWARMX_WORKSPACE_ROOT: workspaceB,
      });
      try {
        const secondStatus = await second.client.callTool({
          name: "swarm",
          arguments: { action: "status", request: {} },
          _meta: { threadId: "workspace-b-observer" },
        });
        expect(textValue(secondStatus)).toMatchObject({ data: { kind: "inactive" } });
        expect(memberBindings(join(root, "home", "swarm"), workspaceA)).toHaveLength(1);
        expect(memberBindings(join(root, "home", "swarm"), workspaceB)).toEqual([]);
        const status = await first.client.callTool({
          name: "swarm",
          arguments: { action: "status", request: {} },
          _meta: { threadId: "workspace-a-lead" },
        });
        expect(textValue(status)).toMatchObject({
          data: {
            kind: "active",
            members: [
              expect.objectContaining({ name: "lead", status: "idle" }),
              expect.objectContaining({ name: "worker", status: "idle" }),
            ],
            tasks: [expect.objectContaining({ status: "in_progress" })],
          },
        });
      } finally {
        await second.client.close();
        await second.host.dispose();
      }
      const archived = await first.client.callTool({
        name: "swarm",
        arguments: { action: "archive", request: {} },
        _meta: { threadId: "workspace-a-lead" },
      });
      expect(archived.isError).not.toBe(true);
    } finally {
      await first.client.close();
      await first.host.dispose();
    }
  });

  it("never crash-recovers live state from another per-Thread MCP in the same workspace", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-live-owner-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtime = fakeBridgeRuntime(workspace);
    const bridge = await startRuntimeBridge(workspace);
    bridge.attach(runtime.value);
    const environment = {
      SWARMX_BRIDGE_TOKEN: bridge.token,
      SWARMX_BRIDGE_URL: bridge.url,
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: workspace.id,
      SWARMX_WORKSPACE_LABEL: workspace.label,
      SWARMX_WORKSPACE_ROOT: workspace.root,
    };
    const owner = await connectHost(environment);
    try {
      await owner.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Live owner team" } },
        _meta: { threadId: "owner-thread" },
      });
      await owner.client.callTool({
        name: "swarm",
        arguments: {
          action: "add_member",
          request: {
            description: "Keeps one live task",
            name: "worker",
            prompt: "Join and wait.",
          },
        },
        _meta: { threadId: "owner-thread" },
      });
      await owner.client.callTool({
        name: "swarm",
        arguments: {
          action: "create_task",
          request: {
            assignedTo: "worker",
            blockedBy: [],
            description: "Must remain live",
            kind: "read",
            subject: "Live task",
            writeScopes: [],
          },
        },
        _meta: { threadId: "owner-thread" },
      });
      const before = new SwarmJournal(join(root, "home", "swarm"), { mode: "client" });
      const beforeState = before.list()[0];
      const projectionMarker = 424_242;
      const beforeDatabase = new DatabaseSync(before.databasePath);
      beforeDatabase
        .prepare("UPDATE swarm_teams SET created_at = ? WHERE team_id = ?")
        .run(projectionMarker, beforeState?.id);
      beforeDatabase.close();
      before.close();
      expect(beforeState?.tasks[0]?.status).toBe("in_progress");

      const auxiliary = await connectHost(environment);
      try {
        const status = await auxiliary.client.callTool({
          name: "swarm",
          arguments: { action: "status", request: {} },
          _meta: { threadId: "unrelated-thread" },
        });
        expect(textValue(status)).toMatchObject({ data: { kind: "inactive" } });
      } finally {
        await auxiliary.client.close();
        await auxiliary.host.dispose();
      }

      const after = new SwarmJournal(join(root, "home", "swarm"), { mode: "client" });
      const afterState = after.list()[0];
      const afterDatabase = new DatabaseSync(after.databasePath);
      const storedProjection = afterDatabase
        .prepare("SELECT created_at FROM swarm_teams WHERE team_id = ?")
        .get(afterState?.id) as { created_at: number } | undefined;
      afterDatabase.close();
      after.close();
      expect(afterState?.revision).toBe(beforeState?.revision);
      expect(afterState?.tasks[0]?.status).toBe("in_progress");
      expect(storedProjection?.created_at).toBe(projectionMarker);
      await owner.client.callTool({
        name: "swarm",
        arguments: { action: "archive", request: {} },
        _meta: { threadId: "owner-thread" },
      });
    } finally {
      await owner.client.close();
      await owner.host.dispose();
      await bridge.dispose();
    }
  });

  it("preserves claims made by two already-open per-Thread MCP processes", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-binding-writers-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtime = fakeBridgeRuntime(workspace);
    const bridge = await startRuntimeBridge(workspace);
    bridge.attach(runtime.value);
    const environment = {
      SWARMX_BRIDGE_TOKEN: bridge.token,
      SWARMX_BRIDGE_URL: bridge.url,
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: workspace.id,
      SWARMX_WORKSPACE_LABEL: workspace.label,
      SWARMX_WORKSPACE_ROOT: workspace.root,
    };
    const first = await connectHost(environment);
    const second = await connectHost(environment);
    try {
      await second.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Second writer team" } },
        _meta: { threadId: "lead-b" },
      });
      await first.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "First writer team" } },
        _meta: { threadId: "lead-a" },
      });
      for (const [connection, lead, name] of [
        [first, "lead-a", "worker-a"],
        [second, "lead-b", "worker-b"],
      ] as const) {
        const added = await connection.client.callTool({
          name: "swarm",
          arguments: {
            action: "add_member",
            request: {
              description: `Native ${name}`,
              name,
              prompt: "Join and wait.",
            },
          },
          _meta: { threadId: lead },
        });
        expect(added.isError).not.toBe(true);
      }
      const claims = memberBindings(join(root, "home", "swarm"), root);
      expect(claims).toHaveLength(2);
      expect(claims).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ conversationId: "codex:member-thread-1" }),
          expect.objectContaining({ conversationId: "codex:member-thread-2" }),
        ]),
      );

      for (const [nativeThread, memberName] of [
        ["member-thread-1", "worker-a"],
        ["member-thread-2", "worker-b"],
      ] as const) {
        const child = await connectHost(environment);
        try {
          const status = await child.client.callTool({
            name: "swarm",
            arguments: { action: "status", request: {} },
            _meta: { threadId: nativeThread },
          });
          expect(textValue(status)).toMatchObject({ data: { memberName } });
        } finally {
          await child.client.close();
          await child.host.dispose();
        }
      }
      for (const [connection, lead] of [
        [first, "lead-a"],
        [second, "lead-b"],
      ] as const) {
        await connection.client.callTool({
          name: "swarm",
          arguments: { action: "archive", request: {} },
          _meta: { threadId: lead },
        });
      }
    } finally {
      await first.client.close();
      await first.host.dispose();
      await second.client.close();
      await second.host.dispose();
      await bridge.dispose();
    }
  });

  it("never archives a native Thread already claimed by another member", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-binding-conflict-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtime = fakeBridgeRuntime(workspace);
    const bridge = await startRuntimeBridge(workspace);
    bridge.attach(runtime.value);
    const connection = await connectHost({
      SWARMX_BRIDGE_TOKEN: bridge.token,
      SWARMX_BRIDGE_URL: bridge.url,
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: workspace.id,
      SWARMX_WORKSPACE_LABEL: workspace.label,
      SWARMX_WORKSPACE_ROOT: workspace.root,
    });
    try {
      await connection.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Original owner" } },
        _meta: { threadId: "lead-a" },
      });
      await connection.client.callTool({
        name: "swarm",
        arguments: {
          action: "add_member",
          request: {
            description: "Owns the original native Thread",
            name: "worker-a",
            prompt: "Join the Team.",
          },
        },
        _meta: { threadId: "lead-a" },
      });
      const originalBinding = memberBindings(join(root, "home", "swarm"), root)[0];
      expect(originalBinding).toEqual(
        expect.objectContaining({ conversationId: "codex:member-thread-1" }),
      );
      await connection.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Conflicting owner" } },
        _meta: { threadId: "lead-b" },
      });
      vi.spyOn(runtime.value, "createProvisionedMember").mockResolvedValueOnce({
        runtime: "codex",
        conversationId: "codex:member-thread-1",
        workspace: { id: workspace.id, label: workspace.label },
        title: "Already owned child",
        archived: false,
        updatedAt: Date.now(),
      });
      runtime.archive.mockClear();

      const conflict = await connection.client.callTool({
        name: "swarm",
        arguments: {
          action: "add_member",
          request: {
            description: "Must not take another member's Thread",
            name: "worker-b",
            prompt: "Join the Team.",
          },
        },
        _meta: { threadId: "lead-b" },
      });

      expect(conflict.isError).toBe(true);
      expect(conflict.content).toEqual([
        expect.objectContaining({
          type: "text",
          text: expect.stringContaining("already belongs to another member"),
        }),
      ]);
      expect(runtime.archive).not.toHaveBeenCalled();
      await expect(runtime.value.read("codex:member-thread-1")).resolves.toMatchObject({
        archived: false,
      });
      expect(memberBindings(join(root, "home", "swarm"), root)).toEqual([originalBinding]);
      const originalStatus = await connection.client.callTool({
        name: "swarm",
        arguments: { action: "status", request: {} },
        _meta: { threadId: "member-thread-1" },
      });
      expect(textValue(originalStatus)).toMatchObject({ data: { memberName: "worker-a" } });

      await connection.client.callTool({
        name: "swarm",
        arguments: { action: "archive", request: {} },
        _meta: { threadId: "lead-b" },
      });
      await connection.client.callTool({
        name: "swarm",
        arguments: { action: "archive", request: {} },
        _meta: { threadId: "lead-a" },
      });
    } finally {
      await connection.client.close();
      await connection.host.dispose();
      await bridge.dispose();
    }
  });

  it("restores a native child Thread as the exact Swarm member after MCP restart", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-member-restart-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtime = fakeBridgeRuntime(workspace);
    const bridge = await startRuntimeBridge(workspace);
    bridge.attach(runtime.value);
    const environment = {
      SWARMX_BRIDGE_TOKEN: bridge.token,
      SWARMX_BRIDGE_URL: bridge.url,
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: workspace.id,
      SWARMX_WORKSPACE_LABEL: workspace.label,
      SWARMX_WORKSPACE_ROOT: workspace.root,
    };
    try {
      const first = await connectHost(environment);
      try {
        await first.client.callTool({
          name: "swarm",
          arguments: { action: "create", request: { name: "Restartable team" } },
          _meta: { threadId: "lead-thread" },
        });
        const added = await first.client.callTool({
          name: "swarm",
          arguments: {
            action: "add_member",
            request: {
              description: "Owns one bounded task",
              name: "worker",
              prompt: "Join the Team and wait.",
            },
          },
          _meta: { threadId: "lead-thread" },
        });
        expect(added.isError).not.toBe(true);
        const childStatus = await first.client.callTool({
          name: "swarm",
          arguments: { action: "status", request: {} },
          _meta: { threadId: "member-thread-1" },
        });
        expect(textValue(childStatus)).toMatchObject({
          action: "status",
          data: { kind: "active", memberName: "worker", role: "legacy" },
        });
      } finally {
        await first.client.close();
        await first.host.dispose();
      }

      const second = await connectHost(environment);
      try {
        runtime.read.mockClear();
        const childStatus = await second.client.callTool({
          name: "swarm",
          arguments: { action: "status", request: {} },
          _meta: { threadId: "member-thread-1" },
        });
        expect(textValue(childStatus)).toMatchObject({
          action: "status",
          data: { kind: "active", memberName: "worker", role: "legacy" },
        });
        expect(new Set(runtime.read.mock.calls.map(([conversationId]) => conversationId))).toEqual(
          new Set(["codex:member-thread-1"]),
        );
        const archived = await second.client.callTool({
          name: "swarm",
          arguments: { action: "archive", request: {} },
          _meta: { threadId: "lead-thread" },
        });
        expect(archived.isError).not.toBe(true);
        expect(runtime.archive).toHaveBeenCalledWith("codex:member-thread-1", expect.anything());
      } finally {
        await second.client.close();
        await second.host.dispose();
      }
    } finally {
      await bridge.dispose();
    }
  });

  it("keeps child-created work pending until the separate lead carrier can deliver it", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-child-scheduling-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtime = fakeBridgeRuntime(workspace);
    const bridge = await startRuntimeBridge(workspace);
    bridge.attach(runtime.value);
    const environment = {
      SWARMX_BRIDGE_TOKEN: bridge.token,
      SWARMX_BRIDGE_URL: bridge.url,
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: workspace.id,
      SWARMX_WORKSPACE_LABEL: workspace.label,
      SWARMX_WORKSPACE_ROOT: workspace.root,
    };
    const lead = await connectHost(environment);
    let child: Awaited<ReturnType<typeof connectHost>> | undefined;
    try {
      await lead.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Split carrier team" } },
        _meta: { threadId: "lead-thread" },
      });
      await lead.client.callTool({
        name: "swarm",
        arguments: {
          action: "add_member",
          request: {
            description: "Creates one task from its own MCP",
            name: "worker",
            prompt: "Join the Team.",
          },
        },
        _meta: { threadId: "lead-thread" },
      });
      child = await connectHost(environment);
      const created = await child.client.callTool({
        name: "swarm",
        arguments: {
          action: "create_task",
          request: {
            assignedTo: "worker",
            blockedBy: [],
            description: "Must wait for lead delivery",
            kind: "read",
            subject: "Deferred cross-process task",
            writeScopes: [],
          },
        },
        _meta: { threadId: "member-thread-1" },
      });
      expect(created.isError).not.toBe(true);
      expect(textValue(created)).toMatchObject({
        data: { tasks: [expect.objectContaining({ status: "pending" })] },
      });
      const journal = new SwarmJournal(join(root, "home", "swarm"), { mode: "client" });
      expect(journal.list()[0]).toMatchObject({
        attempts: [],
        tasks: [expect.objectContaining({ status: "pending" })],
      });
      journal.close();
      await lead.client.callTool({
        name: "swarm",
        arguments: { action: "archive", request: {} },
        _meta: { threadId: "lead-thread" },
      });
    } finally {
      await child?.client.close();
      await child?.host.dispose();
      await lead.client.close();
      await lead.host.dispose();
      await bridge.dispose();
    }
  });

  it("recovers queued child messages only when their exact target carrier invokes Swarm", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-child-messages-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtime = fakeBridgeRuntime(workspace);
    const bridge = await startRuntimeBridge(workspace);
    bridge.attach(runtime.value);
    const environment = {
      SWARMX_BRIDGE_TOKEN: bridge.token,
      SWARMX_BRIDGE_URL: bridge.url,
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: workspace.id,
      SWARMX_WORKSPACE_LABEL: workspace.label,
      SWARMX_WORKSPACE_ROOT: workspace.root,
    };
    const lead = await connectHost(environment);
    let worker: Awaited<ReturnType<typeof connectHost>> | undefined;
    let peer: Awaited<ReturnType<typeof connectHost>> | undefined;
    let racingLead: Awaited<ReturnType<typeof connectHost>> | undefined;
    try {
      await lead.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Message carrier team" } },
        _meta: { threadId: "lead-thread" },
      });
      for (const [name, description] of [
        ["worker", "Sends queued messages"],
        ["peer", "Receives a queued wakeup"],
      ] as const) {
        await lead.client.callTool({
          name: "swarm",
          arguments: {
            action: "add_member",
            request: { description, name, prompt: `Join as ${name}.` },
          },
          _meta: { threadId: "lead-thread" },
        });
      }
      worker = await connectHost(environment);
      const toLead = await worker.client.callTool({
        name: "swarm",
        arguments: {
          action: "send_message",
          request: { content: "quiet lead note", delivery: "quiet", target: "lead" },
        },
        _meta: { threadId: "member-thread-1" },
      });
      expect(textValue(toLead)).toMatchObject({ data: { status: "queued" } });
      racingLead = await connectHost(environment);
      const nativeStart = vi.spyOn(runtime.value, "start");
      nativeStart.mockClear();
      await Promise.all(
        [lead, racingLead].map((connection) =>
          connection.client.callTool({
            name: "swarm",
            arguments: { action: "status", request: {} },
            _meta: { threadId: "lead-thread" },
          }),
        ),
      );
      expect(
        nativeStart.mock.calls.filter(
          ([request]) =>
            request.conversationId === "codex:lead-thread" &&
            request.text.includes("quiet lead note"),
        ),
      ).toHaveLength(1);

      const toPeer = await worker.client.callTool({
        name: "swarm",
        arguments: {
          action: "send_message",
          request: { content: "wake peer", delivery: "wakeup", target: "peer" },
        },
        _meta: { threadId: "member-thread-1" },
      });
      expect(textValue(toPeer)).toMatchObject({ data: { status: "queued" } });
      nativeStart.mockClear();
      await lead.client.callTool({
        name: "swarm",
        arguments: { action: "status", request: {} },
        _meta: { threadId: "lead-thread" },
      });
      expect(
        nativeStart.mock.calls.filter(
          ([request]) =>
            request.conversationId === "codex:member-thread-2" &&
            request.text.includes("wake peer"),
        ),
      ).toHaveLength(0);
      const queued = new SwarmJournal(join(root, "home", "swarm"), { mode: "client" });
      const queuedPeerMessage = queued
        .list()[0]
        ?.messages.find((message) => message.content === "wake peer");
      expect(queuedPeerMessage?.deliveredAt).toBeUndefined();
      expect(queuedPeerMessage?.deliveryStartedAt).toBeUndefined();
      queued.close();
      peer = await connectHost(environment);
      await peer.client.callTool({
        name: "swarm",
        arguments: { action: "status", request: {} },
        _meta: { threadId: "member-thread-2" },
      });

      const journal = new SwarmJournal(join(root, "home", "swarm"), { mode: "client" });
      expect(journal.list()[0]?.messages).toEqual([
        expect.objectContaining({ content: "quiet lead note", deliveredAt: expect.any(Number) }),
        expect.objectContaining({ content: "wake peer", deliveredAt: expect.any(Number) }),
      ]);
      journal.close();
      await lead.client.callTool({
        name: "swarm",
        arguments: { action: "archive", request: {} },
        _meta: { threadId: "lead-thread" },
      });
    } finally {
      await racingLead?.client.close();
      await racingLead?.host.dispose();
      await peer?.client.close();
      await peer?.host.dispose();
      await worker?.client.close();
      await worker?.host.dispose();
      await lead.client.close();
      await lead.host.dispose();
      await bridge.dispose();
    }
  });

  it("does not let per-Thread MCP startup fail an in-flight native member", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-member-startup-race-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtime = fakeBridgeRuntime(workspace);
    const bridge = await startRuntimeBridge(workspace);
    bridge.attach(runtime.value);
    const environment = {
      SWARMX_BRIDGE_TOKEN: bridge.token,
      SWARMX_BRIDGE_URL: bridge.url,
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: workspace.id,
      SWARMX_WORKSPACE_LABEL: workspace.label,
      SWARMX_WORKSPACE_ROOT: workspace.root,
    };
    const first = await connectHost(environment);
    let childHost: Awaited<ReturnType<typeof connectHost>> | undefined;
    let precommitStatus: unknown;
    try {
      await first.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Concurrent startup team" } },
        _meta: { threadId: "lead-thread" },
      });
      const createMember = runtime.value.createProvisionedMember?.bind(runtime.value);
      if (createMember === undefined) throw new Error("Test runtime cannot provision members.");
      vi.spyOn(runtime.value, "createProvisionedMember").mockImplementationOnce(
        async (request, provisioningId, signal) => {
          const conversation = await createMember(request, provisioningId, signal);
          childHost = await connectHost(environment);
          precommitStatus = textValue(
            await childHost.client.callTool({
              name: "swarm",
              arguments: { action: "status", request: {} },
              _meta: { threadId: "member-thread-1" },
            }),
          );
          expect(precommitStatus).toMatchObject({ data: { kind: "inactive" } });
          const journal = new SwarmJournal(join(root, "home", "swarm"), { mode: "client" });
          try {
            expect(journal.list()[0]?.members[1]?.phase).toBe("provisioning");
          } finally {
            journal.close();
          }
          expect(memberBindings(join(root, "home", "swarm"), root)).toEqual([]);
          return conversation;
        },
      );
      const added = await first.client.callTool({
        name: "swarm",
        arguments: {
          action: "add_member",
          request: {
            description: "Starts its own MCP process before the binding commit",
            name: "worker",
            prompt: "Join the Team and wait.",
          },
        },
        _meta: { threadId: "lead-thread" },
      });
      expect(added.isError).not.toBe(true);
      expect(precommitStatus).toBeDefined();
      if (childHost === undefined) throw new Error("Child MCP host was not started.");
      const childStatus = await childHost.client.callTool({
        name: "swarm",
        arguments: { action: "status", request: {} },
        _meta: { threadId: "member-thread-1" },
      });
      expect(textValue(childStatus)).toMatchObject({
        action: "status",
        data: { kind: "active", memberName: "worker", role: "legacy" },
      });
    } finally {
      await childHost?.client.close();
      await childHost?.host.dispose();
      await first.client.close();
      await first.host.dispose();
      await bridge.dispose();
    }
  });

  it("defers restored child observation until the runtime bridge is attached", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-deferred-recovery-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtime = fakeBridgeRuntime(workspace);
    const bridge = await startRuntimeBridge(workspace);
    bridge.attach(runtime.value);
    const environment = {
      SWARMX_BRIDGE_TOKEN: bridge.token,
      SWARMX_BRIDGE_URL: bridge.url,
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: workspace.id,
      SWARMX_WORKSPACE_LABEL: workspace.label,
      SWARMX_WORKSPACE_ROOT: workspace.root,
    };
    const first = await connectHost(environment);
    try {
      await first.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Deferred recovery team" } },
        _meta: { threadId: "lead-thread" },
      });
      const added = await first.client.callTool({
        name: "swarm",
        arguments: {
          action: "add_member",
          request: {
            description: "Persists across MCP startup",
            name: "worker",
            prompt: "Join and wait.",
          },
        },
        _meta: { threadId: "lead-thread" },
      });
      expect(added.isError).not.toBe(true);
    } finally {
      await first.client.close();
      await first.host.dispose();
      await bridge.dispose();
    }

    const unavailable = await createProductMcpHost({
      ...environment,
      SWARMX_BRIDGE_TOKEN: "not-attached",
      SWARMX_BRIDGE_URL: "http://127.0.0.1:1/",
    });
    await unavailable.dispose();
    expect(memberBindings(join(root, "home", "swarm"), root)).toEqual([
      expect.objectContaining({ conversationId: "codex:member-thread-1" }),
    ]);
  });

  it("rolls back a native child Thread when its initial turn is rejected", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-member-rollback-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtime = fakeBridgeRuntime(workspace);
    const bridge = await startRuntimeBridge(workspace);
    bridge.attach(runtime.value);
    const connection = await connectHost({
      SWARMX_BRIDGE_TOKEN: bridge.token,
      SWARMX_BRIDGE_URL: bridge.url,
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: workspace.id,
      SWARMX_WORKSPACE_LABEL: workspace.label,
      SWARMX_WORKSPACE_ROOT: workspace.root,
    });
    try {
      await connection.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Transactional team" } },
        _meta: { threadId: "lead-thread" },
      });
      vi.spyOn(runtime.value, "start").mockRejectedValueOnce(new Error("turn rejected"));
      const added = await connection.client.callTool({
        name: "swarm",
        arguments: {
          action: "add_member",
          request: {
            description: "Must not survive failed provisioning",
            name: "worker",
            prompt: "Join the Team.",
          },
        },
        _meta: { threadId: "lead-thread" },
      });
      expect(added.isError).toBe(true);
      expect(runtime.archive).toHaveBeenCalledWith("codex:member-thread-1", expect.anything());
      expect(memberBindings(join(root, "home", "swarm"), root)).toEqual([]);
      const childStatus = await connection.client.callTool({
        name: "swarm",
        arguments: { action: "status", request: {} },
        _meta: { threadId: "member-thread-1" },
      });
      expect(childStatus.isError).toBe(true);
      expect(childStatus.content).toEqual([
        expect.objectContaining({ type: "text", text: expect.stringContaining("archived") }),
      ]);
    } finally {
      await connection.client.close();
      await connection.host.dispose();
      await bridge.dispose();
    }
  });

  it("archives with an independent cleanup signal before releasing a cancelled creation claim", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-member-cancel-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtime = fakeBridgeRuntime(workspace);
    const bridge = await startRuntimeBridge(workspace);
    bridge.attach(runtime.value);
    const connection = await connectHost({
      SWARMX_BRIDGE_TOKEN: bridge.token,
      SWARMX_BRIDGE_URL: bridge.url,
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: workspace.id,
      SWARMX_WORKSPACE_LABEL: workspace.label,
      SWARMX_WORKSPACE_ROOT: workspace.root,
    });
    try {
      await connection.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Cancelled team" } },
        _meta: { threadId: "lead-thread" },
      });
      let enteredStart!: () => void;
      const startReached = new Promise<void>((resolve) => {
        enteredStart = resolve;
      });
      vi.spyOn(runtime.value, "start").mockImplementationOnce(
        (_request, signal) =>
          new Promise((resolve, reject) => {
            enteredStart();
            signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
            if (signal?.aborted) reject(signal.reason);
            void resolve;
          }),
      );
      const controller = new AbortController();
      const added = connection.client.callTool(
        {
          name: "swarm",
          arguments: {
            action: "add_member",
            request: {
              description: "Must be archived before its claim is released",
              name: "worker",
              prompt: "Join the Team.",
            },
          },
          _meta: { threadId: "lead-thread" },
        },
        undefined,
        { signal: controller.signal },
      );
      await startReached;
      controller.abort(new Error("request cancelled"));

      await expect(added).rejects.toThrow(/cancel/iu);
      await vi.waitFor(() => {
        expect(runtime.archive).toHaveBeenCalledWith("codex:member-thread-1", expect.anything());
      });
      const cleanupSignal = runtime.archive.mock.calls[0]?.[1] as AbortSignal | undefined;
      expect(cleanupSignal?.aborted).toBe(false);
      expect(memberBindings(join(root, "home", "swarm"), root)).toEqual([]);
    } finally {
      await connection.client.close();
      await connection.host.dispose();
      await bridge.dispose();
    }
  });

  it("waits for a committed create handle before cleaning up caller cancellation", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-member-create-cancel-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtime = fakeBridgeRuntime(workspace);
    const bridge = await startRuntimeBridge(workspace);
    bridge.attach(runtime.value);
    const connection = await connectHost({
      SWARMX_BRIDGE_TOKEN: bridge.token,
      SWARMX_BRIDGE_URL: bridge.url,
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: workspace.id,
      SWARMX_WORKSPACE_LABEL: workspace.label,
      SWARMX_WORKSPACE_ROOT: workspace.root,
    });
    try {
      await connection.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Create cancellation team" } },
        _meta: { threadId: "lead-thread" },
      });
      const createMember = runtime.value.createProvisionedMember?.bind(runtime.value);
      if (createMember === undefined) throw new Error("Test runtime cannot provision members.");
      let created!: () => void;
      const nativeCreated = new Promise<void>((resolve) => {
        created = resolve;
      });
      let release!: () => void;
      const createResponseReleased = new Promise<void>((resolve) => {
        release = resolve;
      });
      vi.spyOn(runtime.value, "createProvisionedMember").mockImplementationOnce(
        async (request, provisioningId, signal) => {
          const conversation = await createMember(request, provisioningId, signal);
          created();
          await createResponseReleased;
          return conversation;
        },
      );
      const controller = new AbortController();
      const added = connection.client.callTool(
        {
          name: "swarm",
          arguments: {
            action: "add_member",
            request: {
              description: "Must be cleaned after native create commits",
              name: "worker",
              prompt: "Join the Team.",
            },
          },
          _meta: { threadId: "lead-thread" },
        },
        undefined,
        { signal: controller.signal },
      );
      await nativeCreated;
      controller.abort(new Error("request cancelled after native create"));
      release();

      await expect(added).rejects.toThrow(/cancel/iu);
      await vi.waitFor(() => {
        expect(runtime.archive).toHaveBeenCalledWith("codex:member-thread-1", expect.anything());
      });
      await expect(runtime.value.read("codex:member-thread-1")).resolves.toMatchObject({
        archived: true,
      });
      expect(memberBindings(join(root, "home", "swarm"), root)).toEqual([]);
    } finally {
      await connection.client.close();
      await connection.host.dispose();
      await bridge.dispose();
    }
  });

  it("never reinterprets a failed bound member or cached actor as lead authority", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-revoked-member-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtime = fakeBridgeRuntime(workspace);
    const bridge = await startRuntimeBridge(workspace);
    bridge.attach(runtime.value);
    const connection = await connectHost({
      SWARMX_BRIDGE_TOKEN: bridge.token,
      SWARMX_BRIDGE_URL: bridge.url,
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: workspace.id,
      SWARMX_WORKSPACE_LABEL: workspace.label,
      SWARMX_WORKSPACE_ROOT: workspace.root,
    });
    try {
      await connection.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Revocation team" } },
        _meta: { threadId: "lead-thread" },
      });
      await connection.client.callTool({
        name: "swarm",
        arguments: {
          action: "add_member",
          request: {
            description: "Must lose authority exactly",
            name: "worker",
            prompt: "Join the Team.",
          },
        },
        _meta: { threadId: "lead-thread" },
      });
      await connection.client.callTool({
        name: "swarm",
        arguments: { action: "status", request: {} },
        _meta: { threadId: "member-thread-1" },
      });
      const journal = new SwarmJournal(join(root, "home", "swarm"), { mode: "client" });
      const team = journal.list()[0];
      const member = team?.members.find((candidate) => candidate.name === "worker");
      if (team === undefined || member === undefined) throw new Error("Expected bound worker.");
      journal.append(team.id, {
        type: "member/updated",
        data: { ...member, error: "native lifecycle failed", phase: "failed" },
      });
      journal.close();

      const status = await connection.client.callTool({
        name: "swarm",
        arguments: { action: "status", request: {} },
        _meta: { threadId: "member-thread-1" },
      });
      const create = await connection.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Escalated authority" } },
        _meta: { threadId: "member-thread-1" },
      });
      expect(status.isError).toBe(true);
      expect(create.isError).toBe(true);
      const inspected = new SwarmJournal(join(root, "home", "swarm"), { mode: "client" });
      expect(inspected.list()).toHaveLength(1);
      inspected.close();
    } finally {
      await connection.client.close();
      await connection.host.dispose();
      await bridge.dispose();
    }
  });

  it("fails an externally archived native child before scheduling more work", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-archived-member-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtime = fakeBridgeRuntime(workspace);
    const bridge = await startRuntimeBridge(workspace);
    bridge.attach(runtime.value);
    const connection = await connectHost({
      SWARMX_BRIDGE_TOKEN: bridge.token,
      SWARMX_BRIDGE_URL: bridge.url,
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: workspace.id,
      SWARMX_WORKSPACE_LABEL: workspace.label,
      SWARMX_WORKSPACE_ROOT: workspace.root,
    });
    try {
      await connection.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Archived child team" } },
        _meta: { threadId: "lead-thread" },
      });
      await connection.client.callTool({
        name: "swarm",
        arguments: {
          action: "add_member",
          request: {
            description: "Must fail if its native Thread is archived",
            name: "worker",
            prompt: "Join the Team.",
          },
        },
        _meta: { threadId: "lead-thread" },
      });
      await runtime.value.archive("codex:member-thread-1");
      const nativeStart = vi.spyOn(runtime.value, "start");
      nativeStart.mockClear();

      const status = await connection.client.callTool({
        name: "swarm",
        arguments: { action: "status", request: {} },
        _meta: { threadId: "lead-thread" },
      });
      expect(status.isError).not.toBe(true);
      const inspected = new SwarmJournal(join(root, "home", "swarm"), { mode: "client" });
      const team = inspected.list()[0];
      expect(team?.members).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            name: "worker",
            phase: "failed",
            error: "Continuable member exited unexpectedly",
          }),
        ]),
      );
      inspected.close();
      expect(memberBindings(join(root, "home", "swarm"), root)).toEqual([
        expect.objectContaining({ conversationId: "codex:member-thread-1" }),
      ]);

      const message = await connection.client.callTool({
        name: "swarm",
        arguments: {
          action: "send_message",
          request: { content: "Must not queue", delivery: "quiet", target: "worker" },
        },
        _meta: { threadId: "lead-thread" },
      });
      expect(message.isError).toBe(true);

      const task = await connection.client.callTool({
        name: "swarm",
        arguments: {
          action: "create_task",
          request: {
            assignedTo: "worker",
            blockedBy: [],
            description: "Must not start on an archived child",
            kind: "read",
            subject: "Archived child task",
            writeScopes: [],
          },
        },
        _meta: { threadId: "lead-thread" },
      });
      expect(task.isError).toBe(true);
      expect(nativeStart).not.toHaveBeenCalled();
      await connection.client.callTool({
        name: "swarm",
        arguments: { action: "archive", request: {} },
        _meta: { threadId: "lead-thread" },
      });
    } finally {
      await connection.client.close();
      await connection.host.dispose();
      await bridge.dispose();
    }
  });

  it("keeps Codex PKB item locators stable and bounds native Thread scans", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-conversation-search-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtime = fakeConversationArchiveRuntime(workspace);
    const bridge = await startRuntimeBridge(workspace);
    bridge.attach(runtime.value);
    const connection = await connectHost({
      SWARMX_BRIDGE_TOKEN: bridge.token,
      SWARMX_BRIDGE_URL: bridge.url,
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: workspace.id,
      SWARMX_WORKSPACE_LABEL: workspace.label,
      SWARMX_WORKSPACE_ROOT: workspace.root,
    });
    try {
      runtime.set([
        conversationFixture(workspace, "evidence", [
          { id: "codex:item-stable", text: "stable evidence", createdAt: 10 },
        ]),
      ]);
      const searched = textValue(
        await connection.client.callTool({
          name: "pkb",
          arguments: { action: "search_conversations", request: { query: "stable evidence" } },
        }),
      );
      const locator = (
        searched.data as { items: Array<{ locator: { seq: number; sessionId: string } }> }
      ).items[0]?.locator;
      expect(locator).toBeDefined();

      runtime.set([
        conversationFixture(workspace, "evidence", [
          { id: "codex:item-new", text: "replacement prefix", createdAt: 20 },
          { id: "codex:item-stable", text: "stable evidence", createdAt: 10 },
        ]),
      ]);
      const read = textValue(
        await connection.client.callTool({
          name: "pkb",
          arguments: { action: "read_conversation", request: locator },
        }),
      );
      expect(read).toMatchObject({ data: { text: "stable evidence" } });

      const longNativeId = "n".repeat(512);
      runtime.set([
        conversationFixture(workspace, longNativeId, [
          { id: "codex:item-long", text: "long locator evidence", createdAt: 30 },
        ]),
      ]);
      const longSearch = textValue(
        await connection.client.callTool({
          name: "pkb",
          arguments: { action: "search_conversations", request: { query: "long locator" } },
        }),
      );
      const longLocator = (
        longSearch.data as { items: Array<{ locator: { seq: number; sessionId: string } }> }
      ).items[0]?.locator;
      expect(longLocator?.sessionId).toHaveLength(518);
      const longRead = await connection.client.callTool({
        name: "pkb",
        arguments: { action: "read_conversation", request: longLocator },
      });
      expect(textValue(longRead)).toMatchObject({ data: { text: "long locator evidence" } });

      runtime.read.mockClear();
      runtime.set(
        Array.from({ length: 40 }, (_, index) =>
          conversationFixture(workspace, `thread-${String(index)}`, [
            {
              id: `codex:item-${String(index)}`,
              text: "bounded scan",
              createdAt: 100 - index,
            },
          ]),
        ),
      );
      const bounded = textValue(
        await connection.client.callTool({
          name: "pkb",
          arguments: { action: "search_conversations", request: { query: "bounded scan" } },
        }),
      );
      expect(runtime.read).toHaveBeenCalledTimes(32);
      expect(bounded).toMatchObject({
        data: { diagnostics: [expect.stringContaining("32")] },
      });
    } finally {
      await connection.client.close();
      await connection.host.dispose();
      await bridge.dispose();
    }
  });

  it("rejects durable Swarm creation without a native Thread identity", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-call-authority-"));
    roots.push(root);
    const connection = await connectHost({
      SWARMX_BRIDGE_TOKEN: "bridge-token",
      SWARMX_BRIDGE_URL: "http://127.0.0.1:1/",
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: "workspace-1",
      SWARMX_WORKSPACE_LABEL: "workspace",
      SWARMX_WORKSPACE_ROOT: root,
    });
    try {
      const created = await connection.client.callTool({
        name: "swarm",
        arguments: { action: "create", request: { name: "Call-owned team" } },
      });
      expect(created.isError).toBe(true);
      const status = await connection.client.callTool({
        name: "swarm",
        arguments: { action: "status", request: {} },
      });
      expect(textValue(status)).toMatchObject({
        action: "status",
        data: { kind: "inactive", revision: 0 },
      });
    } finally {
      await connection.client.close();
      await connection.host.dispose();
    }
  });

  it("rejects a native Thread that belongs to another workspace before product dispatch", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-mcp-foreign-thread-"));
    roots.push(root);
    const workspaceA = new WorkspaceAuthority().mint(root);
    const foreignRoot = join(root, "foreign");
    mkdirSync(foreignRoot);
    const workspaceB = new WorkspaceAuthority().mint(foreignRoot);
    const bridge = await startRuntimeBridge(workspaceA);
    bridge.attach(fakeBridgeRuntime(workspaceB).value);
    const connection = await connectHost({
      SWARMX_BRIDGE_TOKEN: bridge.token,
      SWARMX_BRIDGE_URL: bridge.url,
      SWARMX_HOME: join(root, "home"),
      SWARMX_WORKSPACE_ID: workspaceA.id,
      SWARMX_WORKSPACE_LABEL: workspaceA.label,
      SWARMX_WORKSPACE_ROOT: workspaceA.root,
    });
    try {
      for (const request of [
        {
          name: "science_notebook",
          arguments: {
            action: "create_project",
            request: { requestId: randomUUID(), title: "Must not be created" },
          },
        },
        {
          name: "pkb",
          arguments: { action: "search_knowledge", request: { query: "foreign" } },
        },
        {
          name: "swarm",
          arguments: { action: "create", request: { name: "Must not be created" } },
        },
      ]) {
        const result = await connection.client.callTool({
          ...request,
          _meta: { threadId: "foreign-thread" },
        });
        expect(result.isError).toBe(true);
        expect(result.content).toEqual([
          expect.objectContaining({
            type: "text",
            text: expect.stringContaining("does not belong"),
          }),
        ]);
      }
    } finally {
      await connection.client.close();
      await connection.host.dispose();
      await bridge.dispose();
    }
  });
});

async function connectHost(environment: Readonly<Record<string, string>>, sessionId?: string) {
  let ownedBridge: Awaited<ReturnType<typeof startRuntimeBridge>> | undefined;
  let effectiveEnvironment = environment;
  if (environment.SWARMX_BRIDGE_URL === "http://127.0.0.1:1/") {
    const workspace: WorkspaceScope = {
      id: environment.SWARMX_WORKSPACE_ID,
      label: environment.SWARMX_WORKSPACE_LABEL,
      root: realpathSync(environment.SWARMX_WORKSPACE_ROOT),
      token: "product-mcp-test-workspace",
    };
    ownedBridge = await startRuntimeBridge(workspace);
    ownedBridge.attach(fakeBridgeRuntime(workspace).value);
    effectiveEnvironment = {
      ...environment,
      SWARMX_BRIDGE_TOKEN: ownedBridge.token,
      SWARMX_BRIDGE_URL: ownedBridge.url,
    };
  }
  initializeSwarmStorage(environment.SWARMX_HOME);
  const productHost = await createProductMcpHost(effectiveEnvironment);
  const host =
    ownedBridge === undefined
      ? productHost
      : {
          server: productHost.server,
          dispose: () => disposeInOrder([() => productHost.dispose(), () => ownedBridge.dispose()]),
        };
  const client = new Client({ name: "swarmx-test", version: "0.1.0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  serverTransport.sessionId = sessionId;
  await Promise.all([client.connect(clientTransport), host.server.connect(serverTransport)]);
  return { client, host };
}

function textValue(result: { content: readonly unknown[] }): Record<string, unknown> {
  const text = result.content.find(
    (item): item is { type: "text"; text: string } =>
      typeof item === "object" && item !== null && "type" in item && item.type === "text",
  );
  if (text === undefined) throw new Error("MCP tool did not return text content.");
  try {
    return JSON.parse(text.text) as Record<string, unknown>;
  } catch (error) {
    throw new Error(`MCP tool returned non-JSON text: ${text.text}`, { cause: error });
  }
}

function fakeBridgeRuntime(workspace: WorkspaceScope): {
  archive: ReturnType<typeof vi.fn>;
  read: ReturnType<typeof vi.fn>;
  value: ConversationRuntime;
} {
  let nextConversation = 0;
  const summaries = new Map<string, ConversationSummary>();
  const archive = vi.fn(async (conversationId: string) => {
    const summary = summaries.get(conversationId);
    if (summary !== undefined) summaries.set(conversationId, { ...summary, archived: true });
  });
  const read = vi.fn(async (conversationId: string): Promise<ConversationSnapshot> => {
    const summary = summaries.get(conversationId);
    return {
      ...(summary ?? {
        runtime: "codex" as const,
        conversationId,
        workspace: { id: workspace.id, label: workspace.label },
        title: "Codex Thread",
        archived: false,
        updatedAt: Date.now(),
      }),
      turns: [],
    };
  });
  const create = async () => {
    nextConversation += 1;
    const summary: ConversationSummary = {
      runtime: "codex",
      conversationId: `codex:member-thread-${String(nextConversation)}`,
      workspace: { id: workspace.id, label: workspace.label },
      title: "Swarm member",
      archived: false,
      updatedAt: Date.now(),
    };
    summaries.set(summary.conversationId, summary);
    return summary;
  };
  const value = {
    kind: "codex" as const,
    list: async () => [...summaries.values()],
    create,
    createProvisionedMember: async () => create(),
    read,
    start: async () => ({ turnId: `codex:member-turn-${String(nextConversation)}` }),
    steer: async () => undefined,
    interrupt: async () => undefined,
    revise: async () => {
      throw new Error("not used");
    },
    fork: async () => {
      throw new Error("not used");
    },
    archive,
    subscribe: () => () => undefined,
    respondToApproval: async () => undefined,
    dispose: async () => undefined,
  } satisfies ConversationRuntime;
  return { archive, read, value };
}

function fakeConversationArchiveRuntime(_workspace: WorkspaceScope): {
  read: ReturnType<typeof vi.fn>;
  set(fixtures: Array<{ snapshot: ConversationSnapshot; summary: ConversationSummary }>): void;
  value: ConversationRuntime;
} {
  let fixtures: Array<{ snapshot: ConversationSnapshot; summary: ConversationSummary }> = [];
  const read = vi.fn(async (conversationId: string) => {
    const fixture = fixtures.find(
      (candidate) => candidate.summary.conversationId === conversationId,
    );
    if (fixture === undefined) throw new Error("Thread not found");
    return fixture.snapshot;
  });
  const value = {
    kind: "codex" as const,
    list: async () => fixtures.map((fixture) => fixture.summary),
    create: async () => {
      throw new Error("not used");
    },
    read,
    start: async () => {
      throw new Error("not used");
    },
    steer: async () => undefined,
    interrupt: async () => undefined,
    revise: async () => {
      throw new Error("not used");
    },
    fork: async () => {
      throw new Error("not used");
    },
    archive: async () => undefined,
    subscribe: () => () => undefined,
    respondToApproval: async () => undefined,
    dispose: async () => undefined,
  } satisfies ConversationRuntime;
  return {
    read,
    set(value_) {
      fixtures = value_;
    },
    value,
  };
}

function conversationFixture(
  workspace: WorkspaceScope,
  nativeId: string,
  items: Array<{ id: string; text: string; createdAt: number }>,
): { snapshot: ConversationSnapshot; summary: ConversationSummary } {
  const conversationId = `codex:${nativeId}`;
  const summary: ConversationSummary = {
    runtime: "codex",
    conversationId,
    workspace: { id: workspace.id, label: workspace.label },
    title: nativeId,
    archived: false,
    updatedAt: Math.max(0, ...items.map((item) => item.createdAt)),
  };
  return {
    summary,
    snapshot: {
      ...summary,
      turns: [
        {
          id: `codex:turn-${nativeId}`,
          status: "completed",
          items: items.map((item) => ({
            ...item,
            type: "user_message" as const,
            turnId: `codex:turn-${nativeId}`,
          })),
        },
      ],
    },
  };
}
