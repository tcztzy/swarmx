import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  RequestCancelledError,
  SWARMX_LOCAL_FILES_LSP_ID,
  SWARMX_SKILLS_LSP_ID,
  builtInExtensionBundle,
  cancelAcpRequest,
  createExtensionInventory,
  parseExtensionBundle,
  withAcpRequest,
} from "@swarmx/core";
import { afterEach, describe, expect, it } from "vitest";
import { LspHost } from "./lsp-host.js";

const tempRoots: string[] = [];

afterEach(async () => {
  await Promise.all(tempRoots.map((root) => rm(root, { recursive: true, force: true })));
  tempRoots.length = 0;
});

describe("LspHost", () => {
  it("returns built-in skill completions for $ references", async () => {
    const root = await tempRoot();
    const inventory = createExtensionInventory([
      builtInExtensionBundle(),
      parseExtensionBundle({
        schemaVersion: 1,
        id: "geepilot",
        name: "GEEPilot",
        version: "1.0.0",
        capabilities: {
          skills: [
            {
              id: "geepilot.biosecurity",
              name: "Biosecurity",
              path: "skills/biosecurity/SKILL.md",
              canonicalPath: "skills/biosecurity/SKILL.md",
              governanceRef: "docs/skills-governance.md",
              readOnly: true,
              sourcePluginId: "geepilot",
            },
            {
              id: "geepilot.memory",
              name: "Memory",
              path: "skills/memory/SKILL.md",
            },
          ],
        },
      }),
    ]);
    const host = new LspHost();
    expect(host.supportsClaudeOperations(inventory)).toBe(false);

    const response = await host.complete(inventory, {
      serverId: SWARMX_SKILLS_LSP_ID,
      workspaceRoot: root,
      text: "Use $bio",
      position: { line: 0, character: 8 },
      triggerCharacter: "$",
    });

    expect(completionItems(response.result)).toEqual([
      expect.objectContaining({
        label: "$geepilot.biosecurity",
        kind: 18,
        detail: "Skill Biosecurity",
        data: expect.objectContaining({
          kind: "skill",
          skillId: "geepilot.biosecurity",
          path: "skills/biosecurity/SKILL.md",
          canonicalPath: "skills/biosecurity/SKILL.md",
          governanceRef: "docs/skills-governance.md",
          readOnly: true,
          sourcePluginId: "geepilot",
        }),
      }),
    ]);

    const bareResponse = await host.complete(inventory, {
      serverId: SWARMX_SKILLS_LSP_ID,
      workspaceRoot: root,
      text: "Use $",
      position: { line: 0, character: 5 },
      triggerCharacter: "$",
    });
    expect(completionItems(bareResponse.result)).toHaveLength(2);
  });

  it("returns no skill completions outside $ tokens", async () => {
    const root = await tempRoot();
    const inventory = createExtensionInventory([
      builtInExtensionBundle(),
      parseExtensionBundle({
        schemaVersion: 1,
        id: "test",
        name: "Test",
        version: "1.0.0",
        capabilities: {
          skills: [{ id: "test.memory", name: "Memory" }],
        },
      }),
    ]);
    const host = new LspHost();

    await expect(
      host.complete(inventory, {
        serverId: SWARMX_SKILLS_LSP_ID,
        workspaceRoot: root,
        text: "Use memory",
        position: { line: 0, character: 10 },
      }),
    ).resolves.toMatchObject({
      result: { items: [] },
    });
  });

  it("treats a non-empty bare @ token as a local file reference", async () => {
    const root = await tempRoot();
    await mkdir(path.join(root, "src"));
    await writeFile(path.join(root, "README.md"), "# test\n", "utf8");
    await writeFile(path.join(root, "src", "App.tsx"), "export const App = () => null;\n", "utf8");
    const inventory = createExtensionInventory([builtInExtensionBundle()]);
    const host = new LspHost();

    const directoryResponse = await host.complete(inventory, {
      serverId: SWARMX_LOCAL_FILES_LSP_ID,
      workspaceRoot: root,
      text: "Open @s",
      position: { line: 0, character: 7 },
      triggerCharacter: "@",
    });

    expect(completionItems(directoryResponse.result)).toEqual([
      expect.objectContaining({
        label: "@src/",
        kind: 19,
        detail: "Workspace folder",
      }),
    ]);

    const fileResponse = await host.complete(inventory, {
      serverId: SWARMX_LOCAL_FILES_LSP_ID,
      workspaceRoot: root,
      text: "Open @src/",
      position: { line: 0, character: 10 },
      triggerCharacter: "/",
    });

    expect(completionItems(fileResponse.result)).toEqual([
      expect.objectContaining({
        label: "@src/App.tsx",
        kind: 17,
        detail: "Workspace file",
      }),
    ]);
    await expect(
      host.stop({ serverId: SWARMX_LOCAL_FILES_LSP_ID, workspaceRoot: root }),
    ).resolves.toEqual({
      serverId: SWARMX_LOCAL_FILES_LSP_ID,
      stopped: false,
    });
  });

  it("rejects scheme-qualified local file references", async () => {
    const root = await tempRoot();
    await mkdir(path.join(root, "src"));
    const inventory = createExtensionInventory([builtInExtensionBundle()]);
    const host = new LspHost();

    await expect(
      host.complete(inventory, {
        serverId: SWARMX_LOCAL_FILES_LSP_ID,
        workspaceRoot: root,
        text: "Open @file:src/",
        position: { line: 0, character: 15 },
      }),
    ).resolves.toMatchObject({
      result: { items: [] },
    });
  });

  it("does not list workspace files for a bare @ token", async () => {
    const root = await tempRoot();
    await writeFile(path.join(root, "README.md"), "# test\n", "utf8");
    const inventory = createExtensionInventory([builtInExtensionBundle()]);
    const host = new LspHost();

    await expect(
      host.complete(inventory, {
        serverId: SWARMX_LOCAL_FILES_LSP_ID,
        workspaceRoot: root,
        text: "Open @",
        position: { line: 0, character: 6 },
        triggerCharacter: "@",
      }),
    ).resolves.toMatchObject({
      result: { items: [] },
    });
  });

  it("keeps built-in local file completions bounded to @ workspace paths", async () => {
    const root = await tempRoot();
    const inventory = createExtensionInventory([builtInExtensionBundle()]);
    const host = new LspHost();

    await expect(
      host.complete(inventory, {
        serverId: SWARMX_LOCAL_FILES_LSP_ID,
        workspaceRoot: root,
        text: "Open @../",
        position: { line: 0, character: 9 },
      }),
    ).resolves.toMatchObject({
      result: { items: [] },
    });

    await expect(
      host.complete(inventory, {
        serverId: SWARMX_LOCAL_FILES_LSP_ID,
        workspaceRoot: root,
        text: "Open README",
        position: { line: 0, character: 11 },
      }),
    ).resolves.toMatchObject({
      result: { items: [] },
    });
  });

  it("starts a declared stdio LSP server and returns completions", async () => {
    const root = await tempRoot();
    const serverScript = await writeFakeLspServer(root);
    const inventory = createExtensionInventory([
      parseExtensionBundle({
        schemaVersion: 1,
        id: "test",
        name: "Test",
        version: "1.0.0",
        capabilities: {
          lspServers: [
            {
              id: "test-lsp",
              command: [process.execPath, serverScript],
              languages: ["plaintext"],
            },
          ],
        },
      }),
    ]);
    const host = new LspHost();

    try {
      const response = await host.complete(inventory, {
        serverId: "test-lsp",
        workspaceRoot: root,
        text: "Use $",
        position: { line: 0, character: 5 },
        triggerCharacter: "$",
      });

      expect(response.serverId).toBe("test-lsp");
      expect(completionItems(response.result)).toEqual([
        expect.objectContaining({ label: "$memory", detail: "plaintext:Use $" }),
      ]);
      await expect(host.stop({ serverId: "test-lsp", workspaceRoot: root })).resolves.toEqual({
        serverId: "test-lsp",
        stopped: true,
      });
    } finally {
      await host.stop({ serverId: "test-lsp", workspaceRoot: root });
    }
  });

  it("supports command string plus args and languageIds metadata", async () => {
    const root = await tempRoot();
    const serverScript = await writeFakeLspServer(root);
    const inventory = createExtensionInventory([
      parseExtensionBundle({
        schemaVersion: 1,
        id: "test",
        name: "Test",
        version: "1.0.0",
        capabilities: {
          lspServers: [
            {
              id: "geepilot-reference-lsp",
              command: process.execPath,
              args: [serverScript],
              languageIds: ["markdown"],
            },
          ],
        },
      }),
    ]);
    const host = new LspHost();

    try {
      const response = await host.complete(inventory, {
        serverId: "geepilot-reference-lsp",
        workspaceRoot: root,
        documentUri: "prompt.md",
        text: "Ask @",
        position: { line: 0, character: 5 },
        triggerCharacter: "@",
      });

      expect(completionItems(response.result)).toEqual([
        expect.objectContaining({ label: "$memory", detail: "markdown:Ask @" }),
      ]);
    } finally {
      await host.stop({ serverId: "geepilot-reference-lsp", workspaceRoot: root });
    }
  });

  it("rejects missing LSP command metadata", async () => {
    const root = await tempRoot();
    const inventory = createExtensionInventory([
      parseExtensionBundle({
        schemaVersion: 1,
        id: "test",
        name: "Test",
        version: "1.0.0",
        capabilities: {
          lspServers: [{ id: "missing-command" }],
        },
      }),
    ]);
    const host = new LspHost();

    await expect(
      host.complete(inventory, {
        serverId: "missing-command",
        workspaceRoot: root,
        text: "",
        position: { line: 0, character: 0 },
      }),
    ).rejects.toThrow(/does not declare a command/);
  });

  it("V402-V404 performs Claude LSP operations with one-based positions and exact output", async () => {
    const root = await tempRoot();
    const sourcePath = path.join(root, "source.txt");
    await writeFile(sourcePath, "function target() {}\ntarget();\n", "utf8");
    const serverScript = await writeFakeLspServer(root);
    const inventory = lspInventory(serverScript, "plaintext-lsp", ["plaintext"]);
    const host = new LspHost();

    try {
      expect(host.supportsClaudeOperations(inventory)).toBe(true);
      await expect(
        host.operate(inventory, root, {
          operation: "goToDefinition",
          filePath: "source.txt",
          line: 2,
          character: 3,
        }),
      ).resolves.toEqual({
        operation: "goToDefinition",
        result: "source.txt:2:3",
        filePath: "source.txt",
        resultCount: 1,
        fileCount: 1,
      });

      await expect(
        host.operate(inventory, root, {
          operation: "documentSymbol",
          filePath: "source.txt",
          line: 1,
          character: 1,
        }),
      ).resolves.toMatchObject({
        operation: "documentSymbol",
        result: expect.stringContaining("target - source.txt:1:1"),
        resultCount: 2,
        fileCount: 1,
      });
      await expect(
        host.operate(inventory, root, {
          operation: "workspaceSymbol",
          filePath: "source.txt",
          line: 1,
          character: 1,
          query: "target",
        }),
      ).resolves.toMatchObject({
        result: expect.stringContaining("target - source.txt:1:1"),
        resultCount: 1,
        fileCount: 1,
      });
    } finally {
      await host.stop({ serverId: "plaintext-lsp", workspaceRoot: root });
    }
  });

  it("V404 executes two-step incoming and outgoing call hierarchy requests", async () => {
    const root = await tempRoot();
    await writeFile(path.join(root, "source.txt"), "target();\n", "utf8");
    const serverScript = await writeFakeLspServer(root);
    const inventory = lspInventory(serverScript, "call-lsp", ["plaintext"]);
    const host = new LspHost();

    try {
      await expect(
        host.operate(inventory, root, {
          operation: "incomingCalls",
          filePath: "source.txt",
          line: 1,
          character: 1,
        }),
      ).resolves.toMatchObject({
        result: expect.stringContaining("caller - source.txt:2:1"),
        resultCount: 1,
        fileCount: 1,
      });
      await expect(
        host.operate(inventory, root, {
          operation: "outgoingCalls",
          filePath: "source.txt",
          line: 1,
          character: 1,
        }),
      ).resolves.toMatchObject({
        result: expect.stringContaining("callee - source.txt:3:1"),
        resultCount: 1,
        fileCount: 1,
      });
    } finally {
      await host.stop({ serverId: "call-lsp", workspaceRoot: root });
    }
  });

  it("V403 rejects Project escapes and missing or ambiguous file-language routing", async () => {
    const root = await tempRoot();
    const outside = await tempRoot();
    await writeFile(path.join(root, "source.txt"), "inside\n", "utf8");
    await writeFile(path.join(outside, "outside.txt"), "outside\n", "utf8");
    await symlink(path.join(outside, "outside.txt"), path.join(root, "escape.txt"));
    const serverScript = await writeFakeLspServer(root);
    const one = lspInventory(serverScript, "one", ["plaintext"]);
    const ambiguous = createExtensionInventory([
      ...lspBundles(serverScript, [
        ["one", ["plaintext"]],
        ["two", ["plaintext"]],
      ]),
    ]);
    const host = new LspHost();

    await expect(
      host.operate(one, root, {
        operation: "hover",
        filePath: "escape.txt",
        line: 1,
        character: 1,
      }),
    ).rejects.toThrow(/escapes the Project/);
    await expect(
      host.operate(ambiguous, root, {
        operation: "hover",
        filePath: "source.txt",
        line: 1,
        character: 1,
      }),
    ).rejects.toThrow(/Multiple configured LSP servers.*one, two/);
    await expect(
      host.operate(lspInventory(serverScript, "python", ["python"]), root, {
        operation: "hover",
        filePath: "source.txt",
        line: 1,
        character: 1,
      }),
    ).rejects.toThrow(/No configured LSP server supports/);
  });

  it("V404 cancels an in-flight LSP request and keeps the session reusable", async () => {
    const root = await tempRoot();
    await writeFile(path.join(root, "source.txt"), "target();\n", "utf8");
    const serverScript = await writeFakeLspServer(root);
    const inventory = lspInventory(serverScript, "cancel-lsp", ["plaintext"]);
    const host = new LspHost();
    const requestId = "claude-lsp-cancel";

    try {
      const run = withAcpRequest(requestId, () =>
        host.operate(inventory, root, {
          operation: "workspaceSymbol",
          filePath: "source.txt",
          line: 1,
          character: 1,
          query: "hang",
        }),
      );
      await new Promise((resolve) => setTimeout(resolve, 20));
      await expect(cancelAcpRequest(requestId)).resolves.toBe(true);
      await expect(run).rejects.toBeInstanceOf(RequestCancelledError);

      await expect(
        host.operate(inventory, root, {
          operation: "workspaceSymbol",
          filePath: "source.txt",
          line: 1,
          character: 1,
          query: "target",
        }),
      ).resolves.toMatchObject({ resultCount: 1 });
    } finally {
      await host.stop({ serverId: "cancel-lsp", workspaceRoot: root });
    }
  });
});

function lspInventory(serverScript: string, id: string, languages: string[]) {
  return createExtensionInventory([...lspBundles(serverScript, [[id, languages]])]);
}

function lspBundles(serverScript: string, entries: Array<[string, string[]]>) {
  return entries.map(([id, languages]) =>
    parseExtensionBundle({
      schemaVersion: 1,
      id: `test-${id}`,
      name: id,
      version: "1.0.0",
      capabilities: {
        lspServers: [{ id, command: [process.execPath, serverScript], languages }],
      },
    }),
  );
}

async function tempRoot(): Promise<string> {
  const root = await mkdtemp(path.join(tmpdir(), "swarmx-lsp-"));
  tempRoots.push(root);
  return root;
}

async function writeFakeLspServer(root: string): Promise<string> {
  const serverPath = path.join(root, "fake-lsp.mjs");
  await writeFile(serverPath, FAKE_LSP_SERVER, "utf8");
  return serverPath;
}

function completionItems(result: unknown): Array<Record<string, unknown>> {
  if (!result || typeof result !== "object") return [];
  const items = (result as { items?: unknown }).items;
  return Array.isArray(items) ? (items as Array<Record<string, unknown>>) : [];
}

const FAKE_LSP_SERVER = `
const documents = new Map();
let buffer = Buffer.alloc(0);

process.stdin.on("data", (chunk) => {
  buffer = Buffer.concat([buffer, chunk]);
  readMessages();
});

function readMessages() {
  while (true) {
    const headerEnd = buffer.indexOf("\\r\\n\\r\\n");
    if (headerEnd < 0) return;
    const header = buffer.subarray(0, headerEnd).toString("ascii");
    const match = /Content-Length:\\s*(\\d+)/i.exec(header);
    if (!match) process.exit(2);
    const length = Number.parseInt(match[1], 10);
    const bodyStart = headerEnd + 4;
    const bodyEnd = bodyStart + length;
    if (buffer.length < bodyEnd) return;
    const body = buffer.subarray(bodyStart, bodyEnd).toString("utf8");
    buffer = buffer.subarray(bodyEnd);
    handleMessage(JSON.parse(body));
  }
}

function send(message) {
  const body = JSON.stringify(message);
  process.stdout.write("Content-Length: " + Buffer.byteLength(body, "utf8") + "\\r\\n\\r\\n" + body);
}

function handleMessage(message) {
  if (message.method === "initialize") {
    send({
      jsonrpc: "2.0",
      id: message.id,
      result: {
        capabilities: {
          textDocumentSync: 1,
          completionProvider: { triggerCharacters: ["$", "@"] }
        }
      }
    });
    return;
  }
  if (message.method === "initialized") return;
  if (message.method === "textDocument/didOpen") {
    const document = message.params.textDocument;
    documents.set(document.uri, {
      languageId: document.languageId,
      text: document.text
    });
    return;
  }
  if (message.method === "textDocument/didChange") {
    const uri = message.params.textDocument.uri;
    const existing = documents.get(uri) ?? { languageId: "plaintext", text: "" };
    documents.set(uri, {
      languageId: existing.languageId,
      text: message.params.contentChanges[0]?.text ?? ""
    });
    return;
  }
  if (message.method === "textDocument/completion") {
    const uri = message.params.textDocument.uri;
    const document = documents.get(uri) ?? { languageId: "plaintext", text: "" };
    send({
      jsonrpc: "2.0",
      id: message.id,
      result: {
        isIncomplete: false,
        items: [
          {
            label: "$memory",
            kind: 6,
            detail: document.languageId + ":" + document.text
          }
        ]
      }
    });
    return;
  }
  const uri = message.params?.textDocument?.uri;
  const range = (line, character) => ({
    start: { line, character },
    end: { line, character: character + 1 }
  });
  if (message.method === "textDocument/definition") {
    send({ jsonrpc: "2.0", id: message.id, result: [{ uri, range: range(1, 2) }] });
    return;
  }
  if (message.method === "textDocument/references") {
    send({
      jsonrpc: "2.0",
      id: message.id,
      result: [{ uri, range: range(0, 0) }, { uri, range: range(1, 0) }]
    });
    return;
  }
  if (message.method === "textDocument/hover") {
    send({
      jsonrpc: "2.0",
      id: message.id,
      result: { contents: { kind: "markdown", value: "target(): void" } }
    });
    return;
  }
  if (message.method === "textDocument/documentSymbol") {
    send({
      jsonrpc: "2.0",
      id: message.id,
      result: [{
        name: "target",
        kind: 12,
        range: range(0, 0),
        selectionRange: range(0, 0),
        uri,
        children: [{ name: "local", kind: 13, range: range(0, 9), selectionRange: range(0, 9) }]
      }]
    });
    return;
  }
  if (message.method === "workspace/symbol") {
    if (message.params.query === "hang") return;
    const firstUri = documents.keys().next().value;
    send({
      jsonrpc: "2.0",
      id: message.id,
      result: [{ name: message.params.query, kind: 12, location: { uri: firstUri, range: range(0, 0) } }]
    });
    return;
  }
  if (message.method === "textDocument/implementation") {
    send({ jsonrpc: "2.0", id: message.id, result: [{ uri, range: range(0, 0) }] });
    return;
  }
  if (message.method === "textDocument/prepareCallHierarchy") {
    send({
      jsonrpc: "2.0",
      id: message.id,
      result: [{ name: "target", kind: 12, uri, range: range(0, 0), selectionRange: range(0, 0) }]
    });
    return;
  }
  if (message.method === "callHierarchy/incomingCalls") {
    const item = message.params.item;
    send({
      jsonrpc: "2.0",
      id: message.id,
      result: [{ from: { ...item, name: "caller", range: range(1, 0), selectionRange: range(1, 0) }, fromRanges: [range(0, 0)] }]
    });
    return;
  }
  if (message.method === "callHierarchy/outgoingCalls") {
    const item = message.params.item;
    send({
      jsonrpc: "2.0",
      id: message.id,
      result: [{ to: { ...item, name: "callee", range: range(2, 0), selectionRange: range(2, 0) }, fromRanges: [range(0, 0)] }]
    });
    return;
  }
  if (message.method === "shutdown") {
    send({ jsonrpc: "2.0", id: message.id, result: null });
    return;
  }
  if (message.method === "exit") {
    process.exit(0);
  }
  if (Object.prototype.hasOwnProperty.call(message, "id")) {
    send({
      jsonrpc: "2.0",
      id: message.id,
      error: { code: -32601, message: "method not found" }
    });
  }
}
`;
