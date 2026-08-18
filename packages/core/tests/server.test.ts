import { once } from "node:events";
import http from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AuditInput } from "../src/audit.js";
import { createServer, type ServerAuditWriter, type ServerSwarmExecution } from "../src/server.js";
import { Swarm } from "../src/swarm.js";
import type { MessageChunk } from "../src/types.js";

const servers: http.Server[] = [];

afterEach(async () => {
  await Promise.all(servers.splice(0).map(closeServer));
});

describe("server boundary", () => {
  it("serves local loopback clients without wildcard browser CORS", async () => {
    const server = await startServer();
    const response = await request(server, "GET", "/models");

    expect(response.statusCode).toBe(200);
    expect(response.headers["access-control-allow-origin"]).toBeUndefined();
    expect(JSON.parse(response.body)).toMatchObject({
      object: "list",
      data: [{ id: "agent", object: "model" }],
    });
  });

  it("rejects browser origins unless explicitly allowed", async () => {
    const blocked = await startServer();
    const blockedResponse = await request(blocked, "GET", "/models", {
      Origin: "https://app.example",
    });

    expect(blockedResponse.statusCode).toBe(403);
    expect(blockedResponse.body).toContain("Origin not allowed");

    const allowed = await startServer({
      allowedOrigins: ["https://app.example"],
    });
    const allowedResponse = await request(allowed, "GET", "/models", {
      Origin: "https://app.example",
    });

    expect(allowedResponse.statusCode).toBe(200);
    expect(allowedResponse.headers["access-control-allow-origin"]).toBe("https://app.example");
  });

  it("allows Origin null only when trusted desktop bridge mode is explicit", async () => {
    const blocked = await startServer();
    const blockedResponse = await request(blocked, "OPTIONS", "/models", {
      Origin: "null",
    });

    expect(blockedResponse.statusCode).toBe(403);

    const allowed = await startServer({ allowNullOrigin: true });
    const allowedResponse = await request(allowed, "OPTIONS", "/models", {
      Origin: "null",
    });

    expect(allowedResponse.statusCode).toBe(204);
    expect(allowedResponse.headers["access-control-allow-origin"]).toBe("null");
  });

  it("requires bearer tokens for explicit token-protected servers", async () => {
    const server = await startServer({ apiToken: "server-token" });

    const missing = await request(server, "GET", "/models");
    expect(missing.statusCode).toBe(401);

    const wrong = await request(server, "GET", "/models", {
      Authorization: "Bearer wrong",
    });
    expect(wrong.statusCode).toBe(401);

    const ok = await request(server, "GET", "/models", {
      Authorization: "Bearer server-token",
    });
    expect(ok.statusCode).toBe(200);
  });

  it("refuses non-loopback bindings without a token and rejects wildcard origins", () => {
    expect(() => createServer(createTestSwarm(), { port: 0, host: "0.0.0.0" })).toThrow(
      /Non-loopback/,
    );
    expect(() =>
      createServer(createTestSwarm(), {
        port: 0,
        host: "127.0.0.1",
        allowedOrigins: ["*"],
      }),
    ).toThrow(/wildcard origins/);
  });

  it("records correlated request attempts and successful outcomes", async () => {
    const audit = new RecordingAuditWriter();
    const server = await startServer({ audit });
    const response = await request(server, "GET", "/models?ignored=secret-query");

    expect(response.statusCode).toBe(200);
    expect(response.headers["x-request-id"]).toMatch(/^[0-9a-f-]{36}$/);
    expect(audit.events.map((event) => event.outcome)).toEqual(["attempted", "completed"]);
    expect(audit.events[0]?.requestId).toBe(response.headers["x-request-id"]);
    expect(audit.events[1]?.requestId).toBe(response.headers["x-request-id"]);
    expect(audit.events[1]?.metadata).toMatchObject({
      method: "GET",
      path: "/models",
      statusCode: 200,
    });
    expect(audit.events[1]?.metadata?.durationMs).toEqual(expect.any(Number));
  });

  it("records origin and authentication denials without exposing boundary inputs", async () => {
    const originAudit = new RecordingAuditWriter();
    const originServer = await startServer({ audit: originAudit });
    const originResponse = await request(originServer, "GET", "/models", {
      Origin: "https://blocked.example/secret-origin",
    });

    expect(originResponse.statusCode).toBe(403);
    expect(originAudit.events.map((event) => event.outcome)).toEqual(["attempted", "denied"]);
    expect(originAudit.events[1]?.metadata).toMatchObject({ statusCode: 403 });

    const authAudit = new RecordingAuditWriter();
    const authServer = await startServer({ apiToken: "expected-secret", audit: authAudit });
    const authResponse = await request(authServer, "GET", "/models", {
      Authorization: "Bearer supplied-secret",
    });

    expect(authResponse.statusCode).toBe(401);
    expect(authAudit.events.map((event) => event.outcome)).toEqual(["attempted", "denied"]);
    expect(authAudit.events[1]?.metadata).toMatchObject({ statusCode: 401 });
    expect(JSON.stringify([...originAudit.events, ...authAudit.events])).not.toContain("secret");
  });

  it("fails closed before request execution when the audit writer is unavailable", async () => {
    const swarm = createTestSwarm();
    const execute = vi.spyOn(swarm, "execute");
    const server = await startServerWithSwarm(swarm, {
      audit: {
        append() {
          throw new Error("audit unavailable");
        },
      },
    });
    const response = await request(
      server,
      "POST",
      "/chat/completions",
      { "Content-Type": "application/json" },
      JSON.stringify({ messages: [{ role: "user", content: "must not execute" }] }),
    );

    expect(response.statusCode).toBe(503);
    expect(response.headers["x-request-id"]).toMatch(/^[0-9a-f-]{36}$/);
    expect(execute).not.toHaveBeenCalled();
  });

  it("never includes tokens, query parameters, prompts, or responses in audit inputs", async () => {
    const audit = new RecordingAuditWriter();
    const server = await startServer({ audit });
    const response = await request(
      server,
      "POST",
      "/chat/completions?api_key=query-secret",
      {
        Authorization: "Bearer header-secret",
        "Content-Type": "application/json",
        "X-Request-ID": "attacker-controlled-secret",
      },
      JSON.stringify({
        messages: [{ role: "user", content: "raw-prompt-secret" }],
      }),
    );

    expect(response.statusCode).toBe(200);
    const serialized = JSON.stringify(audit.events);
    expect(serialized).not.toContain("query-secret");
    expect(serialized).not.toContain("header-secret");
    expect(serialized).not.toContain("attacker-controlled-secret");
    expect(serialized).not.toContain("raw-prompt-secret");
    expect(serialized).not.toContain("raw-prompt-secret\n");
    expect(response.headers["x-request-id"]).not.toBe("attacker-controlled-secret");
  });

  it("records failed requests without copying parser details", async () => {
    const audit = new RecordingAuditWriter();
    const server = await startServer({ audit });
    const response = await request(
      server,
      "POST",
      "/chat/completions",
      { "Content-Type": "application/json" },
      "invalid-json-secret",
    );

    expect(response.statusCode).toBe(500);
    expect(audit.events.map((event) => event.outcome)).toEqual(["attempted", "failed"]);
    expect(audit.events[1]?.metadata).toMatchObject({ statusCode: 500 });
    expect(JSON.stringify(audit.events)).not.toContain("invalid-json-secret");
  });

  it("forwards execution chunks to SSE clients before execution completes", async () => {
    let releaseSecondChunk!: () => void;
    const secondChunkStarted = new Promise<void>((resolve) => {
      releaseSecondChunk = resolve;
    });
    const first = { role: "assistant", content: "first", kind: "message" } as const;
    const second = { role: "assistant", content: "second", kind: "message" } as const;
    const execute = vi.fn(
      async (
        _arguments_: Record<string, unknown>,
        _context: Record<string, unknown> | undefined,
        onChunk?: (chunk: MessageChunk) => void,
      ): Promise<MessageChunk[]> => {
        onChunk?.(first);
        await secondChunkStarted;
        onChunk?.(second);
        return [first, second];
      },
    );
    const server = await startServerWithSwarm({
      models: [],
      execute,
      listAllSessions: async () => [],
    });
    const address = server.address() as AddressInfo;
    let body = "";
    let sawFirstChunk!: () => void;
    const firstChunkSeen = new Promise<void>((resolve) => {
      sawFirstChunk = resolve;
    });
    const response = new Promise<{ statusCode: number; body: string }>((resolve, reject) => {
      const req = http.request(
        {
          host: "127.0.0.1",
          port: address.port,
          path: "/chat/completions",
          method: "POST",
          headers: { "Content-Type": "application/json" },
        },
        (res) => {
          res.setEncoding("utf8");
          res.on("data", (chunk: string) => {
            body += chunk;
            if (body.includes('"content":"first"')) sawFirstChunk();
          });
          res.on("end", () => resolve({ statusCode: res.statusCode ?? 0, body }));
        },
      );
      req.on("error", reject);
      req.end(JSON.stringify({ stream: true, messages: [{ role: "user", content: "go" }] }));
    });

    let firstChunkTimer: ReturnType<typeof setTimeout> | undefined;
    await expect(
      Promise.race([
        firstChunkSeen,
        new Promise<never>((_, reject) => {
          firstChunkTimer = setTimeout(
            () => reject(new Error("first chunk was not streamed")),
            1_000,
          );
        }),
      ]),
    ).resolves.toBeUndefined();
    if (firstChunkTimer) clearTimeout(firstChunkTimer);
    releaseSecondChunk();
    await expect(response).resolves.toMatchObject({ statusCode: 200 });

    const completed = await response;
    expect(completed.body.indexOf('"content":"first"')).toBeGreaterThanOrEqual(0);
    expect(completed.body.indexOf('"content":"first"')).toBeLessThan(
      completed.body.indexOf('"content":"second"'),
    );
    expect(completed.body).toContain('"finish_reason":"stop"');
    expect(execute).toHaveBeenCalledWith(
      { messages: [{ role: "user", content: "go" }] },
      undefined,
      expect.any(Function),
    );
  });
});

async function startServer(options: Parameters<typeof createServer>[1] = {}): Promise<http.Server> {
  return startServerWithSwarm(createTestSwarm(), options);
}

async function startServerWithSwarm(
  swarm: ServerSwarmExecution,
  options: Parameters<typeof createServer>[1] = {},
): Promise<http.Server> {
  const server = createServer(swarm, {
    port: 0,
    host: "127.0.0.1",
    ...options,
  });
  servers.push(server);
  if (!server.listening) await once(server, "listening");
  return server;
}

function createTestSwarm(): Swarm & ServerSwarmExecution {
  const swarm = new Swarm({
    name: "server_test",
    root: "agent",
    nodes: {
      agent: {
        kind: "agent",
        agent: {
          name: "agent",
          backend: { type: "echo" },
        },
      },
    },
    edges: [],
  });
  return Object.assign(swarm, { models: [{ id: "agent", object: "model" as const }] });
}

function request(
  server: http.Server,
  method: string,
  path: string,
  headers: Record<string, string> = {},
  body?: string,
): Promise<{
  statusCode: number;
  headers: http.IncomingHttpHeaders;
  body: string;
}> {
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Server has no TCP address.");

  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        method,
        hostname: "127.0.0.1",
        port: address.port,
        path,
        headers,
      },
      (res) => {
        let body = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => {
          body += chunk;
        });
        res.on("end", () => {
          resolve({
            statusCode: res.statusCode ?? 0,
            headers: res.headers,
            body,
          });
        });
      },
    );
    req.on("error", reject);
    req.end(body);
  });
}

class RecordingAuditWriter implements ServerAuditWriter {
  readonly events: AuditInput[] = [];

  append(input: AuditInput): void {
    this.events.push(input);
  }
}

function closeServer(server: http.Server): Promise<void> {
  return new Promise((resolve, reject) => {
    if (!server.listening) {
      resolve();
      return;
    }
    server.close((error) => {
      if (error) reject(error);
      else resolve();
    });
  });
}
