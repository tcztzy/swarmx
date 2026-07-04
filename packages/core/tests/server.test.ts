import { once } from "node:events";
import http from "node:http";
import { afterEach, describe, expect, it } from "vitest";
import { createServer } from "../src/server.js";
import { Swarm } from "../src/swarm.js";

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
});

async function startServer(options: Parameters<typeof createServer>[1] = {}): Promise<http.Server> {
  const server = createServer(createTestSwarm(), {
    port: 0,
    host: "127.0.0.1",
    ...options,
  });
  servers.push(server);
  if (!server.listening) await once(server, "listening");
  return server;
}

function createTestSwarm(): Swarm {
  return new Swarm({
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
}

function request(
  server: http.Server,
  method: string,
  path: string,
  headers: Record<string, string> = {},
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
    req.end();
  });
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
