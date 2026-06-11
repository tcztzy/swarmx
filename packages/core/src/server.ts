import http from "node:http";
import type { Swarm } from "./swarm.js";

export interface ServerOptions {
  port?: number;
  host?: string;
}

export function createServer(swarm: Swarm, opts: ServerOptions = {}): http.Server {
  const port = opts.port ?? 3000;
  const host = opts.host ?? "127.0.0.1";

  const server = http.createServer(async (req, res) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

    if (req.method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }

    const url = new URL(req.url ?? "/", `http://${host}:${port}`);

    try {
      if (req.method === "GET" && url.pathname === "/models") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ object: "list", data: listModels(swarm) }));
        return;
      }

      if (req.method === "GET" && url.pathname === "/sessions") {
        res.writeHead(200, { "Content-Type": "application/json" });
        const sessions = await swarm.listAllSessions();
        res.end(JSON.stringify(sessions));
        return;
      }

      if (req.method === "POST" && url.pathname === "/chat/completions") {
        const body = await readBody(req);

        if (body.stream) {
          await handleStream(req, res, swarm, body);
        } else {
          res.writeHead(200, { "Content-Type": "application/json" });
          const result = await handleChat(swarm, body);
          res.end(JSON.stringify(result));
        }
        return;
      }

      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Not found" }));
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      if (!res.headersSent) {
        res.writeHead(500, { "Content-Type": "application/json" });
      }
      res.end(JSON.stringify({ error: message }));
    }
  });

  server.listen(port, host);
  return server;
}

// ── SSE streaming ───────────────────────────────────────────────────────────

async function handleStream(
  _req: http.IncomingMessage,
  res: http.ServerResponse,
  swarm: Swarm,
  body: ChatCompletionRequest,
): Promise<void> {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  });

  const id = `chatcmpl-${Date.now()}`;
  const model = body.model ?? "swarmx";
  const created = Math.floor(Date.now() / 1000);

  try {
    const rootNode = swarm.nodes.get(swarm.root);

    if (rootNode?.kind === "agent" && rootNode.agent) {
      await streamViaAgent(id, created, model, res, rootNode.agent, body);
    } else {
      await streamViaSwarm(id, created, model, res, swarm, body);
    }

    res.write("data: [DONE]\n\n");
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    if (!res.closed) {
      res.write(`data: ${JSON.stringify({ error: errorMsg })}\n\n`);
      res.write("data: [DONE]\n\n");
    }
  }

  res.end();
}

async function streamViaAgent(
  id: string,
  created: number,
  model: string,
  res: http.ServerResponse,
  agent: import("./agent.js").Agent,
  body: ChatCompletionRequest,
): Promise<void> {
  return new Promise((resolve, reject) => {
    agent
      .callStream({ messages: body.messages }, (chunk) => {
        if (res.closed) return;

        if (chunk.kind === "message") {
          const c: ChatCompletionChunk = {
            id,
            object: "chat.completion.chunk",
            created,
            model,
            choices: [
              {
                index: 0,
                delta: {
                  role: chunk.role === "user" ? "user" : "assistant",
                  content: chunk.content,
                },
                finish_reason: null,
              },
            ],
          };
          res.write(`data: ${JSON.stringify(c)}\n\n`);
        }
      })
      .then(() => {
        const final: ChatCompletionChunk = {
          id,
          object: "chat.completion.chunk",
          created,
          model,
          choices: [
            {
              index: 0,
              delta: {},
              finish_reason: "stop",
            },
          ],
        };
        res.write(`data: ${JSON.stringify(final)}\n\n`);
        resolve();
      })
      .catch(reject);
  });
}

async function streamViaSwarm(
  id: string,
  created: number,
  model: string,
  res: http.ServerResponse,
  swarm: Swarm,
  body: ChatCompletionRequest,
): Promise<void> {
  const messages = await swarm.execute({ messages: body.messages });

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    const chunk: ChatCompletionChunk = {
      id,
      object: "chat.completion.chunk",
      created,
      model,
      choices: [
        {
          index: 0,
          delta:
            msg.kind === "message"
              ? { role: msg.role === "user" ? "user" : "assistant", content: msg.content }
              : { role: "assistant" },
          finish_reason: i === messages.length - 1 ? "stop" : null,
        },
      ],
    };
    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
  }
}

// ── Non-streaming ───────────────────────────────────────────────────────────

async function handleChat(
  swarm: Swarm,
  body: ChatCompletionRequest,
): Promise<ChatCompletionResponse> {
  const messages = await swarm.execute({ messages: body.messages });
  const content = messages.map((m) => m.content).join("\n");

  return {
    id: `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: body.model ?? "swarmx",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
      },
    ],
  };
}

// ── Models ──────────────────────────────────────────────────────────────────

function listModels(swarm: Swarm): Array<{ id: string; object: string }> {
  const models: Array<{ id: string; object: string }> = [];
  for (const [name, node] of swarm.nodes) {
    if (node.kind === "agent") {
      models.push({ id: name, object: "model" });
    }
  }
  if (swarm.queen) {
    models.push({ id: swarm.queen.name, object: "model" });
  }
  return models;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function readBody(req: http.IncomingMessage): Promise<ChatCompletionRequest> {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => {
      data += chunk;
    });
    req.on("end", () => {
      try {
        resolve(JSON.parse(data));
      } catch (e) {
        reject(e);
      }
    });
    req.on("error", reject);
  });
}

// ── Types ───────────────────────────────────────────────────────────────────

interface ChatCompletionRequest {
  model?: string;
  messages: Array<{ role: string; content: string }>;
  stream?: boolean;
}

interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: { role: string; content: string };
    finish_reason: string;
  }>;
}

interface ChatCompletionChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: { role?: string; content?: string };
    finish_reason: string | null;
  }>;
}
