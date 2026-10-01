import { readdirSync, readFileSync } from "node:fs";
import { connect } from "node:net";
import { homedir } from "node:os";
import { join } from "node:path";
import { defineToolPlugin } from "openclaw/plugin-sdk/tool-plugin";

const CALL_TIMEOUT_MS = 120_000;
const UNBOUND = "SwarmX bridge is not bound to an active Host execution";

class BridgeError extends Error {
  constructor(message, unbound) {
    super(message);
    this.unbound = unbound;
  }
}

function bridgePaths() {
  const explicit = process.env.SWARMX_OPENCLAW_BRIDGE;
  if (explicit) return [explicit];
  const home = process.env.SWARMX_HOME ?? join(homedir(), ".swarmx");
  const directory = join(home, "openclaw", "bridges");
  try {
    return readdirSync(directory)
      .filter((name) => name.endsWith(".json"))
      .map((name) => join(directory, name));
  } catch {
    return [];
  }
}

function readBridges() {
  const bridges = [];
  for (const path of bridgePaths()) {
    try {
      const parsed = JSON.parse(readFileSync(path, "utf8"));
      if (typeof parsed?.socket === "string" && typeof parsed?.token === "string")
        bridges.push({ ...parsed, path });
    } catch {
      // A Host that is starting or shutting down leaves no usable descriptor.
    }
  }
  return bridges;
}

function availableTools() {
  const names = new Set();
  for (const bridge of readBridges())
    for (const tool of bridge.tools ?? []) if (typeof tool?.name === "string") names.add(tool.name);
  return [...names].sort();
}

function callBridge(bridge, payload) {
  return new Promise((resolve, reject) => {
    const socket = connect(bridge.socket);
    let buffer = "";
    let settled = false;
    const fail = (error) => {
      if (settled) return;
      settled = true;
      socket.destroy();
      reject(error);
    };
    socket.setTimeout(CALL_TIMEOUT_MS, () => fail(new Error("SwarmX Host bridge timed out.")));
    socket.once("error", () => fail(new BridgeError(UNBOUND, true)));
    socket.once("connect", () =>
      socket.write(JSON.stringify({ id: 1, token: bridge.token, ...payload }) + "\n"),
    );
    socket.on("data", (chunk) => {
      buffer += chunk.toString("utf8");
      const index = buffer.indexOf("\n");
      if (index < 0 || settled) return;
      settled = true;
      socket.end();
      let reply;
      try {
        reply = JSON.parse(buffer.slice(0, index));
      } catch (error) {
        reject(error);
        return;
      }
      if (reply.ok) resolve(reply.value);
      else reject(new BridgeError(String(reply.error ?? "SwarmX tool call failed."), false));
    });
  });
}

const parameters = {
  type: "object",
  properties: {
    tool: { type: "string", description: "SwarmX Host tool name, for example memory." },
    args: {
      type: "object",
      description: "Arguments for the selected SwarmX Host tool.",
      additionalProperties: true,
    },
  },
  required: ["tool"],
  additionalProperties: false,
};

export default defineToolPlugin({
  id: "swarmx",
  name: "SwarmX Host tools",
  description: "Call SwarmX Host product tools such as shared Memory from OpenClaw sessions.",
  tools: (tool) => [
    tool({
      name: "swarmx",
      description: "Call a SwarmX Host product tool for the current session.",
      parameters,
      factory: ({ toolContext }) => {
        const names = availableTools();
        const sessionKey = toolContext?.sessionKey;
        return [
          {
            name: "swarmx",
            label: "SwarmX Host tools",
            description: [
              "Call a SwarmX Host product tool bound to this session, for example memory.",
              names.length ? `Available tools: ${names.join(", ")}.` : "",
              "Arguments follow the Host tool schema.",
            ]
              .filter(Boolean)
              .join(" "),
            parameters,
            execute: async (toolCallId, params) => {
              const bridges = readBridges();
              if (!bridges.length) throw new Error("SwarmX Host bridge descriptor is unavailable.");
              if (typeof sessionKey !== "string" || !sessionKey)
                throw new Error("SwarmX tool calls require a gateway session.");
              let lastError;
              for (const bridge of bridges) {
                try {
                  toolContext.assertInvocationCurrent?.();
                  const value = await callBridge(bridge, {
                    tool: params?.tool,
                    args: params?.args ?? {},
                    sessionKey,
                    toolCallId,
                  });
                  return {
                    content: [
                      {
                        type: "text",
                        text: typeof value === "string" ? value : JSON.stringify(value),
                      },
                    ],
                  };
                } catch (error) {
                  lastError = error;
                  if (!error?.unbound) throw error;
                }
              }
              throw lastError ?? new Error(UNBOUND);
            },
          },
        ];
      },
    }),
  ],
});
