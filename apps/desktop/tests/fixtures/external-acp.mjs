import { randomUUID } from "node:crypto";
import { watch } from "node:fs";
import { access, appendFile, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { Readable, Writable } from "node:stream";
import * as acp from "@agentclientprotocol/sdk";

const root = process.env.ACP_FIXTURE_HOME;
const legacyModes = process.env.ACP_FIXTURE_LEGACY_MODES === "1";
const holdExit = process.env.ACP_FIXTURE_HOLD_EXIT === "1";
const holdPermissionTerminal = process.env.ACP_FIXTURE_HOLD_PERMISSION_TERMINAL === "1";
const missingCapability = process.env.ACP_FIXTURE_MISSING_CAPABILITY;
const opaquePermissions = process.env.ACP_FIXTURE_OPAQUE_IDS === "1";
const allowId = opaquePermissions ? "opaque-allow-42" : "yes";
const rejectId = opaquePermissions ? "opaque-deny-99" : "no";
const permissionOptions = [
  { optionId: allowId, name: "Allow once", kind: "allow_once" },
  { optionId: rejectId, name: "Reject once", kind: "reject_once" },
];
const log = (event, data = {}) =>
  appendFile(
    join(root, "events.jsonl"),
    `${JSON.stringify({ event, pid: process.pid, at: Date.now(), ...data })}\n`,
  );
const exists = (path) =>
  access(path).then(
    () => true,
    () => false,
  );
async function gate(name) {
  if (await exists(join(root, name))) return;
  await new Promise((resolve) => {
    const watcher = watch(root, async () => {
      if (await exists(join(root, name))) {
        watcher.close();
        resolve();
      }
    });
    void exists(join(root, name)).then((yes) => {
      if (yes) {
        watcher.close();
        resolve();
      }
    });
  });
}
const sessions = JSON.parse(await readFile(join(root, "sessions.json"), "utf8").catch(() => "{}"));
const save = () => writeFile(join(root, "sessions.json"), JSON.stringify(sessions));
const modes = (currentModeId = "safe") => ({
  currentModeId,
  availableModes: [
    { id: "safe", name: "Safe" },
    { id: "plan", name: "Plan" },
  ],
});
const settings = (model = "local", mode = "safe") => ({
  ...(legacyModes ? { modes: modes(mode) } : {}),
  configOptions: [
    {
      id: "backend",
      name: "Model",
      category: "model",
      type: "select",
      currentValue: model,
      options: [
        { value: "local", name: "Local" },
        { value: "other", name: "Other" },
      ],
    },
    {
      id: "thinking",
      name: "Thinking",
      category: "thought_level",
      type: "select",
      currentValue: "off",
      options: [{ value: "off", name: "Off" }],
    },
  ],
});
const active = new Map();
const app = acp
  .agent()
  .onRequest("initialize", async ({ params }) => {
    await log("initialize", { params });
    const agentCapabilities = { loadSession: true, sessionCapabilities: { list: {}, resume: {} } };
    if (missingCapability === "load") delete agentCapabilities.loadSession;
    else if (missingCapability) delete agentCapabilities.sessionCapabilities[missingCapability];
    return {
      protocolVersion: acp.PROTOCOL_VERSION,
      agentCapabilities,
    };
  })
  .onRequest("session/new", async ({ params }) => {
    await log("new", { params });
    const sessionId = randomUUID();
    sessions[sessionId] = { cwd: params.cwd, updates: [] };
    await save();
    return { sessionId, ...settings() };
  })
  .onRequest("session/list", async ({ params }) => {
    await log("list");
    return {
      sessions: Object.entries(sessions)
        .filter(([, s]) => s.cwd === params.cwd)
        .map(([sessionId, s]) => ({ sessionId, cwd: s.cwd, title: "Native fixture" })),
    };
  })
  .onRequest("session/resume", async ({ params }) => {
    await log("resume", { sessionId: params.sessionId });
    if (!sessions[params.sessionId] || (await exists(join(root, "fail-resume"))))
      throw new Error("Native resume rejected");
    if (await exists(join(root, "pause-resume"))) {
      await log("resume-paused");
      await gate("release-resume");
    }
    return settings(sessions[params.sessionId].model, sessions[params.sessionId].mode);
  })
  .onRequest("session/load", async ({ params, client }) => {
    await log("load", { sessionId: params.sessionId });
    if (!sessions[params.sessionId]) throw new Error("Native history missing");
    for (const update of sessions[params.sessionId].updates)
      await client.notify("session/update", { sessionId: params.sessionId, update });
    return settings(sessions[params.sessionId].model, sessions[params.sessionId].mode);
  })
  .onRequest("session/set_config_option", async ({ params }) => {
    await log("config", { params });
    if (
      !settings().configOptions.some(
        (o) => o.id === params.configId && o.options.some((v) => v.value === params.value),
      )
    )
      throw new Error("Invalid selection");
    if (await exists(join(root, "pause-config"))) {
      await log("config-paused");
      await gate("release-config");
    }
    if (params.configId === "backend") sessions[params.sessionId].model = params.value;
    await save();
    return { configOptions: settings(sessions[params.sessionId].model).configOptions };
  })
  .onRequest("session/set_mode", async ({ params }) => {
    await log("mode", { params });
    if (!legacyModes || !modes().availableModes.some((mode) => mode.id === params.modeId))
      throw new Error("Invalid native mode");
    sessions[params.sessionId].mode = params.modeId;
    await save();
    return {};
  })
  .onRequest("session/prompt", async ({ params, client }) => {
    const { sessionId } = params;
    const run = { cancelled: false };
    active.set(sessionId, run);
    const text = params.prompt.map((x) => (x.type === "text" ? x.text : "")).join("");
    await log("prompt", { sessionId, text });
    if (text === "exit") process.exit(7);
    if (text === "disconnect-hold") {
      process.stdout.end();
      return await new Promise(() => {});
    }
    try {
      if (text === "cancel") {
        await gate("cancel-received");
        await log("terminal-paused");
        await gate("release-terminal");
        await log("terminal", { stopReason: "cancelled" });
        return { stopReason: "cancelled" };
      }
      if (text.startsWith("permission")) {
        const observed = ["permission-observed", "permission-override", "permission-null"].includes(
          text,
        );
        if (observed) {
          // DSH drains tool updates before requesting permission with only a toolCallId.
          for (const update of [
            {
              sessionUpdate: "tool_call",
              toolCallId: "tool-1",
              title: "Run local fixture",
              kind: "execute",
              status: "pending",
              rawInput: { draft: true },
            },
            {
              sessionUpdate: "tool_call_update",
              toolCallId: "tool-1",
              rawInput: { command: "printf '%s\\n' 'ACP exact approval'", cwd: root },
            },
            { sessionUpdate: "tool_call_update", toolCallId: "tool-1", status: "pending" },
          ]) {
            sessions[sessionId].updates.push(update);
            await client.notify("session/update", { sessionId, update });
          }
          await save();
        }
        const response = await client.request("session/request_permission", {
          sessionId,
          toolCall: {
            toolCallId: "tool-1",
            ...(text === "permission"
              ? {
                  title: "Run local fixture",
                  rawInput: { command: "printf '%s\\n' 'ACP exact approval'", cwd: root },
                }
              : text === "permission-override"
                ? {
                    title: "Run replacement fixture",
                    rawInput: { command: "printf replacement", cwd: root },
                  }
                : text === "permission-null"
                  ? { rawInput: null }
                  : {}),
          },
          options: permissionOptions,
        });
        await log("permission-result", { response });
        if (
          run.cancelled ||
          response.outcome.outcome !== "selected" ||
          response.outcome.optionId !== allowId
        ) {
          if (holdPermissionTerminal) {
            await log("terminal-paused");
            await gate("release-terminal");
          }
          return { stopReason: run.cancelled ? "cancelled" : "refusal" };
        }
      }
      const updates = [
        {
          sessionUpdate: "user_message_chunk",
          messageId: randomUUID(),
          content: { type: "text", text },
        },
        {
          sessionUpdate: "tool_call",
          toolCallId: "tool-1",
          title: "Local tool",
          name: "local",
          kind: "execute",
          status: "in_progress",
          rawInput: { text },
        },
        {
          sessionUpdate: "tool_call_update",
          toolCallId: "tool-1",
          status: "completed",
          rawOutput: { ok: true },
        },
        {
          sessionUpdate: "agent_message_chunk",
          messageId: randomUUID(),
          content: { type: "text", text: `native:${text}` },
        },
      ];
      await log("tool");
      for (const update of updates) {
        sessions[sessionId].updates.push(update);
        await client.notify("session/update", { sessionId, update });
      }
      await save();
      await log("terminal", { stopReason: "end_turn" });
      return { stopReason: "end_turn" };
    } finally {
      active.delete(sessionId);
    }
  })
  .onNotification("session/cancel", async ({ params }) => {
    const run = active.get(params.sessionId);
    if (run) run.cancelled = true;
    await log("cancel");
    await writeFile(join(root, "cancel-received"), "");
  });
if (holdExit) {
  process.on("SIGTERM", () => {
    void log("dispose-waiting");
  });
}
const connection = app.connect(
  acp.ndJsonStream(Writable.toWeb(process.stdout), Readable.toWeb(process.stdin)),
);
await log("started", {
  startupMs: Math.round(process.uptime() * 1000),
  hostToken: process.env.SWARMX_MCP_TOKEN === undefined ? null : "[present]",
  inheritedHostApiToken: process.env.SWARMX_API_TOKEN !== undefined,
  inheritedHostMcpToken: process.env.SWARMX_MCP_TOKEN !== undefined,
});
await connection.closed;
if (holdExit) {
  await log("exit-paused");
  await gate("release-exit");
}
await log("closed");
process.exit(0);
