import { Context } from "@deepseek-ai/cordis";
import { createUserMessage } from "@deepseek-ai/dsh-llm";
import type {} from "@deepseek-ai/dsh-llm-retry";
import SessionStore, { type Session, type SessionId } from "@deepseek-ai/dsh-session";
import { afterEach, describe, expect, it } from "vitest";

const contexts: Context[] = [];

afterEach(async () => {
  await Promise.all(contexts.splice(0).map((context) => context.fiber.dispose()));
});

function textOf(session: Session): string[] {
  return session
    .deriveMessages()
    .flatMap((message) =>
      message.content.flatMap((block) => (block.type === "text" ? [block.text] : [])),
    );
}

async function failedSource(): Promise<{ context: Context; source: Session; prefixEnd: number }> {
  const context = new Context();
  contexts.push(context);
  await context.plugin(SessionStore);
  const source = context.sessions.create("append-only-source" as SessionId, {
    meta: { cwd: "/workspace" },
  });
  source.append("turn/start", { turn: 1 });
  source.append(
    "user/message",
    createUserMessage({
      content: [{ type: "text", text: "kept prefix" }],
      source: { kind: "user" },
    }),
    { surfaceOp: "append" },
  );
  source.append("turn/end", { turn: 1, reason: { kind: "completed" } });
  const prefixEnd = source.events.at(-1)?.seq;
  if (prefixEnd === undefined) throw new Error("completed prefix is missing");

  source.append("turn/start", { turn: 2 });
  source.append(
    "user/message",
    createUserMessage({
      content: [{ type: "text", text: "superseded prompt" }],
      source: { kind: "user" },
    }),
    { surfaceOp: "append" },
  );
  source.append("step/start", { turn: 2, step: 1 });
  source.append("llm/retry", {
    retryId: "append-only-retry" as never,
    turn: 2,
    step: 1,
    provider: "fixture",
    mode: "normal",
    policyKey: "fixture-normal",
    retry: 1,
    maxRetries: 1,
    delayMs: 1,
    failure: { code: "TRANSPORT", message: "Connection error" },
  });
  source.append("step/end", { turn: 2, step: 1 });
  source.append("turn/end", {
    turn: 2,
    reason: {
      kind: "error",
      error: { code: "TRANSPORT", message: "Connection error" },
    },
  });
  return { context, source, prefixEnd };
}

describe("V11 append-only branch projections", () => {
  it("retains superseded input and failure events only in the source history", async () => {
    const { context, source, prefixEnd } = await failedSource();
    const sourceBefore = structuredClone(source.events);
    const child = context.sessions.fork(source, prefixEnd, "edited-child" as SessionId);
    child.append("turn/start", { turn: 2 });
    child.append(
      "user/message",
      createUserMessage({
        content: [{ type: "text", text: "edited prompt" }],
        source: { kind: "user" },
      }),
      { surfaceOp: "append" },
    );

    expect(source.events).toEqual(sourceBefore);
    expect(source.events.some((event) => event.type === "llm/retry")).toBe(true);
    expect(source.events.at(-1)).toMatchObject({
      type: "turn/end",
      data: { reason: { kind: "error", error: { message: "Connection error" } } },
    });
    expect(child.header.parentSession).toBe(source.id);
    expect(textOf(child)).toEqual(["kept prefix", "edited prompt"]);
    expect(JSON.stringify(child.events)).not.toContain("superseded prompt");
    expect(JSON.stringify(child.events)).not.toContain("Connection error");
  });

  it("keeps retry failures out of the messages projected for an LLM request", async () => {
    const { source } = await failedSource();

    expect(textOf(source)).toEqual(["kept prefix", "superseded prompt"]);
    expect(JSON.stringify(source.deriveMessages())).not.toContain("Connection error");
    expect(JSON.stringify(source.deriveMessages())).not.toContain("TRANSPORT");
  });
});
