import { randomUUID } from "node:crypto";
import type { SessionId } from "@deepseek-ai/dsh-session";
import type { ToolDefinition, ToolRunContext } from "@deepseek-ai/dsh-tools";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createScienceToolDefinitions,
  registerScienceTools,
  SCIENCE_TOOL_NAMES,
} from "../src/tools.js";
import { createScienceFixture, type ScienceFixture } from "./fixture.js";

const fixtures: ScienceFixture[] = [];

afterEach(async () => {
  await Promise.all(fixtures.splice(0).map((fixture) => fixture.dispose()));
});

function execution(sessionId: SessionId, signal = new AbortController().signal): ToolRunContext {
  return {
    callId: "science-call-1",
    rootCallId: "science-call-1",
    name: "science_record",
    arguments: {},
    agent: { id: sessionId },
    signal,
    deferContext: vi.fn(),
    concludeTurn: vi.fn(),
  } as unknown as ToolRunContext;
}

function named(definitions: readonly ToolDefinition[], name: string): ToolDefinition {
  const definition = definitions.find((candidate) => candidate.name === name);
  if (!definition) throw new Error(`Missing tool ${name}`);
  return definition;
}

describe("T18 aggregate Science tools", () => {
  it("registers exactly seven aggregate tools and disposes every registration", () => {
    const dispose = vi.fn();
    const register = vi.fn(() => dispose);
    const science = {} as never;

    const unregister = registerScienceTools({ science, tools: { register } } as never);

    expect(register.mock.calls.map(([definition]) => definition.name)).toEqual(SCIENCE_TOOL_NAMES);
    unregister();
    expect(dispose).toHaveBeenCalledTimes(7);
  });

  it("returns a durable fact locator and rejects unknown strict input", async () => {
    const fixture = await createScienceFixture();
    fixtures.push(fixture);
    const project = fixture.context.science.createProject(fixture.sessionA, {
      requestId: randomUUID(),
      title: "Tool-driven research",
    });
    const definitions = createScienceToolDefinitions(fixture.context.science);
    const record = named(definitions, "science_record");

    const result = await record.execute(
      {
        action: "create_question",
        request: {
          requestId: randomUUID(),
          projectId: project.id,
          title: "Can tools retain provenance?",
          summary: "Exercise the public service boundary.",
          tags: [],
        },
      },
      execution(fixture.sessionA),
    );
    expect(result).toMatchObject({
      classification: "fact",
      summary: "Created research question",
      locator: {
        sessionId: fixture.sessionA,
        toolCallId: "science-call-1",
        entityKind: "question",
        entityId: expect.any(String),
        journalSeq: expect.any(Number),
      },
      data: { kind: "question" },
    });
    await expect(
      record.execute(
        {
          action: "create_question",
          request: {
            requestId: randomUUID(),
            projectId: project.id,
            title: "Invalid",
            summary: "Unexpected authority must fail.",
            tags: [],
          },
          sql: "SELECT * FROM science_journal",
        },
        execution(fixture.sessionA),
      ),
    ).rejects.toMatchObject({ code: "INVALID_REQUEST" });
  });

  it("forwards cancellation and labels comparisons as inference", async () => {
    const fixture = await createScienceFixture();
    fixtures.push(fixture);
    const project = fixture.context.science.createProject(fixture.sessionA, {
      requestId: randomUUID(),
      title: "Tool cancellation",
    });
    const definitions = createScienceToolDefinitions(fixture.context.science);
    const experimentTool = named(definitions, "science_experiment");
    const controller = new AbortController();
    controller.abort();

    await expect(
      experimentTool.execute(
        {
          action: "define",
          request: {
            requestId: randomUUID(),
            projectId: project.id,
            title: "Cancelled experiment",
            summary: "No fact should be appended.",
            hypothesisIds: [],
            protocol: "none",
            tags: [],
          },
        },
        execution(fixture.sessionA, controller.signal),
      ),
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(fixture.context.science.journalCount()).toBe(1);
  });
});
