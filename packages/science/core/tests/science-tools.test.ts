import { randomUUID } from "node:crypto";
import type { ImageAttachmentRef } from "@deepseek-ai/dsh-attachment";
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

describe("Science model tools", () => {
  it("registers the bounded domain tools and disposes every registration", () => {
    const dispose = vi.fn();
    const register = vi.fn(() => dispose);
    const science = {} as never;

    const unregister = registerScienceTools({
      attachments: { saveImage: vi.fn() },
      science,
      tools: { register },
    } as never);

    expect(register.mock.calls.map(([definition]) => definition.name)).toEqual(SCIENCE_TOOL_NAMES);
    unregister();
    expect(dispose).toHaveBeenCalledTimes(8);
  });

  it("returns a durable fact locator and rejects unknown strict input", async () => {
    const fixture = await createScienceFixture();
    fixtures.push(fixture);
    const project = fixture.context.science.createProject(fixture.sessionA, {
      requestId: randomUUID(),
      title: "Tool-driven research",
    });
    const definitions = createScienceToolDefinitions(fixture.context.science, {
      saveImage: vi.fn(),
    } as never);
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

  it("queries the project Research Object instead of exposing workspace or trace graphs", async () => {
    const fixture = await createScienceFixture();
    fixtures.push(fixture);
    const project = fixture.context.science.createProject(fixture.sessionA, {
      requestId: randomUUID(),
      title: "RO-Crate query",
    });
    const query = named(
      createScienceToolDefinitions(fixture.context.science, { saveImage: vi.fn() } as never),
      "science_query",
    );

    expect(query.parameters.properties?.action).toMatchObject({
      enum: ["research_object", "inspect_annotation"],
    });
    const value = await query.execute(
      { action: "research_object", request: { projectId: project.id } },
      execution(fixture.sessionA),
    );
    expect(value).toMatchObject({
      classification: "fact",
      summary: "Read project Research Object",
      locator: { entityKind: "project", entityId: project.id },
      data: {
        "@context": "https://w3id.org/ro/crate/1.3/context",
        "@graph": expect.any(Array),
      },
    });
  });

  it("forwards cancellation and labels comparisons as inference", async () => {
    const fixture = await createScienceFixture();
    fixtures.push(fixture);
    const project = fixture.context.science.createProject(fixture.sessionA, {
      requestId: randomUUID(),
      title: "Tool cancellation",
    });
    const definitions = createScienceToolDefinitions(fixture.context.science, {
      saveImage: vi.fn(),
    } as never);
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

  it("keeps literature search distinct from local science state and uses a direct request", async () => {
    const searchLiterature = vi.fn(() =>
      Promise.resolve({
        source: "zotero",
        ranking: "zotero-local-v1",
        query: "genome foundation model",
        totalCandidates: 1,
        snapshot: {
          source: "zotero",
          format: "bibtex",
          digest: `sha256:${"a".repeat(64)}`,
          entryCount: 1,
          sourceVersion: "15002",
        },
        results: [],
      }),
    );
    const definitions = createScienceToolDefinitions(
      { searchLiterature } as never,
      { saveImage: vi.fn() } as never,
    );
    const literature = named(definitions, "literature_search");

    expect(literature.description).toContain("local Zotero");
    expect(literature.description).toContain("BibTeX");
    expect(literature.parameters).toMatchObject({
      additionalProperties: false,
      required: ["query"],
      properties: { query: { type: "string" }, limit: { type: "integer" } },
    });
    const value = await literature.execute(
      { query: "genome foundation model", limit: 5 },
      execution("literature-session" as SessionId),
    );
    expect(searchLiterature).toHaveBeenCalledWith(
      "literature-session",
      { query: "genome foundation model", limit: 5 },
      expect.any(AbortSignal),
    );
    expect(value).toMatchObject({
      classification: "inference",
      summary: "Searched local Zotero literature",
      locator: null,
      data: { source: "zotero", ranking: "zotero-local-v1" },
    });
    await expect(
      literature.execute(
        { query: "genome", provider: "semantic-scholar" },
        execution("literature-session" as SessionId),
      ),
    ).rejects.toMatchObject({ code: "INVALID_REQUEST" });
    expect(searchLiterature).toHaveBeenCalledOnce();
  });

  it("re-authorizes an image annotation and renders verified pixels for the model", async () => {
    const fixture = await createScienceFixture();
    fixtures.push(fixture);
    const project = fixture.context.science.createProject(fixture.sessionA, {
      requestId: randomUUID(),
      title: "Annotation discussion",
    });
    const artifact = fixture.context.science.importArtifact(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      name: "point.png",
      dataBase64:
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z8S8AAAAASUVORK5CYII=",
    });
    const attachment = {
      attachmentId: "attachment-1",
      mediaType: "image/png",
      bytes: artifact.size,
      width: 1,
      height: 1,
      name: "point.png",
    } as ImageAttachmentRef;
    const saveImage = vi.fn(() => Promise.resolve(attachment));
    const query = named(
      createScienceToolDefinitions(fixture.context.science, { saveImage } as never),
      "science_query",
    );
    expect(query.description).toContain("never submit artifact_id alone");
    expect(query.parameters.properties?.request).toMatchObject({
      oneOf: expect.arrayContaining([
        expect.objectContaining({
          required: ["type", "id", "comment", "created_at", "target"],
        }),
      ]),
    });
    const request = {
      type: "comment",
      id: "annotation-1",
      comment: "Why is this point isolated?",
      created_at: 1_787_371_200_000,
      target: {
        type: "image_point",
        artifact_id: artifact.id,
        project_id: project.id,
        title: artifact.title,
        digest: artifact.digest,
        mime: "image/png",
        point: { x: 0.25, y: 0.75 },
      },
    } as const;

    const value = await query.execute(
      { action: "inspect_annotation", request },
      execution(fixture.sessionA),
    );
    expect(value).toMatchObject({
      classification: "fact",
      summary: "Inspected science image annotation",
      data: { annotation: request, attachment },
    });
    expect(saveImage).toHaveBeenCalledWith(
      expect.objectContaining({ mediaType: "image/png", data: expect.any(Uint8Array) }),
    );
    expect(query.output.render({ action: "inspect_annotation", request }, value as never)).toEqual([
      { type: "text", text: expect.stringContaining("Why is this point isolated?") },
      { type: "image", attachment },
    ]);

    await expect(
      query.execute(
        {
          action: "inspect_annotation",
          request: {
            ...request,
            target: { ...request.target, digest: `sha256:${"b".repeat(64)}` },
          },
        },
        execution(fixture.sessionA),
      ),
    ).rejects.toMatchObject({ code: "ARTIFACT_SOURCE_CHANGED" });
    expect(saveImage).toHaveBeenCalledTimes(1);
  });
});
