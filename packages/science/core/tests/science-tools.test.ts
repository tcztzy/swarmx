import { randomUUID } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";
import {
  createScienceToolDefinitions,
  registerScienceTools,
  SCIENCE_TOOL_NAMES,
  type ScienceImageAttachment,
  type ScienceToolDefinition,
  type ScienceToolExecution,
} from "../src/tools.js";
import { createScienceFixture, type ScienceFixture } from "./fixture.js";

const fixtures: ScienceFixture[] = [];

afterEach(async () => {
  await Promise.all(fixtures.splice(0).map((fixture) => fixture.dispose()));
});

function execution(actorId: string, signal = new AbortController().signal): ScienceToolExecution {
  return {
    callId: "science-call-1",
    actorId,
    signal,
  };
}

function named(definitions: readonly ScienceToolDefinition[], name: string): ScienceToolDefinition {
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

    const result = await record.invoke(
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
      record.invoke(
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

  it("lets every runtime create the project root through the shared tool", async () => {
    const fixture = await createScienceFixture();
    fixtures.push(fixture);
    const notebook = named(
      createScienceToolDefinitions(fixture.context.science, { saveImage: vi.fn() } as never),
      "science_notebook",
    );

    const created = await notebook.invoke(
      {
        action: "create_project",
        request: { requestId: randomUUID(), title: "Runtime-neutral project" },
      },
      execution(fixture.sessionA),
    );

    expect(created).toMatchObject({
      classification: "fact",
      summary: "Created science project",
      data: { kind: "project", title: "Runtime-neutral project" },
    });
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

    expect(query.parameters.oneOf).toHaveLength(7);
    expect(query.mcpParameters).toMatchObject({ type: "object", oneOf: expect.any(Array) });
    const publishedInput = z.fromJSONSchema(
      query.mcpParameters as Parameters<typeof z.fromJSONSchema>[0],
    );
    expect(() =>
      publishedInput.parse({ action: "head", request: { projectId: project.id } }),
    ).toThrow();
    expect(() => publishedInput.parse({ action: "batch_head", request: { ids: [] } })).toThrow();
    const mcpVariants = (query.mcpParameters as { oneOf?: Record<string, unknown>[] }).oneOf;
    const neighbors = mcpVariants?.find(
      (variant) =>
        (
          (variant.properties as Record<string, unknown> | undefined)?.action as
            | Record<string, unknown>
            | undefined
        )?.const === "neighbors",
    );
    expect(neighbors).toMatchObject({
      properties: {
        action: { const: "neighbors" },
        request: {
          properties: {
            relations: { minItems: 1, maxItems: 16, uniqueItems: true },
            limit: { minimum: 1, maximum: 100 },
          },
        },
      },
    });
    expect(
      publishedInput.parse({ action: "research_object", request: { projectId: project.id } }),
    ).toEqual({ action: "research_object", request: { projectId: project.id } });
    const value = await query.invoke(
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

  it("exposes strict ID-addressed query actions without returning a workspace snapshot", async () => {
    const fixture = await createScienceFixture();
    fixtures.push(fixture);
    const project = fixture.context.science.createProject(fixture.sessionA, {
      requestId: randomUUID(),
      title: "Addressed tool project",
    });
    const query = named(
      createScienceToolDefinitions(fixture.context.science, { saveImage: vi.fn() } as never),
      "science_query",
    );
    const id = `sx:p/${project.id}`;

    const head = await query.invoke(
      { action: "head", request: { id } },
      execution(fixture.sessionA),
    );
    expect(head).toMatchObject({
      classification: "fact",
      summary: "Read science resource head",
      locator: { entityKind: "project", entityId: project.id },
      data: { ref: { id, exactId: `${id}@1`, revision: 1 } },
    });
    const metadata = await query.invoke(
      { action: "get", request: { id: `${id}@1`, projection: "metadata" } },
      execution(fixture.sessionA),
    );
    expect(metadata).toMatchObject({
      classification: "fact",
      data: { ref: { id }, metadata: { kind: "project" } },
    });
    expect(JSON.stringify(head)).not.toContain("projects");
    expect(metadata.data).not.toHaveProperty("projects");
    expect(metadata.data).not.toHaveProperty("notebooks");

    for (const input of [
      { action: "head", request: { id, extra: true } },
      { action: "batch_head", request: { ids: [id], extra: true } },
      { action: "get", request: { id, projection: "metadata", extra: true } },
      { action: "select", request: { id, format: "table", extra: true } },
      { action: "neighbors", request: { id, extra: true } },
      { action: "unknown", request: {} },
    ]) {
      await expect(query.invoke(input, execution(fixture.sessionA))).rejects.toMatchObject({
        code: "INVALID_REQUEST",
      });
    }
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
      experimentTool.invoke(
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
    const value = await literature.invoke(
      { query: "genome foundation model", limit: 5 },
      execution("literature-session"),
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
      literature.invoke(
        { query: "genome", provider: "semantic-scholar" },
        execution("literature-session"),
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
    } as ScienceImageAttachment;
    const saveImage = vi.fn(() => Promise.resolve(attachment));
    const query = named(
      createScienceToolDefinitions(fixture.context.science, { saveImage } as never),
      "science_query",
    );
    expect(query.description).toContain("never submit artifact_id alone");
    const inspectBranch = query.parameters.oneOf?.find(
      (branch) => branch.title === "Image annotation request",
    )?.properties?.request;
    expect(inspectBranch).toMatchObject({
      required: ["type", "id", "comment", "created_at", "target"],
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

    const value = await query.invoke(
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
      query.invoke(
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
