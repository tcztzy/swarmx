import { describe, expect, it, vi } from "vitest";
import { TYPERT_REMOTE } from "../../../science/core/src/remote.js";
import { apply } from "../src/client/index.js";
import { apply as applyHost, name as pluginName } from "../src/index.js";

vi.mock("@deepseek-ai/dsh-client-ui-primitives", () => ({
  IconBranchOutline16: () => null,
  IconDataOutline16: () => null,
  IconEditOutline16: () => null,
  IconEllipsisOutline16: () => null,
  IconFullscreenOutline16: () => null,
  IconSendOutline16: () => null,
  IconTrashOutline16: () => null,
  Menu: () => null,
}));

const artifact = {
  id: "artifact-1",
  projectId: "project-1",
  kind: "figure" as const,
  title: "umap.png",
  digest: `sha256:${"a".repeat(64)}`,
  mime: "image/png",
  size: 100,
  creator: { kind: "session" as const, sessionId: "session-1" },
  runId: null,
  environment: {},
  license: null,
  sourceEntityIds: [],
  createdAt: 1,
  updatedAt: 1,
  revision: 1,
  provenance: { eventId: "event-1", journalSeq: 1, sessionId: "session-1" },
};

describe("V102/V122 Host-global deliverable contract", () => {
  it("keeps only the generic Markdown file-link guidance global", () => {
    expect(pluginName).toBe("swarmx-ui-science");
    const sections: Array<{ name: string; order: number; text: string }> = [];
    const result = applyHost({
      systemPrompt: { section: (section) => sections.push(section) },
    } as never) as unknown as () => void;

    expect(result).toBeUndefined();
    expect(sections.map((section) => section.name)).toEqual(["ui:deliverable-file-references"]);
    const deliverables = sections.find(({ name }) => name === "ui:deliverable-file-references");
    expect(deliverables?.text).toContain("[filename](./workspace-relative/path)");
    expect(deliverables?.text).toContain("Prefer this Markdown link");
    expect(deliverables?.text).not.toContain(
      "Format changed-file references as Markdown inline code",
    );
    expect(deliverables?.text).not.toContain("typst compile");
    expect(deliverables?.text).not.toContain("science_query");
    expect(deliverables?.text).not.toContain("literature_search");
  });
});

describe("V17/V57 Science client registration", () => {
  it("registers the produced-file accumulator before Remote mounting can yield", () => {
    const registerConversationNode = vi.fn();
    const neverMounted = new Promise<never>(() => undefined);
    void apply({
      conversationEvents: { register: registerConversationNode },
      remote: { $mount: vi.fn(() => neverMounted) },
      sessions: { list: { getSnapshot: () => ({ current: undefined, byId: {} }) } },
      sideView: { open: vi.fn() },
      slots: { inject: vi.fn(), register: vi.fn() },
      workspaces: { openPath: vi.fn() },
      provide: vi.fn(),
    } as never);

    expect(registerConversationNode).toHaveBeenCalledOnce();
  });

  it("mounts Chat artifact cards and DetailsPanel content without a Science peer view", async () => {
    const disposeRemote = vi.fn();
    const mounted = vi.fn(() => Promise.resolve(disposeRemote));
    const registrations: Array<{ options: Record<string, unknown>; component: unknown }> = [];
    const register = vi.fn((options: Record<string, unknown>, component: unknown) => {
      registrations.push({ options, component });
      return vi.fn();
    });
    const injectSlot = vi.fn((_name: string, callback: () => void) => callback());
    const injectService = vi.fn((_services: string[], callback: (scope: typeof context) => void) =>
      callback(context),
    );
    const unregisterReference = vi.fn();
    const registerSource = vi.fn(() => unregisterReference);
    const registerConversationNode = vi.fn();
    const provide = vi.fn();
    let inputState = {
      draft: "Existing draft",
      draftRev: 1,
      phase: "plain",
      occurrences: [],
      imageIds: [],
      queue: [],
    };
    const input = {
      state: { getSnapshot: () => inputState },
      setDraft: vi.fn((draft: string) => {
        inputState = { ...inputState, draft, draftRev: inputState.draftRev + 1 };
      }),
      insertReference: vi.fn(() => true),
    };
    const context = {
      conversation: { input: { for: vi.fn(() => input) } },
      conversationEvents: { register: registerConversationNode },
      inject: injectService,
      inputTriggers: { registerSource },
      remote: {
        $mount: mounted,
        science: {
          previewArtifact: vi.fn(() =>
            Promise.resolve({
              ok: true,
              value: {
                kind: "image",
                artifactId: "artifact-1",
                digest: artifact.digest,
                mime: "image/png",
                size: 4,
                dataUrl: "data:image/png;base64,iVBORw==",
              },
            }),
          ),
          getResearchObject: vi.fn(() =>
            Promise.resolve({
              ok: true,
              value: {
                "@context": "https://w3id.org/ro/crate/1.3/context",
                "@graph": [
                  {
                    "@id": "ro-crate-metadata.json",
                    "@type": "CreativeWork",
                    about: { "@id": "urn:uuid:project-1" },
                  },
                  {
                    "@id": "urn:uuid:project-1",
                    "@type": "Dataset",
                    name: "Project",
                    description: "Project Research Object",
                    datePublished: "2026-08-24T00:00:00.000Z",
                    license: "All rights reserved",
                    hasPart: [],
                  },
                ],
              },
            }),
          ),
          previewTypstDocument: vi.fn(() =>
            Promise.resolve({
              ok: true,
              value: {
                relativePath: "paper.typ",
                title: "paper.typ",
                source: "= Paper",
                sourceRevision: `sha256:${"b".repeat(64)}`,
                status: "compiling",
                diagnostics: [],
                pdfBase64: null,
                pdfRevision: null,
                pdfSourceRevision: null,
                pdfSize: null,
                compiledAt: null,
              },
            }),
          ),
          updateTypstSource: vi.fn(),
          resolveTypstSourceAtPoint: vi.fn(),
        },
      },
      provide,
      sessions: {
        binding: vi.fn(() => ({ ctx: {} })),
        list: { getSnapshot: () => ({ current: "session-1", byId: {} }) },
      },
      sideView: { open: vi.fn() },
      slots: { inject: injectSlot, register },
      workspaces: { openPath: vi.fn() },
    };

    const dispose = await apply(context as never);

    expect(mounted).toHaveBeenCalledWith(TYPERT_REMOTE);
    expect(registerSource).toHaveBeenCalledWith(
      expect.objectContaining({ trigger: "@", name: "annotation" }),
    );
    expect(registerConversationNode).toHaveBeenCalledOnce();
    expect(provide).toHaveBeenCalledWith("chatFileMentions", expect.any(Object));
    expect(injectService).toHaveBeenCalledWith(["remote.science"], expect.any(Function));
    expect(injectSlot).toHaveBeenCalledWith(
      "conversation.chat.turnTail.items",
      expect.any(Function),
    );
    expect(injectSlot).toHaveBeenCalledWith("side-view.content", expect.any(Function));
    expect(injectSlot).not.toHaveBeenCalledWith("side-view.tool.actions", expect.any(Function));
    expect(registrations.some(({ options }) => options.name === "conversation.view")).toBe(false);
    expect(registrations.some(({ options }) => options.name === "conversation.chat.turnTail")).toBe(
      false,
    );

    const filesItem = registrations.find(
      ({ options }) =>
        options.name === "conversation.chat.turnTail.items" && options.id === "science-files",
    )?.options as
      | {
          inject(sessionId: string): {
            openFile(path: string): void;
            openTypst(path: string): void;
          };
        }
      | undefined;
    if (!filesItem) throw new Error("Science Files turn-tail registration was not captured");
    filesItem.inject("session-1").openTypst("docs/paper.typ");
    expect(context.sideView.open).toHaveBeenCalledWith("session-1", {
      id: "science-typst:docs/paper.typ",
      kind: "science-typst",
      title: "paper.typ",
      mode: "workbench",
      payload: { relativePath: "docs/paper.typ" },
    });

    const turnTailItem = registrations.find(
      ({ options }) =>
        options.name === "conversation.chat.turnTail.items" && options.id === "science-artifacts",
    )?.options as
      | {
          inject(sessionId: string): {
            loadPreview(artifactId: string, signal?: AbortSignal): Promise<unknown>;
            openArtifact(artifact: typeof artifact): void;
          };
        }
      | undefined;
    if (!turnTailItem) throw new Error("Science turn-tail item registration was not captured");
    const turnTailFace = turnTailItem.inject("session-1");
    await turnTailFace.loadPreview("artifact-1");
    turnTailFace.openArtifact(artifact);
    expect(context.remote.science.previewArtifact).toHaveBeenCalledWith(
      "session-1",
      { artifactId: "artifact-1" },
      undefined,
    );
    expect(context.sideView.open).toHaveBeenCalledWith(
      "session-1",
      expect.objectContaining({
        id: "science-artifact:artifact-1",
        kind: "science-artifact",
        mode: "workbench",
      }),
    );

    const contentRegistration = registrations.find(
      ({ options }) => options.name === "side-view.content" && options.key === "science-artifact",
    )?.options as
      | {
          inject(sessionId: string): {
            loadResearchObject(projectId: string, signal?: AbortSignal): Promise<unknown>;
            addAnnotationToConversation(annotation: Record<string, unknown>): boolean;
          };
        }
      | undefined;
    if (!contentRegistration)
      throw new Error("Artifact DetailsPanel registration was not captured");
    const content = contentRegistration.inject("session-1");
    await content.loadResearchObject("project-1");
    expect(context.remote.science.getResearchObject).toHaveBeenCalledWith(
      "session-1",
      { projectId: "project-1" },
      undefined,
    );
    expect(
      content.addAnnotationToConversation({
        version: 1,
        id: "annotation-1",
        artifactId: "artifact-1",
        projectId: "project-1",
        title: "umap.png",
        digest: artifact.digest,
        mime: "image/png",
        x: 0.25,
        y: 0.75,
        comment: "Why is this cluster separated?",
        createdAt: 1_787_371_200_000,
      }),
    ).toBe(true);
    expect(input.setDraft).toHaveBeenCalledWith("Existing draft @");
    expect(input.insertReference).toHaveBeenCalledOnce();

    await dispose();
    expect(disposeRemote).toHaveBeenCalledOnce();
    expect(unregisterReference).toHaveBeenCalledOnce();
  });
});
