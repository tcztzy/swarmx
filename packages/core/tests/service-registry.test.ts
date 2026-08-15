import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import { Agent } from "../src/agent.js";
import type { AgentContextEngine } from "../src/context-engine.js";
import {
  AblationProfileSchema,
  AgentServiceRegistry,
  createBuiltinAgentServiceRegistry,
  DEFAULT_ABLATION_PROFILE,
  type ServiceTopology,
} from "../src/service-registry.js";

const topology: ServiceTopology = {
  swarmPath: ["evaluation"],
  nodeId: "writer",
  rootNodeId: "writer",
  agentName: "writer",
};

describe("AgentServiceRegistry", () => {
  it("preserves every supplied service in the production profile", () => {
    const contextEngine: AgentContextEngine = { compile: vi.fn() };
    const personalMemory = {
      source: "desktop_settings" as const,
      content: "Remember this preference.",
      updatedAt: "2026-08-15T00:00:00.000Z",
      characterCount: 25,
    };
    const registry = createBuiltinAgentServiceRegistry();
    const memoryTools = [
      {
        name: "Memory",
        inputSchema: { type: "object" },
        async call() {
          return "memory";
        },
      },
    ];

    const activation = registry.activate(DEFAULT_ABLATION_PROFILE, topology, {
      contextEngine,
      personalMemory,
      memoryTools,
      skillInstructions: [],
    });

    expect(activation.services.contextEngine).toBe(contextEngine);
    expect(activation.services.personalMemory).toBe(personalMemory);
    expect(activation.services.memoryTools).toBe(memoryTools);
    expect(activation.receipt).toMatchObject({
      profileId: "production",
      topology,
      entries: [
        { seam: "context_engine", variantId: "production" },
        { seam: "memory", variantId: "production" },
        { seam: "skill_evolution", variantId: "production" },
      ],
    });
  });

  it("removes only the service selected for baseline", () => {
    const contextEngine: AgentContextEngine = { compile: vi.fn() };
    const activation = createBuiltinAgentServiceRegistry().activate(
      {
        schemaVersion: 1,
        profileId: "without_context",
        variants: {
          context_engine: "baseline",
          memory: "production",
          skill_evolution: "production",
        },
      },
      topology,
      { contextEngine, skillInstructions: [] },
    );

    expect(activation.services.contextEngine).toBeUndefined();
    expect(activation.receipt.entries).toContainEqual({
      seam: "context_engine",
      variantId: "baseline",
    });
  });

  it("passes deterministic topology to custom variants", () => {
    const registry = createBuiltinAgentServiceRegistry();
    const resolve = vi.fn(() => ({}));
    registry.register({ seam: "context_engine", variantId: "topology_probe", resolve });
    const profile = {
      schemaVersion: 1 as const,
      profileId: "topology_probe",
      variants: {
        context_engine: "topology_probe",
        memory: "baseline",
        skill_evolution: "baseline",
      },
    };

    registry.activate(profile, topology, {});

    expect(resolve).toHaveBeenCalledWith(expect.objectContaining({ topology }));
  });

  it("lists every unresolved seam before activation", () => {
    const registry = new AgentServiceRegistry();

    expect(() => registry.assertEntriesActivated(DEFAULT_ABLATION_PROFILE)).toThrow(
      /context_engine:production.*memory:production.*skill_evolution:production/s,
    );
  });

  it("rejects duplicate providers and incomplete profiles", () => {
    const registry = createBuiltinAgentServiceRegistry();
    expect(() =>
      registry.register({ seam: "memory", variantId: "production", resolve: () => ({}) }),
    ).toThrow(/already registered/);
    expect(() =>
      AblationProfileSchema.parse({
        schemaVersion: 1,
        profileId: "incomplete",
        variants: { context_engine: "baseline", memory: "baseline" },
      }),
    ).toThrow();
  });

  it("locks providers after a successful activation assertion", () => {
    const registry = createBuiltinAgentServiceRegistry();
    registry.assertEntriesActivated(DEFAULT_ABLATION_PROFILE);

    expect(() =>
      registry.register({ seam: "memory", variantId: "late", resolve: () => ({}) }),
    ).toThrow(/locked/);
  });

  it("rejects a variant that contributes across seam boundaries", () => {
    const registry = createBuiltinAgentServiceRegistry();
    registry.register({
      seam: "context_engine",
      variantId: "leaky",
      resolve: () => ({ personalMemory: undefined }),
    });

    expect(() =>
      registry.activate(
        {
          schemaVersion: 1,
          profileId: "leaky",
          variants: {
            context_engine: "leaky",
            memory: "baseline",
            skill_evolution: "baseline",
          },
        },
        topology,
        {},
      ),
    ).toThrow(/cross-seam fields.*personalMemory/);
  });

  it("applies baseline variants before Agent instructions and tools are assembled", async () => {
    const content = "Use the evolved procedure.";
    const contextEngine: AgentContextEngine = { compile: vi.fn() };
    const agent = new Agent(
      {
        name: "writer",
        instructions: "Base instructions.",
        model: "test-model",
        client: { apiProtocol: "openai_responses" },
      },
      {
        contextEngine,
        personalMemory: {
          source: "desktop_settings",
          content: "Remember this preference.",
          updatedAt: "2026-08-15T00:00:00.000Z",
          characterCount: 25,
        },
        skillInstructions: [
          {
            skillId: "review",
            variantId: "review:active",
            revisionId: "revision_1",
            contentDigest: `sha256:${sha256(content)}`,
            mode: "prompt_fragment",
            content,
          },
        ],
        memoryTools: [
          {
            name: "Memory",
            inputSchema: { type: "object" },
            async call() {
              return "memory";
            },
          },
        ],
        localTools: [
          {
            name: "workspace",
            inputSchema: { type: "object" },
            async call() {
              return "workspace";
            },
          },
        ],
        serviceRegistry: createBuiltinAgentServiceRegistry(),
        ablationProfile: {
          schemaVersion: 1,
          profileId: "minimal_baseline",
          variants: {
            context_engine: "baseline",
            memory: "baseline",
            skill_evolution: "baseline",
          },
        },
        serviceTopology: topology,
      },
    );
    const create = vi.fn(async () => ({
      id: "response",
      status: "completed",
      error: null,
      output: [
        {
          id: "message",
          type: "message",
          role: "assistant",
          status: "completed",
          content: [{ type: "output_text", text: "done", annotations: [] }],
        },
      ],
    }));
    Object.defineProperty(agent.client.responses, "create", { value: create });

    await agent.call({ messages: [{ role: "user", content: "test" }] });

    expect(agent.instructions).toBe("Base instructions.");
    expect(contextEngine.compile).not.toHaveBeenCalled();
    expect(create.mock.calls[0]?.[0]?.tools).toContainEqual(
      expect.objectContaining({ name: "workspace" }),
    );
    expect(create.mock.calls[0]?.[0]?.tools ?? []).not.toContainEqual(
      expect.objectContaining({ name: "Memory" }),
    );
    expect(agent.serviceActivationReceipt).toMatchObject({
      profileId: "minimal_baseline",
      topology,
    });
  });

  it("fails unresolved profiles during Agent construction", () => {
    expect(
      () =>
        new Agent(
          { name: "writer" },
          {
            serviceRegistry: new AgentServiceRegistry(),
            ablationProfile: DEFAULT_ABLATION_PROFILE,
          },
        ),
    ).toThrow(/context_engine:production.*memory:production.*skill_evolution:production/s);
  });

  it("rejects explicit ablation for external ACP Harnesses", () => {
    expect(
      () =>
        new Agent(
          { name: "external", backend: { type: "custom", program: "external-acp" } },
          {
            serviceRegistry: createBuiltinAgentServiceRegistry(),
            ablationProfile: DEFAULT_ABLATION_PROFILE,
          },
        ),
    ).toThrow(/external ACP/i);
  });
});

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}
