import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Context } from "@deepseek-ai/cordis";
import type { Agent } from "@deepseek-ai/dsh-agent";
import type { SkillProviderControl } from "@deepseek-ai/dsh-skill";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  registerWikiSkillProvider,
  TargetedWikiSkillProvider,
  type WikiSkillStageRequest,
  WikiSkillStore,
} from "../src/index.js";

const roots: string[] = [];
const taskSetRevision = `sha256:${"c".repeat(64)}` as const;
const patternRevision = `sha256:${"d".repeat(64)}` as const;

function request(proposalId: string, name: string, model: string): WikiSkillStageRequest {
  return {
    name,
    operation: "create",
    proposalId,
    purposeMarkdown: `# Purpose\n\nUse ${name}.`,
    skillMarkdown: `---\nname: ${name}\ndescription: Use ${name}.\n---\n\n# ${name}`,
    sources: {
      patterns: [{ id: "workspaces/demo/concepts/pattern.md", revision: patternRevision }],
      traces: [{ endSeq: 2, sessionId: "session-1", startSeq: 1 }],
    },
    target: { model, preset: "dsh-science" },
  };
}

async function accept(store: WikiSkillStore, proposal: WikiSkillStageRequest) {
  const staged = await store.stage(proposal);
  const shared = {
    benchmarkId: "fixture",
    runs: 1,
    target: proposal.target,
    taskSetRevision,
  } as const;
  await store.resolve(staged.proposalId, {
    baseline: { ...shared, score: 0 },
    candidate: { ...shared, score: 1, skillRevision: staged.candidateRevision },
  });
}

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-wikiskill-provider-"));
  roots.push(root);
  const store = new WikiSkillStore(root);
  await store.initialize();
  await accept(store, request("20000000-0000-4000-8000-000000000001", "model-a-skill", "a"));
  await accept(store, request("20000000-0000-4000-8000-000000000002", "model-b-skill", "b"));
  await store.stage(request("20000000-0000-4000-8000-000000000003", "staged-only", "a"));
  return { root, store };
}

function context(): Context {
  return {
    get: () => undefined,
    logger: { warn: vi.fn() },
  } as unknown as Context;
}

function agent(model?: string): Agent {
  return {
    options: { model },
    session: { header: { agentPreset: "dsh-science" } },
  } as unknown as Agent;
}

function control() {
  const abort = new AbortController();
  return {
    abort,
    control: { invalidate: vi.fn(), signal: abort.signal } satisfies SkillProviderControl,
  };
}

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { force: true, recursive: true })));
});

describe("V237 exact-Agent WikiSkill provider", () => {
  it("shows only the active root for the current preset+model and delegates DSH loading", async () => {
    const { store } = await fixture();
    const liveAgent = agent("a");
    const lifecycle = control();
    const provider = new TargetedWikiSkillProvider(context(), lifecycle.control, store, liveAgent, {
      watch: false,
    });

    const first = await provider.list({ cwd: "/workspace" });
    expect(first.map(({ name }) => name)).toEqual(["model-a-skill"]);
    expect(first.map(({ name }) => name)).not.toContain("staged-only");
    await expect(provider.get(first[0] as never, {})).resolves.toMatchObject({
      content: "# model-a-skill",
      name: "model-a-skill",
      provider: "wikiskill-active",
    });

    (liveAgent.options as { model?: string }).model = "b";
    const second = await provider.list({ cwd: "/workspace" });
    expect(second.map(({ name }) => name)).toEqual(["model-b-skill"]);
    expect(lifecycle.control.invalidate).toHaveBeenCalledOnce();
    await provider.dispose();
  });

  it("invalidates a cached empty catalog when a target model becomes available", async () => {
    const { store } = await fixture();
    const liveAgent = agent();
    const lifecycle = control();
    const provider = new TargetedWikiSkillProvider(context(), lifecycle.control, store, liveAgent, {
      watch: false,
    });

    await expect(provider.list({ cwd: "/workspace" })).resolves.toEqual([]);
    (liveAgent.options as { model?: string }).model = "a";
    await expect(provider.list({ cwd: "/workspace" })).resolves.toEqual([
      expect.objectContaining({ name: "model-a-skill" }),
    ]);
    expect(lifecycle.control.invalidate).toHaveBeenCalledOnce();
    await provider.dispose();
  });

  it("hides flat and symlinked files that were not promoted by the validation store", async () => {
    const { store } = await fixture();
    const root = store.activeRoot({ model: "a", preset: "dsh-science" });
    await writeFile(
      join(root, "flat-skill.md"),
      "---\nname: flat-skill\ndescription: Unvalidated flat skill.\n---\n\n# Flat",
      "utf8",
    );
    const outside = await mkdtemp(join(tmpdir(), "swarmx-wikiskill-outside-"));
    roots.push(outside);
    const escaped = join(outside, "escaped-skill");
    await mkdir(escaped);
    await writeFile(
      join(escaped, "SKILL.md"),
      "---\nname: escaped-skill\ndescription: Escaped skill.\n---\n\n# Escaped",
      "utf8",
    );
    await symlink(escaped, join(root, "escaped-skill"), "dir");
    const lifecycle = control();
    const provider = new TargetedWikiSkillProvider(
      context(),
      lifecycle.control,
      store,
      agent("a"),
      { watch: false },
    );

    const candidates = await provider.list({ cwd: "/workspace" });
    expect(candidates.map(({ name }) => name)).toEqual(["model-a-skill"]);
    await provider.dispose();
  });

  it("registers in the supplied Agent scope and refreshes before downstream pre-step readers", async () => {
    const { store } = await fixture();
    const liveAgent = agent("a");
    const lifecycle = control();
    let provider: TargetedWikiSkillProvider | undefined;
    let preStep:
      | ((input: { agent: Agent }, next: () => Promise<string>) => Promise<string>)
      | undefined;
    const unregister = vi.fn();
    const removeListener = vi.fn();
    const agentContext = {
      agent: liveAgent,
      get: () => undefined,
      logger: { warn: vi.fn() },
      on: vi.fn((name: string, listener: typeof preStep) => {
        expect(name).toBe("agent/pre-step");
        preStep = listener;
        return removeListener;
      }),
      skills: {
        registerProvider: vi.fn(
          (create: (value: SkillProviderControl) => TargetedWikiSkillProvider) => {
            provider = create(lifecycle.control);
            return unregister;
          },
        ),
      },
    } as unknown as Context;

    const dispose = registerWikiSkillProvider(agentContext, store, { watch: false });
    const next = vi.fn(async () => "downstream");
    await expect(preStep?.({ agent: liveAgent }, next)).resolves.toBe("downstream");
    expect(next).toHaveBeenCalledOnce();
    await expect(provider?.list({ cwd: "/workspace" })).resolves.toEqual([
      expect.objectContaining({ name: "model-a-skill" }),
    ]);

    dispose();
    expect(removeListener).toHaveBeenCalledOnce();
    expect(unregister).toHaveBeenCalledOnce();
  });
});
