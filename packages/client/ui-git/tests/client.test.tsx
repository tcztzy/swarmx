import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { apply } from "../src/client/index.js";
import {
  VersionControlView,
  versionControlSideViewEntry,
} from "../src/client/version-control-view.js";

vi.mock("@deepseek-ai/dsh-client-ui-primitives", () => ({
  IconBranchOutline16: () => <span>branch</span>,
  IconRefreshOutline16: () => <span>refresh</span>,
  Tooltip: ({ children }: { readonly children: unknown }) => children,
}));

const repository = {
  kind: "repository" as const,
  version: "2.50.1",
  objectFormat: "sha1",
  head: "a".repeat(40),
  branch: "main",
  upstream: "origin/main",
  ahead: 1,
  behind: 2,
  clean: false,
  staged: 1,
  unstaged: 0,
  untracked: 1,
  conflicted: 0,
  truncated: false,
  entries: [
    { kind: "ordinary" as const, path: "src/model.ts", index: "M", worktree: "." },
    { kind: "untracked" as const, path: "results/new.csv", index: "?", worktree: "?" },
  ],
};

const dvcInspection = {
  version: "3.67.1",
  root: "analysis",
  dvcYamlDigest: `sha256:${"a".repeat(64)}`,
  dvcLockDigest: null,
  data: {
    categories: [],
    digest: `sha256:${"b".repeat(64)}`,
    entries: 0,
  },
  pipeline: {
    categories: [
      { name: "changed deps", count: 1 },
      { name: "changed outs", count: 1 },
    ],
    digest: `sha256:${"c".repeat(64)}`,
    entries: 2,
  },
  git: {
    version: "2.50.1",
    objectFormat: "sha1",
    head: "a".repeat(40),
    branch: "main",
    upstream: "origin/main",
    ahead: 1,
    behind: 2,
    clean: false,
    staged: 1,
    unstaged: 0,
    untracked: 1,
    conflicted: 0,
  },
};

describe("dsh-ui-git client", () => {
  it("mounts one read-only Version Control action and combines independent Git/DVC loads", async () => {
    const registrations: Array<{ options: Record<string, unknown>; component: unknown }> = [];
    const disposeRemote = vi.fn();
    const context = {
      inject: vi.fn((_services: string[], callback: (scope: typeof context) => void) =>
        callback(context),
      ),
      remote: {
        $mount: vi.fn(() => Promise.resolve(disposeRemote)),
        gitUi: { snapshot: vi.fn(() => Promise.resolve({ ok: true, value: repository })) },
        dvcUi: {
          snapshot: vi.fn(() =>
            Promise.resolve({ ok: true, value: { kind: "project", inspection: dvcInspection } }),
          ),
        },
      },
      sideView: { open: vi.fn() },
      slots: {
        inject: vi.fn((_name: string, callback: () => void) => callback()),
        register: vi.fn((options: Record<string, unknown>, component: unknown) => {
          registrations.push({ options, component });
          return vi.fn();
        }),
      },
    };

    const dispose = await apply(context as never);

    expect(registrations.map(({ options }) => [options.name, options.key ?? options.id])).toEqual([
      ["conversation.session.header.actions", "swarmx-version-control"],
      ["side-view.content", "version-control"],
    ]);
    const header = registrations[0]?.options;
    if (typeof header?.inject !== "function") {
      throw new Error("Version Control header injection is missing");
    }
    const injected = (header.inject as (id: string) => { load: () => Promise<unknown> })(
      "session-1",
    );
    expect(Object.keys(injected)).toEqual(["load", "open"]);
    await expect(injected.load()).resolves.toEqual({ git: repository, dvc: dvcInspection });
    context.remote.dvcUi.snapshot.mockRejectedValueOnce(new Error("DVC probe failed"));
    await expect(injected.load()).resolves.toEqual({ git: repository, dvc: null });
    await dispose();
    expect(disposeRemote).toHaveBeenCalledOnce();
  });

  it("renders default-open Git and available DVC accordions in one panel", () => {
    const snapshot = { git: repository, dvc: dvcInspection };
    const entry = versionControlSideViewEntry(snapshot);
    const markup = renderToStaticMarkup(
      <VersionControlView entry={entry} load={() => Promise.resolve(snapshot)} />,
    );

    expect(markup).toContain("Version Control");
    expect(markup.match(/<details[^>]*open=""/gu)).toHaveLength(2);
    expect(markup).toContain("Changes");
    expect(markup).toContain("main");
    expect(markup).toContain("origin/main");
    expect(markup).toContain("Ahead 1");
    expect(markup).toContain("Behind 2");
    expect(markup).toContain("src/model.ts");
    expect(markup).toContain("results/new.csv");
    expect(markup).toContain("DVC");
    expect(markup).toContain("Pipeline changed");
    expect(markup).toContain("changed deps");
    expect(markup).toContain("changed outs");
    expect(markup).toContain("2 Git changes · 2 DVC changes");
    expect(markup).toContain("No changes");
    expect(markup).toContain("sha256:aaaaaaaaaaaa");
    expect(markup).not.toContain("git add");
    expect(markup).not.toContain("Commit");
    expect(markup).not.toContain("DVC Pull");
    expect(markup).not.toContain("Reproduce");
  });

  it("keeps Git visible when DVC is unavailable", () => {
    const snapshot = { git: repository, dvc: null };
    const markup = renderToStaticMarkup(
      <VersionControlView
        entry={versionControlSideViewEntry(snapshot)}
        load={() => Promise.resolve(snapshot)}
      />,
    );

    expect(markup).toContain("Changes");
    expect(markup).toContain("src/model.ts");
    expect(markup).not.toContain(">DVC<");
  });
});
