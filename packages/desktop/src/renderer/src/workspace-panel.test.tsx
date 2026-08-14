/** @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, expectTypeOf, it, vi } from "vitest";
import type { SwarmxAPI } from "../../shared/desktop-api.js";
import {
  type BrowserState,
  parseUnifiedPatch,
  WorkspacePanel,
  type WorkspacePanelApi,
} from "./workspace-panel.js";

const workspaceTerminal = vi.hoisted(() => {
  const instances: MockTerminal[] = [];
  class MockTerminal {
    cols = 80;
    rows = 24;
    options: { theme?: unknown } = {};
    readonly loadAddon = vi.fn();
    readonly open = vi.fn();
    readonly write = vi.fn();
    readonly writeln = vi.fn();
    readonly focus = vi.fn();
    readonly reset = vi.fn();
    readonly dispose = vi.fn();

    constructor() {
      instances.push(this);
    }

    onData() {
      return { dispose: vi.fn() };
    }
  }
  return { instances, MockTerminal };
});
const workspaceFit = vi.hoisted(() => ({
  MockFitAddon: class MockFitAddon {
    readonly fit = vi.fn();
  },
}));

vi.mock("@xterm/xterm", () => ({ Terminal: workspaceTerminal.MockTerminal }));
vi.mock("@xterm/addon-fit", () => ({ FitAddon: workspaceFit.MockFitAddon }));

beforeEach(() => {
  workspaceTerminal.instances.length = 0;
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
    callback(0);
    return 1;
  });
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("WorkspacePanel", () => {
  it("accepts the canonical desktop API without a duplicate Workspace facade", () => {
    expectTypeOf<SwarmxAPI>().toMatchTypeOf<WorkspacePanelApi>();
  });

  it("opens the four workspace tools from the equal-split launcher", async () => {
    const api = createApi();
    const user = userEvent.setup();
    render(<WorkspacePanel api={api} cwd="/workspace/swarmx" onClose={vi.fn()} />);

    const launcher = screen.getByRole("navigation", { name: "Open workspace tool" });
    expect(within(launcher).getByRole("button", { name: /Review/ })).toBeTruthy();
    expect(within(launcher).getByRole("button", { name: /Terminal/ })).toBeTruthy();
    expect(within(launcher).getByRole("button", { name: /Browser/ })).toBeTruthy();
    expect(within(launcher).getByRole("button", { name: /Files/ })).toBeTruthy();

    await user.click(within(launcher).getByRole("button", { name: /Review/ }));
    await waitFor(() => expect(api.getWorkspaceReview).toHaveBeenCalledWith("/workspace/swarmx"));
    expect(await screen.findByText("src/App.tsx")).toBeTruthy();
    expect(screen.getByText("const next = true;")).toBeTruthy();
    expect(screen.getAllByText("+1")).toHaveLength(2);
    expect(screen.getAllByText("−1")).toHaveLength(2);

    await user.click(screen.getByRole("tab", { name: "Files" }));
    await waitFor(() =>
      expect(api.listWorkspaceDirectory).toHaveBeenCalledWith("", "/workspace/swarmx"),
    );
    await user.click(await screen.findByRole("button", { name: /src/ }));
    expect(api.listWorkspaceDirectory).toHaveBeenLastCalledWith("src", "/workspace/swarmx");
    await user.click(await screen.findByRole("button", { name: /App.tsx/ }));
    expect(api.readWorkspaceFile).toHaveBeenCalledWith("src/App.tsx", "/workspace/swarmx");
    expect(await screen.findByText("export function App() {}")).toBeTruthy();

    await user.click(screen.getByRole("tab", { name: "Browser" }));
    await waitFor(() => expect(api.createBrowser).toHaveBeenCalledTimes(1));
    expect(screen.getByRole("textbox", { name: "Address or search" })).toBeTruthy();
  }, 10_000);

  it("supports the Review keyboard shortcut without eagerly loading other tools", async () => {
    const api = createApi();
    render(<WorkspacePanel api={api} cwd="/workspace/swarmx" onClose={vi.fn()} />);

    fireEvent.keyDown(window, { key: "G", ctrlKey: true, shiftKey: true });

    await waitFor(() => expect(api.getWorkspaceReview).toHaveBeenCalledTimes(1));
    expect(api.createBrowser).not.toHaveBeenCalled();
    expect(api.createTerminal).not.toHaveBeenCalled();
    expect(screen.getByRole("tab", { name: "Review" }).getAttribute("aria-selected")).toBe("true");
  });

  it("preserves a Terminal across tool switches and replaces it when cwd changes", async () => {
    const api = createApi();
    const user = userEvent.setup();
    const view = render(<WorkspacePanel api={api} cwd="/workspace/a" onClose={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: /Terminal/ }));
    await waitFor(() => expect(api.createTerminal).toHaveBeenCalledOnce());
    const firstId = api.createTerminal.mock.calls[0]?.[0].id as string;
    expect(api.createTerminal.mock.calls[0]?.[0]).toMatchObject({ cwd: "/workspace/a" });

    await user.click(screen.getByRole("tab", { name: "Review" }));
    await user.click(screen.getByRole("tab", { name: "Terminal" }));
    expect(api.createTerminal).toHaveBeenCalledOnce();
    expect(api.killTerminal).not.toHaveBeenCalled();

    view.rerender(<WorkspacePanel api={api} cwd="/workspace/b" onClose={vi.fn()} />);
    await waitFor(() => expect(api.createTerminal).toHaveBeenCalledTimes(2));
    expect(api.killTerminal).toHaveBeenCalledWith(firstId);
    expect(api.createTerminal.mock.calls[1]?.[0]).toMatchObject({ cwd: "/workspace/b" });
  });
});

describe("parseUnifiedPatch", () => {
  it("tracks old and new line numbers for unified Git hunks", () => {
    const [hunk] = parseUnifiedPatch(
      [
        "@@ -10,2 +10,2 @@ function run()",
        " const old = false;",
        "-return old;",
        "+return true;",
      ].join("\n"),
    );

    expect(hunk?.lines).toEqual([
      {
        id: "context:10:10:0",
        kind: "context",
        marker: " ",
        content: "const old = false;",
        oldLine: 10,
        newLine: 10,
      },
      {
        id: "deletion:11:1",
        kind: "deletion",
        marker: "-",
        content: "return old;",
        oldLine: 11,
      },
      {
        id: "addition:11:2",
        kind: "addition",
        marker: "+",
        content: "return true;",
        newLine: 11,
      },
    ]);
  });
});

function createApi(): WorkspacePanelApi & Record<string, ReturnType<typeof vi.fn>> {
  const browserState: BrowserState = {
    id: "browser-1",
    url: "https://www.google.com",
    title: "Google",
    loading: false,
    canGoBack: false,
    canGoForward: false,
  };
  return {
    getWorkspaceReview: vi.fn(async () => ({
      root: "/workspace/swarmx",
      branch: "main",
      isRepository: true,
      truncated: false,
      files: [
        {
          path: "src/App.tsx",
          status: " M",
          binary: false,
          additions: 1,
          deletions: 1,
          truncated: false,
          patch: [
            "diff --git a/src/App.tsx b/src/App.tsx",
            "@@ -1 +1 @@",
            "-const next = false;",
            "+const next = true;",
          ].join("\n"),
        },
      ],
    })),
    listWorkspaceDirectory: vi.fn(async (path = "") => ({
      root: "/workspace/swarmx",
      path,
      truncated: false,
      entries:
        path === "src"
          ? [{ name: "App.tsx", path: "src/App.tsx", kind: "file" as const, size: 24 }]
          : [{ name: "src", path: "src", kind: "directory" as const }],
    })),
    readWorkspaceFile: vi.fn(async (path: string) => ({
      root: "/workspace/swarmx",
      path,
      content: "export function App() {}",
      size: 24,
      binary: false,
      truncated: false,
    })),
    createTerminal: vi.fn(async ({ id }: { id: string }) => ({ id, pid: 42 })),
    writeTerminal: vi.fn(async () => ({ written: true })),
    resizeTerminal: vi.fn(async () => ({ resized: true })),
    killTerminal: vi.fn(async () => ({ killed: true })),
    onTerminalData: vi.fn(() => () => undefined),
    onTerminalExit: vi.fn(() => () => undefined),
    createBrowser: vi.fn(async () => browserState),
    navigateBrowser: vi.fn(async () => browserState),
    backBrowser: vi.fn(async () => browserState),
    forwardBrowser: vi.fn(async () => browserState),
    reloadBrowser: vi.fn(async () => browserState),
    setBrowserBounds: vi.fn(async () => ({ updated: true })),
    setBrowserVisible: vi.fn(async () => ({ updated: true })),
    destroyBrowser: vi.fn(async () => ({ destroyed: true })),
    onBrowserState: vi.fn(() => () => undefined),
  };
}
