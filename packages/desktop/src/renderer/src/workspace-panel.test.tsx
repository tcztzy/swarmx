/** @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  type BrowserState,
  WorkspacePanel,
  type WorkspacePanelApi,
  parseUnifiedPatch,
} from "./workspace-panel.js";

vi.mock("@xterm/xterm", () => ({ Terminal: class MockTerminal {} }));
vi.mock("@xterm/addon-fit", () => ({ FitAddon: class MockFitAddon {} }));

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("WorkspacePanel", () => {
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
          status: "M",
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
