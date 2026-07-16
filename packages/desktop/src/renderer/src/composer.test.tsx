/** @vitest-environment jsdom */

import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import type React from "react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Composer } from "./composer.js";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("Composer", () => {
  it("grows to its content height without exposing the native resize handle", () => {
    render(<TestComposer />);
    const input = screen.getByRole("textbox");
    Object.defineProperty(input, "scrollHeight", { configurable: true, value: 92 });

    fireEvent.change(input, { target: { value: "A longer prompt" } });

    expect(input.style.height).toBe("92px");
    expect(input.className).toContain("composer__textarea");
  });

  it("opens a context picker before adding system-selected paths", async () => {
    const selectFilesAndFolders = vi.fn(async () => ["/workspace/src/App.tsx", "/tmp/My Note.md"]);
    render(<TestComposer selectFilesAndFolders={selectFilesAndFolders} />);
    const input = screen.getByRole<HTMLTextAreaElement>("textbox");

    fireEvent.click(screen.getByRole("button", { name: "Add context" }));

    expect(input.value).toBe("");
    expect(selectFilesAndFolders).not.toHaveBeenCalled();
    expect(screen.getByText("Add")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Files and folders" }));
    await act(async () => undefined);
    expect(selectFilesAndFolders).toHaveBeenCalledTimes(1);
    expect(input.value).toBe('@/workspace/src/App.tsx @"/tmp/My Note.md"');
  });

  it("uses bare @ references for local-file completions", async () => {
    vi.useFakeTimers();
    const completeMention = vi.fn(async () => ({
      result: {
        items: [
          {
            label: "@src/",
            detail: "Workspace folder",
            textEdit: { newText: "@src/" },
          },
        ],
      },
    }));
    render(<TestComposer completeMention={completeMention} />);
    const input = screen.getByRole<HTMLTextAreaElement>("textbox");

    fireEvent.change(input, { target: { value: "@" } });
    input.setSelectionRange(1, 1);
    fireEvent.select(input);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });
    expect(completeMention).not.toHaveBeenCalled();
    const bareMentionSuggestions = screen.getByLabelText("Mention suggestions");
    expect(within(bareMentionSuggestions).queryByText("@src/")).toBeNull();

    fireEvent.change(input, { target: { value: "@s" } });
    input.setSelectionRange(2, 2);
    fireEvent.select(input);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });

    expect(completeMention).toHaveBeenCalledWith(
      expect.objectContaining({
        serverId: "swarmx.local-files",
        workspaceRoot: "/workspace",
        position: { line: 0, character: 2 },
      }),
    );
    expect(screen.getByText("@src/")).toBeTruthy();

    fireEvent.keyDown(input, { key: "Enter" });
    expect(input.value).toBe("@src/");
  });

  it("queries and inserts skill options for a bare $ token", async () => {
    vi.useFakeTimers();
    const completeMention = vi.fn(async () => ({
      result: {
        items: [
          {
            label: "$geepilot.memory",
            detail: "Skill Memory",
            textEdit: { newText: "$geepilot.memory" },
          },
        ],
      },
    }));
    render(<TestComposer completeMention={completeMention} />);
    const input = screen.getByRole<HTMLTextAreaElement>("textbox");

    fireEvent.change(input, { target: { value: "$" } });
    input.setSelectionRange(1, 1);
    fireEvent.select(input);
    expect(screen.getByRole("status").textContent).toContain("Loading options");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });

    expect(completeMention).toHaveBeenCalledWith(
      expect.objectContaining({
        serverId: "swarmx.skills",
        triggerCharacter: "$",
        position: { line: 0, character: 1 },
      }),
    );
    expect(screen.getByText("$geepilot.memory")).toBeTruthy();

    fireEvent.keyDown(input, { key: "Enter" });
    expect(input.value).toBe("$geepilot.memory");
  });
});

function TestComposer({
  completeMention = vi.fn(async () => ({ result: { items: [] } })),
  selectFilesAndFolders = vi.fn(async () => []),
}: {
  completeMention?: ReturnType<typeof vi.fn>;
  selectFilesAndFolders?: ReturnType<typeof vi.fn>;
}): React.JSX.Element {
  const [value, setValue] = useState("");
  return (
    <Composer
      value={value}
      placeholder="Message SwarmX"
      disabled={false}
      running={false}
      sendDisabled={false}
      workspaceRoot="/workspace"
      mentionServers={[
        {
          id: "swarmx.local-files",
          name: "Files and folders",
          mentionPrefixes: ["@"],
        },
        { id: "swarmx.skills", name: "Skills", mentionPrefixes: ["$"] },
      ]}
      completeMention={completeMention}
      selectFilesAndFolders={selectFilesAndFolders}
      onChange={setValue}
      onSubmit={() => undefined}
      onStop={() => undefined}
    >
      <span>Model</span>
    </Composer>
  );
}
