import { describe, expect, it, vi } from "vitest";

const hooks = vi.hoisted(() => ({
  layoutEffect: vi.fn(),
  passiveEffect: vi.fn(),
}));

vi.mock("react", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react")>()),
  useEffect: hooks.passiveEffect,
  useLayoutEffect: hooks.layoutEffect,
  useRef: () => ({ current: null }),
  useState: () => [false, vi.fn()],
}));

describe("T30 JupyterLab output lifecycle", () => {
  it("V78 detaches the Lumino widget in the layout cleanup before React removes its host", async () => {
    vi.stubGlobal("window", {});
    const { NotebookOutputArea } = await import("../src/client/notebook-output.js");

    NotebookOutputArea({ outputs: [] });

    expect(hooks.layoutEffect).toHaveBeenCalledOnce();
    expect(hooks.passiveEffect).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });
});
