import { describe, expect, it } from "vitest";
import {
  BrowserEventContracts,
  BrowserInvokeContracts,
  DESKTOP_BROWSER_LIMITS,
  DesktopBrowserStateSchema,
} from "./browser.js";
import { DesktopEventContractRegistry, DesktopInvokeContractRegistry } from "./index.js";

const state = {
  id: "browser-1",
  url: "https://example.com/",
  title: "Example",
  loading: false,
  canGoBack: false,
  canGoForward: true,
};

describe("Browser IPC contracts", () => {
  it("owns exactly eight invokes and one owner-scoped event with stable policies", () => {
    expect(Object.keys(BrowserInvokeContracts)).toEqual([
      "browser:create",
      "browser:navigate",
      "browser:back",
      "browser:forward",
      "browser:reload",
      "browser:setBounds",
      "browser:setVisible",
      "browser:destroy",
    ]);
    expect(
      Object.fromEntries(
        Object.entries(BrowserInvokeContracts).map(([channel, contract]) => [
          channel,
          contract.audit,
        ]),
      ),
    ).toEqual({
      "browser:create": "intent_outcome",
      "browser:navigate": "intent_outcome",
      "browser:back": "intent_outcome",
      "browser:forward": "intent_outcome",
      "browser:reload": "intent_outcome",
      "browser:setBounds": "failure_only",
      "browser:setVisible": "failure_only",
      "browser:destroy": "intent_outcome",
    });
    expect(Object.keys(BrowserEventContracts)).toEqual(["browser:state"]);
    expect(
      Object.keys(DesktopInvokeContractRegistry).filter((channel) =>
        channel.startsWith("browser:"),
      ),
    ).toEqual(Object.keys(BrowserInvokeContracts));
    expect(
      Object.keys(DesktopEventContractRegistry).filter((channel) => channel.startsWith("browser:")),
    ).toEqual(Object.keys(BrowserEventContracts));
  });

  it("preserves optional create and Host-owned address and bounds normalization", () => {
    expect(BrowserInvokeContracts["browser:create"].args.parse([])).toEqual([]);
    expect(BrowserInvokeContracts["browser:create"].args.parse([undefined])).toEqual([undefined]);
    expect(BrowserInvokeContracts["browser:create"].args.parse([{}])).toEqual([{}]);
    expect(
      BrowserInvokeContracts["browser:create"].args.parse([
        {
          id: "  ",
          url: "plain search text",
          bounds: { x: -1.5, y: 100_001.9, width: 20_000.2, height: 0 },
          visible: false,
        },
      ]),
    ).toEqual([
      {
        id: "  ",
        url: "plain search text",
        bounds: { x: -1.5, y: 100_001.9, width: 20_000.2, height: 0 },
        visible: false,
      },
    ]);
    expect(
      BrowserInvokeContracts["browser:navigate"].args.parse([
        { id: "browser-1", url: "release notes" },
      ]),
    ).toEqual([{ id: "browser-1", url: "release notes" }]);
    expect(
      BrowserInvokeContracts["browser:setBounds"].args.parse([
        {
          id: "browser-1",
          bounds: {
            x: Number.NaN,
            y: Number.POSITIVE_INFINITY,
            width: Number.NEGATIVE_INFINITY,
            height: Number.NaN,
          },
        },
      ]),
    ).toEqual([
      {
        id: "browser-1",
        bounds: {
          x: Number.NaN,
          y: Number.POSITIVE_INFINITY,
          width: Number.NEGATIVE_INFINITY,
          height: Number.NaN,
        },
      },
    ]);
  });

  it("rejects malformed tuples and strict request or receipt drift", () => {
    expect(BrowserInvokeContracts["browser:create"].args.safeParse([{}, {}]).success).toBe(false);
    expect(
      BrowserInvokeContracts["browser:navigate"].args.safeParse([
        { id: "browser-1", url: "https://example.com", rawCredential: "secret" },
      ]).success,
    ).toBe(false);
    expect(
      BrowserInvokeContracts["browser:setBounds"].args.safeParse([
        {
          id: "browser-1",
          bounds: { x: "0", y: 0, width: 100, height: 100 },
        },
      ]).success,
    ).toBe(false);
    expect(
      BrowserInvokeContracts["browser:setBounds"].result.safeParse({
        updated: true,
        ignored: true,
      }).success,
    ).toBe(false);
    expect(
      BrowserInvokeContracts["browser:destroy"].result.safeParse({ destroyed: "yes" }).success,
    ).toBe(false);
  });

  it("validates strict state results and events while allowing initial empty text", () => {
    expect(DesktopBrowserStateSchema.parse(state)).toEqual(state);
    expect(
      BrowserEventContracts["browser:state"].payload.parse({
        ...state,
        url: "",
        title: "",
        error: "Page failed to load.",
      }),
    ).toMatchObject({ url: "", title: "", error: "Page failed to load." });
    expect(DesktopBrowserStateSchema.safeParse({ ...state, hostOnly: true }).success).toBe(false);
    expect(DesktopBrowserStateSchema.safeParse({ ...state, loading: "false" }).success).toBe(false);
    expect(DesktopBrowserStateSchema.safeParse({ ...state, error: "" }).success).toBe(false);
  });

  it("enforces byte bounds on identities, inputs, and Renderer-controlled state", () => {
    const over = (bytes: number) => "x".repeat(bytes + 1);
    expect(
      BrowserInvokeContracts["browser:create"].args.safeParse([
        { id: over(DESKTOP_BROWSER_LIMITS.idBytes) },
      ]).success,
    ).toBe(false);
    expect(
      BrowserInvokeContracts["browser:navigate"].args.safeParse([
        { id: "browser-1", url: over(DESKTOP_BROWSER_LIMITS.inputBytes) },
      ]).success,
    ).toBe(false);
    expect(
      DesktopBrowserStateSchema.safeParse({
        ...state,
        url: over(DESKTOP_BROWSER_LIMITS.urlBytes),
      }).success,
    ).toBe(false);
    expect(
      DesktopBrowserStateSchema.safeParse({
        ...state,
        title: over(DESKTOP_BROWSER_LIMITS.titleBytes),
      }).success,
    ).toBe(false);
    expect(
      DesktopBrowserStateSchema.safeParse({
        ...state,
        error: over(DESKTOP_BROWSER_LIMITS.errorBytes),
      }).success,
    ).toBe(false);
  });
});
