import { describe, expect, it } from "vitest";
import { DesktopEventContractRegistry, DesktopInvokeContractRegistry } from "./index.js";
import {
  DesktopTerminalDataEventSchema,
  DesktopTerminalExitEventSchema,
  TerminalEventContracts,
  TerminalInvokeContracts,
} from "./terminal.js";

describe("Terminal IPC contracts", () => {
  it("owns exactly four semantic-only invokes and two owner-scoped events", () => {
    expect(Object.keys(TerminalInvokeContracts)).toEqual([
      "terminal:create",
      "terminal:write",
      "terminal:resize",
      "terminal:kill",
    ]);
    expect(
      Object.fromEntries(
        Object.entries(TerminalInvokeContracts).map(([channel, contract]) => [
          channel,
          contract.audit,
        ]),
      ),
    ).toEqual({
      "terminal:create": "semantic_only",
      "terminal:write": "semantic_only",
      "terminal:resize": "semantic_only",
      "terminal:kill": "semantic_only",
    });
    expect(Object.keys(TerminalEventContracts)).toEqual(["terminal:data", "terminal:exit"]);
    expect(
      Object.keys(DesktopInvokeContractRegistry).filter((channel) =>
        channel.startsWith("terminal:"),
      ),
    ).toEqual(Object.keys(TerminalInvokeContracts));
    expect(
      Object.keys(DesktopEventContractRegistry).filter((channel) =>
        channel.startsWith("terminal:"),
      ),
    ).toEqual(Object.keys(TerminalEventContracts));
  });

  it("preserves Host-owned semantic rejection and dimension normalization", () => {
    expect(
      TerminalInvokeContracts["terminal:create"].args.parse([
        {
          id: "  ",
          cwd: "  ",
          cols: Number.NaN,
          rows: Number.POSITIVE_INFINITY,
        },
      ]),
    ).toEqual([
      {
        id: "  ",
        cwd: "  ",
        cols: Number.NaN,
        rows: Number.POSITIVE_INFINITY,
      },
    ]);
    expect(
      TerminalInvokeContracts["terminal:resize"].args.parse([
        {
          id: "terminal-1",
          cols: Number.NEGATIVE_INFINITY,
          rows: Number.NaN,
        },
      ]),
    ).toEqual([
      {
        id: "terminal-1",
        cols: Number.NEGATIVE_INFINITY,
        rows: Number.NaN,
      },
    ]);
    expect(
      TerminalInvokeContracts["terminal:write"].args.safeParse([
        { id: "terminal-1", data: "x".repeat(1024 * 1024 + 1) },
      ]).success,
    ).toBe(true);
  });

  it("rejects missing tuple items, wrong types, unknown fields, and receipt drift", () => {
    expect(TerminalInvokeContracts["terminal:create"].args.safeParse([]).success).toBe(false);
    expect(
      TerminalInvokeContracts["terminal:create"].args.safeParse([{ cwd: "/workspace" }]).success,
    ).toBe(false);
    expect(
      TerminalInvokeContracts["terminal:write"].args.safeParse([
        { id: "terminal-1", data: "secret", rawCredential: "not transported" },
      ]).success,
    ).toBe(false);
    expect(
      TerminalInvokeContracts["terminal:resize"].args.safeParse([
        { id: "terminal-1", cols: "80", rows: 24 },
      ]).success,
    ).toBe(false);
    expect(
      TerminalInvokeContracts["terminal:create"].result.safeParse({
        id: "terminal-1",
        pid: 42,
        cwd: "/secret",
      }).success,
    ).toBe(false);
    expect(
      TerminalInvokeContracts["terminal:create"].result.safeParse({
        id: "terminal-1",
        pid: Number.NaN,
      }).success,
    ).toBe(false);
    expect(
      TerminalInvokeContracts["terminal:create"].result.safeParse({ id: "", pid: 42 }).success,
    ).toBe(false);
    expect(
      TerminalInvokeContracts["terminal:kill"].result.safeParse({ killed: "yes" }).success,
    ).toBe(false);
  });

  it("validates strict data and exit events without interpreting terminal content", () => {
    expect(
      DesktopTerminalDataEventSchema.parse({ id: "terminal-1", data: "\u001b[31mraw\u001b[0m" }),
    ).toEqual({
      id: "terminal-1",
      data: "\u001b[31mraw\u001b[0m",
    });
    expect(
      DesktopTerminalExitEventSchema.parse({ id: "terminal-1", exitCode: 0, signal: 1 }),
    ).toEqual({ id: "terminal-1", exitCode: 0, signal: 1 });
    expect(
      TerminalEventContracts["terminal:exit"].payload.parse({
        id: "terminal-1",
        exitCode: 0,
      }),
    ).toEqual({ id: "terminal-1", exitCode: 0 });
    expect(
      DesktopTerminalDataEventSchema.safeParse({
        id: "terminal-1",
        data: "raw",
        cwd: "/secret",
      }).success,
    ).toBe(false);
    expect(
      DesktopTerminalExitEventSchema.safeParse({ id: "terminal-1", exitCode: "0" }).success,
    ).toBe(false);
    expect(
      DesktopTerminalExitEventSchema.safeParse({
        id: "terminal-1",
        exitCode: Number.POSITIVE_INFINITY,
      }).success,
    ).toBe(false);
  });
});
