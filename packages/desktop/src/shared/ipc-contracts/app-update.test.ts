import { describe, expect, it } from "vitest";
import {
  AppUpdateEventContracts,
  AppUpdateInvokeContracts,
  DesktopUpdateStateSchema,
} from "./app-update.js";
import { composeContractMaps } from "./base.js";

describe("App Update IPC contracts", () => {
  it("owns the invoke argument, result, event, and audit contracts", () => {
    expect(Object.keys(AppUpdateInvokeContracts)).toEqual([
      "appUpdate:getState",
      "appUpdate:install",
    ]);
    expect(AppUpdateInvokeContracts["appUpdate:getState"].audit).toBe("failure_only");
    expect(AppUpdateInvokeContracts["appUpdate:install"].audit).toBe("intent_outcome");
    expect(AppUpdateInvokeContracts["appUpdate:getState"].args.parse([])).toEqual([]);
    expect(() =>
      AppUpdateInvokeContracts["appUpdate:getState"].args.parse(["unexpected"]),
    ).toThrow();
    expect(AppUpdateEventContracts["appUpdate:state"].payload).toBe(DesktopUpdateStateSchema);
  });

  it("accepts bounded states and rejects malformed or widened transport data", () => {
    expect(
      DesktopUpdateStateSchema.parse({
        phase: "downloading",
        currentVersion: "3.2.0",
        latestVersion: "3.3.0",
        progress: 42,
      }),
    ).toEqual({
      phase: "downloading",
      currentVersion: "3.2.0",
      latestVersion: "3.3.0",
      progress: 42,
    });
    expect(() =>
      DesktopUpdateStateSchema.parse({ phase: "unknown", currentVersion: "3.2.0" }),
    ).toThrow();
    expect(() =>
      DesktopUpdateStateSchema.parse({
        phase: "hidden",
        currentVersion: "3.2.0",
        unexpected: true,
      }),
    ).toThrow();
  });

  it("rejects duplicate feature channels during registry composition", () => {
    expect(() =>
      composeContractMaps(AppUpdateInvokeContracts, {
        "appUpdate:getState": AppUpdateInvokeContracts["appUpdate:getState"],
      }),
    ).toThrow(/duplicated/i);
  });
});
