import { describe, expect, it } from "vitest";
import { preflightExtensionComposition } from "../src/extension-composition.js";

const emptyCapabilities = {
  software: [],
  skills: [],
  mcpServers: [],
  models: [],
  modelSupplies: [],
  providers: [],
  harnesses: [],
  agents: [],
  appConnectors: [],
  uiContributions: [],
  commands: [],
  lspServers: [],
  hooks: [],
  monitors: [],
  outputStyles: [],
  settings: [],
  assets: [],
  permissions: [],
  authPolicies: [],
  marketplaceSources: [],
  pluginCatalog: [],
};

function bundle(
  id: string,
  composition: Record<string, unknown> = {},
  capabilities: Record<string, unknown> = {},
) {
  return {
    schemaVersion: 1,
    id,
    name: id,
    version: "1.0.0",
    trust: "verified",
    source: { type: "marketplace", marketplace: "official", package: id },
    integrity: `sha256:${id}`,
    composition,
    capabilities: { ...emptyCapabilities, ...capabilities },
  };
}

describe("Extension composition preflight", () => {
  it("closes dependencies and returns one deterministic, explained load order", () => {
    const input = {
      selectedExtensionIds: ["writer"],
      bundles: [
        bundle("writer", {
          phase: "runtime",
          provides: ["feature:writer"],
          requires: ["feature:foundation"],
        }),
        bundle("foundation", {
          phase: "preflight",
          provides: ["feature:foundation"],
        }),
      ],
    };

    const first = preflightExtensionComposition(input);
    const second = preflightExtensionComposition({
      ...input,
      bundles: [...input.bundles].reverse(),
    });

    expect(first).toEqual(second);
    expect(first).toMatchObject({
      status: "ready",
      loadOrder: ["foundation", "writer"],
      sideEffects: [],
      extensions: [
        { id: "foundation", loadReason: "required_by:writer" },
        { id: "writer", loadReason: "selected" },
      ],
    });
  });

  it.each([
    {
      name: "missing dependency",
      bundles: [bundle("a", { provides: ["feature:a"], requires: ["feature:missing"] })],
      selectedExtensionIds: ["a"],
      code: "missing_dependency",
    },
    {
      name: "dependency cycle",
      bundles: [
        bundle("a", { provides: ["feature:a"], requires: ["feature:b"] }),
        bundle("b", { provides: ["feature:b"], requires: ["feature:a"] }),
      ],
      selectedExtensionIds: ["a"],
      code: "dependency_cycle",
    },
    {
      name: "duplicate capability",
      bundles: [
        bundle("a", { provides: ["feature:shared"] }),
        bundle("b", { provides: ["feature:shared"] }),
      ],
      selectedExtensionIds: ["a", "b"],
      code: "duplicate_capability",
    },
    {
      name: "duplicate tool",
      bundles: [bundle("a", { tools: ["Read"] }), bundle("b", { tools: ["Read"] })],
      selectedExtensionIds: ["a", "b"],
      code: "duplicate_tool",
    },
    {
      name: "provider conflict",
      bundles: [
        bundle("a", {}, { providers: [{ id: "gateway", label: "A", kind: "openai" }] }),
        bundle("b", {}, { providers: [{ id: "gateway", label: "B", kind: "openai" }] }),
      ],
      selectedExtensionIds: ["a", "b"],
      code: "provider_conflict",
    },
    {
      name: "declared conflict",
      bundles: [
        bundle("a", { provides: ["feature:a"], conflicts: ["feature:b"] }),
        bundle("b", { provides: ["feature:b"] }),
      ],
      selectedExtensionIds: ["a", "b"],
      code: "explicit_conflict",
    },
    {
      name: "missing order target",
      bundles: [bundle("a", { after: ["not-installed"] })],
      selectedExtensionIds: ["a"],
      code: "missing_order_target",
    },
    {
      name: "ambiguous sensitive order",
      bundles: [
        bundle("a", { phase: "runtime", orderSensitive: true }),
        bundle("b", { phase: "runtime", orderSensitive: true }),
      ],
      selectedExtensionIds: ["a", "b"],
      code: "ambiguous_order",
    },
    {
      name: "protected kernel replacement",
      bundles: [bundle("a", { provides: ["kernel:audit-policy"] })],
      selectedExtensionIds: ["a"],
      code: "protected_capability",
    },
    {
      name: "invalid runtime phase",
      bundles: [bundle("a", { phase: "after-start" })],
      selectedExtensionIds: ["a"],
      code: "invalid_phase",
    },
  ])("blocks $name with an actionable issue", ({ bundles, selectedExtensionIds, code }) => {
    const result = preflightExtensionComposition({ bundles, selectedExtensionIds });
    expect(result.status).toBe("blocked");
    expect(result.issues).toContainEqual(
      expect.objectContaining({ severity: "error", code, message: expect.any(String) }),
    );
  });

  it("rejects an untrusted sensitive permission even when a grant is present", () => {
    const extension = {
      ...bundle(
        "networker",
        {},
        {
          permissions: [
            {
              id: "network:any",
              kind: "network",
              access: "network",
              required: true,
            },
          ],
        },
      ),
      trust: "untrusted",
    };
    const result = preflightExtensionComposition({
      bundles: [extension],
      selectedExtensionIds: ["networker"],
      installedExtensions: [
        {
          pluginId: "networker",
          name: "networker",
          state: "enabled",
          enabled: true,
          trust: "untrusted",
          currentRevision: {
            revisionId: "networker@1",
            version: "1.0.0",
            contentDigest: "sha256:networker",
            sourceId: "official",
          },
          requestedPermissionIds: ["network:any"],
          grantedPermissionIds: ["network:any"],
        },
      ],
    });

    expect(result.status).toBe("blocked");
    expect(result.issues.map((issue) => issue.code)).toContain("untrusted_sensitive_permission");
  });

  it("does not let manifest trust claims authorize an executable bundle without installation", () => {
    for (const trust of ["local", "verified", "builtin"] as const) {
      const result = preflightExtensionComposition({
        bundles: [
          {
            ...bundle(
              `forged-${trust}`,
              {},
              {
                commands: [{ id: "run", name: "Run", command: ["forged-command"] }],
              },
            ),
            trust,
          },
        ],
        selectedExtensionIds: [`forged-${trust}`],
      });

      expect(result.status).toBe("blocked");
      expect(result.extensions[0]).toMatchObject({ trust: "untrusted" });
      expect(result.issues).toContainEqual(
        expect.objectContaining({
          code: "executable_not_installed",
          extensionId: `forged-${trust}`,
        }),
      );
    }
  });

  it("allows an executable bundle only when host observation and installed authority agree", () => {
    const bundleInput = bundle(
      "approved",
      {},
      {
        commands: [{ id: "run", name: "Run", command: ["approved-command"] }],
      },
    );
    const approved = {
      ...bundleInput,
      hostObservation: {
        source: { type: "path", locator: "/extensions/approved/extension.json" },
        contentDigest: "sha256:approved-content",
      },
    };
    const result = preflightExtensionComposition({
      bundles: [approved],
      selectedExtensionIds: ["approved"],
      installedExtensions: [
        {
          pluginId: "approved",
          name: "approved",
          state: "enabled",
          enabled: true,
          trust: "verified",
          currentRevision: {
            revisionId: "approved@1",
            version: "1.0.0",
            contentDigest: "sha256:approved-content",
            sourceId: "local-source",
          },
          requestedPermissionIds: [],
          grantedPermissionIds: [],
        },
      ],
    });

    expect(result).toMatchObject({
      status: "ready",
      extensions: [
        {
          id: "approved",
          trust: "verified",
          source: { type: "path", locator: "/extensions/approved/extension.json" },
          integrity: "sha256:approved-content",
        },
      ],
    });
  });

  it("blocks an executable bundle whose installed revision is disabled", () => {
    const approved = {
      ...bundle(
        "disabled-executable",
        {},
        { commands: [{ id: "run", name: "Run", command: ["run"] }] },
      ),
      hostObservation: {
        source: { type: "path", locator: "/extensions/disabled/extension.json" },
        contentDigest: "sha256:disabled-content",
      },
    };
    const result = preflightExtensionComposition({
      bundles: [approved],
      selectedExtensionIds: ["disabled-executable"],
      installedExtensions: [
        {
          pluginId: "disabled-executable",
          name: "disabled-executable",
          state: "disabled",
          enabled: true,
          trust: "verified",
          currentRevision: {
            revisionId: "disabled-executable@1",
            version: "1.0.0",
            contentDigest: "sha256:disabled-content",
            sourceId: "local-source",
          },
          requestedPermissionIds: [],
          grantedPermissionIds: [],
        },
      ],
    });

    expect(result.status).toBe("blocked");
    expect(result.issues.map((issue) => issue.code)).toEqual(
      expect.arrayContaining(["extension_disabled", "executable_not_installed"]),
    );
  });

  it("blocks executable dependencies introduced by the transitive capability closure", () => {
    const result = preflightExtensionComposition({
      bundles: [
        bundle(
          "consumer",
          { requires: ["feature:foundation"] },
          { commands: [{ id: "consumer-run", name: "Consumer", command: ["consumer"] }] },
        ),
        bundle(
          "foundation",
          { provides: ["feature:foundation"] },
          { lspServers: [{ id: "foundation-lsp", command: ["foundation-lsp"] }] },
        ),
      ],
      selectedExtensionIds: ["consumer"],
    });

    expect(result.status).toBe("blocked");
    expect(result.extensions.map((extension) => extension.id)).toEqual(["foundation", "consumer"]);
    expect(result.issues.filter((issue) => issue.code === "executable_not_installed")).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ extensionId: "foundation" }),
        expect.objectContaining({ extensionId: "consumer" }),
      ]),
    );
  });

  it.each([
    [
      "custom Harness",
      {
        harnesses: [
          {
            id: "harness",
            label: "Harness",
            modelControl: "direct",
            modelCompatibility: "any",
            backend: { type: "custom", program: "harness" },
          },
        ],
      },
    ],
    ["stdio MCP", { mcpServers: [{ id: "mcp", server: { type: "stdio", command: "mcp" } }] }],
    ["LSP", { lspServers: [{ id: "lsp", command: ["lsp"] }] }],
    ["Hook", { hooks: [{ id: "hook", event: "onStart", command: ["hook"] }] }],
    ["Command", { commands: [{ id: "command", name: "Command", command: ["command"] }] }],
    [
      "Software command",
      { software: [{ id: "software", name: "Software", command: ["software"] }] },
    ],
    [
      "connector entrypoint",
      {
        appConnectors: [
          { id: "connector", name: "Connector", kind: "test", entrypoint: "connector" },
        ],
      },
    ],
  ])("recognizes %s as executable authority", (_name, capabilities) => {
    const id = `executable-${String(_name)
      .replaceAll(/[^A-Za-z0-9]+/gu, "-")
      .toLowerCase()}`;
    const result = preflightExtensionComposition({
      bundles: [bundle(id, {}, capabilities)],
      selectedExtensionIds: [id],
    });

    expect(result.status).toBe("blocked");
    expect(result.issues).toContainEqual(
      expect.objectContaining({ code: "executable_not_installed", extensionId: id }),
    );
  });

  it("blocks a missing required grant and previews requested authority", () => {
    const result = preflightExtensionComposition({
      bundles: [
        bundle(
          "writer",
          {},
          {
            permissions: [
              { id: "project:write", kind: "filesystem", access: "write", required: true },
            ],
          },
        ),
      ],
      selectedExtensionIds: ["writer"],
    });

    expect(result).toMatchObject({
      status: "blocked",
      extensions: [
        {
          id: "writer",
          permissions: {
            requested: ["project:write"],
            granted: [],
            missing: ["project:write"],
          },
        },
      ],
    });
    expect(result.issues.map((issue) => issue.code)).toContain("permission_missing");
  });

  it("does not turn an external Harness's native permissions into SwarmX grants", () => {
    const external = bundle(
      "external",
      {},
      {
        harnesses: [
          {
            id: "external",
            label: "External",
            modelControl: "session",
            modelCompatibility: "any",
            backend: { type: "custom", program: "external-agent" },
          },
        ],
      },
    );
    const result = preflightExtensionComposition({
      bundles: [
        {
          ...external,
          hostObservation: {
            source: { type: "path", locator: "/extensions/external/extension.json" },
            contentDigest: "sha256:external-content",
          },
        },
      ],
      selectedExtensionIds: ["external"],
      installedExtensions: [
        {
          pluginId: "external",
          name: "external",
          state: "enabled",
          enabled: true,
          trust: "verified",
          currentRevision: {
            revisionId: "external@1",
            version: "1.0.0",
            contentDigest: "sha256:external-content",
            sourceId: "local-source",
          },
          requestedPermissionIds: [],
          grantedPermissionIds: [],
        },
      ],
    });

    expect(result.status).toBe("ready");
    expect(result.extensions[0]?.permissions.requested).toEqual([]);
  });

  it("is a pure read-only projection", () => {
    const input = {
      bundles: [bundle("a")],
      selectedExtensionIds: ["a"],
    };
    const before = structuredClone(input);

    expect(preflightExtensionComposition(input).sideEffects).toEqual([]);
    expect(input).toEqual(before);
  });
});
