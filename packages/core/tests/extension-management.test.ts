import { describe, expect, it } from "vitest";
import {
  ExtensionLifecycleManager,
  ExtensionMarketplaceCatalogSchema,
  ExtensionMarketplaceSourceSchema,
  planExtensionAction,
} from "../src/extension-management.js";

const revision1 = {
  revisionId: "paper-tools@1.0.0",
  version: "1.0.0",
  contentDigest: "sha256:one",
  sourceId: "official",
};
const revision2 = {
  revisionId: "paper-tools@1.1.0",
  version: "1.1.0",
  contentDigest: "sha256:two",
  sourceId: "official",
};
const candidate = {
  pluginId: "paper-tools",
  name: "Paper tools",
  trust: "verified" as const,
  revision: revision2,
};

describe("Extension lifecycle management", () => {
  it("accepts credential-free HTTPS catalogs and rejects unsafe remote sources", () => {
    expect(
      ExtensionMarketplaceSourceSchema.parse({
        id: "official",
        name: "Official",
        kind: "remote_catalog",
        location: "https://plugins.swarmx.dev/catalog.json",
        trust: "verified",
      }),
    ).toMatchObject({ enabled: true, trust: "verified" });
    expect(() =>
      ExtensionMarketplaceSourceSchema.parse({
        id: "bad",
        name: "Bad",
        kind: "remote_catalog",
        location: "http://user:password@example.test/catalog.json",
      }),
    ).toThrow(/HTTPS/);
  });

  it("normalizes plugins and entries into one secret-safe marketplace candidate list", () => {
    expect(
      ExtensionMarketplaceCatalogSchema.parse({ schemaVersion: 1, plugins: [candidate] }),
    ).toMatchObject({ candidates: [candidate] });
    expect(() =>
      ExtensionMarketplaceCatalogSchema.parse({
        schemaVersion: 1,
        plugins: [{ ...candidate, accessToken: "do-not-store" }],
      }),
    ).toThrow(/secret/i);
  });

  it("requires explicit confirmation for install, update, trust, uninstall, and rollback", () => {
    expect(
      planExtensionAction({ action: "install", pluginId: "paper-tools", candidate }),
    ).toMatchObject({ allowed: false, requiresConfirmation: true });
    expect(
      planExtensionAction({
        action: "install",
        pluginId: "paper-tools",
        candidate,
        confirmed: true,
      }),
    ).toMatchObject({ allowed: true, targetRevision: revision2 });
  });

  it("updates from upstream without destroying the previous immutable revision", () => {
    const manager = new ExtensionLifecycleManager(
      [
        {
          pluginId: "paper-tools",
          name: "Paper tools",
          state: "enabled",
          enabled: true,
          trust: "verified",
          currentRevision: revision1,
        },
      ],
      () => "2026-07-14T10:00:00.000Z",
    );
    const receipt = manager.apply({
      action: "update",
      pluginId: "paper-tools",
      candidate,
      confirmed: true,
    });

    expect(receipt).toMatchObject({ status: "applied", after: { currentRevision: revision2 } });
    expect(receipt.after?.previousRevisions).toEqual([revision1]);
  });

  it("rolls back to a retained revision and blocks updates while pinned", () => {
    const manager = new ExtensionLifecycleManager(
      [
        {
          pluginId: "paper-tools",
          name: "Paper tools",
          state: "pinned",
          enabled: true,
          trust: "verified",
          currentRevision: revision2,
          previousRevisions: [revision1],
          pinnedRevisionId: revision2.revisionId,
        },
      ],
      () => "2026-07-14T10:00:00.000Z",
    );

    expect(
      manager.plan({ action: "update", pluginId: "paper-tools", candidate, confirmed: true }),
    ).toMatchObject({ allowed: false, reason: expect.stringMatching(/Pinned/) });
    expect(
      manager.apply({ action: "rollback", pluginId: "paper-tools", confirmed: true }),
    ).toMatchObject({
      status: "applied",
      after: {
        state: "pinned",
        pinnedRevisionId: revision1.revisionId,
        currentRevision: revision1,
      },
    });
  });

  it("rejects inline credentials at every action boundary", () => {
    expect(() =>
      planExtensionAction({
        action: "install",
        pluginId: "paper-tools",
        candidate,
        confirmed: true,
        apiKey: "do-not-store",
      }),
    ).toThrow(/secret/i);
  });

  it("requires confirmation and trusted state for permission expansion", () => {
    const installed = {
      pluginId: "paper-tools",
      name: "Paper tools",
      state: "enabled" as const,
      enabled: true,
      trust: "verified" as const,
      currentRevision: revision1,
      requestedPermissionIds: ["project:read", "project:write"],
      grantedPermissionIds: ["project:read"],
    };

    expect(
      planExtensionAction(
        {
          action: "grant_permissions",
          pluginId: "paper-tools",
          permissionIds: ["project:read", "project:write"],
        },
        installed,
      ),
    ).toMatchObject({ allowed: false, requiresConfirmation: true, authorityChange: "expand" });
    expect(
      planExtensionAction(
        {
          action: "grant_permissions",
          pluginId: "paper-tools",
          permissionIds: ["project:read", "project:write"],
          confirmed: true,
        },
        installed,
      ),
    ).toMatchObject({ allowed: true, authorityChange: "expand" });
    expect(
      planExtensionAction(
        {
          action: "grant_permissions",
          pluginId: "paper-tools",
          permissionIds: ["project:write"],
          confirmed: true,
        },
        { ...installed, trust: "untrusted" },
      ),
    ).toMatchObject({ allowed: false, reason: expect.stringMatching(/trusted/i) });
  });

  it("applies permission reduction without expansion confirmation", () => {
    const manager = new ExtensionLifecycleManager([
      {
        pluginId: "paper-tools",
        name: "Paper tools",
        state: "enabled",
        enabled: true,
        trust: "verified",
        currentRevision: revision1,
        requestedPermissionIds: ["project:read", "project:write"],
        grantedPermissionIds: ["project:read", "project:write"],
      },
    ]);

    expect(
      manager.apply({
        action: "grant_permissions",
        pluginId: "paper-tools",
        permissionIds: ["project:read"],
      }),
    ).toMatchObject({
      status: "applied",
      after: { grantedPermissionIds: ["project:read"] },
    });
  });

  it("revokes trust, disables execution, and clears grants", () => {
    const manager = new ExtensionLifecycleManager([
      {
        pluginId: "paper-tools",
        name: "Paper tools",
        state: "enabled",
        enabled: true,
        trust: "verified",
        currentRevision: revision1,
        requestedPermissionIds: ["project:read"],
        grantedPermissionIds: ["project:read"],
      },
    ]);

    expect(
      manager.apply({ action: "revoke_trust", pluginId: "paper-tools", confirmed: true }),
    ).toMatchObject({
      status: "applied",
      after: { trust: "untrusted", enabled: false, grantedPermissionIds: [] },
    });
  });

  it("never lets candidate metadata silently raise trust or permissions", () => {
    const manager = new ExtensionLifecycleManager([
      {
        pluginId: "paper-tools",
        name: "Paper tools",
        state: "enabled",
        enabled: true,
        trust: "local",
        currentRevision: revision1,
        requestedPermissionIds: ["project:read"],
        grantedPermissionIds: ["project:read"],
      },
    ]);

    const receipt = manager.apply({
      action: "update",
      pluginId: "paper-tools",
      candidate: {
        ...candidate,
        trust: "verified",
        requestedPermissionIds: ["project:read", "project:write"],
      },
      confirmed: true,
    });
    expect(receipt.after).toMatchObject({
      trust: "local",
      requestedPermissionIds: ["project:read", "project:write"],
      grantedPermissionIds: ["project:read"],
    });
  });

  it("fails closed before authority expansion when audit intent cannot be written", () => {
    const audit = () => {
      throw new Error("audit unavailable");
    };
    const manager = new ExtensionLifecycleManager(
      [
        {
          pluginId: "paper-tools",
          name: "Paper tools",
          state: "enabled",
          enabled: true,
          trust: "verified",
          currentRevision: revision1,
          requestedPermissionIds: ["project:read"],
          grantedPermissionIds: [],
        },
      ],
      () => "2026-07-14T10:00:00.000Z",
      audit,
    );

    expect(() =>
      manager.apply({
        action: "grant_permissions",
        pluginId: "paper-tools",
        permissionIds: ["project:read"],
        confirmed: true,
      }),
    ).toThrow("audit unavailable");
    expect(manager.list()[0]?.grantedPermissionIds).toEqual([]);
  });

  it("fails closed when authority expansion has no audit intent writer", () => {
    const manager = new ExtensionLifecycleManager([
      {
        pluginId: "paper-tools",
        name: "Paper tools",
        state: "enabled",
        enabled: true,
        trust: "verified",
        currentRevision: revision1,
        requestedPermissionIds: ["project:read"],
        grantedPermissionIds: [],
      },
    ]);

    expect(() =>
      manager.apply({
        action: "grant_permissions",
        pluginId: "paper-tools",
        permissionIds: ["project:read"],
        confirmed: true,
      }),
    ).toThrow(/audit intent/i);
    expect(manager.list()[0]?.grantedPermissionIds).toEqual([]);
  });

  it("rejects Extension-authored trust and credential-policy changes", () => {
    expect(
      planExtensionAction(
        {
          action: "trust",
          pluginId: "paper-tools",
          actor: "extension",
          confirmed: true,
        },
        {
          pluginId: "paper-tools",
          name: "Paper tools",
          state: "disabled",
          enabled: false,
          trust: "untrusted",
          currentRevision: revision1,
        },
      ),
    ).toMatchObject({ allowed: false, reason: expect.stringMatching(/Extension/i) });
    expect(() =>
      planExtensionAction({
        action: "grant_permissions",
        pluginId: "paper-tools",
        confirmed: true,
        approvalPolicy: "allow_all",
      }),
    ).toThrow(/secret|unrecognized|policy/i);
  });

  it("keeps built-in trust kernel-owned", () => {
    expect(
      planExtensionAction({
        action: "install",
        pluginId: "forged-builtin",
        confirmed: true,
        candidate: {
          ...candidate,
          pluginId: "forged-builtin",
          trust: "builtin",
          revision: { ...candidate.revision, revisionId: "forged@1" },
        },
      }),
    ).toMatchObject({ allowed: false, reason: expect.stringMatching(/kernel-owned/i) });
    expect(
      planExtensionAction(
        { action: "trust", pluginId: "builtin", confirmed: true },
        {
          pluginId: "builtin",
          name: "Built in",
          state: "enabled",
          enabled: true,
          trust: "builtin",
          currentRevision: revision1,
        },
      ),
    ).toMatchObject({ allowed: false, reason: expect.stringMatching(/kernel-owned/i) });
  });
});
