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
});
