import { describe, expect, it } from "vitest";
import {
  parseDependencyInstallReceipt,
  parseManagedDependency,
  parseManagedDependencyManifest,
  planDependencyAction,
  resolveManagedDependency,
  selectDependencyPlatform,
} from "../src/dependencies.js";

const SHA256 = "a".repeat(64);

function managedRipgrep(overrides: Record<string, unknown> = {}) {
  return {
    id: "ripgrep",
    kind: "managed-binary",
    owner: "server",
    version: "14.1.1",
    version_source: "fixed",
    install_root: "$GEEPILOT_SERVER_DATA_ROOT/bin",
    platforms: {
      "darwin-aarch64": {
        url: "https://downloads.example.invalid/ripgrep-14.1.1-aarch64-apple-darwin.tar.gz",
        sha256: SHA256,
        archive_member: "ripgrep-14.1.1-aarch64-apple-darwin/rg",
        target_name: "rg",
        executable: true,
      },
    },
    ...overrides,
  };
}

describe("managed dependency primitives", () => {
  it("parses a GEEPilot-style manifest and selects platform entries", () => {
    const manifest = parseManagedDependencyManifest({
      schema: 1,
      dependencies: [managedRipgrep()],
    });

    expect(manifest.schemaVersion).toBe(1);
    expect(manifest.dependencies[0]).toMatchObject({
      id: "ripgrep",
      versionSource: "fixed",
      installRoot: "$GEEPILOT_SERVER_DATA_ROOT/bin",
    });
    expect(resolveManagedDependency(manifest, "ripgrep").version).toBe("14.1.1");
    expect(selectDependencyPlatform(manifest.dependencies[0], "darwin-aarch64")).toMatchObject({
      sha256: SHA256,
      archiveMember: "ripgrep-14.1.1-aarch64-apple-darwin/rg",
      targetName: "rg",
    });
  });

  it("rejects invalid managed download policy before installation", () => {
    expect(() =>
      parseManagedDependencyManifest({
        schema: 1,
        dependencies: [
          managedRipgrep({
            platforms: {
              "darwin-aarch64": {
                url: "https://downloads.example.invalid/rg.tar.gz",
                archive_member: "rg",
              },
            },
          }),
        ],
      }),
    ).toThrow(/sha256/);

    expect(() =>
      parseManagedDependencyManifest({
        schema: 1,
        dependencies: [managedRipgrep({ version: "latest" })],
      }),
    ).toThrow(/exact version/);

    expect(() =>
      parseManagedDependencyManifest({
        schema: 1,
        dependencies: [
          managedRipgrep({
            platforms: {
              "darwin-aarch64": {
                url: "https://user:password@downloads.example.invalid/rg.tar.gz",
                sha256: SHA256,
                archive_member: "../rg",
                target_name: "bin/rg",
              },
            },
          }),
        ],
      }),
    ).toThrow(/embedded credentials|archive_member|target_name/);
  });

  it("rejects duplicate ids and inline secret-looking metadata", () => {
    expect(() =>
      parseManagedDependencyManifest({
        schema: 1,
        dependencies: [managedRipgrep(), managedRipgrep()],
      }),
    ).toThrow(/Duplicate dependency id/);

    expect(() =>
      parseManagedDependencyManifest({
        schema: 1,
        dependencies: [managedRipgrep({ apiKey: "sk-test" })],
      }),
    ).toThrow(/inline secret field.*apiKey/);
  });

  it("plans side-effect-free actions from detection state and dependency kind", () => {
    const ripgrep = parseManagedDependency(managedRipgrep());

    expect(
      planDependencyAction(ripgrep, {
        detection: {
          dependency_id: "ripgrep",
          status: "detected",
          source: "path",
          path: "/opt/homebrew/bin/rg",
          version: "14.1.1",
        },
      }),
    ).toMatchObject({
      action: "use_existing",
      dependencyId: "ripgrep",
    });

    expect(
      planDependencyAction(ripgrep, {
        platformKey: "darwin-aarch64",
        detection: {
          dependency_id: "ripgrep",
          status: "missing",
          source: "missing",
        },
      }),
    ).toMatchObject({
      action: "install_managed",
      platform: "darwin-aarch64",
      selectedPlatform: { sha256: SHA256 },
    });

    expect(
      planDependencyAction(ripgrep, {
        detection: {
          dependency_id: "ripgrep",
          status: "failed",
          source: "managed",
          message: "receipt hash mismatch",
        },
      }),
    ).toMatchObject({
      action: "unavailable",
      reason: "receipt hash mismatch",
    });
  });

  it("requires explicit user action for external harness CLIs", () => {
    const codexCli = parseManagedDependency({
      id: "codex-cli",
      kind: "external-harness-cli",
      owner: "harness",
      version: "vendor-managed",
      version_source: "external-vendor",
      trust_model: "OpenAI vendor installer or user package manager",
    });

    expect(
      planDependencyAction(codexCli, {
        detection: {
          dependency_id: "codex-cli",
          status: "missing",
          source: "missing",
        },
      }),
    ).toMatchObject({
      action: "requires_user_action",
      reason: expect.stringContaining("external harness vendor"),
    });
  });

  it("validates install receipts without allowing credentials", () => {
    expect(
      parseDependencyInstallReceipt({
        dependency_id: "ripgrep",
        version: "14.1.1",
        platform: "darwin-aarch64",
        source_url: "https://downloads.example.invalid/rg.tar.gz",
        sha256: SHA256,
        installed_path: "$GEEPILOT_SERVER_DATA_ROOT/bin/rg",
        installed_at: "2026-07-03T00:00:00.000Z",
        swarmx_version: "3.0.0",
      }),
    ).toMatchObject({
      dependencyId: "ripgrep",
      sourceUrl: "https://downloads.example.invalid/rg.tar.gz",
      installedPath: "$GEEPILOT_SERVER_DATA_ROOT/bin/rg",
    });

    expect(() =>
      parseDependencyInstallReceipt({
        dependency_id: "ripgrep",
        version: "14.1.1",
        platform: "darwin-aarch64",
        source_url: "https://user:password@downloads.example.invalid/rg.tar.gz",
        sha256: SHA256,
        installed_path: "$GEEPILOT_SERVER_DATA_ROOT/bin/rg",
        installed_at: "2026-07-03T00:00:00.000Z",
      }),
    ).toThrow(/URLs must not contain embedded credentials/);

    expect(() =>
      parseDependencyInstallReceipt({
        dependency_id: "ripgrep",
        version: "14.1.1",
        platform: "darwin-aarch64",
        source_url: "https://downloads.example.invalid/rg.tar.gz",
        sha256: SHA256,
        installed_path: "$GEEPILOT_SERVER_DATA_ROOT/bin/rg",
        installed_at: "2026-07-03T00:00:00.000Z",
        bearerToken: "secret-token",
      }),
    ).toThrow(/inline secret field.*bearerToken/);
  });
});
