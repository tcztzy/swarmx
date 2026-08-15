import { describe, expect, it } from "vitest";
import {
  type ProtectedSandboxProfile,
  ProtectedSandboxProfileSchema,
  ProtectedSandboxRegistry,
} from "./sandbox-policy.js";

const profile: ProtectedSandboxProfile = {
  id: "codex",
  image: "node:22-slim",
  imageDigest: "sha256:253da19867dd03e2f817f433d7782adefd2a2bac8729fcd4ebc6770665167a24",
  command: ["npx", "--yes", "@agentclientprotocol/codex-acp@1.1.2"],
  environmentAllowlist: ["OPENAI_API_KEY", "LANG"],
  mounts: [
    { source: "project", access: "rw" },
    { source: "temporary", access: "rw" },
    { source: "credential", access: "ro" },
  ],
  network: "none",
  resources: { cpu: 2, memoryMiB: 4096, temporaryMiB: 1024 },
};

describe("ProtectedSandboxRegistry", () => {
  it("validates host profiles and keeps them immutable", () => {
    const registry = new ProtectedSandboxRegistry([profile]);
    const registered = registry.get("codex");

    expect(registered).toMatchObject({
      imageDigest: profile.imageDigest,
      network: "none",
      resources: profile.resources,
    });
    expect(() => registry.register(profile)).toThrow(/already registered/);
    if (!registered) throw new Error("Expected the profile to be registered.");
    const firstMount = registered.mounts[0];
    if (!firstMount) throw new Error("Expected a Project mount.");
    expect(() => {
      Object.assign(firstMount, { source: "credential" });
    }).toThrow();
  });

  it("fails protected resolution without a profile or runtime and never downgrades", () => {
    const registry = new ProtectedSandboxRegistry();

    expect(
      registry.resolve({
        strategy: "protected_required",
        profileId: "missing",
        runtimeReady: true,
      }),
    ).toMatchObject({ mode: "protected", ready: false, profileId: "missing" });
    expect(
      new ProtectedSandboxRegistry([profile]).resolve({
        strategy: "protected_required",
        profileId: "codex",
        runtimeReady: false,
        runtimeId: "apple_container",
      }),
    ).toMatchObject({ mode: "protected", ready: false, runtimeId: "apple_container" });
  });

  it("reports native mode only for an explicit native_allowed strategy", () => {
    const resolution = new ProtectedSandboxRegistry([profile]).resolve({
      strategy: "native_allowed",
      profileId: "codex",
      runtimeReady: false,
    });

    expect(resolution).toMatchObject({
      strategy: "native_allowed",
      mode: "native",
      ready: true,
    });
  });

  it("rejects profiles that weaken the fixed mount contract", () => {
    expect(() =>
      ProtectedSandboxProfileSchema.parse({
        ...profile,
        mounts: [
          { source: "project", access: "ro" },
          { source: "temporary", access: "rw" },
        ],
      }),
    ).toThrow(/writable Project mount/);
  });
});
