import { describe, expect, it } from "vitest";
import {
  assertSecretRefPolicy,
  evaluateSecretFileMode,
  parseSecretRef,
  parseSecretStatus,
  parseSecretVaultDocument,
  parseSecretWriteRequest,
  redactSecretVaultDocument,
  redactSecretWriteRequest,
  secretStatusFromVault,
  secretValueFromVault,
} from "../src/secrets.js";

const vault = {
  schema_version: 1,
  service: "SwarmX",
  file_mode: "0600",
  secrets: {
    OPENAI_API_KEY: {
      value: "sk-runtime",
      purpose: "provider_api_key",
      updated_at: "2026-07-03T00:00:00.000Z",
      metadata: {
        providerProfileId: "openai",
      },
    },
  },
};

describe("secret primitives", () => {
  it("parses vault documents, looks up runtime values, and redacts safely", () => {
    const parsed = parseSecretVaultDocument(vault);

    expect(parsed).toMatchObject({
      schemaVersion: 1,
      service: "SwarmX",
      fileMode: "0600",
      secrets: {
        OPENAI_API_KEY: {
          value: "sk-runtime",
          purpose: "provider_api_key",
          updatedAt: "2026-07-03T00:00:00.000Z",
        },
      },
    });
    expect(
      secretValueFromVault({
        vault,
        ref: {
          source: "local_auth_file",
          key: "OPENAI_API_KEY",
          purpose: "provider_api_key",
        },
      }),
    ).toBe("sk-runtime");

    const redacted = redactSecretVaultDocument(vault);
    expect(redacted.secrets.OPENAI_API_KEY?.value).toBe("[redacted]");
    expect(JSON.stringify(redacted)).not.toContain("sk-runtime");
  });

  it("reports secret status without returning values", () => {
    const status = secretStatusFromVault({
      vault,
      ref: {
        source: "local_auth_file",
        key: "OPENAI_API_KEY",
        purpose: "provider_api_key",
      },
      checkedAt: "2026-07-03T00:00:00.000Z",
    });

    expect(status).toMatchObject({
      ref: {
        source: "local_auth_file",
        key: "OPENAI_API_KEY",
        purpose: "provider_api_key",
      },
      configured: true,
      checkedAt: "2026-07-03T00:00:00.000Z",
    });
    expect(JSON.stringify(status)).not.toContain("sk-runtime");

    expect(() =>
      parseSecretStatus({
        ref: { source: "local_auth_file", key: "OPENAI_API_KEY" },
        configured: true,
        value: "sk-runtime",
      }),
    ).toThrow(/must not include secret values/);
  });

  it("evaluates restrictive secret file modes without touching files", () => {
    expect(evaluateSecretFileMode({ mode: "0600", path: "~/.swarmx/auth.json" })).toMatchObject({
      path: "~/.swarmx/auth.json",
      mode: "0600",
      expectedMode: "0600",
      secure: true,
    });
    expect(evaluateSecretFileMode({ mode: 0o600 })).toMatchObject({
      mode: "0600",
      secure: true,
    });
    expect(evaluateSecretFileMode({ mode: "0644" })).toMatchObject({
      mode: "0644",
      expectedMode: "0600",
      secure: false,
    });
    expect(evaluateSecretFileMode({})).toMatchObject({
      expectedMode: "0600",
      secure: false,
    });
  });

  it("allows values only in explicit write requests and redacts them", () => {
    const request = parseSecretWriteRequest({
      ref: {
        source: "server_keychain",
        key: "OPENAI_API_KEY",
        purpose: "provider_api_key",
      },
      value: "sk-write",
      actor: "tester",
      requested_at: "2026-07-03T00:00:00.000Z",
    });

    expect(request).toMatchObject({
      ref: {
        source: "server_keychain",
        key: "OPENAI_API_KEY",
        purpose: "provider_api_key",
      },
      value: "sk-write",
      requestedAt: "2026-07-03T00:00:00.000Z",
    });
    expect(redactSecretWriteRequest(request).value).toBe("[redacted]");

    expect(() =>
      parseSecretWriteRequest({
        ref: { source: "local_auth_file", key: "OPENAI_API_KEY" },
        value: "sk-write",
        apiKey: "sk-duplicate",
      }),
    ).toThrow(/inline secret field.*apiKey/);
  });

  it("enforces request-only policy for remote-compute passwords", () => {
    expect(
      parseSecretRef({
        source: "prompt",
        key: "tuqiao-password",
        purpose: "remote_compute_password",
      }),
    ).toMatchObject({
      source: "prompt",
      purpose: "remote_compute_password",
    });
    expect(
      assertSecretRefPolicy({
        source: "request_only",
        key: "tuqiao-password",
        purpose: "remote_compute_password",
      }),
    ).toMatchObject({
      source: "request_only",
    });
    expect(() =>
      parseSecretRef({
        source: "local_auth_file",
        key: "tuqiao-password",
        purpose: "remote_compute_password",
      }),
    ).toThrow(/request-only/);
  });

  it("rejects secret-looking metadata outside vault and write values", () => {
    expect(() =>
      parseSecretVaultDocument({
        schemaVersion: 1,
        service: "SwarmX",
        secrets: {
          OPENAI_API_KEY: {
            value: "sk-runtime",
            metadata: {
              bearerToken: "bad-token",
            },
          },
        },
      }),
    ).toThrow(/inline secret field.*bearerToken/);
    expect(() =>
      parseSecretRef({
        source: "env",
        key: "OPENAI_API_KEY",
        apiKey: "sk-test",
      }),
    ).toThrow(/inline secret field.*apiKey/);
  });
});
