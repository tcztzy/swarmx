import { describe, expect, it } from "vitest";
import { stableHash, stableJson } from "../src/canonical-json.js";
import { findInlineSecretFields, isForbiddenSecretKey } from "../src/secret-scanner.js";

describe("canonical JSON", () => {
  it("keeps stable key ordering, undefined filtering, and hashing", () => {
    const value = { nested: { z: 1, a: true }, omitted: undefined, first: "value" };

    expect(stableJson(value)).toBe('{"first":"value","nested":{"a":true,"z":1}}');
    expect(stableHash("SwarmX")).toBe("466ae01b839931a5");
  });
});

describe("inline secret scanning", () => {
  it("uses one recursive policy while allowing references and redacted metadata", () => {
    expect(
      findInlineSecretFields({
        apiToken: "plain",
        nested: [{ authToken: "plain" }],
        secretRef: { source: "env", key: "OPENAI_API_KEY" },
        credentialRefs: ["provider:openai"],
        password: "[redacted]",
      }),
    ).toEqual([
      { key: "apiToken", path: ["apiToken"] },
      { key: "authToken", path: ["nested", 0, "authToken"] },
    ]);
    expect(isForbiddenSecretKey("hostLogin")).toBe(true);
  });

  it("supports explicit vault paths without weakening other metadata", () => {
    expect(
      findInlineSecretFields(
        {
          value: "allowed vault value",
          metadata: { bearerToken: "plain", password: "[redacted]" },
        },
        { allowRedacted: false, skippedPathPrefixes: [["value"]] },
      ),
    ).toEqual([
      { key: "bearerToken", path: ["metadata", "bearerToken"] },
      { key: "password", path: ["metadata", "password"] },
    ]);
  });
});
