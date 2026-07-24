import { describe, expect, it } from "vitest";
import { classifyProviderError, providerErrorMessage } from "./provider-error.js";

describe("Provider error presentation", () => {
  it("V506 classifies only allowlisted transient Provider signatures", () => {
    expect(
      classifyProviderError("Our servers are currently overloaded. Please try again later."),
    ).toMatchObject({
      type: "provider_error",
      code: "overloaded",
      retryable: true,
    });
    expect(classifyProviderError("HTTP 429: Too many requests")).toMatchObject({
      code: "rate_limited",
    });
    expect(classifyProviderError("upstream request timed out")).toMatchObject({
      code: "temporarily_unavailable",
    });

    expect(classifyProviderError("workspace_shell timed out after 30 seconds")).toBeNull();
    expect(classifyProviderError("Project sandbox is unavailable")).toBeNull();
    expect(classifyProviderError("Unknown harness: local-agent")).toBeNull();
  });

  it("V507 creates a fixed-copy message without retaining the raw Provider error", () => {
    const message = providerErrorMessage(
      "overloaded_error request=req_private_123 bearer=provider_secret_456",
    );

    expect(message).toMatchObject({
      role: "system",
      kind: "message",
      render: { source: "provider", status: "failed" },
      structuredContent: {
        type: "provider_error",
        code: "overloaded",
        title: "Provider is temporarily busy",
      },
    });
    expect(JSON.stringify(message)).not.toContain("req_private_123");
    expect(JSON.stringify(message)).not.toContain("provider_secret_456");
  });
});
