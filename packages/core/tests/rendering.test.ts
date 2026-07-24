import { describe, expect, it } from "vitest";
import {
  normalizeMessageChunk,
  normalizeMessageChunks,
  parseNormalizedRenderEvent,
  sanitizeRenderPayload,
} from "../src/rendering.js";

describe("normalized render events", () => {
  it("normalizes message chunks into deterministic render events", () => {
    const first = normalizeMessageChunk({
      role: "assistant",
      kind: "message",
      content: "Use `HF_HOME` for cache.",
      agent: "analysis_lead",
    });
    const second = normalizeMessageChunk({
      role: "assistant",
      kind: "message",
      content: "Use `HF_HOME` for cache.",
      agent: "analysis_lead",
    });

    expect(first.eventId).toBe(second.eventId);
    expect(first).toMatchObject({
      kind: "message",
      status: "completed",
      source: "swarmx.message",
      agent: "analysis_lead",
      title: "Message from analysis_lead",
      content: "Use `HF_HOME` for cache.",
      provenance: { agent: "analysis_lead" },
    });
  });

  it("redacts tool-call inputs and keeps raw payloads behind references", () => {
    const event = normalizeMessageChunk(
      {
        role: "assistant",
        kind: "tool_call",
        content: JSON.stringify({
          command: "curl",
          apiKey: "sk-test",
          nested: { password: "secret" },
          secretRef: "OPENAI_API_KEY",
        }),
        agent: "runner",
        toolName: "terminal",
      },
      {
        rawPayloadRef: "run_1/tool_call_1.json",
        provenance: { host: "codex", adapter: "acp", pluginId: "geepilot" },
      },
    );

    expect(event.status).toBe("running");
    expect(event.title).toBe("Tool call: terminal");
    expect(event.rawPayloadRef).toBe("run_1/tool_call_1.json");
    expect(event.input).toEqual({
      command: "curl",
      apiKey: "[redacted]",
      nested: { password: "[redacted]" },
      secretRef: "OPENAI_API_KEY",
    });
    expect(JSON.stringify(event)).not.toContain("sk-test");
    expect(JSON.stringify(event)).not.toContain('secret"}');
  });

  it("V502 normalizes request-scoped tool progress as running output", () => {
    const event = normalizeMessageChunk(
      {
        role: "tool",
        kind: "tool_progress",
        content: "building\n",
        toolName: "exec_command",
        structuredContent: { output: "building\n", stream: "stdout", mode: "append" },
      },
      { invocationId: "call_build" },
    );

    expect(event).toMatchObject({
      invocationId: "call_build",
      kind: "tool_progress",
      status: "running",
      title: "Tool progress: exec_command",
      output: { output: "building\n", stream: "stdout", mode: "append" },
    });
  });

  it("keeps Agent provenance on Harness x Model identity", () => {
    expect(
      parseNormalizedRenderEvent({
        eventId: "rne_agent_identity",
        kind: "message",
        status: "completed",
        source: "test",
        title: "Agent message",
        summary: "Agent message",
        provenance: {
          harnessId: "codex",
          modelId: "gpt-5",
          modelSupplyId: "openai-gpt-5",
        },
      }).provenance,
    ).toMatchObject({ harnessId: "codex", modelId: "gpt-5" });

    expect(() =>
      parseNormalizedRenderEvent({
        eventId: "rne_legacy_provider_identity",
        kind: "message",
        status: "completed",
        source: "test",
        title: "Agent message",
        summary: "Agent message",
        provenance: {
          harnessId: "codex",
          modelId: "gpt-5",
          providerProfileId: "openai",
        },
      }),
    ).toThrow(/providerProfileId.*invalid/);
  });

  it("carries caller-supplied artifact refs without copying unsafe metadata", () => {
    const event = normalizeMessageChunk(
      {
        role: "tool",
        kind: "tool_result",
        content: JSON.stringify({ status: "succeeded", summary: "wrote report" }),
        agent: "runner",
        toolName: "writer",
      },
      {
        artifacts: [
          {
            artifactId: "art_report",
            kind: "report",
            title: "Report",
            path: "autonomy/reports/report.html",
            byteCount: 1200,
          },
        ],
        rawPayloadRef: "run_1/tool_result_1.json",
      },
    );

    expect(event.artifacts).toEqual([
      expect.objectContaining({
        artifactId: "art_report",
        kind: "report",
        path: "autonomy/reports/report.html",
      }),
    ]);
    expect(event.rawPayloadRef).toBe("run_1/tool_result_1.json");

    expect(() =>
      normalizeMessageChunk(
        {
          role: "tool",
          kind: "tool_result",
          content: "{}",
          toolName: "writer",
        },
        {
          artifacts: [{ kind: "json", metadata: { apiKey: "sk-test" } }],
        },
      ),
    ).toThrow(/inline secret field.*apiKey/);
  });

  it("marks failed tool results from structured and textual failures", () => {
    const structured = normalizeMessageChunk({
      role: "tool",
      kind: "tool_result",
      content: JSON.stringify({ status: "failed", error: "exit 1" }),
      agent: "runner",
      toolName: "pnpm",
    });
    const textual = normalizeMessageChunk({
      role: "tool",
      kind: "tool_result",
      content: "Command failed with exit code 1",
      agent: "runner",
      toolName: "pnpm",
    });

    expect(structured.status).toBe("failed");
    expect(structured.output).toEqual({ status: "failed", error: "exit 1" });
    expect(textual.status).toBe("failed");
    expect(textual.output).toEqual({ text: "Command failed with exit code 1" });
  });

  it("V501 prioritizes structured exit status over ordinary output text", () => {
    const succeeded = normalizeMessageChunk({
      role: "tool",
      kind: "tool_result",
      content: "Process exited with code 0\nOutput:\nFailed telemetry sends",
      structuredContent: {
        exit_code: 0,
        output: "Failed telemetry sends",
      },
      agent: "runner",
      toolName: "exec_command",
    });
    const failed = normalizeMessageChunk({
      role: "tool",
      kind: "tool_result",
      content: "Process exited with code 7\nOutput:\nNo diagnostic text",
      structuredContent: {
        exit_code: 7,
        output: "No diagnostic text",
      },
      agent: "runner",
      toolName: "exec_command",
    });

    expect(succeeded.status).toBe("succeeded");
    expect(failed.status).toBe("failed");
  });

  it("V363 renders structured tool content instead of model-facing text", () => {
    const event = normalizeMessageChunk({
      role: "tool",
      kind: "tool_result",
      content: "File updated successfully.",
      structuredContent: {
        filePath: "README.md",
        userModified: false,
      },
      toolName: "Write",
    });

    expect(event.output).toEqual({ filePath: "README.md", userModified: false });
    expect(event.summary).toContain("README.md");
  });

  it("normalizes arrays with stable raw payload references", () => {
    const events = normalizeMessageChunks([
      { role: "assistant", kind: "thinking", content: "Planning" },
      { role: "assistant", kind: "message", content: "Done" },
    ]);

    expect(events.map((event) => event.rawPayloadRef)).toEqual([
      "message-chunk:0",
      "message-chunk:1",
    ]);
    expect(events.map((event) => event.kind)).toEqual(["thinking", "message"]);
  });

  it("rejects unredacted render-event secrets but accepts sanitized payloads", () => {
    expect(sanitizeRenderPayload({ apiKey: "sk-test", value: 1 })).toEqual({
      apiKey: "[redacted]",
      value: 1,
    });

    expect(() =>
      parseNormalizedRenderEvent({
        eventId: "rne_bad_secret",
        kind: "artifact",
        status: "completed",
        source: "test",
        title: "Artifact",
        summary: "Unsafe",
        artifacts: [{ kind: "json", metadata: { apiKey: "sk-test" } }],
      }),
    ).toThrow(/inline secret field.*apiKey/);

    expect(
      parseNormalizedRenderEvent({
        eventId: "rne_safe_artifact",
        kind: "artifact",
        status: "completed",
        source: "test",
        title: "Artifact",
        summary: "Safe",
        artifacts: [{ kind: "json", metadata: { apiKey: "[redacted]" } }],
      }).artifacts[0]?.metadata,
    ).toEqual({ apiKey: "[redacted]" });
  });
});
