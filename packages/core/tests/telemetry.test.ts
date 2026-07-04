import { describe, expect, it, vi } from "vitest";
import {
  buildTelemetryEnvelope,
  createTelemetryClient,
  createTelemetryIngestHandler,
  evaluateTelemetryIngest,
  parseTelemetryIngestAcceptedRecord,
  parseTelemetryIngestDecision,
  resolveTelemetryConfig,
  resolveTelemetryIngestConfig,
  sanitizeTelemetryPayload,
  telemetryHeaders,
  telemetryStatus,
} from "../src/telemetry.js";
import type { TelemetryEnvelope } from "../src/telemetry.js";

describe("telemetry primitives", () => {
  it("keeps telemetry disabled until explicit opt-in and endpoint are configured", () => {
    expect(resolveTelemetryConfig({ env: {} })).toMatchObject({
      enabled: false,
      source: "swarmx",
    });
    expect(
      resolveTelemetryConfig({
        env: {
          SWARMX_TELEMETRY_ENABLED: "true",
          SWARMX_TELEMETRY_URL: "https://telemetry.example/events",
          SWARMX_TELEMETRY_TOKEN: "client-token",
          SWARMX_INSTALLATION_ID: "install-1",
        },
      }),
    ).toMatchObject({
      enabled: true,
      url: "https://telemetry.example/events",
      token: "client-token",
      installationId: "install-1",
    });
    expect(
      telemetryStatus({
        enabled: true,
        source: "swarmx",
      }),
    ).toMatchObject({
      enabled: true,
      configured: false,
      reason: "telemetry url is not configured",
    });
  });

  it("builds v1 envelopes while redacting secrets and omitting raw content", () => {
    const envelope = buildTelemetryEnvelope({
      schemaVersion: "geepilot.telemetry.v1",
      timestamp: "2026-07-03T00:00:00.000Z",
      eventType: "agent_prompt_completed",
      source: "swarmx-desktop",
      installationId: "install-1",
      sessionId: "session-1",
      release: {
        appVersion: "3.0.0",
        platform: "darwin",
        arch: "arm64",
      },
      payload: {
        provider: "openai_responses",
        apiKey: "sk-test",
        prompt: "raw user prompt",
        nested: {
          stackTrace: "full stack",
          runLog: "raw run log",
          credentialRef: "keychain:item",
        },
      },
    });

    expect(envelope.schemaVersion).toBe("geepilot.telemetry.v1");
    expect(envelope.eventId).toMatch(/^evt_/);
    expect(envelope.payload).toEqual({
      provider: "openai_responses",
      apiKey: "[redacted]",
      prompt: "[omitted]",
      nested: {
        stackTrace: "[omitted]",
        runLog: "[omitted]",
        credentialRef: "keychain:item",
      },
    });
    expect(JSON.stringify(envelope)).not.toContain("sk-test");
    expect(JSON.stringify(envelope)).not.toContain("raw user prompt");
    expect(JSON.stringify(envelope)).not.toContain("raw run log");
  });

  it("rejects task product telemetry events and unsafe raw envelopes", () => {
    expect(() =>
      buildTelemetryEnvelope({
        timestamp: "2026-07-03T00:00:00.000Z",
        eventType: "task_created",
        source: "swarmx",
        installationId: "install-1",
      }),
    ).toThrow(/not allowed in v1/);

    expect(() =>
      buildTelemetryEnvelope({
        timestamp: "2026-07-03T00:00:00.000Z",
        eventType: "conversation_message_submitted",
        source: "swarmx",
        installationId: "install-1",
        payload: sanitizeTelemetryPayload({ messageText: "raw text" }) as Record<string, unknown>,
      }),
    ).not.toThrow();
  });

  it("sends through an injected sender and adds bearer headers without copying the token", async () => {
    const sent: Array<{
      envelope: TelemetryEnvelope;
      headers: Record<string, string>;
      url: string;
    }> = [];
    const client = createTelemetryClient({
      config: {
        enabled: true,
        url: "https://telemetry.example/events",
        token: "client-token",
        source: "swarmx-desktop",
        installationId: "install-1",
      },
      now: () => "2026-07-03T00:00:00.000Z",
      sender: {
        async send(envelope, options) {
          sent.push({ envelope, headers: options.headers, url: options.url });
        },
      },
    });

    const result = await client.send({
      eventType: "app_session_started",
      payload: { mode: "local" },
    });

    expect(result.status).toBe("sent");
    expect(sent).toHaveLength(1);
    expect(sent[0]?.url).toBe("https://telemetry.example/events");
    expect(sent[0]?.headers).toEqual({
      "Content-Type": "application/json",
      Authorization: "Bearer client-token",
    });
    expect(JSON.stringify(sent[0]?.envelope)).not.toContain("client-token");
    expect(telemetryHeaders({ enabled: true, source: "swarmx", token: "client-token" })).toEqual({
      "Content-Type": "application/json",
      Authorization: "Bearer client-token",
    });
  });

  it("does not throw on send failure and appends failed sends to outbox", async () => {
    const append = vi.fn(async () => undefined);
    const client = createTelemetryClient({
      config: {
        enabled: true,
        url: "https://telemetry.example/events",
        source: "swarmx-desktop",
        installationId: "install-1",
      },
      now: () => "2026-07-03T00:00:00.000Z",
      sender: {
        async send() {
          throw new Error("network unavailable");
        },
      },
      outbox: {
        append,
        count: async () => 1,
      },
    });

    const result = await client.send({
      eventType: "telemetry_send_failed",
      payload: { endpointConfigured: true },
    });

    expect(result.status).toBe("outboxed");
    expect(result.error).toBe("network unavailable");
    expect(append).toHaveBeenCalledWith(
      expect.objectContaining({ eventType: "telemetry_send_failed" }),
      "network unavailable",
    );
    await expect(client.status()).resolves.toMatchObject({
      enabled: true,
      configured: true,
      outboxCount: 1,
    });
  });

  it("skips sends when enabled state, endpoint, or installation id is missing", async () => {
    const sender = { send: vi.fn(async () => undefined) };
    const disabled = createTelemetryClient({
      config: { enabled: false, source: "swarmx" },
      sender,
    });
    const missingInstall = createTelemetryClient({
      config: {
        enabled: true,
        url: "https://telemetry.example/events",
        source: "swarmx",
      },
      sender,
    });

    await expect(disabled.send({ eventType: "app_session_started" })).resolves.toMatchObject({
      status: "skipped",
      reason: "telemetry is disabled",
    });
    await expect(missingInstall.send({ eventType: "app_session_started" })).resolves.toMatchObject({
      status: "skipped",
      reason: "installation id is not configured",
    });
    expect(sender.send).not.toHaveBeenCalled();
  });

  it("keeps telemetry ingest disabled until explicit server opt-in", () => {
    const config = resolveTelemetryIngestConfig({ env: {} });
    const envelope = buildTelemetryEnvelope({
      timestamp: "2026-07-03T00:00:00.000Z",
      eventType: "app_session_started",
      source: "swarmx-desktop",
      installationId: "install-1",
      payload: { mode: "local" },
    });

    expect(config).toMatchObject({
      enabled: false,
      acceptedSchemaVersions: ["swarmx.telemetry.v1"],
    });
    expect(evaluateTelemetryIngest({ body: envelope }, config)).toMatchObject({
      status: "rejected",
      statusCode: 404,
      reason: "telemetry ingest is disabled",
    });
  });

  it("requires ingest bearer tokens and appends accepted telemetry records through an injected store", async () => {
    const envelope = buildTelemetryEnvelope({
      schemaVersion: "geepilot.telemetry.v1",
      eventId: "evt_prompt_done",
      timestamp: "2026-07-03T00:00:00.000Z",
      eventType: "agent_prompt_completed",
      source: "geepilot-desktop",
      installationId: "install-1",
      payload: {
        provider: "openai_responses",
        status: "completed",
      },
    });
    const config = resolveTelemetryIngestConfig({
      prefix: "GEEPILOT",
      env: {
        GEEPILOT_TELEMETRY_INGEST_ENABLED: "true",
        GEEPILOT_TELEMETRY_INGEST_TOKEN: "ingest-token",
        GEEPILOT_TELEMETRY_ACCEPTED_SCHEMA_VERSIONS: "geepilot.telemetry.v1",
      },
    });
    const append = vi.fn(async () => ({ storageRef: "telemetry/events.jsonl#1" }));
    const handler = createTelemetryIngestHandler({
      config,
      now: () => "2026-07-03T00:01:00.000Z",
      store: { append },
    });

    expect(evaluateTelemetryIngest({ body: envelope }, config)).toMatchObject({
      status: "rejected",
      statusCode: 401,
      reason: "telemetry ingest token is missing or invalid",
    });
    expect(
      evaluateTelemetryIngest(
        {
          headers: { authorization: "Bearer wrong-token" },
          body: envelope,
        },
        config,
      ),
    ).toMatchObject({
      status: "rejected",
      statusCode: 401,
    });

    const decision = await handler.ingest({
      headers: {
        authorization: "Bearer ingest-token",
        "user-agent": "vitest",
      },
      body: envelope,
      sourceIp: "127.0.0.1",
    });

    expect(decision.status).toBe("accepted");
    expect(decision.statusCode).toBe(202);
    expect(append).toHaveBeenCalledTimes(1);
    expect(decision.record).toMatchObject({
      receivedAt: "2026-07-03T00:01:00.000Z",
      sourceIp: "127.0.0.1",
      userAgent: "vitest",
      storageRef: "telemetry/events.jsonl#1",
      envelope: {
        eventId: "evt_prompt_done",
        timestamp: "2026-07-03T00:00:00.000Z",
      },
    });
    expect(parseTelemetryIngestAcceptedRecord(decision.record).ingestId).toMatch(/^ing_/);
    expect(parseTelemetryIngestDecision(decision).status).toBe("accepted");
    expect(JSON.stringify(decision.record)).not.toContain("ingest-token");
  });

  it("rejects unsupported telemetry schema versions and unsafe raw ingest payloads before append", async () => {
    const config = resolveTelemetryIngestConfig({
      settings: {
        enabled: true,
        acceptedSchemaVersions: ["geepilot.telemetry.v1"],
      },
    });
    const append = vi.fn(async () => undefined);
    const handler = createTelemetryIngestHandler({ config, store: { append } });

    const unsupported = buildTelemetryEnvelope({
      timestamp: "2026-07-03T00:00:00.000Z",
      eventType: "app_session_started",
      source: "swarmx-desktop",
      installationId: "install-1",
      payload: { mode: "local" },
    });
    const unsafeRawEnvelope = {
      schemaVersion: "geepilot.telemetry.v1",
      eventId: "evt_raw_payload",
      timestamp: "2026-07-03T00:00:00.000Z",
      eventType: "agent_prompt_completed",
      source: "geepilot-desktop",
      installationId: "install-1",
      payload: { prompt: "raw prompt text" },
    };

    await expect(handler.ingest({ body: unsupported })).resolves.toMatchObject({
      status: "rejected",
      statusCode: 400,
      reason: 'unsupported telemetry schema version "swarmx.telemetry.v1"',
    });
    await expect(handler.ingest({ body: unsafeRawEnvelope })).resolves.toMatchObject({
      status: "rejected",
      statusCode: 400,
      reason: "telemetry envelope is invalid",
      error: expect.stringContaining("Telemetry payload contains unsafe field"),
    });
    expect(append).not.toHaveBeenCalled();
  });

  it("returns rejected decisions when injected telemetry append fails", async () => {
    const envelope = buildTelemetryEnvelope({
      timestamp: "2026-07-03T00:00:00.000Z",
      eventType: "app_session_started",
      source: "swarmx-desktop",
      installationId: "install-1",
      payload: { mode: "local" },
    });
    const handler = createTelemetryIngestHandler({
      config: resolveTelemetryIngestConfig({
        settings: { enabled: true },
      }),
      store: {
        async append() {
          throw new Error("disk full");
        },
      },
    });

    await expect(handler.ingest({ body: envelope })).resolves.toMatchObject({
      status: "rejected",
      statusCode: 500,
      reason: "telemetry append failed",
      error: "disk full",
    });
  });
});
