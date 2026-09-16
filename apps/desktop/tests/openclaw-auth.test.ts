import { createHash, verify } from "node:crypto";
import { mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, expect, it } from "vitest";
import { openClawAuth } from "../src/agents/openclaw-auth.js";

let home: string;
beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "swarmx-openclaw-auth-"));
});
afterEach(() => rmSync(home, { recursive: true, force: true }));

it("persists a private Ed25519 identity and signs SDK challenge payloads", () => {
  const auth = openClawAuth(home, "wss://gateway.example");
  const identity = auth.loadOrCreateDeviceIdentity();
  expect(openClawAuth(home, "wss://gateway.example").loadOrCreateDeviceIdentity()).toEqual(
    identity,
  );
  const raw = Buffer.from(auth.publicKeyRawBase64UrlFromPem(identity.publicKeyPem), "base64url");
  expect(identity.deviceId).toBe(createHash("sha256").update(raw).digest("hex"));
  expect(
    verify(
      null,
      Buffer.from("SDK challenge"),
      identity.publicKeyPem,
      Buffer.from(auth.signDevicePayload(identity.privateKeyPem, "SDK challenge"), "base64url"),
    ),
  ).toBe(true);
  expect(statSync(join(home, "openclaw", "device.json")).mode & 0o777).toBe(0o600);
});

it("isolates device tokens by Gateway origin, device and role and removes revoked tokens", () => {
  const auth = openClawAuth(home, "wss://gateway.example/a");
  const input = { deviceId: "device", role: "operator" };
  auth.storeDeviceAuthToken({ ...input, token: "private token", scopes: ["operator.read"] });
  expect(openClawAuth(home, "wss://gateway.example/a/").loadDeviceAuthToken(input)).toEqual({
    token: "private token",
    scopes: ["operator.read"],
  });
  expect(openClawAuth(home, "wss://other.example/a").loadDeviceAuthToken(input)).toBeNull();
  expect(openClawAuth(home, "wss://gateway.example/b").loadDeviceAuthToken(input)).toBeNull();
  expect(auth.loadDeviceAuthToken({ ...input, deviceId: "other" })).toBeNull();
  expect(auth.loadDeviceAuthToken({ ...input, role: "node" })).toBeNull();
  for (const file of readdirSync(join(home, "openclaw")))
    expect(statSync(join(home, "openclaw", file)).mode & 0o777).toBe(0o600);
  auth.clearDeviceAuthToken(input);
  expect(auth.loadDeviceAuthToken(input)).toBeNull();
});

it("rejects damaged identity data without silently replacing the paired device", () => {
  const auth = openClawAuth(home, "wss://gateway.example");
  auth.loadOrCreateDeviceIdentity();
  const path = join(home, "openclaw", "device.json");
  writeFileSync(path, "broken");
  expect(() => auth.loadOrCreateDeviceIdentity()).toThrow();
  expect(readFileSync(path, "utf8")).toBe("broken");
});
