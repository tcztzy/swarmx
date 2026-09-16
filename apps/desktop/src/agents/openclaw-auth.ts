import { createHash, createPublicKey, generateKeyPairSync, sign } from "node:crypto";
import { readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { type GatewayClientHostDeps, gatewayOriginScope } from "@openclaw/gateway-client";
import { z } from "zod";
import { writePrivateJson } from "../host/settings-store.js";

const identitySchema = z.object({
  deviceId: z.string(),
  publicKeyPem: z.string(),
  privateKeyPem: z.string(),
});
const tokenSchema = z.object({ token: z.string(), scopes: z.array(z.string()) });

function read(path: string): unknown {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    return undefined;
  }
}

export function openClawAuth(productHome: string, url: string) {
  const directory = join(productHome, "openclaw");
  const identityPath = join(directory, "device.json");
  const tokenPath = (deviceId: string, role: string) =>
    join(
      directory,
      `${createHash("sha256")
        .update(JSON.stringify([gatewayOriginScope(url), deviceId, role]))
        .digest("hex")}.json`,
    );
  return {
    loadOrCreateDeviceIdentity() {
      const saved = read(identityPath);
      if (saved !== undefined) return identitySchema.parse(saved);
      const { publicKey, privateKey } = generateKeyPairSync("ed25519");
      const raw = z.string().parse(publicKey.export({ format: "jwk" }).x);
      const identity = {
        deviceId: createHash("sha256").update(Buffer.from(raw, "base64url")).digest("hex"),
        publicKeyPem: publicKey.export({ type: "spki", format: "pem" }).toString(),
        privateKeyPem: privateKey.export({ type: "pkcs8", format: "pem" }).toString(),
      };
      writePrivateJson(identityPath, identity);
      return identity;
    },
    signDevicePayload: (privateKey, payload) =>
      sign(null, Buffer.from(payload), privateKey).toString("base64url"),
    publicKeyRawBase64UrlFromPem: (publicKey) =>
      z.string().parse(createPublicKey(publicKey).export({ format: "jwk" }).x),
    loadDeviceAuthToken({ deviceId, role }) {
      const saved = read(tokenPath(deviceId, role));
      return saved === undefined ? null : tokenSchema.parse(saved);
    },
    storeDeviceAuthToken({ deviceId, role, token, scopes }) {
      writePrivateJson(tokenPath(deviceId, role), { token, scopes });
    },
    clearDeviceAuthToken({ deviceId, role }) {
      rmSync(tokenPath(deviceId, role), { force: true });
    },
  } satisfies GatewayClientHostDeps;
}
