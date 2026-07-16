import { readFile, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

const DEFAULT_REFRESH_SKEW_MS = 5 * 60 * 1_000;

export interface CodexAccessTokenProvider {
  available(): Promise<boolean>;
  resolve(): Promise<string>;
}

export interface CodexAccessTokenResolverOptions {
  env?: NodeJS.ProcessEnv;
  authPath?: string;
  refresh: () => Promise<unknown>;
  now?: () => number;
  refreshSkewMs?: number;
}

export class CodexAccessTokenResolver implements CodexAccessTokenProvider {
  private readonly env: NodeJS.ProcessEnv;
  private readonly authPath: string;
  private readonly refresh: () => Promise<unknown>;
  private readonly now: () => number;
  private readonly refreshSkewMs: number;
  private refreshPromise?: Promise<void>;

  constructor(options: CodexAccessTokenResolverOptions) {
    this.env = options.env ?? process.env;
    this.authPath = options.authPath ?? defaultCodexAuthPath(this.env);
    this.refresh = options.refresh;
    this.now = options.now ?? Date.now;
    this.refreshSkewMs = options.refreshSkewMs ?? DEFAULT_REFRESH_SKEW_MS;
  }

  async available(): Promise<boolean> {
    if (this.env.CODEX_ACCESS_TOKEN?.trim()) return true;
    try {
      await this.readManagedAccessToken();
      return true;
    } catch {
      return false;
    }
  }

  async resolve(): Promise<string> {
    const environmentToken = this.env.CODEX_ACCESS_TOKEN?.trim();
    if (environmentToken) return environmentToken;

    const token = await this.readManagedAccessToken();
    if (!tokenNeedsRefresh(token, this.now(), this.refreshSkewMs)) return token;

    await this.refreshManagedToken();
    const refreshed = await this.readManagedAccessToken();
    if (tokenNeedsRefresh(refreshed, this.now(), 0)) {
      throw new Error("Codex sign-in expired. Sign in again with the official Codex client.");
    }
    return refreshed;
  }

  private async refreshManagedToken(): Promise<void> {
    if (!this.refreshPromise) {
      this.refreshPromise = this.refresh()
        .then(() => undefined)
        .catch(() => {
          throw new Error("Codex access token refresh failed. Open Codex and sign in again.");
        })
        .finally(() => {
          this.refreshPromise = undefined;
        });
    }
    return this.refreshPromise;
  }

  private async readManagedAccessToken(): Promise<string> {
    let metadata: Awaited<ReturnType<typeof stat>>;
    try {
      metadata = await stat(this.authPath);
    } catch {
      throw new Error("Codex sign-in was not found. Sign in with Codex or set CODEX_ACCESS_TOKEN.");
    }
    if (!metadata.isFile() || (process.platform !== "win32" && (metadata.mode & 0o077) !== 0)) {
      throw new Error("Codex auth storage must be a private file readable only by its owner.");
    }

    let document: unknown;
    try {
      document = JSON.parse(await readFile(this.authPath, "utf8")) as unknown;
    } catch {
      throw new Error("Codex sign-in data could not be read safely.");
    }
    const token = nestedString(document, "tokens", "access_token")?.trim();
    if (!token) {
      throw new Error("Codex access token is unavailable. Open Codex and sign in again.");
    }
    return token;
  }
}

function defaultCodexAuthPath(env: NodeJS.ProcessEnv): string {
  const configuredHome = env.CODEX_HOME?.trim();
  if (!configuredHome) return join(homedir(), ".codex", "auth.json");
  const expandedHome =
    configuredHome === "~" ? homedir() : configuredHome.replace(/^~\//, `${homedir()}/`);
  return join(expandedHome, "auth.json");
}

function tokenNeedsRefresh(token: string, now: number, skewMs: number): boolean {
  const expiresAt = jwtExpiration(token);
  return expiresAt !== undefined && expiresAt <= now + skewMs;
}

function jwtExpiration(token: string): number | undefined {
  const payload = token.split(".")[1];
  if (!payload) return undefined;
  try {
    const claims = JSON.parse(Buffer.from(payload, "base64url").toString("utf8")) as Record<
      string,
      unknown
    >;
    return typeof claims.exp === "number" && Number.isFinite(claims.exp)
      ? claims.exp * 1_000
      : undefined;
  } catch {
    return undefined;
  }
}

function nestedString(value: unknown, parent: string, key: string): string | undefined {
  if (!isRecord(value)) return undefined;
  const nested = value[parent];
  if (!isRecord(nested)) return undefined;
  return typeof nested[key] === "string" ? nested[key] : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
