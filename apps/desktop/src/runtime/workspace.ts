import { createHash, randomUUID } from "node:crypto";
import { existsSync, realpathSync } from "node:fs";
import { isAbsolute, relative, resolve, sep } from "node:path";
import type { WorkspaceScope } from "./contracts.js";

interface OwnedWorkspace {
  root: string;
  token: string;
}

export class WorkspaceAuthority {
  private readonly scopes = new Map<string, OwnedWorkspace>();

  mint(root: string, label?: string): WorkspaceScope {
    const canonical = realpathSync(root);
    const id = createHash("sha256").update(canonical).digest("hex").slice(0, 24);
    const token = randomUUID();
    this.scopes.set(id, { root: canonical, token });
    return {
      id,
      label: label ?? canonical.split(sep).at(-1) ?? "workspace",
      root: canonical,
      token,
    };
  }

  resolve(scope: WorkspaceScope, relativePath: string): string {
    const owned = this.scopes.get(scope.id);
    if (owned === undefined || owned.token !== scope.token || owned.root !== scope.root) {
      throw new Error("Unknown workspace scope.");
    }
    if (!relativePath || isAbsolute(relativePath)) {
      throw new Error("Workspace resources require a non-empty relative path.");
    }
    const candidate = resolve(owned.root, relativePath);
    if (!isContained(owned.root, candidate)) {
      throw new Error("Workspace resources require a contained relative path.");
    }
    const nearest = nearestExistingPath(candidate);
    const canonicalNearest = realpathSync(nearest);
    if (!isContained(owned.root, canonicalNearest)) {
      throw new Error("Workspace resource escapes workspace through a symbolic link.");
    }
    return candidate;
  }
}

function isContained(root: string, candidate: string): boolean {
  const remainder = relative(root, candidate);
  return remainder === "" || (!remainder.startsWith(`..${sep}`) && remainder !== "..");
}

function nearestExistingPath(path: string): string {
  let current = path;
  while (!existsSync(current)) {
    const parent = resolve(current, "..");
    if (parent === current) throw new Error(`No existing ancestor for workspace path "${path}".`);
    current = parent;
  }
  return current;
}
