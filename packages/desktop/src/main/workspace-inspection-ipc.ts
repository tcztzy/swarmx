import { WorkspaceInspectionInvokeContracts } from "../shared/ipc-contracts/workspace-inspection.js";
import type { DesktopIpcRegistrar } from "./ipc-router.js";
import type {
  WorkspaceDirectoryListing,
  WorkspaceReviewSnapshot,
  WorkspaceTextFile,
} from "./workspace-tools.js";

export interface WorkspaceInspectionTools {
  readonly root: string;
  review(): Promise<WorkspaceReviewSnapshot>;
  listDirectory(path?: string): Promise<WorkspaceDirectoryListing>;
  readFile(path: string): Promise<WorkspaceTextFile>;
}

export interface WorkspaceInspectionHost {
  workspaceRoot: string;
  normalizeWorkingDirectory(cwd?: string): Promise<string | undefined>;
  toolsFor(cwd?: string): WorkspaceInspectionTools;
}

export function registerWorkspaceInspectionIpc(
  registrar: DesktopIpcRegistrar,
  host: WorkspaceInspectionHost,
): void {
  const resolveTools = async (cwd?: string) =>
    host.toolsFor(await host.normalizeWorkingDirectory(cwd));

  registrar.register(
    "workspace:root",
    WorkspaceInspectionInvokeContracts["workspace:root"],
    () => host.workspaceRoot,
  );
  registrar.register(
    "workspace:review",
    WorkspaceInspectionInvokeContracts["workspace:review"],
    async (_event, [{ cwd }]) => toDesktopReview(await (await resolveTools(cwd)).review()),
  );
  registrar.register(
    "workspace:listDirectory",
    WorkspaceInspectionInvokeContracts["workspace:listDirectory"],
    async (_event, [{ path, cwd }]) => {
      const workspace = await resolveTools(cwd);
      return toDesktopListing(workspace.root, await workspace.listDirectory(path));
    },
  );
  registrar.register(
    "workspace:readFile",
    WorkspaceInspectionInvokeContracts["workspace:readFile"],
    async (_event, [{ path, cwd }]) => {
      const workspace = await resolveTools(cwd);
      return toDesktopPreview(workspace.root, await workspace.readFile(path));
    },
  );
}

function toDesktopReview(snapshot: WorkspaceReviewSnapshot) {
  return {
    root: snapshot.root,
    branch: snapshot.branch,
    isRepository: snapshot.isRepository,
    files: snapshot.files.map((file) => ({
      path: file.path,
      ...(file.previousPath === undefined ? {} : { previousPath: file.previousPath }),
      status: file.status,
      patch: file.patch,
      binary: file.binary,
      additions: file.additions,
      deletions: file.deletions,
      truncated: file.truncated,
      ...(file.error === undefined ? {} : { error: file.error }),
    })),
    truncated: snapshot.truncated,
    ...(snapshot.error === undefined ? {} : { error: snapshot.error }),
  };
}

function toDesktopListing(root: string, listing: WorkspaceDirectoryListing) {
  return {
    root,
    path: listing.path,
    entries: listing.entries.map((entry) => ({
      name: entry.name,
      path: entry.path,
      kind: entry.kind,
    })),
    truncated: listing.truncated,
  };
}

function toDesktopPreview(root: string, preview: WorkspaceTextFile) {
  return {
    root,
    path: preview.path,
    content: preview.content,
    size: preview.size,
    binary: false,
    truncated: preview.truncated,
  };
}
