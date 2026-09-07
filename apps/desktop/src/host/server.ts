import { randomBytes, randomUUID, timingSafeEqual } from "node:crypto";
import { createReadStream, realpathSync, statSync } from "node:fs";
import { realpath, stat } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { extname, relative, resolve, sep } from "node:path";
import { MemoryError } from "@swarmx/memory";
import {
  importArtifactRequestSchema,
  MAX_SCIENCE_IMPORT_BYTES,
  ScienceError,
} from "@swarmx/science";
import type { ViteDevServer } from "vite";
import { ZodError, z } from "zod";
import { RunControlSchema } from "../execution-record.js";
import { LanguageSchema, ProjectSchema } from "../settings.js";
import { loadAgUiHistory } from "./ag-ui.js";
import { handleMcp } from "./mcp.js";
import { ProductServices, type Workspace } from "./product-services.js";
import { ProjectStore, resolveWorkspace } from "./workspace-settings.js";

const HOST = "127.0.0.1";
const COOKIE = "swarmx_session";
const BODY_LIMIT = 1024 * 1024;
const LAUNCH_TTL_MS = 60_000;
const LogQuery = z
  .strictObject({
    after: z.coerce.number().int().nonnegative().default(0),
    limit: z.coerce.number().int().min(1).max(1000).default(200),
    session: z.string().min(1).max(2048).optional(),
    run: z.string().min(1).max(2048).optional(),
    descendants: z
      .enum(["true", "false"])
      .default("false")
      .transform((value) => value === "true"),
  })
  .refine(
    (query) => !query.descendants || query.session !== undefined,
    "Descendants require a session.",
  );
const DvcMutation = z.discriminatedUnion("action", [
  z.strictObject({
    action: z.literal("pull"),
    request: z.strictObject({
      remote: z.string().optional(),
      targets: z.array(z.string()).max(32).optional(),
    }),
  }),
  z.strictObject({
    action: z.literal("reproduce"),
    request: z.strictObject({
      pull: z.boolean().optional(),
      remote: z.string().optional(),
      targets: z.array(z.string()).max(32).optional(),
    }),
  }),
]);

const CONTENT_TYPES: Readonly<Record<string, string>> = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".ico": "image/x-icon",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
};

export interface StartHostOptions {
  readonly development?: boolean | undefined;
  readonly products: ProductServices;
  readonly rendererRoot: string;
  readonly workspace: Workspace;
}

export interface SwarmXHost {
  readonly products: ProductServices;
  readonly internalToken: string;
  readonly internalUrl: string;
  issueLaunchUrl(): string;
  dispose(): Promise<void>;
}

export async function startHost(options: StartHostOptions): Promise<SwarmXHost> {
  const rendererRoot = realpathSync(options.rendererRoot);
  if (!statSync(rendererRoot).isDirectory())
    throw new Error("Renderer build directory is missing.");
  const sessionToken = secret();
  const internalToken = process.env.SWARMX_API_TOKEN ?? secret();
  const launchTokens = new Map<string, number>();
  const shutdown = new AbortController();
  const operations = new Set<Promise<void>>();
  let origin = "";
  let hostHeader = "";
  let closing = false;
  let switching = false;
  let rendererDevServer: ViteDevServer | undefined;
  const nonce = options.development ? secret() : undefined;
  const projects = new ProjectStore(options.products.options.productHome);
  projects.register(options.workspace);
  const owners = new Map([[options.workspace.id, options.products]]);
  const loadingProjects = new Map<string, Promise<ProductServices>>();
  const loadProject = async (id: string): Promise<ProductServices> => {
    if (closing) throw new HttpError(503, "SwarmX Host is closing.");
    const existing = owners.get(id);
    if (existing) return existing;
    const pending = loadingProjects.get(id);
    if (pending) return pending;
    const project = projects.read().projects.find((project) => project.id === id);
    if (!project) throw new HttpError(404, "Project not found.");
    const operation = (async () => {
      if (
        (await realpath(project.root)) !== project.root ||
        !(await stat(project.root)).isDirectory()
      )
        throw new HttpError(
          409,
          "Project directory has moved. Add its current directory as a project.",
        );
      const products = await ProductServices.create({
        productHome: options.products.options.productHome,
        workspace: project,
      });
      try {
        await products.attachAgents(origin, internalToken);
        owners.set(id, products);
        return products;
      } catch (error) {
        await products.dispose();
        throw error;
      }
    })();
    loadingProjects.set(id, operation);
    try {
      return await operation;
    } finally {
      loadingProjects.delete(id);
    }
  };
  const server = createServer((request, response) => {
    securityHeaders(response, nonce);
    const operation = route(request, response, {
      ...options,
      rendererRoot,
      rendererDevServer,
      sessionToken,
      internalToken,
      launchTokens,
      origin,
      hostHeader,
      signal: shutdown.signal,
      projects,
      loadProject,
      switchWorkspace: async (path: string) => {
        if (switching || options.products.busy || operations.size > 1)
          throw new HttpError(409, "Stop active executions before switching workspace.");
        switching = true;
        try {
          const workspace = projects.register(await resolveWorkspace(path));
          const products = await loadProject(workspace.id);
          projects.select(workspace.id);
          options = { ...options, products, workspace };
          return workspace;
        } finally {
          switching = false;
        }
      },
      switching,
    });
    operations.add(operation);
    void operation
      .catch((error: unknown) => sendError(response, error))
      .finally(() => operations.delete(operation));
  });
  if (nonce) {
    const { createServer: createViteServer } = await import("vite");
    rendererDevServer = await createViteServer({
      root: rendererRoot,
      configFile: resolve(rendererRoot, "vite.config.ts"),
      server: { middlewareMode: true, ws: { server } },
      html: { cspNonce: nonce },
    });
  }
  try {
    await new Promise<void>((done, reject) => {
      server.once("error", reject);
      server.listen(0, HOST, done);
    });
  } catch (error) {
    await rendererDevServer?.close();
    throw error;
  }
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("Host has no TCP address.");
  hostHeader = `${HOST}:${String(address.port)}`;
  origin = `http://${hostHeader}`;
  return {
    get products() {
      return options.products;
    },
    internalToken,
    internalUrl: origin,
    issueLaunchUrl() {
      if (closing) throw new Error("SwarmX Host is closing.");
      const token = secret();
      launchTokens.set(token, Date.now() + LAUNCH_TTL_MS);
      return `${origin}/?token=${encodeURIComponent(token)}`;
    },
    async dispose() {
      if (closing) return;
      closing = true;
      shutdown.abort(new Error("SwarmX Host is closing."));
      await rendererDevServer?.close();
      server.closeAllConnections();
      await new Promise<void>((done, reject) =>
        server.close((error) => (error === undefined ? done() : reject(error))),
      );
      await Promise.allSettled([...loadingProjects.values()]);
      const disposalCount = owners.size;
      const results = await Promise.allSettled([
        ...operations,
        ...[...owners.values()].map((products) => products.dispose()),
      ]);
      for (const result of results.slice(-disposalCount))
        if (result.status === "rejected") throw result.reason;
      launchTokens.clear();
    },
  };
}

interface RouteContext extends StartHostOptions {
  readonly rendererDevServer: ViteDevServer | undefined;
  readonly projects: ProjectStore;
  loadProject(id: string): Promise<ProductServices>;
  readonly switching: boolean;
  switchWorkspace(path: string): Promise<Workspace>;
  readonly rendererRoot: string;
  readonly sessionToken: string;
  readonly internalToken: string;
  readonly launchTokens: Map<string, number>;
  readonly origin: string;
  readonly hostHeader: string;
  readonly signal: AbortSignal;
}

async function route(
  request: IncomingMessage,
  response: ServerResponse,
  context: RouteContext,
): Promise<void> {
  if (request.headers.host !== context.hostHeader) throw new HttpError(403, "Invalid Host header.");
  if (context.switching) throw new HttpError(409, "Workspace is changing. Reconnect shortly.");
  const url = new URL(request.url ?? "/", context.origin);
  const projectPath = /^\/projects\/([a-zA-Z0-9_-]+)(?=\/|$)/u.exec(url.pathname);
  if (projectPath) url.pathname = url.pathname.slice(projectPath[0].length) || "/";
  const bindProject = async () => {
    if (!projectPath?.[1]) return;
    const products = await context.loadProject(projectPath[1]);
    context = { ...context, products, workspace: products.options.workspace };
  };
  if (url.pathname.startsWith("/a2a/")) {
    if (request.method !== "GET") authorizeBearer(request, context.internalToken);
    await bindProject();
    await a2aRoute(request, response, url, context);
    return;
  }
  if (url.pathname === "/mcp") {
    await bindProject();
    const credential = url.searchParams.get("acp");
    if (credential !== null && !context.products.acpExecutions.has(credential))
      throw new HttpError(401, "Unknown or expired ACP credential.");
    authorizeBearer(request, credential ?? context.internalToken);
    await handleMcp(
      request,
      response,
      context.products.toolManifest,
      async (name, args, signal) => {
        let sessionId = url.searchParams.get("session") ?? undefined;
        let runId = url.searchParams.get("run") ?? undefined;
        const acpToken = url.searchParams.get("acp");
        if (acpToken !== null) {
          const bound = context.products.acpExecutions.get(acpToken);
          if (!bound) throw new Error("ACP tool endpoint is not bound to an active execution.");
          ({ sessionId, runId } = bound);
        }
        const callId = randomUUID();
        if (!sessionId)
          throw new Error("Product MCP calls require an active session and execution identity.");
        return context.products.callTool(name, args, {
          actorId: sessionId ?? "agent",
          sessionId,
          runId,
          callId,
          signal,
        });
      },
    );
    return;
  }
  if (exchangeLaunchToken(request, response, url, context)) return;
  if (!safeEqual(cookie(request, COOKIE), context.sessionToken)) {
    throw new HttpError(401, "SwarmX browser session is required.");
  }
  if (!validOrigin(request, context.origin)) throw new HttpError(403, "Invalid Origin header.");
  await bindProject();
  if (url.pathname === "/" && !projectPath) {
    response.writeHead(303, { location: `/projects/${context.projects.read().activeId}/` });
    response.end();
    return;
  }
  if (url.pathname.startsWith("/api/")) {
    if (context.switching) throw new HttpError(409, "Workspace is changing. Reconnect shortly.");
    response.setHeader("cache-control", "no-store");
    await apiRoute(request, response, url, context);
    return;
  }
  if (context.rendererDevServer) {
    const vite = context.rendererDevServer;
    await new Promise<void>((done, reject) => {
      response.once("finish", done);
      response.once("close", done);
      vite.middlewares(request, response, (error?: unknown) =>
        reject(error ?? new HttpError(404, "Not found.")),
      );
    });
  } else {
    await serveStatic(request, response, url, context.rendererRoot);
  }
}

async function a2aRoute(
  request: IncomingMessage,
  response: ServerResponse,
  url: URL,
  context: RouteContext,
): Promise<void> {
  const match = /^\/a2a\/([^/]+)(?:\/\.well-known\/agent-card\.json)?$/u.exec(url.pathname);
  if (match?.[1] === undefined) throw new HttpError(404, "A2A Agent not found.");
  const id = decodeURIComponent(match[1]);
  if (request.method === "GET" && url.pathname.endsWith("/.well-known/agent-card.json")) {
    sendJson(response, 200, context.products.a2aCard(id));
    return;
  }
  if (request.method !== "POST" || url.pathname.endsWith("/.well-known/agent-card.json")) {
    throw new HttpError(405, "Method not allowed.");
  }
  authorizeBearer(request, context.internalToken);
  const version = request.headers["a2a-version"];
  if (typeof version !== "string") throw new HttpError(400, "A2A-Version is required.");
  const body = z.record(z.string(), z.unknown()).parse(await readJson(request));
  sendJson(response, 200, await context.products.handleA2A(id, body, version));
}

async function apiRoute(
  request: IncomingMessage,
  response: ServerResponse,
  url: URL,
  context: RouteContext,
): Promise<void> {
  if (url.pathname === "/api/v1/projects" && request.method === "GET") {
    sendJson(response, 200, context.projects.read());
    return;
  }
  if (url.pathname === "/api/v1/projects" && request.method === "POST") {
    sendJson(response, 201, await context.projects.add(await readJson(request)));
    return;
  }
  const openProject = /^\/api\/v1\/projects\/([^/]+)\/open$/u.exec(url.pathname)?.[1];
  if (openProject && request.method === "POST") {
    const id = ProjectSchema.shape.id.parse(openProject);
    await context.loadProject(id);
    sendJson(response, 200, context.projects.select(id));
    return;
  }
  if (url.pathname === "/api/v1/memory" && request.method === "GET") {
    sendJson(response, 200, await context.products.learning.status());
    return;
  }
  if (url.pathname === "/api/v1/memory/settings" && request.method === "PUT") {
    const input = await readJson(request);
    sendJson(
      response,
      200,
      await context.products.journal.tool(
        "memory.configure",
        input,
        { actorId: "renderer", callId: randomUUID() },
        async () => context.products.settings.writeMemory(input),
      ),
    );
    return;
  }
  if (url.pathname === "/api/v1/memory/notes" && request.method === "PUT") {
    sendJson(
      response,
      200,
      await context.products.learning.edit({
        action: "update_core_memory",
        request: await readJson(request),
      }),
    );
    return;
  }
  const pendingMemory = /^\/api\/v1\/memory\/pending\/([^/]+)$/u.exec(url.pathname)?.[1];
  if (pendingMemory && request.method === "POST") {
    const { action } = z
      .strictObject({ action: z.enum(["approve", "reject"]) })
      .parse(await readJson(request));
    await context.products.learning.decide(z.string().uuid().parse(pendingMemory), action);
    sendJson(response, 200, { action });
    return;
  }
  if (url.pathname === "/api/v1/memory/graph" && request.method === "GET") {
    sendJson(response, 200, await context.products.memory.vault.graph(context.workspace.root));
    return;
  }
  if (url.pathname === "/api/v1/memory/concept" && request.method === "GET") {
    sendJson(
      response,
      200,
      await context.products.memory.vault.load(
        context.workspace.root,
        z.string().min(1).max(1024).parse(url.searchParams.get("id")),
      ),
    );
    return;
  }
  if (url.pathname === "/api/v1/memory/review" && request.method === "POST") {
    const { sessionId, focus } = z
      .strictObject({
        sessionId: z.string().min(1).max(2048),
        focus: z.string().max(2000).default(""),
      })
      .parse(await readJson(request));
    context.products.learning.review(sessionId, focus);
    sendJson(response, 202, { state: "running" });
    return;
  }
  if (url.pathname === "/api/v1/language" && request.method === "PUT") {
    const { language } = z
      .strictObject({ language: LanguageSchema })
      .parse(await readJson(request));
    context.products.settings.writeLanguage(language);
    sendJson(response, 200, { language });
    return;
  }
  if (url.pathname === "/api/v1/workspace" && request.method === "PUT") {
    const { root } = z
      .strictObject({ root: z.string().min(1).max(4096) })
      .parse(await readJson(request));
    sendJson(response, 200, await context.switchWorkspace(root));
    return;
  }
  if (url.pathname === "/api/v1/settings" && request.method === "GET") {
    sendJson(response, 200, { ...context.products.settings.read(), workspace: context.workspace });
    return;
  }
  if (url.pathname === "/api/v1/settings" && request.method === "PUT") {
    if (context.products.busy)
      throw new HttpError(409, "Stop active executions before changing permissions.");
    const policy = await readJson(request);
    sendJson(
      response,
      200,
      await context.products.journal.tool(
        "settings.update",
        policy,
        { actorId: "renderer", callId: randomUUID() },
        async () => context.products.updatePolicy(policy),
      ),
    );
    return;
  }
  if (url.pathname === "/api/v1/environment" && request.method === "GET") {
    sendJson(response, 200, context.products.environment.status());
    return;
  }
  if (url.pathname === "/api/v1/environment" && request.method === "POST") {
    const { action } = z
      .strictObject({ action: z.enum(["setup", "inspect", "cancel"]) })
      .parse(await readJson(request));
    const result = await context.products.journal.tool(
      `environment.${action}`,
      {},
      { actorId: "renderer", callId: randomUUID() },
      async () => {
        if (action === "setup") {
          if (context.products.busy)
            throw new HttpError(409, "Stop active executions before environment setup.");
          return context.products.environment.setup(context.signal);
        }
        if (action === "inspect") return context.products.environment.inspect();
        context.products.environment.cancelSetup();
        return { cancelled: true };
      },
    );
    sendJson(response, 200, result);
    return;
  }
  const tool = /^\/api\/v1\/tools\/([a-z_]+)$/u.exec(url.pathname)?.[1];
  if (tool && request.method === "POST") {
    if (
      !tool.startsWith("science_") ||
      !context.products.toolManifest.some((entry) => entry.name === tool)
    )
      throw new HttpError(404, "Science tool not found.");
    const input = await readJson(request);
    const cancelled = new AbortController();
    const abort = () => {
      if (!response.writableEnded) cancelled.abort(new Error("Research request disconnected."));
    };
    response.once("close", abort);
    try {
      sendJson(
        response,
        200,
        await context.products.callTool(tool, input, {
          actorId: "renderer",
          callId: randomUUID(),
          signal: AbortSignal.any([context.signal, cancelled.signal]),
        }),
      );
    } finally {
      response.off("close", abort);
    }
    return;
  }
  if (url.pathname === "/api/v1/research-object" && request.method === "GET") {
    sendJson(
      response,
      200,
      context.products.science.getResearchObject("renderer", {
        projectId: z.string().uuid().parse(url.searchParams.get("project")),
      }),
    );
    return;
  }
  if (url.pathname === "/api/v1/notebook-executions" && request.method === "GET") {
    sendJson(
      response,
      200,
      context.products.science.getNotebookExecutions(
        "renderer",
        { projectId: z.string().uuid().parse(url.searchParams.get("project")) },
        context.signal,
      ),
    );
    return;
  }
  if (url.pathname === "/api/v1/artifact-preview" && request.method === "GET") {
    sendJson(
      response,
      200,
      context.products.science.previewArtifact("renderer", {
        artifactId: z.string().uuid().parse(url.searchParams.get("id")),
      }),
    );
    return;
  }
  if (url.pathname === "/api/v1/artifacts" && request.method === "POST") {
    const input = importArtifactRequestSchema.parse(
      await readJson(request, Math.ceil(MAX_SCIENCE_IMPORT_BYTES / 3) * 4 + 8192),
    );
    sendJson(
      response,
      200,
      await context.products.journal.tool(
        "science.import",
        { ...input, dataBase64: `[${input.dataBase64.length} base64 characters]` },
        { actorId: "renderer", callId: randomUUID() },
        async () => context.products.science.importArtifact("renderer", input, context.signal),
      ),
    );
    return;
  }
  const contentId = /^\/api\/v1\/artifacts\/([^/]+)\/content$/u.exec(url.pathname)?.[1];
  if (contentId && request.method === "GET") {
    const { artifact, bytes } = context.products.science.readArtifactContent(
      "renderer",
      { artifactId: z.string().uuid().parse(decodeURIComponent(contentId)) },
      context.signal,
    );
    const extension =
      (
        { "image/png": ".png", "image/svg+xml": ".svg", "application/pdf": ".pdf" } as Record<
          string,
          string
        >
      )[artifact.mime] ?? "";
    response.setHeader("content-type", "application/octet-stream");
    response.setHeader(
      "content-disposition",
      `attachment; filename*=UTF-8''${encodeURIComponent(artifact.title.endsWith(extension) ? artifact.title : artifact.title + extension)}`,
    );
    response.end(bytes);
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/v1/logs") {
    sendJson(response, 200, {
      ...context.products.journal.read(LogQuery.parse(Object.fromEntries(url.searchParams))),
      activeRunIds: context.products.journal.activeRuns().map((run) => run.runId),
    });
    return;
  }
  const runId = /^\/api\/v1\/runs\/([^/]+)$/u.exec(url.pathname)?.[1];
  if (request.method === "POST" && runId !== undefined) {
    const command = RunControlSchema.parse(await readJson(request));
    const run = context.products.journal
      .activeRuns()
      .find((run) => run.runId === decodeURIComponent(runId));
    if (!run?.agent || run.sessionId === null)
      throw new HttpError(409, "This execution is no longer active. Refresh its status.");
    if (run.pendingInteractions)
      throw new HttpError(
        409,
        "Answer or cancel the pending confirmation in the parent conversation first.",
      );
    const agent = await context.products.agent(
      z.string().parse(run.attributes["swarmx.harness.name"]),
    );
    if (command.action === "steer") await agent.steer(run.sessionId, command.text, run.runId);
    else await agent.interrupt(run.sessionId, run.runId);
    sendJson(response, 200, { runId: run.runId });
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/ag-ui") {
    await (await context.products.agUi(url.searchParams.get("agent") ?? "swarm")).handle(
      request,
      response,
    );
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/v1/bootstrap") {
    let sessions: Awaited<ReturnType<ProductServices["rootAgent"]["list"]>> = [];
    let sessionError: string | undefined;
    try {
      sessions = await context.products.rootAgent.list();
    } catch (error) {
      sessionError = `Native Agent unavailable: ${error instanceof Error ? error.message : String(error)}. Research and settings remain available.`;
    }
    sendJson(response, 200, {
      agents: context.products.availableAgents,
      defaultHarness: context.products.defaultHarness,
      language: context.products.settings.readLanguage(),
      sessions,
      ...(sessionError ? { sessionError } : {}),
      workspace: context.workspace,
      projects: context.projects.read().projects,
    });
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/v1/models") {
    const agent = await context.products.agent(url.searchParams.get("agent") ?? "swarm");
    sendJson(response, 200, await agent.models(url.searchParams.get("session") ?? undefined));
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/v1/capabilities") {
    const agent = await context.products.agent(url.searchParams.get("agent") ?? "swarm");
    sendJson(response, 200, agent.capabilities);
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/v1/sessions") {
    sendJson(
      response,
      200,
      await (await context.products.agent(url.searchParams.get("agent") ?? "swarm")).list(),
    );
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/v1/sessions") {
    sendJson(response, 200, {
      sessionId: await (
        await context.products.agent(url.searchParams.get("agent") ?? "swarm")
      ).create(),
    });
    return;
  }
  const session = /^\/api\/v1\/sessions\/([^/]+)$/u.exec(url.pathname)?.[1];
  if (request.method === "GET" && session !== undefined) {
    sendJson(
      response,
      200,
      await loadAgUiHistory(
        await context.products.agent(url.searchParams.get("agent") ?? "swarm"),
        decodeURIComponent(session),
      ),
    );
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/v1/science") {
    sendJson(response, 200, context.products.science.getWorkspace("renderer", context.signal));
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/v1/swarm") {
    sendJson(response, 200, context.products.listSwarms());
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/v1/git") {
    sendJson(
      response,
      200,
      (await context.products.dvc.inspect(context.workspace.root, context.signal)).git,
    );
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/v1/dvc") {
    sendJson(
      response,
      200,
      await context.products.dvc.inspect(context.workspace.root, context.signal),
    );
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/v1/dvc") {
    const input = DvcMutation.parse(await readJson(request));
    const result = await context.products.journal.tool(
      `dvc.${input.action}`,
      input.request,
      { actorId: "renderer", callId: randomUUID() },
      async () => {
        if (input.action === "pull")
          return context.products.dvc.pull(
            context.workspace.root,
            {
              ...(input.request.remote === undefined ? {} : { remote: input.request.remote }),
              ...(input.request.targets === undefined ? {} : { targets: input.request.targets }),
            },
            context.signal,
          );
        const reproduction = await context.products.dvc.reproduce(
          context.workspace.root,
          {
            ...(input.request.pull === undefined ? {} : { pull: input.request.pull }),
            ...(input.request.remote === undefined ? {} : { remote: input.request.remote }),
            ...(input.request.targets === undefined ? {} : { targets: input.request.targets }),
          },
          context.signal,
        );
        try {
          return {
            source: reproduction.source,
            result: reproduction.result,
            after: reproduction.after,
          };
        } finally {
          await reproduction.dispose();
        }
      },
    );
    sendJson(response, 200, result);
    return;
  }
  throw new HttpError(404, "Not found.");
}

function exchangeLaunchToken(
  request: IncomingMessage,
  response: ServerResponse,
  url: URL,
  context: RouteContext,
): boolean {
  if (request.method !== "GET" || url.pathname !== "/" || !url.searchParams.has("token")) {
    return false;
  }
  const token = url.searchParams.get("token") ?? "";
  const expiry = context.launchTokens.get(token);
  context.launchTokens.delete(token);
  if (expiry === undefined || expiry < Date.now())
    throw new HttpError(401, "Launch token expired.");
  response.statusCode = 303;
  response.setHeader("location", `/projects/${context.projects.read().activeId}/`);
  response.setHeader(
    "set-cookie",
    `${COOKIE}=${context.sessionToken}; HttpOnly; SameSite=Strict; Path=/`,
  );
  response.end();
  return true;
}

async function serveStatic(
  request: IncomingMessage,
  response: ServerResponse,
  url: URL,
  rendererRoot: string,
): Promise<void> {
  if (request.method !== "GET" && request.method !== "HEAD") {
    throw new HttpError(405, "Method not allowed.");
  }
  let pathname: string;
  try {
    pathname = decodeURIComponent(url.pathname);
  } catch {
    throw new HttpError(400, "Malformed static path.");
  }
  if (pathname.includes("\0") || pathname.split("/").includes("..")) {
    throw new HttpError(404, "Not found.");
  }
  const candidate = resolve(rendererRoot, pathname === "/" ? "index.html" : pathname.slice(1));
  if (!contained(rendererRoot, candidate)) throw new HttpError(404, "Not found.");
  let canonical: string;
  try {
    canonical = await realpath(candidate);
  } catch {
    throw new HttpError(404, "Not found.");
  }
  if (!contained(rendererRoot, canonical) || !(await stat(canonical)).isFile()) {
    throw new HttpError(404, "Not found.");
  }
  response.statusCode = 200;
  response.setHeader(
    "content-type",
    CONTENT_TYPES[extname(canonical)] ?? "application/octet-stream",
  );
  if (request.method === "HEAD") {
    response.end();
    return;
  }
  await new Promise<void>((done, reject) => {
    const stream = createReadStream(canonical);
    stream.once("error", reject);
    response.once("finish", done);
    stream.pipe(response);
  });
}

async function readJson(request: IncomingMessage, limit = BODY_LIMIT): Promise<unknown> {
  if (!(request.headers["content-type"] ?? "").startsWith("application/json")) {
    throw new HttpError(415, "Expected an application/json body.");
  }
  const chunks: Buffer[] = [];
  let bytes = 0;
  for await (const chunk of request) {
    const buffer = Buffer.from(chunk);
    bytes += buffer.byteLength;
    if (bytes > limit) throw new HttpError(413, "Request body is too large.");
    chunks.push(buffer);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch (error) {
    throw new HttpError(400, "Request body is not valid JSON.", { cause: error });
  }
}

function authorizeBearer(request: IncomingMessage, token: string): void {
  if (!safeEqual(request.headers.authorization?.replace(/^Bearer /u, ""), token)) {
    throw new HttpError(403, "Invalid Host bearer token.");
  }
}

function validOrigin(request: IncomingMessage, origin: string): boolean {
  return request.headers.origin === undefined
    ? request.method === "GET" || request.method === "HEAD"
    : request.headers.origin === origin;
}

function contained(root: string, candidate: string): boolean {
  const path = relative(root, candidate);
  return path === "" || (path !== ".." && !path.startsWith(`..${sep}`));
}

function cookie(request: IncomingMessage, name: string): string | undefined {
  for (const entry of (request.headers.cookie ?? "").split(";")) {
    const [key, ...value] = entry.trim().split("=");
    if (key === name) return value.join("=");
  }
  return undefined;
}

function safeEqual(left: string | undefined, right: string): boolean {
  if (left === undefined) return false;
  const leftBytes = Buffer.from(left);
  const rightBytes = Buffer.from(right);
  return leftBytes.length === rightBytes.length && timingSafeEqual(leftBytes, rightBytes);
}

function secret(): string {
  return randomBytes(32).toString("base64url");
}

function sendJson(response: ServerResponse, status: number, value: unknown): void {
  if (response.headersSent || response.destroyed) return;
  response.statusCode = status;
  response.setHeader("content-type", "application/json; charset=utf-8");
  response.end(`${JSON.stringify(value)}\n`);
}

function sendError(response: ServerResponse, error: unknown): void {
  if (response.headersSent || response.destroyed) {
    if (!response.destroyed) response.end();
    return;
  }
  const status =
    error instanceof HttpError
      ? error.status
      : error instanceof ZodError
        ? 400
        : error instanceof ScienceError || error instanceof MemoryError
          ? error.code.endsWith("NOT_FOUND")
            ? 404
            : error.code.includes("REVISION")
              ? 409
              : 400
          : 500;
  sendJson(response, status, {
    error: error instanceof Error ? error.message : String(error),
    ...(error instanceof ZodError ? { issues: error.issues } : {}),
  });
}

function securityHeaders(response: ServerResponse, nonce?: string): void {
  const nonceSource = nonce ? ` 'nonce-${nonce}'` : "";
  response.setHeader(
    "content-security-policy",
    `default-src 'self'; script-src 'self'${nonceSource}; style-src 'self'${nonceSource}; style-src-attr 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'`,
  );
  response.setHeader("referrer-policy", "no-referrer");
  response.setHeader("x-content-type-options", "nosniff");
  response.setHeader("x-frame-options", "DENY");
}

class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
  }
}
