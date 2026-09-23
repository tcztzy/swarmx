import { randomUUID } from "node:crypto";
import { importArtifactRequestSchema } from "@swarmx/science/types";
import { z } from "zod";
import type { Interaction, NativeAgent } from "../agents/types.js";
import {
  actionableMessage,
  type LogsEvidencePayloadSchema,
  type LogsQuerySchema,
} from "../bridge-contract.js";
import { RunControlSchema } from "../execution-record.js";
import { LanguageSchema } from "../settings.js";
import {
  WorkCommandSchema,
  WorkReadRequestSchema,
  WorkReadSchema,
  type WorkRunSchema,
} from "../work.js";
import { loadAgUiHistory } from "./ag-ui.js";
import type { ProductServices } from "./product-services.js";
import { HttpError, type SwarmXHost } from "./server.js";

const ARTIFACT_EXTENSIONS: Readonly<Record<string, string>> = {
  "image/png": ".png",
  "image/svg+xml": ".svg",
  "application/pdf": ".pdf",
};

/** Renderer-facing operations over one SwarmX Host. Electron IPC is the only carrier. */
export class HostOperations {
  private readonly activeWork = new Map<
    string,
    {
      cycleId: string;
      controller: AbortController;
      interactions: Map<string, { request: Interaction; resolve: (answer: unknown) => void }>;
    }
  >();

  constructor(private readonly host: SwarmXHost) {}

  activeProducts(): Promise<ProductServices> {
    this.host.signal.throwIfAborted();
    return Promise.resolve(this.host.products);
  }

  async bootstrap() {
    const products = this.host.products;
    let sessions: Awaited<ReturnType<ProductServices["rootAgent"]["list"]>> = [];
    let sessionError: string | undefined;
    try {
      sessions = await products.rootAgent.list();
    } catch (error) {
      sessionError = `Native Agent unavailable: ${error instanceof Error ? error.message : String(error)}. Research and settings remain available.`;
    }
    return {
      agents: products.availableAgents,
      defaultHarness: products.defaultHarness,
      language: products.settings.readLanguage(),
      sessions,
      ...(sessionError ? { sessionError } : {}),
      cwd: products.options.cwd,
    };
  }

  async callTool(name: string, args: unknown, callId: string, signal: AbortSignal) {
    const products = await this.activeProducts();
    return products.callTool(name, args, { actorId: "renderer", callId, signal });
  }

  async agent(id: string): Promise<NativeAgent> {
    return this.host.products.agent(id);
  }

  async cancelAgUi(id: string, threadId: string): Promise<void> {
    return this.host.products.cancelAgUi(id, threadId);
  }

  async settings() {
    const products = this.host.products;
    return { ...products.settings.read(), cwd: products.options.cwd };
  }

  async updateSettings(raw: unknown) {
    const products = this.host.products;
    if (products.busy)
      throw new HttpError(409, "Stop active executions before changing permissions.");
    await products.journal.tool(
      "settings.update",
      raw,
      { actorId: "renderer", callId: randomUUID() },
      async () => products.updatePolicy(raw),
    );
    return this.settings();
  }

  async writeLanguage(raw: unknown) {
    const products = this.host.products;
    const language = LanguageSchema.parse(raw);
    products.settings.writeLanguage(language);
    return { language };
  }

  async workRead(raw: unknown = {}) {
    this.host.signal.throwIfAborted();
    const { cycleId } = WorkReadRequestSchema.parse(raw);
    const work = this.host.products.work;
    const cycles = work.cycles();
    const selected = cycleId ?? cycles[0]?.id;
    return WorkReadSchema.parse({
      cycles,
      snapshot: selected === undefined ? null : work.snapshot(selected),
      activeWorkIds: [...this.activeWork.keys()],
      interactions: [...this.activeWork].flatMap(([workId, { interactions }]) =>
        [...interactions.values()].map(({ request: { id, title, schema } }) => ({
          workId,
          id,
          title,
          schema,
        })),
      ),
    });
  }

  async workCommand(raw: unknown) {
    this.host.signal.throwIfAborted();
    const command = WorkCommandSchema.parse(raw);
    const products = this.host.products;
    const work = products.work;
    let cycleId: string;
    switch (command.action) {
      case "createCycle":
        cycleId = work.createCycle(command.request).id;
        break;
      case "createItem":
        cycleId = work.createItem(command.request).cycleId;
        break;
      case "setBudget":
        if (
          [...this.activeWork.values()].some((entry) => entry.cycleId === command.request.cycleId)
        )
          throw new HttpError(409, "Stop active work before changing its budget.");
        cycleId = work.setBudget(command.request).id;
        break;
      case "revise":
        if (this.activeWork.has(command.request.id))
          throw new HttpError(409, "Stop active work before changing its criteria.");
        cycleId = work.revise(command.request).cycleId;
        break;
      case "accept": {
        const feedback = work.accept({
          ...command.request,
          source: "user",
          layer: "user",
          evaluator: "desktop-user",
          evaluatorVersion: "v1",
        });
        cycleId = work.attempt(feedback.attemptId).cycleId;
        break;
      }
      case "reconcileCharge":
        cycleId = work.attempt(work.reconcileCharge(command.request).reservationId).cycleId;
        break;
      case "reconcileOutcome":
        cycleId = work.reconcileOutcome(command.request).cycleId;
        break;
      case "stop": {
        const active = this.activeWork.get(command.workId);
        if (!active) throw new HttpError(409, "This work is not active in this Host.");
        cycleId = active.cycleId;
        active.controller.abort(new Error("Work stopped by the user."));
        break;
      }
      case "respond": {
        const active = this.activeWork.get(command.workId);
        const pending = active?.interactions.get(command.interactionId);
        if (!active || !pending)
          throw new HttpError(409, "This confirmation is no longer pending. Refresh work status.");
        cycleId = active.cycleId;
        active.interactions.delete(command.interactionId);
        pending.resolve(command.cancel ? undefined : command.answer);
        break;
      }
      case "startNext":
        cycleId = command.cycleId;
        for (const item of work.pending(cycleId)) {
          const selected = item.mode === "managed" ? item.supervisor : item.configuration;
          if (
            this.activeWork.has(item.id) ||
            (selected && !this.host.products.admittedWork(selected)) ||
            item.dependencies.some((id) => work.item(id).state !== "accepted")
          )
            continue;
          if ((await this.startWork(item.id)).reservation) break;
        }
        break;
      case "start":
        cycleId = work.item(command.workId).cycleId;
        await this.startWork(command.workId, command.options);
        break;
    }
    return this.workRead({ cycleId });
  }

  private async startWork(workId: string, options?: z.infer<typeof WorkRunSchema>) {
    if (this.activeWork.has(workId)) throw new HttpError(409, "This work is already active.");
    const cycleId = this.host.products.work.item(workId).cycleId;
    const controller = new AbortController();
    const signal = AbortSignal.any([this.host.signal, controller.signal]);
    const interactions = new Map<
      string,
      { request: Interaction; resolve: (answer: unknown) => void }
    >();
    this.activeWork.set(workId, { cycleId, controller, interactions });
    try {
      return await this.host.products.runWork(
        workId,
        signal,
        (request, nativeSignal) => {
          const lifetime = nativeSignal ? AbortSignal.any([signal, nativeSignal]) : signal;
          if (lifetime.aborted) return Promise.resolve(undefined);
          if (interactions.has(request.id)) throw new Error("Duplicate pending interaction ID.");
          const answer = Promise.withResolvers<unknown>();
          const cancel = () => {
            interactions.delete(request.id);
            answer.resolve(undefined);
          };
          interactions.set(request.id, { request, resolve: answer.resolve });
          lifetime.addEventListener("abort", cancel, { once: true });
          return answer.promise.finally(() => {
            interactions.delete(request.id);
            lifetime.removeEventListener("abort", cancel);
          });
        },
        options,
      );
    } finally {
      controller.abort(new Error("Work execution ended."));
      this.activeWork.delete(workId);
    }
  }

  async environment() {
    return this.host.products.environment.status();
  }

  async environmentAction(action: "setup" | "inspect" | "cancel") {
    const products = this.host.products;
    return products.journal.tool(
      `environment.${action}`,
      {},
      { actorId: "renderer", callId: randomUUID() },
      async () => {
        if (action === "setup") {
          if (products.busy)
            throw new HttpError(409, "Stop active executions before environment setup.");
          return products.environment.setup(this.host.signal);
        }
        if (action === "inspect") return products.environment.inspect();
        products.environment.cancelSetup();
        return { cancelled: true };
      },
    );
  }

  async listSessions(agent: string) {
    return (await this.agent(agent)).list();
  }

  async createSession(agent: string) {
    return { sessionId: await (await this.agent(agent)).create() };
  }

  async history(agent: string, sessionId: string) {
    try {
      return await loadAgUiHistory(await this.agent(agent), sessionId);
    } catch (error) {
      throw new Error(actionableMessage(error));
    }
  }

  async models(agent: string, session?: string) {
    return (await this.agent(agent)).models(session);
  }

  async logs(query: z.infer<typeof LogsQuerySchema>) {
    const products = this.host.products;
    return {
      ...products.journal.read(query),
      activeRunIds: products.journal.activeRuns().map((run) => run.runId),
    };
  }

  async evidence({ sources }: z.infer<typeof LogsEvidencePayloadSchema>) {
    return (await this.activeProducts()).journal.evidence(sources);
  }

  async controlRun(runId: string, raw: unknown) {
    const command = RunControlSchema.parse(raw);
    const products = this.host.products;
    const run = products.journal.activeRuns().find((entry) => entry.runId === runId);
    if (!run?.agent || run.sessionId === null)
      throw new HttpError(409, "This execution is no longer active. Refresh its status.");
    if (run.pendingInteractions)
      throw new HttpError(
        409,
        "Answer or cancel the pending confirmation in the parent conversation first.",
      );
    const agent = await products.agent(z.string().parse(run.attributes["swarmx.harness.name"]));
    if (command.action === "steer") await agent.steer(run.sessionId, command.text, run.runId);
    else await agent.interrupt(run.sessionId, run.runId);
    return { runId: run.runId };
  }

  async scienceWorkspace() {
    const products = this.host.products;
    return products.science.getWorkspace("renderer", this.host.signal);
  }

  async researchObject(projectId: string) {
    const products = this.host.products;
    return products.science.getResearchObject("renderer", { projectId });
  }

  async notebookExecutions(projectId: string, includeArtifactId?: string) {
    const products = this.host.products;
    return products.science.getNotebookExecutions(
      "renderer",
      { projectId, ...(includeArtifactId === undefined ? {} : { includeArtifactId }) },
      this.host.signal,
    );
  }

  async artifactPreview(id: string) {
    const products = this.host.products;
    return products.science.previewArtifact("renderer", { artifactId: id });
  }

  async artifactContent(id: string) {
    const products = this.host.products;
    const { artifact, bytes } = products.science.readArtifactContent(
      "renderer",
      { artifactId: id },
      this.host.signal,
    );
    const extension = ARTIFACT_EXTENSIONS[artifact.mime] ?? "";
    return {
      name: artifact.title.endsWith(extension) ? artifact.title : `${artifact.title}${extension}`,
      mime: artifact.mime,
      bytes,
    };
  }

  async importArtifact(raw: unknown) {
    const products = this.host.products;
    const input = importArtifactRequestSchema.parse(raw);
    return products.journal.tool(
      "science.import",
      { ...input, dataBase64: `[${input.dataBase64.length} base64 characters]` },
      { actorId: "renderer", callId: randomUUID() },
      async () => products.science.importArtifact("renderer", input, this.host.signal),
    );
  }
}
