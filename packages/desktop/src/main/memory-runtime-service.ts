import { readFile } from "node:fs/promises";
import path from "node:path";
import type {
  GlobalMemoryBackend,
  GlobalMemoryDeleteInput,
  GlobalMemoryWriteInput,
  MemoryBackend,
  MemoryCreateInput,
  MemoryDeleteInput,
  MemoryDiffInput,
  MemoryGetVersionInput,
  MemoryHistoryInput,
  MemoryRestoreInput,
  MemorySearchInput,
  MemoryUpdateInput,
} from "@swarmx/core/memory";
import {
  type MemoryRuntimeEnvironmentHost,
  MemoryRuntimeEnvironmentService,
  MemoryRuntimeManifestSchema,
} from "@swarmx/runtime";
import { MemoryRuntimeBackend } from "./memory-runtime-backend.js";
import { MemoryRuntimeHost } from "./memory-runtime-host.js";

export interface MemoryRuntimeServiceOptions {
  manifestPath: string;
  memoryRoot: string;
  environmentHost?: MemoryRuntimeEnvironmentHost;
  createHost?: (
    launch: Awaited<ReturnType<MemoryRuntimeEnvironmentService["launchSpec"]>>,
  ) => MemoryRuntimeHost;
}

export class MemoryRuntimeService implements MemoryBackend, GlobalMemoryBackend {
  private readonly options: MemoryRuntimeServiceOptions;
  private initialized?: Promise<{ backend: MemoryRuntimeBackend; host: MemoryRuntimeHost }>;

  constructor(options: MemoryRuntimeServiceOptions) {
    this.options = options;
  }

  async list() {
    return (await this.backend()).list();
  }

  async get(id: string) {
    return (await this.backend()).get(id);
  }

  async search(input: MemorySearchInput) {
    return (await this.backend()).search(input);
  }

  async create(input: MemoryCreateInput) {
    return (await this.backend()).create(input);
  }

  async update(input: MemoryUpdateInput) {
    return (await this.backend()).update(input);
  }

  async delete(input: MemoryDeleteInput) {
    return (await this.backend()).delete(input);
  }

  async graph() {
    return (await this.backend()).graph();
  }

  async history(input: MemoryHistoryInput) {
    return (await this.backend()).history(input);
  }

  async getVersion(input: MemoryGetVersionInput) {
    return (await this.backend()).getVersion(input);
  }

  async diff(input: MemoryDiffInput) {
    return (await this.backend()).diff(input);
  }

  async restore(input: MemoryRestoreInput) {
    return (await this.backend()).restore(input);
  }

  async getGlobalMemory() {
    return (await this.backend()).getGlobalMemory();
  }

  async saveGlobalMemory(input: GlobalMemoryWriteInput) {
    return (await this.backend()).saveGlobalMemory(input);
  }

  async forgetGlobalMemory(input: GlobalMemoryDeleteInput) {
    return (await this.backend()).forgetGlobalMemory(input);
  }

  async close(): Promise<void> {
    if (!this.initialized) return;
    const initialized = await this.initialized.catch(() => undefined);
    this.initialized = undefined;
    await initialized?.host.close();
  }

  private async backend(): Promise<MemoryRuntimeBackend> {
    if (!this.initialized) this.initialized = this.initialize();
    return (await this.initialized).backend;
  }

  private async initialize(): Promise<{
    backend: MemoryRuntimeBackend;
    host: MemoryRuntimeHost;
  }> {
    const manifestPath = path.resolve(this.options.manifestPath);
    const manifest = MemoryRuntimeManifestSchema.parse(
      JSON.parse(await readFile(manifestPath, "utf8")),
    );
    const environment = new MemoryRuntimeEnvironmentService(
      manifest,
      this.options.environmentHost,
      {
        manifestRoot: path.dirname(manifestPath),
      },
    );
    const status = await environment.status();
    if (!status.ready) {
      throw new Error(`Memory runtime is unavailable (${status.reason ?? status.state}).`);
    }
    const launch = await environment.launchSpec(status, {
      memoryRoot: path.resolve(this.options.memoryRoot),
    });
    const host = this.options.createHost?.(launch) ?? new MemoryRuntimeHost({ launch });
    const backend = new MemoryRuntimeBackend(host);
    return { backend, host };
  }
}
