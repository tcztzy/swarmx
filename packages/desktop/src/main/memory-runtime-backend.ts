import {
  buildMemoryGraph,
  type GlobalMemoryBackend,
  type GlobalMemoryDeleteInput,
  GlobalMemoryDeleteInputSchema,
  type GlobalMemoryWriteInput,
  GlobalMemoryWriteInputSchema,
  type MemoryBackend,
  type MemoryCreateInput,
  MemoryCreateInputSchema,
  type MemoryDeleteInput,
  MemoryDeleteInputSchema,
  type MemoryDiffInput,
  MemoryDiffInputSchema,
  type MemoryGetVersionInput,
  MemoryGetVersionInputSchema,
  type MemoryHistoryInput,
  MemoryHistoryInputSchema,
  type MemoryRestoreInput,
  MemoryRestoreInputSchema,
  type MemorySearchInput,
  MemorySearchInputSchema,
  type MemoryUpdateInput,
  MemoryUpdateInputSchema,
} from "@swarmx/core/memory";
import {
  MEMORY_RUNTIME_PROTOCOL_VERSION,
  type MemoryRuntimeRequest,
  type MemoryRuntimeResult,
} from "@swarmx/core/memory-runtime-protocol";
import { globalMemoryState } from "@swarmx/core/personal-memory";

interface MemoryRuntimeRequester {
  request<Request extends MemoryRuntimeRequest>(
    request: Request,
  ): Promise<MemoryRuntimeResult<Request>>;
}

export class MemoryRuntimeBackend implements MemoryBackend, GlobalMemoryBackend {
  constructor(private readonly runtime: MemoryRuntimeRequester) {}

  async list() {
    const result = await this.runtime.request({
      protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
      operation: "list",
    });
    return result.pages;
  }

  async get(id: string) {
    const parsed = MemoryGetVersionInputSchema.shape.id.parse(id);
    const result = await this.runtime.request({
      protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
      operation: "get",
      id: parsed,
    });
    return result.page;
  }

  async search(input: MemorySearchInput) {
    const parsed = MemorySearchInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
      operation: "search",
    });
    return result.pages;
  }

  async create(input: MemoryCreateInput) {
    const parsed = MemoryCreateInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
      operation: "create",
    });
    return result.page;
  }

  async update(input: MemoryUpdateInput) {
    const parsed = MemoryUpdateInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
      operation: "update",
    });
    return result.page;
  }

  async delete(input: MemoryDeleteInput) {
    const parsed = MemoryDeleteInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
      operation: "delete",
    });
    return result.page;
  }

  async graph() {
    const result = await this.runtime.request({
      protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
      operation: "snapshot",
    });
    return buildMemoryGraph(result.generation, result.pages);
  }

  async history(input: MemoryHistoryInput) {
    const parsed = MemoryHistoryInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
      operation: "history",
    });
    return result.versions;
  }

  async getVersion(input: MemoryGetVersionInput) {
    const parsed = MemoryGetVersionInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
      operation: "get_version",
    });
    return result.version;
  }

  async diff(input: MemoryDiffInput) {
    const parsed = MemoryDiffInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
      operation: "diff",
    });
    return result.diff;
  }

  async restore(input: MemoryRestoreInput) {
    const parsed = MemoryRestoreInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
      operation: "restore",
    });
    return result.page;
  }

  async getGlobalMemory() {
    const result = await this.runtime.request({
      protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
      operation: "global_get",
    });
    return globalMemoryState(result);
  }

  async saveGlobalMemory(input: GlobalMemoryWriteInput) {
    const parsed = GlobalMemoryWriteInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
      operation: "global_save",
    });
    return result.file;
  }

  async forgetGlobalMemory(input: GlobalMemoryDeleteInput) {
    const parsed = GlobalMemoryDeleteInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
      operation: "global_forget",
    });
    return result.file;
  }
}
