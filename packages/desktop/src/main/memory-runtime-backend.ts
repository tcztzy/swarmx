import {
  buildMemoryGraph,
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
import type {
  MemoryRuntimeRequest,
  MemoryRuntimeResult,
} from "@swarmx/core/memory-runtime-protocol";

interface MemoryRuntimeRequester {
  request<Request extends MemoryRuntimeRequest>(
    request: Request,
  ): Promise<MemoryRuntimeResult<Request>>;
}

export class MemoryRuntimeBackend implements MemoryBackend {
  constructor(private readonly runtime: MemoryRuntimeRequester) {}

  async list() {
    const result = await this.runtime.request({ protocolVersion: 1, operation: "list" });
    return result.pages;
  }

  async get(id: string) {
    const parsed = MemoryGetVersionInputSchema.shape.id.parse(id);
    const result = await this.runtime.request({
      protocolVersion: 1,
      operation: "get",
      id: parsed,
    });
    return result.page;
  }

  async search(input: MemorySearchInput) {
    const parsed = MemorySearchInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: 1,
      operation: "search",
    });
    return result.pages;
  }

  async create(input: MemoryCreateInput) {
    const parsed = MemoryCreateInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: 1,
      operation: "create",
    });
    return result.page;
  }

  async update(input: MemoryUpdateInput) {
    const parsed = MemoryUpdateInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: 1,
      operation: "update",
    });
    return result.page;
  }

  async delete(input: MemoryDeleteInput) {
    const parsed = MemoryDeleteInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: 1,
      operation: "delete",
    });
    return result.page;
  }

  async graph() {
    const result = await this.runtime.request({ protocolVersion: 1, operation: "snapshot" });
    return buildMemoryGraph(result.generation, result.pages);
  }

  async history(input: MemoryHistoryInput) {
    const parsed = MemoryHistoryInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: 1,
      operation: "history",
    });
    return result.versions;
  }

  async getVersion(input: MemoryGetVersionInput) {
    const parsed = MemoryGetVersionInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: 1,
      operation: "get_version",
    });
    return result.version;
  }

  async diff(input: MemoryDiffInput) {
    const parsed = MemoryDiffInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: 1,
      operation: "diff",
    });
    return result.diff;
  }

  async restore(input: MemoryRestoreInput) {
    const parsed = MemoryRestoreInputSchema.parse(input);
    const result = await this.runtime.request({
      ...parsed,
      protocolVersion: 1,
      operation: "restore",
    });
    return result.page;
  }
}
