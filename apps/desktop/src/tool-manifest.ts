/** Shared product-tool manifest contract between the Host and native integrations. */
export interface ToolManifestEntry {
  readonly name: string;
  readonly description: string;
  readonly inputSchema: Record<string, unknown>;
}
