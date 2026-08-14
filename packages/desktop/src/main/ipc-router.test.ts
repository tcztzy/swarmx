import { describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { createDesktopIpcRegistrar, createSemanticAuditReceipt } from "./ipc-router.js";

describe("Desktop IPC contract registrar", () => {
  const contract = {
    kind: "invoke",
    args: z.tuple([z.object({ id: z.string().min(1) }).strict()]),
    result: z.object({ ok: z.literal(true) }).strict(),
    audit: "intent_outcome",
  } as const;

  it("validates arguments before effects and validates sync and async results", async () => {
    let registered: ((event: never, ...args: unknown[]) => unknown) | undefined;
    const effect = vi.fn((id: string) => ({ ok: true as const, id }));
    const registrar = createDesktopIpcRegistrar({
      auditPolicy: () => "intent_outcome",
      registerAuthorized: (_channel, handler) => {
        registered = handler as typeof registered;
      },
    });
    registrar.register("feature:run", contract, (_event, [input]) => {
      effect(input.id);
      return { ok: true as const };
    });

    const receipt = createSemanticAuditReceipt();
    expect(() => registered?.({} as never, receipt, { id: "", secret: "not echoed" })).toThrow(
      "feature:run arguments failed validation",
    );
    expect(effect).not.toHaveBeenCalled();
    expect(registered?.({} as never, receipt, { id: "valid" })).toEqual({ ok: true });

    registrar.register("feature:sync-invalid", contract, () => ({ ok: false }) as never);
    expect(() => registered?.({} as never, receipt, { id: "valid" })).toThrow(
      "feature:sync-invalid result failed validation",
    );

    registrar.register("feature:async", contract, async () => ({ ok: false }) as never);
    await expect(
      Promise.resolve(registered?.({} as never, receipt, { id: "valid" })),
    ).rejects.toThrow("feature:async result failed validation");
  });

  it("rejects a feature contract whose policy differs from the authoritative router", () => {
    const registrar = createDesktopIpcRegistrar({
      auditPolicy: () => "failure_only",
      registerAuthorized: vi.fn(),
    });
    expect(() =>
      registrar.register("feature:run", contract, () => ({ ok: true as const })),
    ).toThrow(/inconsistent audit policy/i);
  });

  it("keeps semantic audit receipts isolated between dispatches", () => {
    const first = createSemanticAuditReceipt();
    const second = createSemanticAuditReceipt();
    second.recordSemanticAudit();
    expect(first.semanticAuditRecorded).toBe(false);
    expect(second.semanticAuditRecorded).toBe(true);
    first.recordSemanticAudit();
    expect(first.semanticAuditRecorded).toBe(true);
  });
});
