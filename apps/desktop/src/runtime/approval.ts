import type { ApprovalRequest, ApprovalResponse } from "./contracts.js";

interface PendingApproval {
  request: ApprovalRequest;
  resolve(response: ApprovalResponse): void;
  reject(error: Error): void;
}

function approvalKey(request: ApprovalRequest | ApprovalResponse): string {
  const identity = JSON.stringify([
    request.runtime,
    request.conversationId,
    request.turnId,
    request.itemId,
    request.approvalId,
  ]);
  if (identity === undefined) throw new Error("Approval identity could not be encoded.");
  return identity;
}

export class ApprovalRegistry {
  private readonly pending = new Map<string, PendingApproval>();
  private disposed = false;

  request(request: ApprovalRequest): Promise<ApprovalResponse> {
    if (this.disposed) return Promise.reject(new Error("Approval registry is disposed."));
    const key = approvalKey(request);
    if (this.pending.has(key)) {
      return Promise.reject(new Error(`Approval "${request.approvalId}" is already pending.`));
    }
    return new Promise<ApprovalResponse>((resolve, reject) => {
      this.pending.set(key, { request: structuredClone(request), resolve, reject });
    });
  }

  list(runtime: string, conversationId: string): ApprovalRequest[] {
    return [...this.pending.values()]
      .filter(
        ({ request }) => request.runtime === runtime && request.conversationId === conversationId,
      )
      .map(({ request }) => structuredClone(request));
  }

  respond(response: ApprovalResponse): void {
    const key = approvalKey(response);
    const pending = this.pending.get(key);
    if (pending === undefined) {
      const sameId = [...this.pending.values()].find(
        ({ request }) => request.approvalId === response.approvalId,
      );
      throw new Error(
        sameId === undefined
          ? `Approval "${response.approvalId}" is not pending.`
          : `Approval "${response.approvalId}" response does not match its scoped identity.`,
      );
    }
    if (!pending.request.choices.includes(response.decision)) {
      throw new Error(
        `Approval decision "${response.decision}" is not allowed for "${response.approvalId}".`,
      );
    }
    this.pending.delete(key);
    pending.resolve(structuredClone(response));
  }

  reject(request: ApprovalRequest, reason: string): boolean {
    const key = approvalKey(request);
    const pending = this.pending.get(key);
    if (pending === undefined) return false;
    this.pending.delete(key);
    pending.reject(new Error(reason));
    return true;
  }

  rejectConversation(runtime: string, conversationId: string, reason: string): ApprovalRequest[] {
    const rejected: ApprovalRequest[] = [];
    for (const [key, pending] of this.pending) {
      if (
        pending.request.runtime === runtime &&
        pending.request.conversationId === conversationId
      ) {
        this.pending.delete(key);
        pending.reject(new Error(reason));
        rejected.push(structuredClone(pending.request));
      }
    }
    return rejected;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const pending of this.pending.values()) {
      pending.reject(new Error("Approval registry was disposed before a decision."));
    }
    this.pending.clear();
  }
}
