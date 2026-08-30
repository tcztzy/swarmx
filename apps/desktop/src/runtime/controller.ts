import type {
  ConversationRuntime,
  ConversationSnapshot,
  ConversationSummary,
  UserMessageItem,
} from "./contracts.js";

export class ConversationController {
  constructor(private readonly runtime: ConversationRuntime) {}

  async retry(
    conversationId: string,
    userItemId: string,
    signal?: AbortSignal,
  ): Promise<ConversationSummary> {
    const { item, snapshot } = await this.findUserItem(conversationId, userItemId, signal);
    const request = {
      conversationId: snapshot.conversationId,
      beforeTurnId: item.turnId,
      text: item.text,
    };
    return signal === undefined
      ? this.runtime.revise(request)
      : this.runtime.revise(request, signal);
  }

  async edit(
    conversationId: string,
    userItemId: string,
    text: string,
    signal?: AbortSignal,
  ): Promise<ConversationSummary> {
    const { item, snapshot } = await this.findUserItem(conversationId, userItemId, signal);
    const request = {
      conversationId: snapshot.conversationId,
      beforeTurnId: item.turnId,
      text,
    };
    return signal === undefined
      ? this.runtime.revise(request)
      : this.runtime.revise(request, signal);
  }

  fork(
    conversationId: string,
    beforeTurnId: string,
    signal?: AbortSignal,
  ): Promise<ConversationSummary> {
    return this.runtime.fork({ conversationId, beforeTurnId }, signal);
  }

  private async findUserItem(
    conversationId: string,
    userItemId: string,
    signal?: AbortSignal,
  ): Promise<{ item: UserMessageItem; snapshot: ConversationSnapshot }> {
    const snapshot = await (signal === undefined
      ? this.runtime.read(conversationId)
      : this.runtime.read(conversationId, signal));
    for (const turn of snapshot.turns) {
      const item = turn.items.find(
        (candidate): candidate is UserMessageItem =>
          candidate.id === userItemId && candidate.type === "user_message",
      );
      if (item !== undefined) return { item, snapshot };
    }
    throw new Error(
      `User message "${userItemId}" is not present in conversation "${conversationId}".`,
    );
  }
}
