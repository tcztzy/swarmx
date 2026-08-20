import { describe, expect, it, vi } from "vitest";
import { conversationIcons } from "../src/client/icons.js";

const primitiveIcons = vi.hoisted(() => ({
  edit: () => null,
  retry: () => null,
}));

vi.mock("@deepseek-ai/dsh-client-ui-primitives", () => ({
  IconEditOutline16: primitiveIcons.edit,
  IconRefreshOutline16: primitiveIcons.retry,
}));

describe("V13 conversation icon mapping", () => {
  it("keeps the complete action icon configuration in one semantic map", () => {
    expect(Object.keys(conversationIcons)).toEqual(["edit", "retry"]);
    expect(conversationIcons.edit).toBe(primitiveIcons.edit);
    expect(conversationIcons.retry).toBe(primitiveIcons.retry);
  });
});
