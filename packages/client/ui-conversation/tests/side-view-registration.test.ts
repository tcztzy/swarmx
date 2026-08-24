import { describe, expect, it, vi } from "vitest";
import { registerSideView } from "../src/client/side-view-registration.js";

describe("V48/V49/V99 Side View registration lifecycle", () => {
  it("shadows only details, publishes the service, and does not redeclare upstream children", () => {
    const registrations: Array<{ options: Record<string, unknown>; component: unknown }> = [];
    const cleanups: Array<() => void> = [];
    const disposeService = vi.fn();
    const context = {
      conversationEvents: { register: vi.fn() },
      effect: vi.fn((setup: () => undefined | (() => void)) => {
        const cleanup = setup();
        if (cleanup) cleanups.push(cleanup);
      }),
      layout: { openDetails: vi.fn(), closeDetails: vi.fn() },
      reflect: {
        provide: vi.fn(() => disposeService),
      },
      slots: {
        inject: vi.fn((_name: string, callback: () => void) => callback()),
        register: vi.fn((options: Record<string, unknown>, component: unknown) => {
          registrations.push({ options, component });
          return vi.fn();
        }),
      },
    };

    registerSideView(context as never);

    expect(context.reflect.provide).toHaveBeenCalledWith("sideView", expect.any(Object));
    const details = registrations.find(({ options }) => options.name === "details");
    expect(details?.options).toEqual(
      expect.objectContaining({
        name: "details",
        priority: -10,
        children: {
          "side-view.content": { kind: "keyed", scope: "session" },
        },
      }),
    );
    expect(
      registrations.some(
        ({ options }) => options.name === "side-view.content" && options.key === "tool",
      ),
    ).toBe(false);
    const turnTail = registrations.find(
      ({ options }) => options.name === "conversation.chat.turnTail",
    );
    expect(turnTail?.options).toEqual(
      expect.objectContaining({
        children: {
          "conversation.chat.turnTail.items": { kind: "list", scope: "session" },
        },
      }),
    );
    expect(
      registrations.some(({ options }) =>
        ["conversation", "conversation.session"].includes(String(options.name)),
      ),
    ).toBe(false);
    expect(JSON.stringify(registrations)).not.toContain("Inspect");

    for (const cleanup of cleanups.reverse()) cleanup();
    expect(context.layout.closeDetails).toHaveBeenCalled();
    expect(disposeService).toHaveBeenCalledOnce();
  });
});
