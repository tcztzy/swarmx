import { describe, expect, it, vi } from "vitest";
import { apply } from "../src/client/index.js";

describe("dsh-ui-dvc client", () => {
  it("mounts only the read-only DVC Remote for the shared Version Control view", async () => {
    const disposeRemote = vi.fn();
    const context = {
      remote: {
        $mount: vi.fn(() => Promise.resolve(disposeRemote)),
      },
    };

    const dispose = await apply(context as never);

    expect(context.remote.$mount).toHaveBeenCalledOnce();
    await dispose();
    expect(disposeRemote).toHaveBeenCalledOnce();
  });
});
