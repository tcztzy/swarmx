/** @vitest-environment jsdom */

import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { APP_ICON_URL, AppBrandIcon } from "./app-brand.js";
import { HARNESSES, HarnessBrandIcon } from "./harness-presentation.js";

afterEach(cleanup);

describe("SwarmX application icon", () => {
  it("uses the packaged renderer asset for product branding", () => {
    const { container } = render(<AppBrandIcon className="product-mark" />);
    const image = container.querySelector("img");

    expect(APP_ICON_URL).toMatch(/^data:image\/svg\+xml/);
    expect(image?.getAttribute("src")).toBe(APP_ICON_URL);
    expect(image?.getAttribute("class")).toBe("product-mark");
    expect(image?.getAttribute("alt")).toBe("");
    expect(image?.getAttribute("aria-hidden")).toBe("true");
  });

  it("uses the application icon for the SwarmX harness", () => {
    const swarmx = HARNESSES.find((harness) => harness.id === "swarmx");
    if (!swarmx) throw new Error("SwarmX harness is missing");

    const { container } = render(<HarnessBrandIcon harness={swarmx} />);
    const image = container.querySelector<HTMLImageElement>('[data-harness-icon="swarmx"]');

    expect(image?.getAttribute("src")).toBe(APP_ICON_URL);
  });
});
