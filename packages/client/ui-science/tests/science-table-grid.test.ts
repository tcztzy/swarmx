import { describe, expect, it } from "vitest";
import { scienceColumnTypeToGrid } from "../src/client/science-table-grid.js";

describe("T28 typed artifact table", () => {
  it("V70 maps Science string columns to AG Grid's published text data type", () => {
    expect(scienceColumnTypeToGrid("string")).toBe("text");
    expect(scienceColumnTypeToGrid("number")).toBe("number");
    expect(scienceColumnTypeToGrid("boolean")).toBe("boolean");
  });
});
