import { describe, expect, it } from "vitest";
import { TYPERT_REMOTE } from "../src/remote.js";
import { TYPERT } from "../src/typert.js";

describe("T13 strict Science Remote contract", () => {
  it("publishes the same strict, abortable descriptors to Host and Client", () => {
    expect(TYPERT.package).toBe("@swarmx/dsh-science");
    expect(TYPERT.face).toBe("host");
    expect(TYPERT.invocations).toEqual(TYPERT_REMOTE.descriptors);
    expect(TYPERT_REMOTE.descriptors.map((descriptor) => descriptor.method)).toEqual([
      "compareRuns",
      "createHypothesis",
      "createDocument",
      "createFigure",
      "createNotebook",
      "createProject",
      "createQuestion",
      "defineExperiment",
      "executeNotebookCell",
      "exportProject",
      "finishRun",
      "getResearchObject",
      "importArtifact",
      "linkEvidence",
      "modifyDocument",
      "modifyFigureCode",
      "previewArtifact",
      "previewTypstDocument",
      "recordClaim",
      "registerArtifact",
      "resolveTypstSourceAtPoint",
      "searchLiterature",
      "startRun",
      "updateTypstSource",
    ]);

    for (const descriptor of TYPERT_REMOTE.descriptors) {
      expect(descriptor.parameters.every((parameter) => parameter.codec.mode === "strict")).toBe(
        true,
      );
      expect(descriptor.result.mode).toBe("strict");
      expect(descriptor.cancellation).toEqual({ parameter: "signal" });
    }
  });
});
