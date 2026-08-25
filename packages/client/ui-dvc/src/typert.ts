import { DVC_UI_INVOCATIONS } from "./remote-contract.js";

/** Host-side Typert contribution consumed by the DSH registry loader. */
export const TYPERT = Object.freeze({
  package: "@swarmx/dsh-ui-dvc",
  face: "host",
  schemas: Object.freeze([]),
  invocations: DVC_UI_INVOCATIONS,
  model: Object.freeze({
    services: Object.freeze([]),
    events: Object.freeze([]),
    objects: Object.freeze([]),
  }),
});

export default TYPERT;
