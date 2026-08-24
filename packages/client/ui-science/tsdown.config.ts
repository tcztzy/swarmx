import { fileURLToPath } from "node:url";
import { clientBundle } from "../tsdown.client.ts";

export default clientBundle("@swarmx/dsh-ui-science", ["lib/types/index.js"], {
  "@swarmx/annotation": fileURLToPath(
    new URL("../../core/annotation/lib/types/index.js", import.meta.url),
  ),
  "@swarmx/dsh-science/remote": fileURLToPath(
    new URL("../../science/core/lib/types/remote.js", import.meta.url),
  ),
  "@swarmx/dsh-science/types": fileURLToPath(
    new URL("../../science/core/lib/types/contracts.js", import.meta.url),
  ),
});
