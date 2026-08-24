import { fileURLToPath } from "node:url";
import { clientBundle } from "../tsdown.client.ts";

export default clientBundle("@swarmx/dsh-ui-conversation", ["lib/types/index.js"], {
  "@swarmx/annotation": fileURLToPath(
    new URL("../../core/annotation/lib/types/index.js", import.meta.url),
  ),
});
