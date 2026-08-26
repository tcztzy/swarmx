import { fileURLToPath } from "node:url";
import { clientBundle } from "../tsdown.client.ts";

export default clientBundle("@swarmx/dsh-ui-swarm", ["lib/types/index.js"], {
  "@swarmx/dsh-swarm/contracts": fileURLToPath(
    new URL("../../core/swarm/lib/types/contracts.js", import.meta.url),
  ),
  "@swarmx/dsh-swarm/remote": fileURLToPath(
    new URL("../../core/swarm/lib/types/remote.js", import.meta.url),
  ),
});
