import type { SwarmxAPI } from "../../shared/desktop-api.js";

declare global {
  interface Window {
    swarmxAPI: SwarmxAPI;
  }
}

export const api = window.swarmxAPI;
