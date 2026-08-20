/**
 * Node-side half of the re-run actions plugin.
 *
 * The browser half owns all the behavior; this side exists because a DSH client
 * plugin is a dual-half package — the host scans enabled entries for a
 * `dsh.client` declaration and serves the matching `./client` bundle. The host
 * has nothing to do here, so the plugin body is empty.
 * @module @swarmx/dsh-ui-conversation
 */

/** Stable Cordis plugin name. */
export const name = "swarmx-conversation";

/** Mount the host half. Registration lives entirely in the browser bundle. */
export function apply(): void {
  // Intentionally empty: see the module comment.
}
