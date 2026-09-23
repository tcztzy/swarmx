import { rebuild } from "@electron/rebuild";

export default async function beforeBuild({ appDir, electronVersion, arch }) {
  await rebuild({ buildPath: appDir, electronVersion, arch });
  // pnpm deploy already supplied the complete production tree, including SDK peers.
  return false;
}
