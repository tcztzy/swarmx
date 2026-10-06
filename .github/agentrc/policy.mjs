// Packages use the repository's TypeScript project build plus tsdown bundles.
// Script presence is not evidence that a build has passed.
export default {
  name: "swarmx-build-commands",
  criteria: {
    add: [
      {
        id: "build-script",
        title: "Package build or bundle command present",
        pillar: "build-system",
        level: 1,
        scope: "app",
        impact: "high",
        effort: "low",
        check: async (_context, app) => {
          const scripts = app?.scripts;
          const found = [scripts?.build, scripts?.bundle].some(
            (script) => typeof script === "string" && script.trim().length > 0,
          );
          return {
            status: found ? "pass" : "fail",
            evidence: app?.manifestPath ? [app.manifestPath] : [],
            reason: found
              ? undefined
              : "Missing a nonempty build or bundle command in package.json.",
          };
        },
      },
    ],
  },
};
