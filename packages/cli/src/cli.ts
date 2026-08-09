#!/usr/bin/env node
import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { createInterface } from "node:readline";
import type {
  AuditInput,
  SkillInstructionDelivery,
  SwarmConfig,
  SwarmRuntimeOptions,
} from "@swarmx/core";
import {
  AuditStore,
  createServer,
  HARNESSES,
  listSessionSummaries,
  SWARMX_VERSION,
  Swarm,
} from "@swarmx/core";
import { Command } from "commander";
import { runAuditCommand } from "./audit-command.js";
import { runDoctorCommand } from "./doctor.js";
import { type EvalRunOptions, errorEvalResult, formatEvalResult, runEval } from "./eval-run.js";
import {
  errorName,
  runEvolutionDecide,
  runEvolutionEvaluate,
  runEvolutionEvolve,
  runEvolutionPromote,
  runEvolutionRollback,
  runEvolutionStatus,
} from "./evolution-command.js";
import { createSendSwarmConfig } from "./send-config.js";

const program = new Command();
const cliAudit = new AuditStore();

program.name("swarmx").description("SwarmX multi-agent orchestration CLI").version(SWARMX_VERSION);

program
  .command("doctor")
  .description("Check SwarmX runtime health and optionally repair fixable issues")
  .option("--harness <id>", "Check only one harness")
  .option("--json", "Print a structured JSON report", false)
  .option("--fix", "Preview and apply the repair plan after confirmation", false)
  .option("-y, --yes", "Confirm the displayed repair plan non-interactively", false)
  .action(async (opts: { harness?: string; json?: boolean; fix?: boolean; yes?: boolean }) => {
    const requestId = cliRequestId();
    recordCliAudit("cli.doctor", "attempted", requestId, {
      fix: opts.fix === true,
      scopedHarness: Boolean(opts.harness),
    });
    try {
      process.exitCode = await runDoctorCommand(opts);
      recordCliAudit("cli.doctor", process.exitCode === 0 ? "completed" : "failed", requestId, {
        fix: opts.fix === true,
      });
    } catch (err) {
      recordCliAudit("cli.doctor", "failed", requestId, { errorType: errorName(err) });
      console.error("Error:", err instanceof Error ? err.message : err);
      process.exitCode = 2;
    }
  });

program
  .command("send <message>")
  .description("Send a one-shot prompt to a SwarmX agent")
  .option("-c, --config <path>", "Path to swarm config JSON")
  .option("-h, --harness <name>", "Harness to use (swarmx, claude_code, opencode, etc.)", "swarmx")
  .option("-m, --model <runtime-id>", "Request-scoped model/runtime id")
  .option("-e, --effort <level>", "Request-scoped reasoning effort")
  .option(
    "--resolve-skill <binding...>",
    "Resolve the evolved active Skill revision(s) as <skillId>:<variantId> for this new execution",
  )
  .option("--evolution-root <path>", "Override the skill evolution ledger root")
  .action(
    async (
      message: string,
      opts: {
        config?: string;
        harness?: string;
        model?: string;
        effort?: string;
        resolveSkill?: string[];
        evolutionRoot?: string;
      },
    ) => {
      const requestId = cliRequestId();
      recordAgentRunAudit("cli_send", "attempted", requestId, {
        hasConfig: Boolean(opts.config),
        customHarness: Boolean(opts.harness && opts.harness !== "swarmx"),
        modelSpecified: Boolean(opts.model),
        effortSpecified: Boolean(opts.effort),
        resolvesEvolvedSkills: Boolean(opts.resolveSkill?.length),
      });
      try {
        let swarm: Swarm;

        if (opts.config) {
          const config = JSON.parse(readFileSync(opts.config, "utf-8")) as SwarmConfig;
          swarm = new Swarm(config, await cliSkillRuntimeOptions(opts, config));
        } else {
          const sendConfig = createSendSwarmConfig({
            harnessId: opts.harness ?? "swarmx",
            model: opts.model,
            effort: opts.effort,
          });
          swarm = new Swarm(sendConfig, await cliSkillRuntimeOptions(opts, sendConfig));
        }

        const result = await swarm.execute({
          messages: [{ role: "user", content: message }],
        });

        for (const msg of result) {
          if (msg.content) {
            process.stdout.write(msg.content);
          }
        }
        process.stdout.write("\n");
        recordAgentRunAudit("cli_send", "completed", requestId);
      } catch (err) {
        recordAgentRunAudit("cli_send", "failed", requestId, { errorType: errorName(err) });
        console.error("Error:", err instanceof Error ? err.message : err);
        process.exit(1);
      }
    },
  );

program
  .command("eval-run [message]")
  .description("Run a SwarmX eval sample and print a structured JSON result")
  .option("-c, --config <path>", "Path to swarm config JSON")
  .option("--input-json <json>", "Structured eval arguments JSON object")
  .option("--input-file <path>", "Path to structured eval arguments JSON object")
  .option("--pretty", "Pretty-print JSON output", false)
  .option(
    "--skill-delivery <json>",
    "Request-scoped prompt_fragment Skill delivery metadata (skillId, variantId, revisionId, contentDigest)",
  )
  .option("--skill-content-path <path>", "Digest-verified Skill Markdown content file")
  .option(
    "--resolve-skill <binding...>",
    "Resolve the evolved active Skill revision(s) as <skillId>:<variantId> for this execution",
  )
  .option("--evolution-root <path>", "Override the skill evolution ledger root")
  .action(async (message: string | undefined, opts: EvalRunOptions) => {
    const requestId = cliRequestId();
    recordAgentRunAudit("eval", "attempted", requestId, {
      hasInlineMessage: Boolean(message),
      hasConfig: Boolean(opts.config),
      hasInputFile: Boolean(opts.inputFile),
      hasSkillDelivery: Boolean(opts.skillDelivery),
      resolvesEvolvedSkills: Boolean(opts.resolveSkill?.length),
    });
    try {
      const result = await runEval(message, opts);
      process.stdout.write(formatEvalResult(result, opts.pretty));
      recordAgentRunAudit("eval", "completed", requestId);
    } catch (err) {
      recordAgentRunAudit("eval", "failed", requestId, { errorType: errorName(err) });
      process.stdout.write(formatEvalResult(errorEvalResult(err), opts.pretty));
      process.exitCode = 1;
    }
  });

program
  .command("serve")
  .description("Start OpenAI-compatible HTTP server")
  .option("-p, --port <port>", "Port to listen on", "3000")
  .option("-c, --config <path>", "Path to swarm config JSON")
  .option("--host <host>", "Host to bind", "127.0.0.1")
  .option("--api-token <token>", "Bearer token required for server requests")
  .option("--allowed-origin <origin...>", "Browser origin allowed to call the server")
  .option("--allow-null-origin", "Allow trusted desktop bridge requests with Origin: null", false)
  .action(
    async (opts: {
      port?: string;
      config?: string;
      host?: string;
      apiToken?: string;
      allowedOrigin?: string[];
      allowNullOrigin?: boolean;
    }) => {
      const requestId = cliRequestId();
      recordCliAudit("cli.serve", "attempted", requestId, {
        nonDefaultHost: Boolean(opts.host && opts.host !== "127.0.0.1"),
        hasConfig: Boolean(opts.config),
        authenticated: Boolean(opts.apiToken),
        allowedOriginCount: opts.allowedOrigin?.length ?? 0,
      });
      try {
        let swarm: Swarm;

        if (opts.config) {
          const config = JSON.parse(readFileSync(opts.config, "utf-8")) as SwarmConfig;
          swarm = new Swarm(config);
        } else {
          swarm = new Swarm({
            name: "default",
            root: "agent",
            nodes: {
              agent: {
                kind: "agent",
                agent: {
                  name: "agent",
                  instructions: "You are a helpful assistant.",
                },
              },
            },
            edges: [],
          });
        }

        const port = Number.parseInt(opts.port ?? "3000", 10);
        createServer(swarm, {
          port,
          host: opts.host,
          apiToken: opts.apiToken,
          allowedOrigins: opts.allowedOrigin,
          allowNullOrigin: opts.allowNullOrigin,
          audit: cliAudit,
        });
        recordCliAudit("cli.serve", "completed", requestId, { port });
        console.log(`SwarmX server listening on http://${opts.host ?? "127.0.0.1"}:${port}`);
        console.log("Endpoints:");
        console.log("  GET  /models");
        console.log("  POST /chat/completions");
        console.log("  GET  /sessions");
      } catch (err) {
        recordCliAudit("cli.serve", "failed", requestId, { errorType: errorName(err) });
        console.error("Error:", err instanceof Error ? err.message : err);
        process.exit(1);
      }
    },
  );

program
  .command("audit")
  .description("Verify, query, or export the local audit chain")
  .option("--verify", "Verify the complete chain without listing events", false)
  .option("--json", "Print structured JSON", false)
  .option("--output <path>", "Export verified JSONL to a file")
  .option("--limit <count>", "Maximum events to print or export", "100")
  .option("--category <category>", "Filter by event category")
  .option("--action <action>", "Filter by action")
  .option("--outcome <outcome>", "Filter by outcome")
  .option("--actor-id <id>", "Filter by actor id")
  .option("--target-id <id>", "Filter by target id")
  .option("--session-id <id>", "Filter by Session id")
  .option("--task-id <id>", "Filter by task id")
  .option("--request-id <id>", "Filter by request id")
  .option("--from <timestamp>", "Filter at or after an ISO timestamp")
  .option("--to <timestamp>", "Filter at or before an ISO timestamp")
  .option("--oldest-first", "Return oldest matching events first", false)
  .action(
    (options: {
      verify?: boolean;
      json?: boolean;
      output?: string;
      limit?: string;
      category?: string;
      action?: string;
      outcome?: string;
      actorId?: string;
      targetId?: string;
      sessionId?: string;
      taskId?: string;
      requestId?: string;
      from?: string;
      to?: string;
      oldestFirst?: boolean;
    }) => {
      try {
        const result = runAuditCommand(
          { ...options, reverse: options.oldestFirst !== true },
          cliAudit,
        );
        process.stdout.write(result.output);
        if (result.exitCode !== 0) process.exitCode = result.exitCode;
      } catch (error) {
        console.error("Error:", error instanceof Error ? error.message : error);
        process.exitCode = 1;
      }
    },
  );

const sessionsCommand = program
  .command("sessions")
  .description("List local sessions")
  .allowExcessArguments(false);

sessionsCommand.action(() => {
  const sessions = listSessionSummaries();
  if (sessions.length === 0) {
    console.log("No sessions found.");
    return;
  }
  for (const s of sessions) {
    console.log(`[${s.id.slice(0, 8)}] ${s.title} (${s.harness}) - ${s.messageCount} messages`);
  }
});

program
  .command("harnesses")
  .description("List available agent harnesses")
  .action(() => {
    for (const [id, h] of Object.entries(HARNESSES)) {
      console.log(`${id}: ${h.label}`);
      console.log(`  Model control: ${h.modelControl}`);
      console.log(
        `  Compatible model APIs: ${
          h.modelCompatibility === "any" ? "any" : h.supportedModelApis.join(", ") || "none"
        }`,
      );
      console.log();
    }
  });

program
  .command("repl")
  .description("Start interactive REPL session")
  .option("-c, --config <path>", "Path to swarm config JSON")
  .action(async (opts: { config?: string }) => {
    const requestId = cliRequestId();
    recordCliAudit("cli.repl.session", "attempted", requestId, {
      hasConfig: Boolean(opts.config),
    });
    let swarm: Swarm;

    try {
      if (opts.config) {
        const config = JSON.parse(readFileSync(opts.config, "utf-8")) as SwarmConfig;
        swarm = new Swarm(config);
      } else {
        swarm = new Swarm({
          name: "default",
          root: "agent",
          nodes: {
            agent: {
              kind: "agent",
              agent: {
                name: "agent",
                instructions: "You are a helpful assistant.",
              },
            },
          },
          edges: [],
        });
      }
    } catch (error) {
      recordCliAudit("cli.repl.session", "failed", requestId, { errorType: errorName(error) });
      throw error;
    }

    console.log("SwarmX REPL — type /help for commands, /quit to exit");
    console.log(`Swarm: ${swarm.name}, root: ${swarm.root}`);
    console.log();

    const rl = createInterface({
      input: process.stdin,
      output: process.stdout,
      prompt: "> ",
    });

    rl.prompt();

    rl.on("line", async (line: string) => {
      const trimmed = line.trim();
      if (!trimmed) {
        rl.prompt();
        return;
      }

      if (trimmed === "/quit" || trimmed === "/exit") {
        rl.close();
        return;
      }

      if (trimmed === "/help") {
        console.log("Commands:");
        console.log("  /quit, /exit  Exit REPL");
        console.log("  /help         Show this help");
        console.log();
        rl.prompt();
        return;
      }

      const turnRequestId = cliRequestId();
      recordAgentRunAudit("repl", "attempted", turnRequestId);
      try {
        process.stdout.write("... ");
        const result = await swarm.execute({
          messages: [{ role: "user", content: trimmed }],
        });
        process.stdout.write("\r");

        for (const msg of result) {
          if (msg.content) {
            const prefix = msg.agent ? `[${msg.agent}] ` : "";
            process.stdout.write(`${prefix}${msg.content}\n`);
          }
        }
        console.log();
        recordAgentRunAudit("repl", "completed", turnRequestId);
      } catch (err) {
        recordAgentRunAudit("repl", "failed", turnRequestId, { errorType: errorName(err) });
        console.error("Error:", err instanceof Error ? err.message : err);
      }
      rl.prompt();
    });

    rl.on("close", () => {
      recordCliAudit("cli.repl.session", "completed", requestId);
      console.log("Goodbye.");
      process.exit(0);
    });
  });

const evolutionCommand = program
  .command("evolution")
  .description("Skill self-improvement loop: evolve, evaluate, promote, rollback");

evolutionCommand
  .command("digest")
  .description("Compute the optimizer launch environment digest for a request file")
  .option("--worker <path>", "Path to the Python worker source")
  .option("--python <path>", "Python interpreter used for the worker")
  .option("--optimizer <id>", "Optimizer id (deterministic.v1 or dspy.gepa.v1)", "deterministic.v1")
  .action(async (opts: { worker?: string; python?: string; optimizer?: string }) => {
    const requestId = cliRequestId();
    recordCliAudit("skill.evolution.digest", "attempted", requestId, {
      optimizer: opts.optimizer,
    });
    try {
      const {
        computeSkillEvolutionLaunchDigest,
        discoverEvolutionPythonPath,
        defaultPythonPath,
        resolveLockedDspyVersion,
      } = await import("./evolution-command.js");
      const isGepa = opts.optimizer === "dspy.gepa.v1";
      const pythonPath =
        opts.python ?? (isGepa ? await discoverEvolutionPythonPath() : defaultPythonPath());
      const dspyVersion = isGepa ? await resolveLockedDspyVersion({ pythonPath }) : undefined;
      const digest = await computeSkillEvolutionLaunchDigest({
        workerPath: opts.worker,
        pythonPath,
        dependencyGroups: isGepa ? ["evolution"] : [],
        dspyVersion,
      });
      console.log(digest);
      recordCliAudit("skill.evolution.digest", "completed", requestId, {});
    } catch (error) {
      recordCliAudit("skill.evolution.digest", "failed", requestId, {
        errorType: errorName(error),
      });
      console.error("Error:", error instanceof Error ? error.message : error);
      process.exitCode = 2;
    }
  });

evolutionCommand
  .command("evolve <request-file>")
  .description("Run a granted skill optimization WorkItem and ingest the candidate")
  .option("--task-root <path>", "Override the durable task runtime root")
  .option("--evolution-root <path>", "Override the skill evolution ledger root")
  .option("--worker <path>", "Path to the Python worker source")
  .option("--python <path>", "Python interpreter for the worker")
  .option("--cwd <path>", "Worker working/artifact root")
  .option(
    "--model-command <cmd>",
    'Local credential-free command for proposer "gateway" model calls (JSON on stdin, {content, usage:{totalTokens}} on stdout)',
  )
  .action(
    async (
      requestFile: string,
      opts: {
        taskRoot?: string;
        evolutionRoot?: string;
        worker?: string;
        python?: string;
        cwd?: string;
        modelCommand?: string;
      },
    ) => {
      const requestId = cliRequestId();
      recordCliAudit("skill.evolution.evolve", "attempted", requestId, {
        requestFile,
        gatewayProposer: Boolean(opts.modelCommand),
      });
      try {
        const result = await runEvolutionEvolve(requestFile, {
          taskRoot: opts.taskRoot,
          evolutionRoot: opts.evolutionRoot,
          workerPath: opts.worker,
          python: opts.python,
          cwd: opts.cwd,
          modelCommand: opts.modelCommand,
        });
        console.log(
          `Optimization WorkItem ${result.workItemId} produced candidate ${result.candidateId}.`,
        );
        recordCliAudit("skill.evolution.evolve", "completed", requestId, {
          candidateId: result.candidateId,
        });
      } catch (error) {
        recordCliAudit("skill.evolution.evolve", "failed", requestId, {
          errorType: errorName(error),
        });
        console.error("Error:", error instanceof Error ? error.message : error);
        process.exitCode = 2;
      }
    },
  );

evolutionCommand
  .command("evaluate <candidate-id>")
  .description("Run paired baseline/candidate evaluation on a hidden holdout")
  .option("--holdout <path>", "Hidden holdout JSONL with caseId/input/target fields")
  .option(
    "--evidence <path>",
    "Record evidence JSON produced by an independent evaluator (Inspect)",
  )
  .option("-c, --config <path>", "Swarm config used for both executions")
  .option("--seed <number>", "Paired evaluation seed", "0")
  .option("--task-root <path>", "Override the durable task runtime root")
  .option("--evolution-root <path>", "Override the skill evolution ledger root")
  .action(
    async (
      candidateId: string,
      opts: {
        holdout?: string;
        evidence?: string;
        config?: string;
        seed?: string;
        taskRoot?: string;
        evolutionRoot?: string;
      },
    ) => {
      const requestId = cliRequestId();
      recordCliAudit("skill.evolution.evaluate", "attempted", requestId, { candidateId });
      try {
        const manifest = await runEvolutionEvaluate(candidateId, {
          holdoutPath: opts.holdout,
          evidencePath: opts.evidence,
          configPath: opts.config,
          seed: Number.parseInt(opts.seed ?? "0", 10),
          taskRoot: opts.taskRoot,
          evolutionRoot: opts.evolutionRoot,
        });
        process.stdout.write(`${JSON.stringify(manifest, null, 2)}\n`);
        recordCliAudit("skill.evolution.evaluate", "completed", requestId, {
          candidateId,
          verdict: manifest.verdict,
        });
      } catch (error) {
        recordCliAudit("skill.evolution.evaluate", "failed", requestId, {
          errorType: errorName(error),
        });
        console.error("Error:", error instanceof Error ? error.message : error);
        process.exitCode = 2;
      }
    },
  );

evolutionCommand
  .command("status [skill-id]")
  .description("List skill evolution candidates, evaluations, and active revisions")
  .option("--evolution-root <path>", "Override the skill evolution ledger root")
  .action((_skillId: string | undefined, opts: { evolutionRoot?: string }) => {
    const requestId = cliRequestId();
    recordCliAudit("skill.evolution.status", "attempted", requestId, {});
    try {
      process.stdout.write(runEvolutionStatus({ evolutionRoot: opts.evolutionRoot }));
      recordCliAudit("skill.evolution.status", "completed", requestId, {});
    } catch (error) {
      recordCliAudit("skill.evolution.status", "failed", requestId, {
        errorType: errorName(error),
      });
      console.error("Error:", error instanceof Error ? error.message : error);
      process.exitCode = 2;
    }
  });

evolutionCommand
  .command("promote <candidate-id>")
  .description("Promote a staged candidate with compare-and-swap after human approval")
  .requiredOption("--actor <id>", "Human actor approving the promotion")
  .requiredOption("--reason <text>", "Approval reason")
  .option("--yes", "Confirm the promotion non-interactively", false)
  .option("--evolution-root <path>", "Override the skill evolution ledger root")
  .action(
    async (
      candidateId: string,
      opts: { actor: string; reason: string; yes?: boolean; evolutionRoot?: string },
    ) => {
      const requestId = cliRequestId();
      recordCliAudit("skill.evolution.promote", "attempted", requestId, { candidateId });
      try {
        process.stdout.write(
          runEvolutionPromote(candidateId, {
            actor: opts.actor,
            reason: opts.reason,
            yes: opts.yes,
            evolutionRoot: opts.evolutionRoot,
          }),
        );
        recordCliAudit("skill.evolution.promote", "completed", requestId, { candidateId });
      } catch (error) {
        recordCliAudit("skill.evolution.promote", "failed", requestId, {
          errorType: errorName(error),
        });
        console.error("Error:", error instanceof Error ? error.message : error);
        process.exitCode = 2;
      }
    },
  );

evolutionCommand
  .command("reject <candidate-id>")
  .description("Reject a candidate without promotion")
  .requiredOption("--actor <id>", "Human actor")
  .requiredOption("--reason <text>", "Rejection reason")
  .option("--evolution-root <path>", "Override the skill evolution ledger root")
  .action(
    async (
      candidateId: string,
      opts: { actor: string; reason: string; evolutionRoot?: string },
    ) => {
      const requestId = cliRequestId();
      recordCliAudit("skill.evolution.reject", "attempted", requestId, { candidateId });
      try {
        process.stdout.write(
          runEvolutionDecide(candidateId, "reject", {
            actor: opts.actor,
            reason: opts.reason,
            evolutionRoot: opts.evolutionRoot,
          }),
        );
        recordCliAudit("skill.evolution.reject", "completed", requestId, { candidateId });
      } catch (error) {
        recordCliAudit("skill.evolution.reject", "failed", requestId, {
          errorType: errorName(error),
        });
        console.error("Error:", error instanceof Error ? error.message : error);
        process.exitCode = 2;
      }
    },
  );

evolutionCommand
  .command("quarantine <candidate-id>")
  .description("Quarantine a candidate that failed static or secret checks")
  .requiredOption("--actor <id>", "Human actor")
  .requiredOption("--reason <text>", "Quarantine reason")
  .option("--evolution-root <path>", "Override the skill evolution ledger root")
  .action(
    async (
      candidateId: string,
      opts: { actor: string; reason: string; evolutionRoot?: string },
    ) => {
      const requestId = cliRequestId();
      recordCliAudit("skill.evolution.quarantine", "attempted", requestId, { candidateId });
      try {
        process.stdout.write(
          runEvolutionDecide(candidateId, "quarantine", {
            actor: opts.actor,
            reason: opts.reason,
            evolutionRoot: opts.evolutionRoot,
          }),
        );
        recordCliAudit("skill.evolution.quarantine", "completed", requestId, { candidateId });
      } catch (error) {
        recordCliAudit("skill.evolution.quarantine", "failed", requestId, {
          errorType: errorName(error),
        });
        console.error("Error:", error instanceof Error ? error.message : error);
        process.exitCode = 2;
      }
    },
  );

evolutionCommand
  .command("rollback <skill-id>")
  .description("Restore a retained revision with compare-and-swap")
  .requiredOption("--revision <id>", "Retained revision id to restore")
  .requiredOption("--actor <id>", "Human actor")
  .requiredOption("--reason <text>", "Rollback reason")
  .option("--yes", "Confirm the rollback non-interactively", false)
  .option("--evolution-root <path>", "Override the skill evolution ledger root")
  .action(
    async (
      skillId: string,
      opts: {
        revision: string;
        actor: string;
        reason: string;
        yes?: boolean;
        evolutionRoot?: string;
      },
    ) => {
      const requestId = cliRequestId();
      recordCliAudit("skill.evolution.rollback", "attempted", requestId, { skillId });
      try {
        process.stdout.write(
          runEvolutionRollback(skillId, {
            revision: opts.revision,
            actor: opts.actor,
            reason: opts.reason,
            yes: opts.yes,
            evolutionRoot: opts.evolutionRoot,
          }),
        );
        recordCliAudit("skill.evolution.rollback", "completed", requestId, { skillId });
      } catch (error) {
        recordCliAudit("skill.evolution.rollback", "failed", requestId, {
          errorType: errorName(error),
        });
        console.error("Error:", error instanceof Error ? error.message : error);
        process.exitCode = 2;
      }
    },
  );

type CliAuditOutcome = "attempted" | "completed" | "failed";
type CliAgentRunSurface = "cli_send" | "eval" | "repl";

async function cliSkillRuntimeOptions(
  opts: {
    resolveSkill?: string[];
    evolutionRoot?: string;
  },
  config: SwarmConfig,
): Promise<SwarmRuntimeOptions> {
  if (!opts.resolveSkill?.length) return {};
  const { nativeAgentTargetId, resolveActiveSkillDeliveriesForAgent } = await import(
    "./evolution-command.js"
  );
  const { parseSkillBinding } = await import("./eval-run.js");
  const bindings = opts.resolveSkill.map(parseSkillBinding);
  const perAgent: Record<string, readonly SkillInstructionDelivery[]> = {};
  for (const node of Object.values(config.nodes)) {
    if (node.kind !== "agent" || !node.agent) continue;
    const backend = node.agent.backend?.type ?? "swarmx";
    if (backend !== "swarmx" && backend !== "echo") continue;
    const deliveries = await resolveActiveSkillDeliveriesForAgent({
      bindings,
      agentName: node.agent.name,
      targetAgentId: nativeAgentTargetId(node.agent),
      evolutionRoot: opts.evolutionRoot,
    });
    const agentDeliveries = deliveries[node.agent.name];
    if (agentDeliveries?.length) perAgent[node.agent.name] = agentDeliveries;
  }
  return Object.keys(perAgent).length > 0 ? { agent: { skillInstructionsByAgent: perAgent } } : {};
}

function recordAgentRunAudit(
  surface: CliAgentRunSurface,
  outcome: CliAuditOutcome,
  requestId: string,
  metadata: Record<string, unknown> = {},
): void {
  cliAudit.append(buildCliAuditInput("agent.run", outcome, requestId, { ...metadata, surface }));
}

function recordCliAudit(
  action: string,
  outcome: CliAuditOutcome,
  requestId: string,
  metadata: Record<string, unknown> = {},
): void {
  cliAudit.append(buildCliAuditInput(action, outcome, requestId, metadata));
}

function buildCliAuditInput(
  action: string,
  outcome: CliAuditOutcome,
  requestId: string,
  metadata: Record<string, unknown>,
): AuditInput {
  return {
    category: "system",
    action,
    outcome,
    actor: { kind: "user", id: "cli" },
    requestId,
    metadata,
  };
}

function cliRequestId(): string {
  return `cli_${randomUUID()}`;
}

program.parse();
