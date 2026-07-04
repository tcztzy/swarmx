#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { Agent, HARNESSES, Swarm, createServer, getHarness, listSessions } from "@swarmx/core";
import type { AgentConfig, MessageChunk, SwarmConfig } from "@swarmx/core";
import { Command } from "commander";
import { type EvalRunOptions, errorEvalResult, formatEvalResult, runEval } from "./eval-run.js";

const program = new Command();

program.name("swarmx").description("SwarmX multi-agent orchestration CLI").version("3.0.0");

program
  .command("send <message>")
  .description("Send a one-shot prompt to a SwarmX agent")
  .option("-c, --config <path>", "Path to swarm config JSON")
  .option("-h, --harness <name>", "Harness to use (swarmx, claude_code, opencode, etc.)", "swarmx")
  .action(async (message: string, opts: { config?: string; harness?: string }) => {
    try {
      let swarm: Swarm;

      if (opts.config) {
        const config = JSON.parse(readFileSync(opts.config, "utf-8")) as SwarmConfig;
        swarm = new Swarm(config);
      } else {
        const harness = getHarness(opts.harness ?? "swarmx");
        const agent = new Agent({
          name: "agent",
          instructions: "You are a helpful assistant.",
          backend: harness?.backend,
        });
        swarm = new Swarm({
          name: "default",
          root: "agent",
          nodes: {
            agent: {
              kind: "agent",
              agent: { name: "agent", instructions: "You are a helpful assistant." },
            },
          },
          edges: [],
        });
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
    } catch (err) {
      console.error("Error:", err instanceof Error ? err.message : err);
      process.exit(1);
    }
  });

program
  .command("eval-run [message]")
  .description("Run a SwarmX eval sample and print a structured JSON result")
  .option("-c, --config <path>", "Path to swarm config JSON")
  .option("--input-json <json>", "Structured eval arguments JSON object")
  .option("--input-file <path>", "Path to structured eval arguments JSON object")
  .option("--pretty", "Pretty-print JSON output", false)
  .action(async (message: string | undefined, opts: EvalRunOptions) => {
    try {
      const result = await runEval(message, opts);
      process.stdout.write(formatEvalResult(result, opts.pretty));
    } catch (err) {
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
        const server = createServer(swarm, {
          port,
          host: opts.host,
          apiToken: opts.apiToken,
          allowedOrigins: opts.allowedOrigin,
          allowNullOrigin: opts.allowNullOrigin,
        });
        console.log(`SwarmX server listening on http://${opts.host ?? "127.0.0.1"}:${port}`);
        console.log("Endpoints:");
        console.log("  GET  /models");
        console.log("  POST /chat/completions");
        console.log("  GET  /sessions");
      } catch (err) {
        console.error("Error:", err instanceof Error ? err.message : err);
        process.exit(1);
      }
    },
  );

program
  .command("sessions")
  .description("List local sessions")
  .action(() => {
    const sessions = listSessions();
    if (sessions.length === 0) {
      console.log("No sessions found.");
      return;
    }
    for (const s of sessions) {
      console.log(
        `[${s.id.slice(0, 8)}] ${s.title} (${s.harness}) - ${s.messages.length} messages`,
      );
    }
  });

program
  .command("harnesses")
  .description("List available agent harnesses")
  .action(() => {
    for (const [id, h] of Object.entries(HARNESSES)) {
      console.log(`${id}: ${h.label}`);
      console.log(`  Compatible: ${h.compatibleProviders.join(", ")}`);
      console.log();
    }
  });

program
  .command("repl")
  .description("Start interactive REPL session")
  .option("-c, --config <path>", "Path to swarm config JSON")
  .action(async (opts: { config?: string }) => {
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
      } catch (err) {
        console.error("Error:", err instanceof Error ? err.message : err);
      }
      rl.prompt();
    });

    rl.on("close", () => {
      console.log("Goodbye.");
      process.exit(0);
    });
  });

program.parse();
