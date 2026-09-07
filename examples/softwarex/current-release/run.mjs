import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { setTimeout } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { ProductServices } from "../../../apps/desktop/dist/host/product-services.js";
import { startHost } from "../../../apps/desktop/dist/host/server.js";
import {
  ProjectStore,
  resolveWorkspace,
} from "../../../apps/desktop/dist/host/workspace-settings.js";

const repo = fileURLToPath(new URL("../../../", import.meta.url));
const root = resolve(process.argv[2] ?? "artifacts/paper-current");
const workspaceRoot = resolve(root, "workspace");
const productHome = resolve(root, "home");
const evidence = resolve(root, "evidence");
await mkdir(workspaceRoot, { recursive: true });
await mkdir(evidence, { recursive: true });
const workspace = await resolveWorkspace(workspaceRoot);
new ProjectStore(productHome).register({ ...workspace, label: "Germination provenance example" });
const products = await ProductServices.create({ productHome, workspace });
const host = await startHost({
  products,
  workspace,
  rendererRoot: resolve(repo, "apps/desktop/dist/renderer"),
});
const writeJson = (name, value) =>
  writeFile(resolve(evidence, name), JSON.stringify(value, null, 2) + "\n");
const hash = (bytes) => "sha256:" + createHash("sha256").update(bytes).digest("hex");
const exposeUi = (stage, sessionId) =>
  writeFile(
    resolve(root, "ui.json"),
    JSON.stringify({ stage, sessionId, url: host.issueLaunchUrl() }),
    { mode: 0o600 },
  );
let stopping = false;
async function stop() {
  if (stopping) return;
  stopping = true;
  await host.dispose();
}
process.once("SIGINT", () => void stop());
process.once("SIGTERM", () => void stop());

try {
  products.settings.writeLanguage("en");
  products.settings.writeMemory({ enabled: true, autoReview: false, writeApproval: true });
  const environment = await products.environment.setup();
  await writeJson("environment.json", environment);
  const science = products.science;
  assert.equal(science.getWorkspace("renderer").projects.length, 0, "Use a fresh run directory.");
  const project = science.createProject("renderer", {
    requestId: randomUUID(),
    title: "Synthetic germination analysis",
  });
  const csv = await readFile(new URL("../swarm-memory/data/germination.csv", import.meta.url));
  await writeFile(resolve(evidence, "germination.csv"), csv);
  const input = science.importArtifact("renderer", {
    requestId: randomUUID(),
    projectId: project.id,
    name: "germination.csv",
    dataBase64: csv.toString("base64"),
  });
  const question = science.createQuestion("renderer", {
    requestId: randomUUID(),
    projectId: project.id,
    title: "Can a descriptive result be traced and retrieved?",
    summary:
      "Compute synthetic group means and preserve input, figure, verification and a sourced finding.",
    tags: ["synthetic", "provenance"],
  });
  const source = [
    "import json",
    "import os",
    "",
    "import matplotlib.pyplot as plt",
    "import pandas as pd",
    "",
    "data = pd.read_csv(os.environ['SWARMX_SCIENCE_INPUT_0'])",
    "data['percent'] = 100 * data.germinated / data.total",
    "means = data.groupby('treatment').percent.mean()",
    "assert means['control'] == 79 and means['primed'] == 87",
    "fig, ax = plt.subplots(figsize=(6.4, 4.2), layout='constrained')",
    "ax.bar([0, 1], [means['control'], means['primed']], width=0.52, color=['#9ca3af', '#374151'])",
    "for i, group in enumerate(['control', 'primed']):",
    "    ax.scatter([i-0.08, i, i+0.08], data.loc[data.treatment == group, 'percent'], s=22, color='white', edgecolor='black', zorder=3)",
    "    ax.text(i, means[group]+4, f'{means[group]:.0f}%', ha='center', fontsize=13)",
    "ax.set(xticks=[0, 1], xticklabels=['Control (n = 3)', 'Primed (n = 3)'], ylabel='Germination (%)', ylim=(0, 100), title='Synthetic germination fixture')",
    "ax.spines[['top', 'right']].set_visible(False)",
    "fig.supxlabel('Difference: 8 percentage points | Demonstration data only', fontsize=9)",
    "fig.savefig('germination.png', dpi=180)",
    "plt.close(fig)",
    "print(json.dumps({'controlMeanPercent': float(means['control']), 'primedMeanPercent': float(means['primed']), 'differencePercentagePoints': float(means['primed']-means['control']), 'replicatesPerGroup': 3, 'scope': 'synthetic demonstration; no biological inference'}))",
  ].join("\n");
  await writeFile(resolve(evidence, "analysis.py"), source + "\n");
  await products.attachAgents(host.internalUrl, host.internalToken, undefined, "codex");
  const sessionId = await products.rootAgent.create();
  await exposeUi("running", sessionId);
  const responses = [];
  const observer = {
    text(_id, text, role = "assistant") {
      if (role === "assistant") responses.push(text);
    },
    tool(_id, name, _input, output) {
      if (output !== undefined) console.log("Tool completed: " + name);
    },
    raw() {},
    async interact(request) {
      assert.match(
        request.title,
        /Allow the swarmx MCP server to run tool "(?:science_[a-z_]+|memory|swarm)"\?/,
      );
      assert.deepEqual(request.schema, { type: "object", properties: {} });
      return {};
    },
  };
  const prompt = [
    "Analyze the imported synthetic germination dataset and preserve a traceable result. This is a publication demonstration, not biological evidence. Use only the SwarmX MCP product tools, no shell, file editing, external services or other tools. Use English. Do not create another project.",
    "Exact API shapes (request objects are strict): science_notebook create takes {action:'create',request:{requestId:UUID,projectId:ID,title:TEXT}}; execute takes {action:'execute',request:{requestId:UUID,notebookId:ID,source:PYTHON_SOURCE,inputArtifactIds:[ID],outputArtifact:OBJECT_OR_NULL}}. The field is source, never code; execute has no projectId field. Swarm delegation takes {action:'send_message',agentId:'codex',text:CHILD_PROMPT} directly, with no request wrapper or requestId. Tell the child these exact Science request shapes. science_query head takes {action:'head',request:{id:'sx:a/ARTIFACT_ID'}}. Memory takes {action:'create_memory',request:{type:'Finding',scope:'workspace',title:TEXT,description:TEXT,body:MARKDOWN,sources:[{id:'figure',resource:EXACT_ID,title:TEXT}]}}. science_export takes {action:'project',request:{requestId:UUID,projectId:ID}}. Do not try help actions or consult unrelated history.",
    "Project ID: " +
      project.id +
      "\nDataset artifact ID: " +
      input.id +
      "\nDataset SHA256: " +
      input.digest,
    "1. Create a Science notebook in that project titled Germination figure, then execute the supplied Python code through science_notebook, with inputArtifactIds containing the dataset ID, and outputArtifact:{relativePath:'germination.png',kind:'figure',title:'Synthetic germination: 79% versus 87%',mime:'image/png',license:'CC0-1.0'}. Use fresh UUID requestIds. Keep the returned notebook and artifact IDs.\n" +
      source,
    "2. Delegate a separate arithmetic check via swarm send_message to agentId codex. The child must create a separate Science notebook in project " +
      project.id +
      " and execute Python using only csv, hashlib and json to read the same input artifact " +
      input.id +
      " through SWARMX_SCIENCE_INPUT_0. It must compute means, difference and input digest, with outputArtifact:null, and compare against 79%,87%,8 percentage points and " +
      input.digest +
      ". Give the child those IDs and instructions. Tell it to use only science_notebook, not to delegate or save memory. Require the actual output before continuing.",
    "3. Use science_record record_claim with projectId, hypothesisId:null,status:'supported',title:'Synthetic group means differ by 8 percentage points',summary stating the numerical result and no biological inference,tags:['synthetic']. Use link_evidence with projectId,claimId,relation:'supports',title:'Recomputed input and registered figure',summary of the separate check,tags:['synthetic'],sourceEntityIds containing input and figure artifacts and both notebooks.",
    "4. Use memory create_memory to propose a workspace Finding. Title:Synthetic germination fixture result; description:Checked means and input identity for a synthetic provenance demonstration. Body:79%,87%,8 percentage points,n=3 each,input digest,no biological inference. Cite the figure using sources:[{id:'figure',resource:'sx:a/ACTUAL_FIGURE_ARTIFACT_ID@1',title:'Registered germination figure'}] and a [^figure] footnote. First verify the exact figure reference using science_query head with id:'sx:a/ACTUAL_FIGURE_ARTIFACT_ID'. Use actual returned identifiers. Approval is enabled; report the pending proposal honestly.",
    "5. Export using science_export project. Give a concise final summary of the numerical result, separate check, saved figure and provenance, and pending memory approval. Do not claim approval happened.",
  ].join("\n\n");
  await writeFile(resolve(root, "prompt.txt"), prompt);
  console.log("Starting native analysis and delegated verification.");
  const result = await products.rootAgent.start(sessionId, prompt, observer);
  assert.equal(result.stopReason, "end_turn");
  await writeFile(resolve(root, "analysis-response.txt"), responses.join(""));
  assert.equal(
    (await products.learning.status()).pending.length,
    1,
    "Expected one finding awaiting author approval.",
  );
  await exposeUi("awaiting-memory-approval", sessionId);
  console.log("Awaiting author approval of the finding in browser Settings.");
  while ((await products.learning.status()).pending.length) await setTimeout(1000);
  const memoryGraph = await products.memory.vault.graph(workspaceRoot);
  assert.equal(memoryGraph.nodes.length, 1);
  const memoryId = memoryGraph.nodes[0].id;
  await writeFile(
    resolve(evidence, "finding.md"),
    await readFile(resolve(productHome, "memory/vault", memoryId)),
  );
  const recallId = await products.rootAgent.create();
  responses.length = 0;
  const recall = await products.rootAgent.start(
    recallId,
    "Retrieve the saved result for the synthetic germination fixture using memory. Read the full concept. Report the two group means, difference, replicate count, input identity and scientific limitation. Do not rerun analysis, use external tools or change records. Use English.",
    observer,
  );
  assert.equal(recall.stopReason, "end_turn");
  const recallText = responses.join("");
  for (const value of ["79", "87", "8"]) assert.ok(recallText.includes(value));
  const snapshot = science.getWorkspace("renderer");
  assert.equal(snapshot.artifacts.length, 2);
  const figure = snapshot.artifacts.find((artifact) => artifact.kind === "figure");
  assert.ok(figure);
  const executions = science.getNotebookExecutions("renderer", { projectId: project.id });
  assert.equal(executions.length, 2);
  assert.ok(executions.every((execution) => execution.status === "succeeded"));
  assert.ok(snapshot.records.some((record) => record.kind === "claim"));
  const figureBytes = science.readArtifactContent("renderer", { artifactId: figure.id }).bytes;
  assert.equal(hash(figureBytes), figure.digest);
  await writeFile(resolve(evidence, "germination.png"), figureBytes);
  const exported = science.exportProject("renderer", {
    requestId: randomUUID(),
    projectId: project.id,
  });
  await writeFile(resolve(evidence, "ro-crate-metadata.json"), exported.content);
  const events = products.journal.read({ limit: 100000 }).events;
  const runs = events.filter(({ event }) => event.type === "RUN_STARTED");
  const delegated = runs.filter((run) => run.causedBy !== null);
  assert.equal(delegated.length, 1, "Expected one recorded delegation.");
  await writeJson(
    "native-runs.json",
    runs.map((run) => ({
      id: run.id,
      runId: run.runId,
      sessionId: run.sessionId,
      causedBy: run.causedBy,
      parentRunId: events.find((event) => event.id === run.causedBy)?.runId ?? null,
      finished: events.some(
        (event) => event.runId === run.runId && event.event.type === "RUN_FINISHED",
      ),
    })),
  );
  const sourceRevision = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repo,
    encoding: "utf8",
  }).trim();
  await writeJson("result.json", {
    sourceRevision,
    dataKind: "synthetic",
    result: {
      controlMeanPercent: 79,
      primedMeanPercent: 87,
      differencePercentagePoints: 8,
      replicatesPerGroup: 3,
    },
    projectId: project.id,
    questionId: question.id,
    inputArtifact: input,
    figureArtifact: figure,
    executions,
    notebooks: snapshot.notebooks,
    nativeRunCount: runs.length,
    delegatedRunCount: delegated.length,
    approval: "author approved through authenticated browser",
    memory: { id: memoryId, revision: memoryGraph.nodes[0].revision },
    recall: recallText,
    codeUnmodified:
      execFileSync("git", ["diff", "HEAD", "--", "apps/desktop/src", "packages"], {
        cwd: repo,
        encoding: "utf8",
      }).length === 0,
  });
  const files = [
    "analysis.py",
    "environment.json",
    "finding.md",
    "germination.csv",
    "germination.png",
    "native-runs.json",
    "result.json",
    "ro-crate-metadata.json",
  ];
  await writeFile(
    resolve(evidence, "MANIFEST.sha256"),
    (
      await Promise.all(
        files.map(
          async (name) => hash(await readFile(resolve(evidence, name))).slice(7) + "  " + name,
        ),
      )
    ).join("\n") + "\n",
  );
  await exposeUi("complete", sessionId);
  console.log("Example completed and evidence exported. Host remains available for screenshots.");
} catch (error) {
  await stop();
  throw error;
}
