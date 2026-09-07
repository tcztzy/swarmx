import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { afterEach, describe, expect, it } from "vitest";
import { ProductServices } from "../src/host/product-services.js";
import { ResearchEnvironment } from "../src/host/research-environment.js";
import { SettingsStore } from "../src/host/workspace-settings.js";
import { DEFAULT_POLICY, EnvironmentSchema } from "../src/settings.js";

const roots: string[] = [];
const runtimes: ResearchEnvironment[] = [];
afterEach(async () => {
  await Promise.all(runtimes.splice(0).map((runtime) => runtime.close()));
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-environment-"));
  roots.push(root);
  const workspace = join(root, "workspace");
  const staging = join(root, "staging");
  await mkdir(workspace);
  await mkdir(staging);
  const settings = new SettingsStore(root, "workspace");
  const runtime = new ResearchEnvironment(settings, workspace, staging);
  runtimes.push(runtime);
  return { root, workspace, staging, settings, runtime };
}

describe("research settings", () => {
  it("rejects legacy native policy without modifying the saved settings", async () => {
    const { root, settings } = await fixture();
    settings.write(settings.read());
    const { tools: _tools, ...policy } = DEFAULT_POLICY;
    const legacy = JSON.stringify({
      policy: { ...policy, approval: "untrusted" },
      environment: null,
    });
    await writeFile(settings.path, legacy);
    expect(() => new SettingsStore(root, "workspace")).toThrow(
      "Remove policy.approval and set policy.tools explicitly",
    );
    expect(await readFile(settings.path, "utf8")).toBe(legacy);
  });
  it("persists bounded policy and rejects corrupt or widened configuration", async () => {
    const { root, settings } = await fixture();
    settings.write({ policy: { ...DEFAULT_POLICY, filesystem: "read-only" }, environment: null });
    expect(new SettingsStore(root, "workspace").read().policy.filesystem).toBe("read-only");
    expect(new SettingsStore(root, "other").read().policy.filesystem).toBe("workspace-write");
    expect(() =>
      settings.write({ ...settings.read(), policy: { ...DEFAULT_POLICY, cpus: 0 } }),
    ).toThrow();
    await writeFile(settings.path, '{"policy":"unrestricted"}');
    expect(() => new SettingsStore(root, "workspace")).toThrow();
  });

  it("fails closed without an environment", async () => {
    const { runtime } = await fixture();
    expect(runtime.status().state).toBe("missing");
    await expect(runtime.resolveExecutable("python3")).rejects.toThrow("Settings");
    await expect(runtime.resolveExecutable("../../python3")).rejects.toThrow("inside");
  });
});

describe.skipIf(!process.env.SWARMX_TEST_DOCKER_IMAGE)("real Docker research boundary", () => {
  it("sets up the pinned Jupyter image and records its actual architecture and Python packages", async () => {
    const { settings, runtime } = await fixture();
    const environment = await runtime.setup();
    expect(environment.packages.some((name) => /^jupyterlab==/iu.test(name))).toBe(true);
    expect(environment.packages.some((name) => /^numpy==/iu.test(name))).toBe(true);
    expect(await runtime.inspect()).toEqual(environment);
    expect(settings.read().environment).toEqual(environment);
    expect(runtime.status()).toMatchObject({ state: "ready", activeProcesses: 0 });
  }, 120000);
  it("replays an imported dataset into versioned figures with inspectable code, image and RO-Crate evidence", async () => {
    const { root, workspace } = await fixture();
    const products = await ProductServices.create({
      productHome: root,
      workspace: { id: "research", label: "Research", root: workspace },
    });
    try {
      const inspected = await promisify(execFile)("docker", [
        "image",
        "inspect",
        "--format",
        '{"imageId":"{{.Id}}","platform":"{{.Os}}/{{.Architecture}}"}',
        process.env.SWARMX_TEST_DOCKER_IMAGE ?? "",
      ]);
      const image = EnvironmentSchema.pick({ imageId: true, platform: true }).parse(
        JSON.parse(inspected.stdout),
      );
      products.settings.write({
        policy: DEFAULT_POLICY,
        environment: {
          imageId: image.imageId,
          recipeDigest: `sha256:${"0".repeat(64)}`,
          platform: image.platform,
          createdAt: new Date().toISOString(),
          pythonVersion: "fixture",
          packages: [],
        },
      });
      const science = products.science;
      const project = science.createProject("renderer", {
        requestId: randomUUID(),
        title: "Reproducibility fixture",
      });
      const input = science.importArtifact("renderer", {
        requestId: randomUUID(),
        projectId: project.id,
        name: "measurements.csv",
        dataBase64: Buffer.from("time,response\n1,2\n2,4\n3,6\n").toString("base64"),
      });
      const notebook = science.createNotebook("renderer", {
        requestId: randomUUID(),
        projectId: project.id,
        title: "Figure source",
      });
      const source =
        "import os,pandas as pd,matplotlib.pyplot as plt\ndata = pd.read_csv(os.environ['SWARMX_SCIENCE_INPUT_0'])\nassert data.response.mean() == 4\nfig,ax = plt.subplots()\nax.plot(data.time,data.response,color='blue')\nfig.savefig('figure.svg')\nplt.close(fig)\nprint('mean=4')";
      const request = {
        notebookId: notebook.id,
        inputArtifactIds: [input.id],
        outputArtifact: {
          relativePath: "figure.svg",
          kind: "figure" as const,
          title: "Response",
          mime: "image/svg+xml",
          license: null,
        },
      };
      const original = await science.executeNotebookCell("renderer", {
        ...request,
        requestId: randomUUID(),
        source,
      });
      expect(original.status).toBe("succeeded");
      expect(original.stdout.text).toBe("mean=4\n");
      expect(original.environment.runtimeImage).toBe(image.imageId);
      expect(original.environment.runtimePolicy).toContain("network=none");
      const id = original.artifact?.id ?? "";
      const bytes = science.readArtifactContent("renderer", { artifactId: id }).bytes;
      expect(science.previewArtifact("renderer", { artifactId: id })).toMatchObject({
        kind: "image",
      });
      const edited = await science.executeNotebookCell("renderer", {
        ...request,
        requestId: randomUUID(),
        source: source.replace("color='blue'", "color='red'"),
      });
      expect(edited.status).toBe("succeeded");
      expect(edited.artifact?.digest).not.toBe(original.artifact?.digest);
      expect(edited.notebook.cells.filter((cell) => cell.kind === "code")).toHaveLength(2);
      expect(science.readArtifactContent("renderer", { artifactId: id }).bytes).toEqual(bytes);
      const failed = await science.executeNotebookCell("renderer", {
        ...request,
        requestId: randomUUID(),
        source: "raise ValueError('intentional failure')",
        outputArtifact: { ...request.outputArtifact, reproducibilityMetadata: false },
      });
      expect(failed.status).toBe("failed");
      expect(failed.artifact).toBeNull();
      const crate = science.getResearchObject("renderer", { projectId: project.id });
      const notebookEntity = crate["@graph"].find(
        (entity) => entity["@id"] === `urn:uuid:${notebook.id}`,
      );
      expect(notebookEntity?.isBasedOn).toContainEqual({ "@id": `urn:uuid:${input.id}` });
      expect(edited.notebook.cells[0]?.runtimeEnvironment.runtimeImage).toBe(image.imageId);
      expect(science.getWorkspace("renderer").artifacts).toHaveLength(3);
    } finally {
      await products.dispose();
    }
  }, 90000);

  it("confines reads, writes, network and credentials; preserves declared inputs and kills children", async () => {
    const { root, workspace, staging, settings, runtime } = await fixture();
    const exec = promisify(execFile);
    const inspected = await exec("docker", [
      "image",
      "inspect",
      "--format",
      '{"imageId":"{{.Id}}","platform":"{{.Os}}/{{.Architecture}}"}',
      process.env.SWARMX_TEST_DOCKER_IMAGE ?? "",
    ]);
    const image = EnvironmentSchema.pick({ imageId: true, platform: true }).parse(
      JSON.parse(inspected.stdout),
    );
    settings.write({
      policy: DEFAULT_POLICY,
      environment: {
        imageId: image.imageId,
        recipeDigest: `sha256:${"0".repeat(64)}`,
        platform: image.platform,
        createdAt: new Date().toISOString(),
        pythonVersion: "test",
        packages: [],
      },
    });
    await writeFile(join(root, "secret"), "outside workspace");
    await writeFile(join(staging, "input.csv"), "value\n42\n");
    await symlink(root, join(workspace, "escape"));
    const run = async (source: string, signal?: AbortSignal) =>
      runtime.spawn({
        argv: ["python3", "-c", source],
        cwd: workspace,
        env: {
          OPENAI_API_KEY: "must-not-enter",
          SWARMX_SCIENCE_INPUT_0: join(staging, "input.csv"),
        },
        graceMs: 300,
        signal,
        stdio: { stdin: "ignore", stdout: { maxBytes: 4096 }, stderr: { maxBytes: 4096 } },
      });
    const execution = await run(
      [
        "import os,socket,pathlib",
        "assert 'OPENAI_API_KEY' not in os.environ",
        `assert not pathlib.Path(${JSON.stringify(join(root, "secret"))}).exists()`,
        "assert not pathlib.Path('escape/secret').exists()",
        "assert pathlib.Path(os.environ['SWARMX_SCIENCE_INPUT_0']).read_text() == 'value\\n42\\n'",
        "try: socket.create_connection(('1.1.1.1',443),timeout=1); raise AssertionError('network open')",
        "except OSError: pass",
        "pathlib.Path('result.txt').write_text('isolated')",
      ].join("\n"),
    );
    expect(await execution.done).toMatchObject({ exitCode: 0 });
    expect(await readFile(join(workspace, "result.txt"), "utf8")).toBe("isolated");
    settings.write({ ...settings.read(), policy: { ...DEFAULT_POLICY, filesystem: "read-only" } });
    const readonly = await run("open('blocked','w').write('bad')");
    expect((await readonly.done).exitCode).not.toBe(0);
    await expect(
      runtime.spawn({
        argv: ["python3"],
        cwd: join(workspace, "escape"),
        graceMs: 100,
        stdio: { stdin: "ignore", stdout: "pipe", stderr: "pipe" },
      }),
    ).rejects.toThrow("outside");
    const controller = new AbortController();
    const running = await run(
      "import subprocess,time; subprocess.Popen(['sleep','60']); print('ready',flush=True); time.sleep(60)",
      controller.signal,
    );
    await new Promise<void>((resolve) => running.stdout?.once("data", () => resolve()));
    controller.abort();
    await running.done;
    expect(runtime.status().activeProcesses).toBe(0);
    const containers = await exec("docker", ["ps", "-aq", "--filter", `volume=${workspace}`]);
    expect(containers.stdout.trim()).toBe("");
  }, 60000);
});
