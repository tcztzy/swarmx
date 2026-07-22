import { spawn } from "node:child_process";
import { constants } from "node:fs";
import { access, stat } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { type AgentBackend, HARNESSES, SWARMX_VERSION } from "@swarmx/core";

const VERSION_TIMEOUT_MS = 8_000;
const INSTALL_TIMEOUT_MS = 15 * 60_000;
const CONTAINER_SETUP_TIMEOUT_MS = 20 * 60_000;
const CONTAINER_SERVICE_TIMEOUT_MS = 2 * 60_000;
const OUTPUT_LIMIT = 64 * 1024;
const DEFAULT_CONTAINER_CPUS = "2";
const DEFAULT_CONTAINER_MEMORY = "4G";

export type HarnessRequirementStatus = "ready" | "missing" | "unsupported" | "failed";
export type HarnessEnvironmentHarnessState = "ready" | "needs_setup" | "unsupported";
export type ContainerRuntimeId = "apple_container";
export type ContainerRuntimeStatus =
  | "ready"
  | "missing"
  | "unsupported"
  | "service_stopped"
  | "failed";
export type HarnessProtectionMode = "protected" | "native";

export interface HarnessRuntimeRequirement {
  id: string;
  label: string;
  command: string;
  status: HarnessRequirementStatus;
  installable: boolean;
  requiredBy: string[];
  path?: string;
  version?: string;
  note?: string;
}

export interface HarnessEnvironmentHarness {
  harnessId: string;
  harnessLabel: string;
  command?: string;
  installable?: boolean;
  path?: string;
  /** Three-part version reported by this harness's installed CLI, when available. */
  version?: string;
  status: HarnessEnvironmentHarnessState;
  requirements: string[];
  executionMode: HarnessProtectionMode;
  protectionRequired: boolean;
  containerRuntimeId?: ContainerRuntimeId;
  note?: string;
}

export interface HarnessContainerRuntime {
  id: ContainerRuntimeId;
  label: string;
  command: string;
  status: ContainerRuntimeStatus;
  supported: boolean;
  installable: boolean;
  serviceReady: boolean;
  preferred: boolean;
  path?: string;
  version?: string;
  note?: string;
}

export interface HarnessProtectionSummary {
  mode: HarnessProtectionMode;
  ready: boolean;
  requiredHarnessIds: string[];
  selectedRuntimeId?: ContainerRuntimeId;
  note?: string;
}

export interface HarnessEnvironmentStatus {
  checkedAt: string;
  path: string;
  ready: boolean;
  setupAvailable: boolean;
  containerRuntimes: HarnessContainerRuntime[];
  protection: HarnessProtectionSummary;
  requirements: HarnessRuntimeRequirement[];
  harnesses: HarnessEnvironmentHarness[];
}

export interface HarnessVersionCheck {
  harnessId: string;
  version?: string;
}

export interface HarnessEnvironmentSetupRequest {
  harnessId?: string;
  harnessToolId?: string;
  requirementIds?: string[];
  containerRuntimeId?: ContainerRuntimeId;
  includeContainerRuntime?: boolean;
}

export interface HarnessEnvironmentSetupResult {
  success: boolean;
  status: HarnessEnvironmentStatus;
  installedRequirementIds: string[];
  skippedRequirementIds: string[];
  failedRequirementIds: string[];
  installedContainerRuntimeIds: ContainerRuntimeId[];
  skippedContainerRuntimeIds: ContainerRuntimeId[];
  failedContainerRuntimeIds: ContainerRuntimeId[];
  log: string[];
  error?: string;
}

export interface ProtectedHarnessBackendResult {
  success: boolean;
  backend?: AgentBackend;
  mode: HarnessProtectionMode;
  runtimeId?: ContainerRuntimeId;
  error?: string;
}

interface RequirementDefinition {
  id: string;
  label: string;
  command: string;
  versionArgs: string[];
  notes?: Partial<Record<NodeJS.Platform | "default", string>>;
  installers: Partial<Record<NodeJS.Platform | "default", InstallerDefinition>>;
}

interface InstallerDefinition {
  script: string;
  timeoutMs?: number;
}

interface ContainerRuntimeDefinition {
  id: ContainerRuntimeId;
  label: string;
  command: string;
  installers: Partial<Record<NodeJS.Platform | "default", InstallerDefinition>>;
}

interface ProtectedHarnessDefinition {
  image: string;
  command: string[];
  note?: string;
}

interface CommandResult {
  exitCode: number | null;
  stdout: string;
  stderr: string;
  error?: string;
}

export interface HarnessEnvironmentHost {
  env?: NodeJS.ProcessEnv;
  platform?: NodeJS.Platform;
  arch?: string;
  macosVersion?: string;
  protectionMode?: HarnessProtectionMode;
  homeDir?: string;
  now?: () => Date;
  findExecutable?: (
    command: string,
    envPath: string,
    platform: NodeJS.Platform,
  ) => Promise<string | null>;
  runCommand?: (
    program: string,
    args: string[],
    options: { env: NodeJS.ProcessEnv; timeoutMs: number },
  ) => Promise<CommandResult>;
}

const HARNESS_REQUIREMENTS: Record<string, string[]> = {
  swarmx: [],
  claude_code: ["claude_code"],
  codex: ["codex"],
  pi: ["pi"],
  opencode: ["opencode"],
  hermes: ["hermes"],
  openclaw: ["openclaw"],
};

const HARNESS_VERSION_COMMANDS: Record<string, string> = {
  swarmx: "swarmx",
  claude_code: "claude",
  codex: "codex",
  pi: "pi",
  opencode: "opencode",
  hermes: "hermes",
  openclaw: "openclaw",
};

const REQUIREMENTS: RequirementDefinition[] = [
  {
    id: "node",
    label: "Node.js runtime",
    command: "node",
    versionArgs: ["--version"],
    notes: {
      default: "Install an active Node.js LTS release with your preferred version manager.",
    },
    installers: {},
  },
  {
    id: "claude_code",
    label: "Claude Code",
    command: "claude",
    versionArgs: ["--version"],
    installers: {
      default: { script: "npm install --global @anthropic-ai/claude-code" },
    },
  },
  {
    id: "codex",
    label: "Codex",
    command: "codex",
    versionArgs: ["--version"],
    installers: {
      default: { script: "npm install --global @openai/codex" },
    },
  },
  {
    id: "pi",
    label: "Pi coding agent",
    command: "pi",
    versionArgs: ["--version"],
    notes: {
      default: "Requires Node.js 22.19 or newer; configure provider login with Pi itself.",
    },
    installers: {
      default: {
        script: "npm install --global --ignore-scripts @earendil-works/pi-coding-agent",
      },
    },
  },
  {
    id: "opencode",
    label: "OpenCode CLI",
    command: "opencode",
    versionArgs: ["--version"],
    installers: {
      default: { script: "npm install --global opencode-ai" },
    },
  },
  {
    id: "hermes",
    label: "Hermes Agent CLI",
    command: "hermes",
    versionArgs: ["--version"],
    installers: {
      default: { script: "curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash" },
      win32: { script: "iex (irm https://hermes-agent.nousresearch.com/install.ps1)" },
    },
  },
  {
    id: "openclaw",
    label: "OpenClaw CLI",
    command: "openclaw",
    versionArgs: ["--version"],
    notes: {
      default: "CLI detection does not prove Gateway authentication or reachability.",
    },
    installers: {
      default: {
        script: "curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard",
      },
      win32: {
        script: "& ([scriptblock]::Create((iwr -useb https://openclaw.ai/install.ps1))) -NoOnboard",
      },
    },
  },
];

const REQUIREMENT_BY_ID = new Map(REQUIREMENTS.map((requirement) => [requirement.id, requirement]));

const APPLE_CONTAINER_INSTALL_SCRIPT = String.raw`set -euo pipefail
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
release_url="$(curl -fsSLI -o /dev/null -w '%{url_effective}' https://github.com/apple/container/releases/latest)"
version="\${release_url##*/}"
case "$version" in
  ''|latest|releases) echo "Could not resolve latest apple/container release." >&2; exit 1 ;;
esac
pkg="$tmpdir/container-\${version}-installer-signed.pkg"
curl -fL --retry 3 -o "$pkg" "https://github.com/apple/container/releases/download/\${version}/container-\${version}-installer-signed.pkg"
pkg_escaped="$(printf "%q" "$pkg")"
osascript -e "do shell script \"/usr/sbin/installer -pkg $pkg_escaped -target /\" with administrator privileges"`;

const CONTAINER_RUNTIMES: ContainerRuntimeDefinition[] = [
  {
    id: "apple_container",
    label: "Apple Container",
    command: "container",
    installers: {
      darwin: { script: APPLE_CONTAINER_INSTALL_SCRIPT, timeoutMs: CONTAINER_SETUP_TIMEOUT_MS },
    },
  },
];

const PROTECTED_HARNESS_DEFINITIONS: Record<string, ProtectedHarnessDefinition> = {
  claude_code: {
    image: "node:22-slim",
    command: ["npx", "--yes", "@agentclientprotocol/claude-agent-acp@0.58.1"],
  },
  codex: {
    image: "node:22-slim",
    command: ["npx", "--yes", "@agentclientprotocol/codex-acp@1.1.2"],
  },
};

const CONTAINER_ENV_KEYS = [
  "ANTHROPIC_API_KEY",
  "ANTHROPIC_AUTH_TOKEN",
  "ANTHROPIC_BASE_URL",
  "ANTHROPIC_DEFAULT_HAIKU_MODEL",
  "ANTHROPIC_DEFAULT_OPUS_MODEL",
  "ANTHROPIC_DEFAULT_SONNET_MODEL",
  "ANTHROPIC_MODEL",
  "ANTHROPIC_CUSTOM_MODEL_OPTION",
  "CLAUDE_CODE_EFFORT_LEVEL",
  "CLAUDE_CODE_SUBAGENT_MODEL",
  "CLAUDE_MODEL_CONFIG",
  "CODEX_API_KEY",
  "CODEX_CONFIG",
  "MODEL_PROVIDER",
  "OPENAI_API_KEY",
  "OPENAI_BASE_URL",
  "OPENAI_MODEL",
  "OPENCODE_CONFIG",
  "OPENCODE_CONFIG_CONTENT",
  "OLLAMA_BASE_URL",
  "OLLAMA_HOST",
  "OLLAMA_MODEL",
  "YALLM_DEFAULT_PROVIDER",
  "LANG",
  "TERM",
];

export function configureDesktopHarnessEnvironment(
  env: NodeJS.ProcessEnv = process.env,
  homeDir: string = os.homedir(),
): string {
  const mergedPath = mergePath(env.PATH ?? "", commonExecutableDirs(homeDir));
  env.PATH = mergedPath;
  return mergedPath;
}

export class HarnessEnvironmentService {
  private readonly env: NodeJS.ProcessEnv;
  private readonly platform: NodeJS.Platform;
  private readonly arch: string;
  private readonly macosVersion?: string;
  private readonly protectionMode: HarnessProtectionMode;
  private readonly homeDir: string;
  private readonly now: () => Date;
  private readonly findExecutable: NonNullable<HarnessEnvironmentHost["findExecutable"]>;
  private readonly runCommand: NonNullable<HarnessEnvironmentHost["runCommand"]>;
  private readonly harnessVersionCache = new Map<string, string | undefined>();
  private readonly harnessVersionChecks = new Map<string, Promise<string | undefined>>();
  private setupInFlight: Promise<HarnessEnvironmentSetupResult> | null = null;

  constructor(host: HarnessEnvironmentHost = {}) {
    this.env = host.env ?? process.env;
    this.platform = host.platform ?? process.platform;
    this.arch = host.arch ?? os.arch();
    this.macosVersion = host.macosVersion;
    this.protectionMode =
      host.protectionMode ??
      (this.env.SWARMX_HARNESS_PROTECTION === "native" ? "native" : "protected");
    this.homeDir = host.homeDir ?? os.homedir();
    this.now = host.now ?? (() => new Date());
    this.findExecutable = host.findExecutable ?? findExecutableOnPath;
    this.runCommand = host.runCommand ?? runCommand;
    configureDesktopHarnessEnvironment(this.env, this.homeDir);
  }

  async status(): Promise<HarnessEnvironmentStatus> {
    configureDesktopHarnessEnvironment(this.env, this.homeDir);
    const envPath = this.env.PATH ?? "";
    const [requirements, containerRuntimes, harnessVersions] = await Promise.all([
      Promise.all(REQUIREMENTS.map((requirement) => this.detectRequirement(requirement, envPath))),
      this.detectContainerRuntimes(envPath),
      this.detectHarnessVersions(envPath),
    ]);
    const protection = this.protectionSummary(containerRuntimes);
    const nodeRuntime = requirements.find((requirement) => requirement.id === "node");
    const harnesses = Object.entries(HARNESSES).map(([harnessId, harness]) => {
      const requirementIds = HARNESS_REQUIREMENTS[harnessId] ?? [];
      const harnessRequirement = requirements.find((requirement) =>
        requirementIds.includes(requirement.id),
      );
      const command = HARNESS_VERSION_COMMANDS[harnessId] ?? harnessId;
      const protectionRequired = this.harnessRequiresProtection(harnessId);
      const executionMode =
        this.protectionMode === "protected" && protectionRequired ? "protected" : "native";
      if (executionMode === "protected") {
        const definition = PROTECTED_HARNESS_DEFINITIONS[harnessId];
        const unsupported = !definition || containerRuntimes.some((runtime) => !runtime.supported);
        const ready = Boolean(definition && protection.ready);
        return {
          harnessId,
          harnessLabel: harness.label,
          command,
          installable: harnessRequirement?.installable ?? false,
          path: harnessRequirement?.path,
          version: harnessVersions.get(harnessId),
          status: ready ? "ready" : unsupported ? "unsupported" : "needs_setup",
          requirements: requirementIds,
          executionMode,
          protectionRequired,
          containerRuntimeId: protection.selectedRuntimeId,
          note: ready
            ? "Protected by container runtime."
            : (definition?.note ?? protection.note ?? "Container runtime setup required."),
        } satisfies HarnessEnvironmentHarness;
      }

      const harnessRequirements = requirements.filter((requirement) =>
        requirementIds.includes(requirement.id),
      );
      const unsupported = harnessRequirements.some(
        (requirement) => requirement.status === "unsupported",
      );
      const ready = harnessRequirements.every((requirement) => requirement.status === "ready");
      return {
        harnessId,
        harnessLabel: harness.label,
        command,
        installable: harnessRequirement?.installable ?? false,
        path: harnessRequirement?.path,
        version:
          harnessVersions.get(harnessId) ?? (harnessId === "swarmx" ? SWARMX_VERSION : undefined),
        status: ready ? "ready" : unsupported ? "unsupported" : "needs_setup",
        requirements: requirementIds,
        executionMode,
        protectionRequired,
        note:
          harnessId === "openclaw"
            ? "CLI ready; a configured and reachable OpenClaw Gateway is also required."
            : harnessId === "pi"
              ? "Runs natively; provider login, settings, and sessions remain owned by Pi."
              : harnessId === "opencode" || harnessId === "hermes"
                ? "Runs natively so the installed CLI and user configuration remain available."
                : requirementIds.length === 0
                  ? "Built in."
                  : undefined,
      } satisfies HarnessEnvironmentHarness;
    });

    return {
      checkedAt: this.now().toISOString(),
      path: envPath,
      ready: nodeRuntime?.status === "ready",
      setupAvailable: Boolean(nodeRuntime?.status !== "ready" && nodeRuntime?.installable),
      containerRuntimes,
      protection,
      requirements,
      harnesses,
    };
  }

  async harnessVersion(harnessId: string, refresh = false): Promise<HarnessVersionCheck> {
    if (!Object.hasOwn(HARNESSES, harnessId)) return { harnessId };
    configureDesktopHarnessEnvironment(this.env, this.homeDir);
    const version = await this.detectHarnessVersion(harnessId, this.env.PATH ?? "", refresh);
    return { harnessId, ...(version ? { version } : {}) };
  }

  async setup(
    request: HarnessEnvironmentSetupRequest = {},
  ): Promise<HarnessEnvironmentSetupResult> {
    if (this.setupInFlight) return this.setupInFlight;
    this.setupInFlight = this.runSetup(request).finally(() => {
      this.setupInFlight = null;
    });
    return this.setupInFlight;
  }

  async protectedBackendForHarness(
    harnessId: string,
    backend: AgentBackend,
    options: { workspaceDir?: string } = {},
  ): Promise<ProtectedHarnessBackendResult> {
    if (backend.type !== "custom" || !this.harnessRequiresProtection(harnessId)) {
      return { success: true, backend, mode: "native" };
    }

    if (this.protectionMode !== "protected") {
      return { success: true, backend, mode: "native" };
    }

    const status = await this.status();
    if (!status.protection.ready || !status.protection.selectedRuntimeId) {
      return {
        success: false,
        mode: "protected",
        runtimeId: status.protection.selectedRuntimeId,
        error: status.protection.note ?? "Protected harness runtime is not ready.",
      };
    }

    const definition = PROTECTED_HARNESS_DEFINITIONS[harnessId];
    if (!definition) {
      return {
        success: false,
        mode: "protected",
        runtimeId: status.protection.selectedRuntimeId,
        error: `Harness "${harnessId}" does not declare a protected container image.`,
      };
    }

    const credentialArgs = await this.protectedCredentialArgs(harnessId);
    return {
      success: true,
      backend: this.containerWrappedBackend(
        definition,
        options.workspaceDir ?? process.cwd(),
        credentialArgs,
      ),
      mode: "protected",
      runtimeId: status.protection.selectedRuntimeId,
    };
  }

  guessProtectedHarnessId(backend: AgentBackend): string | null {
    if (backend.type !== "custom") return null;
    for (const [harnessId, harness] of Object.entries(HARNESSES)) {
      if (harness.enabled === false) continue;
      if (harness.backend.type !== "custom") continue;
      if (harness.backend.program !== backend.program) continue;
      const expectedArgs = harness.backend.args ?? [];
      const actualArgs = backend.args ?? [];
      if (
        expectedArgs.length === actualArgs.length &&
        expectedArgs.every((arg, i) => arg === actualArgs[i])
      ) {
        return harnessId;
      }
    }
    const commandLine = [backend.program, ...(backend.args ?? [])].join(" ");
    if (commandLine.includes("@agentclientprotocol/codex-acp")) return "codex";
    if (commandLine.includes("@agentclientprotocol/claude-agent-acp")) return "claude_code";
    if (commandLine.includes("pi-acp")) return "pi";
    if (backend.program === "opencode") return "opencode";
    if (backend.program === "hermes") return "hermes";
    if (backend.program === "openclaw") return "openclaw";
    return null;
  }

  private async runSetup(
    request: HarnessEnvironmentSetupRequest,
  ): Promise<HarnessEnvironmentSetupResult> {
    const before = await this.status();
    const selectedIds = this.selectedRequirementIds(request);
    const installedRequirementIds: string[] = [];
    const skippedRequirementIds: string[] = [];
    const failedRequirementIds: string[] = [];
    const installedContainerRuntimeIds: ContainerRuntimeId[] = [];
    const skippedContainerRuntimeIds: ContainerRuntimeId[] = [];
    const failedContainerRuntimeIds: ContainerRuntimeId[] = [];
    const log: string[] = [];

    if (this.shouldSetupContainerRuntime(request)) {
      const containerResult = await this.setupContainerRuntime(before, log);
      installedContainerRuntimeIds.push(...containerResult.installed);
      skippedContainerRuntimeIds.push(...containerResult.skipped);
      failedContainerRuntimeIds.push(...containerResult.failed);
    }

    for (const requirementId of selectedIds) {
      const current = before.requirements.find((requirement) => requirement.id === requirementId);
      const definition = REQUIREMENT_BY_ID.get(requirementId);
      if (!definition || current?.status === "ready") {
        skippedRequirementIds.push(requirementId);
        continue;
      }

      const installer = installerForPlatform(definition, this.platform);
      if (!installer) {
        skippedRequirementIds.push(requirementId);
        continue;
      }

      const command = shellCommand(this.platform, installer.script);
      log.push(`Starting ${definition.label} setup.`);
      const result = await this.runCommand(command.program, command.args, {
        env: this.env,
        timeoutMs: installer.timeoutMs ?? INSTALL_TIMEOUT_MS,
      });
      appendCommandLog(log, result);
      if (result.exitCode === 0 && !result.error) {
        installedRequirementIds.push(requirementId);
        configureDesktopHarnessEnvironment(this.env, this.homeDir);
      } else {
        failedRequirementIds.push(requirementId);
      }
    }

    this.invalidateHarnessVersions(request, selectedIds);
    const status = await this.status();
    const stillMissing = selectedIds.filter((requirementId) => {
      const requirement = status.requirements.find((item) => item.id === requirementId);
      return requirement?.status !== "ready";
    });

    return {
      success:
        failedRequirementIds.length === 0 &&
        stillMissing.length === 0 &&
        failedContainerRuntimeIds.length === 0,
      status,
      installedRequirementIds,
      skippedRequirementIds,
      failedRequirementIds: uniqueStrings([...failedRequirementIds, ...stillMissing]),
      installedContainerRuntimeIds,
      skippedContainerRuntimeIds,
      failedContainerRuntimeIds,
      log,
      error:
        failedRequirementIds.length > 0 ||
        stillMissing.length > 0 ||
        failedContainerRuntimeIds.length > 0
          ? "One or more runtime tools are still unavailable."
          : undefined,
    };
  }

  private selectedRequirementIds(request: HarnessEnvironmentSetupRequest): string[] {
    if (request.requirementIds?.length) {
      return uniqueStrings(request.requirementIds.filter((id) => REQUIREMENT_BY_ID.has(id)));
    }

    if (request.harnessToolId) {
      return REQUIREMENT_BY_ID.has(request.harnessToolId) ? [request.harnessToolId] : [];
    }

    if (request.harnessId) {
      if (
        this.protectionMode === "protected" &&
        this.harnessRequiresProtection(request.harnessId)
      ) {
        return [];
      }
      return uniqueStrings(HARNESS_REQUIREMENTS[request.harnessId] ?? []);
    }

    return [];
  }

  private async detectContainerRuntimes(envPath: string): Promise<HarnessContainerRuntime[]> {
    return Promise.all(
      CONTAINER_RUNTIMES.map((runtime) => this.detectContainerRuntime(runtime, envPath)),
    );
  }

  private async detectContainerRuntime(
    runtime: ContainerRuntimeDefinition,
    envPath: string,
  ): Promise<HarnessContainerRuntime> {
    if (runtime.id !== "apple_container") {
      return {
        id: runtime.id,
        label: runtime.label,
        command: runtime.command,
        status: "unsupported",
        supported: false,
        installable: false,
        serviceReady: false,
        preferred: false,
        note: "Unsupported container runtime.",
      };
    }

    const support = await this.appleContainerSupport();
    const installer = installerForPlatform(runtime, this.platform);
    if (!support.supported) {
      return {
        id: runtime.id,
        label: runtime.label,
        command: runtime.command,
        status: "unsupported",
        supported: false,
        installable: false,
        serviceReady: false,
        preferred: true,
        note: support.note,
      };
    }

    const foundPath = await this.findExecutable(runtime.command, envPath, this.platform);
    if (!foundPath) {
      return {
        id: runtime.id,
        label: runtime.label,
        command: runtime.command,
        status: installer ? "missing" : "unsupported",
        supported: true,
        installable: Boolean(installer),
        serviceReady: false,
        preferred: true,
        note: installer
          ? "Apple Container must be installed and its system service started."
          : "No installer is available for this platform.",
      };
    }

    const version = await this.readVersion(foundPath, ["--version"]);
    if (version.status !== "ready") {
      return {
        id: runtime.id,
        label: runtime.label,
        command: runtime.command,
        status: "failed",
        supported: true,
        installable: Boolean(installer),
        serviceReady: false,
        preferred: true,
        path: foundPath,
        note: version.note,
      };
    }

    const service = await this.runCommand(foundPath, ["system", "status"], {
      env: this.env,
      timeoutMs: VERSION_TIMEOUT_MS,
    });
    const serviceReady = service.exitCode === 0 && /\bstatus\s+running\b/i.test(service.stdout);
    return {
      id: runtime.id,
      label: runtime.label,
      command: runtime.command,
      status: serviceReady ? "ready" : "service_stopped",
      supported: true,
      installable: true,
      serviceReady,
      preferred: true,
      path: foundPath,
      version: version.version,
      note: serviceReady
        ? "Apple Container system service is running."
        : (firstLine(service.stderr) ??
          firstLine(service.stdout) ??
          "Apple Container service is stopped."),
    };
  }

  private async appleContainerSupport(): Promise<{ supported: boolean; note?: string }> {
    if (this.platform !== "darwin") {
      return { supported: false, note: "Apple Container is only supported on macOS." };
    }
    if (this.arch !== "arm64") {
      return { supported: false, note: "Apple Container requires Apple silicon." };
    }
    const version = this.macosVersion ?? (await this.detectMacosVersion());
    const major = Number.parseInt(version.split(".")[0] ?? "", 10);
    if (!Number.isFinite(major)) {
      return { supported: false, note: "Could not verify the macOS version for Apple Container." };
    }
    if (major < 26) {
      return { supported: false, note: "Apple Container requires macOS 26 or newer." };
    }
    return { supported: true };
  }

  private async detectMacosVersion(): Promise<string> {
    const result = await this.runCommand("sw_vers", ["-productVersion"], {
      env: this.env,
      timeoutMs: VERSION_TIMEOUT_MS,
    });
    return firstLine(result.stdout) ?? "";
  }

  private protectionSummary(
    containerRuntimes: HarnessContainerRuntime[],
  ): HarnessProtectionSummary {
    const requiredHarnessIds = Object.keys(PROTECTED_HARNESS_DEFINITIONS);
    if (this.protectionMode !== "protected") {
      return {
        mode: "native",
        ready: true,
        requiredHarnessIds,
        note: "Protected harness execution is disabled.",
      };
    }

    const appleContainer = containerRuntimes.find((runtime) => runtime.id === "apple_container");
    if (appleContainer?.status === "ready") {
      return {
        mode: "protected",
        ready: true,
        requiredHarnessIds,
        selectedRuntimeId: "apple_container",
        note: "Protected harness execution uses Apple Container.",
      };
    }

    return {
      mode: "protected",
      ready: false,
      requiredHarnessIds,
      selectedRuntimeId: appleContainer?.supported ? "apple_container" : undefined,
      note:
        appleContainer?.note ??
        "Protected harness execution requires Apple Container setup on this Mac.",
    };
  }

  private shouldSetupContainerRuntime(request: HarnessEnvironmentSetupRequest): boolean {
    if (this.protectionMode !== "protected") return false;
    if (request.harnessToolId) return false;
    if (request.containerRuntimeId || request.includeContainerRuntime) return true;
    if (request.harnessId && this.harnessRequiresProtection(request.harnessId)) return true;
    return false;
  }

  private containerRuntimeCanSetup(runtime: HarnessContainerRuntime): boolean {
    return (
      runtime.supported &&
      (runtime.installable || runtime.status === "service_stopped") &&
      runtime.status !== "ready"
    );
  }

  private async setupContainerRuntime(
    status: HarnessEnvironmentStatus,
    log: string[],
  ): Promise<{
    installed: ContainerRuntimeId[];
    skipped: ContainerRuntimeId[];
    failed: ContainerRuntimeId[];
  }> {
    const runtimeId: ContainerRuntimeId = "apple_container";
    const runtime = status.containerRuntimes.find((item) => item.id === runtimeId);
    const installed: ContainerRuntimeId[] = [];
    const skipped: ContainerRuntimeId[] = [];
    const failed: ContainerRuntimeId[] = [];

    if (!runtime || runtime.status === "ready") {
      skipped.push(runtimeId);
      return { installed, skipped, failed };
    }
    if (!this.containerRuntimeCanSetup(runtime)) {
      skipped.push(runtimeId);
      log.push(runtime?.note ?? "Apple Container is not supported on this machine.");
      return { installed, skipped, failed };
    }

    if (runtime.status === "missing") {
      const definition = CONTAINER_RUNTIMES.find((item) => item.id === runtimeId);
      const installer = definition ? installerForPlatform(definition, this.platform) : undefined;
      if (!installer) {
        skipped.push(runtimeId);
        return { installed, skipped, failed };
      }
      const command = shellCommand(this.platform, installer.script);
      log.push("Starting Apple Container setup.");
      const installResult = await this.runCommand(command.program, command.args, {
        env: this.env,
        timeoutMs: installer.timeoutMs ?? CONTAINER_SETUP_TIMEOUT_MS,
      });
      appendCommandLog(log, installResult);
      if (installResult.exitCode !== 0 || installResult.error) {
        failed.push(runtimeId);
        return { installed, skipped, failed };
      }
      installed.push(runtimeId);
      configureDesktopHarnessEnvironment(this.env, this.homeDir);
    }

    log.push("Starting Apple Container system service.");
    const startResult = await this.runCommand("container", ["system", "start"], {
      env: this.env,
      timeoutMs: CONTAINER_SERVICE_TIMEOUT_MS,
    });
    appendCommandLog(log, startResult);
    if (startResult.exitCode !== 0 || startResult.error) {
      failed.push(runtimeId);
    }

    return { installed, skipped, failed };
  }

  private harnessRequiresProtection(harnessId: string): boolean {
    return Boolean(PROTECTED_HARNESS_DEFINITIONS[harnessId]);
  }

  private async protectedCredentialArgs(harnessId: string): Promise<string[]> {
    if (harnessId !== "codex") return [];
    const authPath = this.codexAuthPath();
    try {
      const metadata = await stat(authPath);
      if (!metadata.isFile() || (this.platform !== "win32" && (metadata.mode & 0o077) !== 0)) {
        return [];
      }
    } catch {
      return [];
    }
    return ["--volume", `${authPath}:/tmp/auth.json:ro`, "--env", "CODEX_HOME=/tmp"];
  }

  private codexAuthPath(): string {
    const configuredHome = this.env.CODEX_HOME?.trim();
    if (!configuredHome) return path.join(this.homeDir, ".codex", "auth.json");
    const expandedHome =
      configuredHome === "~"
        ? this.homeDir
        : configuredHome.startsWith("~/")
          ? path.join(this.homeDir, configuredHome.slice(2))
          : path.resolve(configuredHome);
    return path.join(expandedHome, "auth.json");
  }

  private containerWrappedBackend(
    definition: ProtectedHarnessDefinition,
    workspaceDir: string,
    credentialArgs: string[] = [],
  ): AgentBackend {
    const workspace = path.resolve(workspaceDir);
    return {
      type: "custom",
      program: "container",
      args: [
        "run",
        "--rm",
        "-i",
        "--init",
        "--progress",
        "none",
        "--cpus",
        DEFAULT_CONTAINER_CPUS,
        "--memory",
        DEFAULT_CONTAINER_MEMORY,
        "--platform",
        "linux/arm64",
        "--workdir",
        workspace,
        "--mount",
        `type=bind,source=${workspace},target=${workspace}`,
        "--tmpfs",
        "/tmp",
        ...credentialArgs,
        ...containerEnvArgs(),
        definition.image,
        ...definition.command,
      ],
    };
  }

  private async detectRequirement(
    requirement: RequirementDefinition,
    envPath: string,
  ): Promise<HarnessRuntimeRequirement> {
    const requiredBy = Object.entries(HARNESS_REQUIREMENTS)
      .filter(([, requirementIds]) => requirementIds.includes(requirement.id))
      .map(([harnessId]) => harnessId);
    const installer = installerForPlatform(requirement, this.platform);
    const foundPath = await this.findExecutable(requirement.command, envPath, this.platform);
    if (!foundPath) {
      return {
        id: requirement.id,
        label: requirement.label,
        command: requirement.command,
        status: installer ? "missing" : "unsupported",
        installable: Boolean(installer),
        requiredBy,
        note: noteForPlatform(requirement, this.platform),
      };
    }

    const version = await this.readVersion(foundPath, requirement.versionArgs);
    return {
      id: requirement.id,
      label: requirement.label,
      command: requirement.command,
      status: version.status,
      installable: Boolean(installer),
      requiredBy,
      path: foundPath,
      version: version.version,
      note: version.note ?? noteForPlatform(requirement, this.platform),
    };
  }

  private async detectHarnessVersions(envPath: string): Promise<Map<string, string>> {
    const versions = await Promise.all(
      Object.keys(HARNESS_VERSION_COMMANDS).map(
        async (harnessId) =>
          [harnessId, await this.detectHarnessVersion(harnessId, envPath)] as const,
      ),
    );
    const detectedVersions = new Map<string, string>();
    for (const [harnessId, version] of versions) {
      if (version) detectedVersions.set(harnessId, version);
    }
    return detectedVersions;
  }

  private async detectHarnessVersion(
    harnessId: string,
    envPath: string,
    refresh = false,
  ): Promise<string | undefined> {
    if (!refresh && this.harnessVersionCache.has(harnessId)) {
      return this.harnessVersionCache.get(harnessId);
    }
    if (!refresh) {
      const pending = this.harnessVersionChecks.get(harnessId);
      if (pending) return pending;
    }

    const check = (async () => {
      if (harnessId === "swarmx") return SWARMX_VERSION;
      const command = HARNESS_VERSION_COMMANDS[harnessId];
      if (!command) return undefined;
      const commandPath = await this.findExecutable(command, envPath, this.platform);
      if (!commandPath) return undefined;

      const result = await this.runCommand(commandPath, ["--version"], {
        env: this.env,
        timeoutMs: VERSION_TIMEOUT_MS,
      });
      if (result.exitCode !== 0 || result.error) return undefined;
      return threePartVersion(result.stdout, result.stderr);
    })();
    this.harnessVersionChecks.set(harnessId, check);
    try {
      const version = await check;
      this.harnessVersionCache.set(harnessId, version);
      return version;
    } finally {
      if (this.harnessVersionChecks.get(harnessId) === check) {
        this.harnessVersionChecks.delete(harnessId);
      }
    }
  }

  private invalidateHarnessVersions(
    request: HarnessEnvironmentSetupRequest,
    requirementIds: string[],
  ): void {
    const affectedHarnessIds = request.harnessToolId
      ? [request.harnessToolId]
      : request.harnessId
        ? [request.harnessId]
        : Object.entries(HARNESS_REQUIREMENTS)
            .filter(([, ids]) => ids.some((id) => requirementIds.includes(id)))
            .map(([harnessId]) => harnessId);
    for (const harnessId of affectedHarnessIds) {
      this.harnessVersionCache.delete(harnessId);
    }
  }

  private async readVersion(
    commandPath: string,
    args: string[],
  ): Promise<{ status: "ready" | "failed"; version?: string; note?: string }> {
    const result = await this.runCommand(commandPath, args, {
      env: this.env,
      timeoutMs: VERSION_TIMEOUT_MS,
    });
    if (result.exitCode !== 0 || result.error) {
      return {
        status: "failed",
        note: result.error ?? firstLine(result.stderr) ?? "Version check failed.",
      };
    }
    const version = threePartVersion(result.stdout, result.stderr);
    return version
      ? { status: "ready", version }
      : { status: "failed", note: "Version check did not report a semantic version." };
  }
}

function commonExecutableDirs(homeDir: string): string[] {
  return [
    path.join(homeDir, ".hermes", "hermes-agent"),
    path.join(homeDir, ".hermes", "hermes-agent", "venv", "bin"),
    path.join(homeDir, ".hermes", "hermes-agent", ".venv", "bin"),
    path.join(homeDir, ".local", "bin"),
    path.join(homeDir, ".npm-global", "bin"),
    path.join(homeDir, ".yarn", "bin"),
    path.join(homeDir, ".local", "share", "pnpm"),
    path.join(homeDir, "Library", "pnpm"),
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
  ];
}

function mergePath(existingPath: string, extraDirs: string[]): string {
  return uniqueStrings([
    ...extraDirs,
    ...existingPath.split(path.delimiter).filter((part) => part.trim().length > 0),
  ]).join(path.delimiter);
}

async function findExecutableOnPath(
  command: string,
  envPath: string,
  platform: NodeJS.Platform,
): Promise<string | null> {
  const extensions = executableExtensions(platform);
  for (const dir of envPath.split(path.delimiter)) {
    if (!dir) continue;
    for (const extension of extensions) {
      const candidate = path.join(dir, `${command}${extension}`);
      try {
        await access(candidate, constants.X_OK);
        return candidate;
      } catch {
        // Try the next PATH candidate.
      }
    }
  }
  return null;
}

function executableExtensions(platform: NodeJS.Platform): string[] {
  if (platform !== "win32") return [""];
  const raw = process.env.PATHEXT ?? ".EXE;.CMD;.BAT;.COM";
  return uniqueStrings(["", ...raw.split(";").map((part) => part.toLowerCase())]);
}

function installerForPlatform(
  requirement: RequirementDefinition | ContainerRuntimeDefinition,
  platform: NodeJS.Platform,
): InstallerDefinition | undefined {
  return requirement.installers[platform] ?? requirement.installers.default;
}

function noteForPlatform(
  requirement: RequirementDefinition,
  platform: NodeJS.Platform,
): string | undefined {
  return requirement.notes?.[platform] ?? requirement.notes?.default;
}

function shellCommand(
  platform: NodeJS.Platform,
  script: string,
): { program: string; args: string[] } {
  if (platform === "win32") {
    return {
      program: "powershell.exe",
      args: ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
    };
  }
  return {
    program: "bash",
    args: ["-lc", script],
  };
}

function containerEnvArgs(): string[] {
  return [
    "--env",
    "HOME=/tmp/swarmx-home",
    "--env",
    "XDG_CACHE_HOME=/tmp/swarmx-cache",
    "--env",
    "PATH=/usr/local/bin:/usr/bin:/bin:/tmp/swarmx-home/.local/bin",
    ...CONTAINER_ENV_KEYS.flatMap((key) => ["--env", key]),
  ];
}

/** Translate host-loopback service URLs for an Apple Container request. */
export function containerHostBridgeUrl(value: string): string {
  try {
    const url = new URL(value);
    if (!["localhost", "127.0.0.1", "[::1]", "::1"].includes(url.hostname)) return value;
    url.hostname = "host.docker.internal";
    return url.toString().replace(/\/$/, value.endsWith("/") ? "/" : "");
  } catch {
    return value;
  }
}

/** Return a copy suitable for a protected process without exposing extra env keys. */
export function containerHostBridgeEnv(env: Record<string, string>): Record<string, string> {
  const translated = { ...env };
  for (const key of ["ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "OLLAMA_BASE_URL", "OLLAMA_HOST"]) {
    if (translated[key]) translated[key] = containerHostBridgeUrl(translated[key]);
  }
  return translated;
}

function runCommand(
  program: string,
  args: string[],
  options: { env: NodeJS.ProcessEnv; timeoutMs: number },
): Promise<CommandResult> {
  return new Promise((resolve) => {
    const child = spawn(program, args, {
      env: options.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let settled = false;
    const timeout = setTimeout(() => {
      if (settled) return;
      child.kill();
      settled = true;
      resolve({
        exitCode: null,
        stdout,
        stderr,
        error: `Command timed out after ${options.timeoutMs}ms.`,
      });
    }, options.timeoutMs);

    child.stdout?.setEncoding("utf8");
    child.stdout?.on("data", (chunk: string) => {
      stdout = limitOutput(`${stdout}${chunk}`);
    });
    child.stderr?.setEncoding("utf8");
    child.stderr?.on("data", (chunk: string) => {
      stderr = limitOutput(`${stderr}${chunk}`);
    });
    child.on("error", (error) => {
      if (settled) return;
      clearTimeout(timeout);
      settled = true;
      resolve({ exitCode: null, stdout, stderr, error: error.message });
    });
    child.on("close", (exitCode) => {
      if (settled) return;
      clearTimeout(timeout);
      settled = true;
      resolve({ exitCode, stdout, stderr });
    });
  });
}

function appendCommandLog(log: string[], result: CommandResult): void {
  const stdout = result.stdout.trim();
  const stderr = result.stderr.trim();
  if (stdout) log.push(stdout);
  if (stderr) log.push(stderr);
  if (result.error) log.push(result.error);
  if (result.exitCode !== 0 && result.exitCode !== null) {
    log.push(`Command exited with code ${result.exitCode}.`);
  }
}

function limitOutput(value: string): string {
  return value.length > OUTPUT_LIMIT ? value.slice(-OUTPUT_LIMIT) : value;
}

function firstLine(value: string): string | undefined {
  return value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find(Boolean);
}

function threePartVersion(stdout: string, stderr: string): string | undefined {
  return `${stdout}\n${stderr}`.match(/\d+\.\d+\.\d+/)?.[0];
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values));
}
