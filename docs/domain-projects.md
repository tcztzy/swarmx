> 迁移说明：SwarmX 已停止提供 builtin Pi；下文既有 Pi 命令、项目布局与验收记录保留为历史证据，`SWARMX_AGENT=pi` 不再可启动。当前默认使用外部 Codex；独立 Pi 或其他框架 Agent 请按 [ACP 配置](acp.md) 显式连接并授权。旧会话与结果文件不会自动转换，Host 日志不能替代原生会话。

# 在独立项目目录中运行 SwarmX

SwarmX 是通用多智能体系统，提供原生 Agent 与模型接入、任务调度、桌面、权限和运行记录。
独立项目维护自己的指令、程序、依赖、数据和验收规则。`@swarmx/science` 提供通用实验记录、
文件和 Python 执行功能；`@swarmx/swarm` 只组合已有 Agent，不负责启动整个应用。

## 启动

先按 [README](../README.md) 安装 SwarmX 的构建依赖并完成 `pnpm install --frozen-lockfile`。
Node.js 和 pnpm 要求以根目录 `package.json` 为准；构建需要 Rust 和本机链接工具。
只有使用 Science 的 Python 执行时才需要 Docker。项目程序的运行环境需单独安装。
使用 Pi 时，还需按 [Agent 平台说明](runtime-platform.md) 完成原生登录及模型配置。

将下面的示例路径替换为实际的项目和 SwarmX 仓库路径；项目目录必须已经存在：

```sh
SWARMX_CWD=/absolute/path/to/project \
SWARMX_HOME="$HOME/.swarmx-project" \
SWARMX_AGENT=pi \
pnpm -C /absolute/path/to/swarmx start
```

`start` 先构建再启动桌面；日常开发可将 `start` 换成 `dev`。启动前退出已打开的 SwarmX
桌面：桌面采用单实例运行，第二次启动只会聚焦已有窗口，不会切换其项目目录。
可在设置中核对当前执行目录。

`SWARMX_CWD` 在 Host 启动时确定，解析符号链接后的真实目录用于会话、Science 和运行记录
的范围检查。一次 Host 启动只服务一个目录；移动项目会改变其目录身份。
`SWARMX_HOME` 默认是 `~/.swarmx`，可为独立项目选择专用私有目录；它不改变原生 Agent
的登录、全局技能或原生会话存储，也不是文件系统沙箱。

外部应用使用已有 ACP stdio 入口。先构建一次：

```sh
pnpm -C /absolute/path/to/swarmx build
```

然后让 ACP 客户端以如下命令启动 SwarmX 子进程，并连接其 stdin/stdout：

```sh
SWARMX_CWD=/absolute/path/to/project \
SWARMX_HOME="$HOME/.swarmx-project" \
SWARMX_AGENT=pi \
pnpm -C /absolute/path/to/swarmx --silent acp
```

这是源码仓库提供的公开启动脚本，不是一个已发布的独立 `swarmx` CLI 包。
客户端使用官方 `@agentclientprotocol/sdk`，依次完成 initialize、session/new 和
session/prompt，并处理流式更新、审批或问题以及终态。session/new 的 `cwd` 必须与 Host
目录一致；不同目录和客户端注入的 MCP 服务器会被拒绝。stdout 只用于协议，诊断写入 stderr。
模型和推理设置通过原生配置选项选择；权限协商和运行 ID 见 [ACP 约定](acp.md)。
单独在终端运行 ACP 命令只会等待客户端，不会自动开始任务。

外部应用通过公开入口使用 SwarmX，不导入 `apps/desktop/dist/platform.js` 等内部实现。

### 外部客户端续接会话

客户端保存 `session/new` 返回的完整 `sessionId`，每轮只发送新增输入。
同一连接可以继续调用 `session/prompt`；重新连接后，先 initialize，再调用
`session/resume`，或在需要显示历史时调用 `session/load`，随后发送新的 prompt。
两者都使用原来的 `sessionId`、相同的项目 `cwd` 和空 `mcpServers`。
原生 Agent 保存上下文；客户端不应将本地显示的历史重新拼进每轮输入，或为追问创建新会话。
只有 Agent 广告支持续接时才能采用此流程；例如 DSH 没有持久会话续接能力。
原生会话缺失、续接失败或执行失败须明确报告，不能静默替换为新会话。

模型与推理设置在每次连接中通过返回的 `configOptions` 和
`session/set_config_option` 选择。客户端须转发审批和问题；若界面尚不支持交互，
应返回取消或拒绝，不能自动批准。取消后等待原 prompt 的终态。
结束 stdio 客户端时还需关闭所拥有子进程的 stdin，并等待进程退出；
SDK 连接关闭本身不保证底层进程退出。

例如 GEEPilot 通过这一入口使用 SwarmX 的原生 Agent、会话、授权、取消与执行记录；
它的任务技能负责生物学问题、方法选择、质量检查与结果解释。
BioV 负责确定性的生物计算、数据访问、软件环境和分析产物，其技能说明这些底层软件的用法。
BioV CLI 或 MCP 按原生 Agent 的项目配置接入，不能通过 ACP 注入 MCP，
也不需要将 BioV 计算复制到 SwarmX。Agent 的 `end_turn` 与 BioV 的计算检查结果分别保存。

## 项目指令、技能和程序

默认 Pi 保留原生资源发现方式。最小项目可以采用：

```text
project/
  AGENTS.md
  .agents/skills/project-smoke/SKILL.md
  scripts/smoke.mjs
  input.txt
  outputs/
```

`AGENTS.md` 写明项目规则和程序入口。技能文件需要 `name`、`description` 的 YAML 元数据；
正文说明适用条件、输入检查、执行命令和结果解释。例如，无害示例的技能内容为：

```markdown
---
name: project-smoke
description: Run the project's harmless file-output check.
---
Run `node scripts/smoke.mjs` from the project directory and inspect `outputs/smoke.json`.
```

`scripts/smoke.mjs` 内容如下；先在 `input.txt` 中写入 `hello`：

```js
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

const result = { cwd: process.cwd(), message: readFileSync("input.txt", "utf8").trim() };
mkdirSync("outputs", { recursive: true });
writeFileSync("outputs/smoke.json", JSON.stringify(result));
console.log("PROJECT_SMOKE_OK");
```

在该目录启动 SwarmX 后，新建 Pi 对话并要求“读取 project-smoke 技能，执行其中的检查，
核对生成的 JSON”。预期结果是 `outputs/smoke.json` 的 `cwd` 为项目真实路径、`message`
为 `hello`，并出现 `PROJECT_SMOKE_OK` 工具输出。模型需要实际选择并执行工具，技能发现
本身不能证明程序已运行。

Pi 会发现 `.agents/skills` 和 `.pi/skills`，先加载技能元数据，按需读取正文；祖先目录和
用户配置中的指令、技能仍可能生效。其他 Agent 使用各自的原生规则，不承诺共享同一技能
目录；见 [原生 Agent](native-agents.md)。项目可直接提供自己的 CLI，不必注册为
SwarmX 产品工具。具体命令、软件和数据要求由项目文档说明。

Pi 的原生命令工具在项目目录运行并继承启动环境。Python 项目可在技能中使用明确的
`.venv/bin/python` 或既有环境管理命令。SwarmX 不会自动安装项目依赖、下载项目数据或激活
虚拟环境。原生工具继续采用原生权限配置；Host 的工具授权不构成原生命令的文件系统限制。

Science notebook/figure 的 Python 使用独立的 Docker 环境，运行时无网络，镜像只读。
它不会自动使用项目的虚拟环境或宿主软件，不应作为任意项目 CLI 的隐式执行环境。

## 输出和运行记录

项目 CLI 应明确指定输出路径，并保存输入标识、参数、软件版本、退出状态及计算结果。
普通输出文件留在项目中；只有显式导入或调用 `science_record` 的 `register_artifact`，
才会保存为 Science 的文件快照。Science RO-Crate 元数据导出与文件下载是不同操作，
见 [科研工作台](product-readiness.md)。

Host 将所观察到的输入、工具、交互和运行终态写入
`$SWARMX_HOME/logs/execution.sqlite`。桌面可按会话检查执行记录；协商 SwarmX 扩展后，ACP
会发送 `update.sessionUpdate = "session_info_update"` 的 `session/update` 通知，
其 `params._meta.swarmx.execution` 提供 `runId` 等关联字段。
ACP 客户端也可以保存自己的协议更新和结果；目前没有公开 ACP 历史日志读取方法。
原生会话历史继续由原生 Agent 管理。

先正常退出所有使用该 product home 的 SwarmX 桌面和 ACP 进程，待数据库关闭后，
可以用 SQLite 命令行只读导出当前存储格式的完整 JSONL：

```sh
sqlite3 -readonly "file:$HOME/.swarmx-project/logs/execution.sqlite?immutable=1" \
  'SELECT record_json FROM execution_events ORDER BY seq;' \
  > /absolute/path/to/project/outputs/swarmx-execution.jsonl
```

导出前须创建 `outputs` 目录。`immutable=1` 只适用于已正常关闭、没有写入者的数据库；
不要对正在运行的 Host 使用此命令。此命令包含该私有目录内所有项目和会话的记录，可能有私密内容；
选择要分享的运行后再发布。数据库及 `record_json` 的存储版本目前为 1，读取程序应检查版本；
这不是新增的网络 API。复制数据库时须一致地处理 WAL，不能只复制正在使用的主文件。
详细字段、覆盖范围和证据导出见 [执行日志](execution-log.md)。

运行耗时包括模型、工具和等待；`end_turn` 只表示 Agent 正常结束，不证明任务结果正确。
失败、取消和缺少终态须分别报告。人工干预可从已观察到的审批、问题回复和 steering 事件
统计，不能将未记录的操作算作零次干预。日志不覆盖 SwarmX 之外的操作，也不代替项目程序
自身的计算记录。

比较不同 Agent 配置时，应明确两组实际差异，控制模型、推理设置、工具、数据和任务输入。
分开项目目录、会话和 product home，并控制原生全局技能、项目指令、Memory 和学习设置。
仅设置两个 `SWARMX_HOME` 不能保证对照独立。

## 验证范围

`apps/desktop/tests/pi-native.test.ts` 的独立项目用例通过官方 ACP SDK 调用 Host，使用真实
Pi SDK 加载项目指令、读取技能、运行上述本地程序，并检查文件和执行记录。
模型响应由本地确定性 provider 提供，不访问模型服务。这验证运行约定，不证明真实模型
会正确执行项目任务，也不代替项目自身的测试或其他 Agent 的技能发现验证。
