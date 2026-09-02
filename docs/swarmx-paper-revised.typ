// MANUSCRIPT RULE / 稿件规则：All reader-visible manuscript content is written in English. Every English title, heading, paragraph, list item, figure/table caption, and metadata value is immediately preceded by a `// 中文：...` comment containing its corresponding Chinese rendering. Typst control code, URLs, identifiers, equations, and pure bibliographic metadata are exempt.

#set document(
  title: "SwarmX: A Local-First Desktop Environment for Conversation-Centered Agentic Scientific Work",
  author: "TO BE COMPLETED BEFORE SUBMISSION",
)
#set page(
  paper: "a4",
  margin: (top: 2.0cm, bottom: 2.0cm, left: 2.25cm, right: 2.25cm),
  numbering: "1",
)
#set text(
  font: ("New Computer Modern", "Source Han Serif SC"),
  size: 10pt,
  lang: "en",
)
#set par(justify: true, leading: 0.52em, first-line-indent: 1.45em)
#set heading(numbering: "1.")
#show heading.where(level: 1): it => {
  set text(size: 13pt, weight: "bold")
  it
}
#show heading.where(level: 2): it => {
  set text(size: 11pt, weight: "bold")
  it
}
#show heading.where(level: 3): it => {
  set text(size: 10pt, weight: "bold")
  it
}
#show figure.caption: set text(size: 8.6pt)
#show raw: set text(font: "New Computer Modern", size: 8.4pt)
#show link: set text(fill: rgb("#1f4e79"))

#let cite(n) = text(fill: rgb("#1f4e79"))[\[#n\]]
#let architecture-box(body, fill-color: rgb("#f7f9fb")) = box(
  width: 100%,
  inset: (x: 0.8em, y: 0.55em),
  fill: fill-color,
  stroke: (paint: rgb("#617181"), thickness: 0.6pt),
  radius: 4pt,
  body,
)
// 中文：SwarmX：面向对话中心智能体科研工作的本地优先桌面环境
#align(center)[
  #text(size: 17pt, weight: "bold")[
    SwarmX: A Local-First Desktop Environment for
    Conversation-Centered Agentic Scientific Work
  ]
]

#v(0.45em)
// 中文：作者：投稿前待填写
#align(center)[#text(size: 11pt)[Author(s): TO BE COMPLETED BEFORE SUBMISSION]]
// 中文：作者单位：投稿前待填写
#align(center)[#text(size: 9.5pt)[Affiliation: TO BE COMPLETED BEFORE SUBMISSION]]
// 中文：通讯作者与支持邮箱：投稿前待填写
#align(center)[#text(size: 9.5pt)[Corresponding author and support email: TO BE COMPLETED BEFORE SUBMISSION]]

#v(0.8em)

// 中文：摘要
= Abstract

// 中文：科研智能体日益需要跨代码、数据、图件、论文和个人知识开展工作，但这些活动通常被分散在聊天界面、Notebook、文件浏览器和知识库之间。这种割裂使人们难以保持智能体动作、生成工件、后续解释和论文修订之间的连续关系。SwarmX 是一个开源、本地优先的桌面环境，它把对话视为科研工作的统一协调面，而不是唯一事实来源。系统扩展既有智能体运行时，而不替换其会话、权限和智能体循环基础设施；新增能力包括非破坏性分支修订、可序列化侧栏检查、结构化引用、科学工作层和私有个人知识库。其核心设计将五类事实域明确分离：交互日志、可变工作区文件、只追加科学记录、不可变工件以及带来源的长期知识。所有科研操作仍由对话中的智能体介导，并受本地主机权限与工作区边界约束；生成结果返回同一对话中的可检查工件，而不是开启一个平行科研工作区。项目溯源和长期知识可通过开放、所有者可读的表示离开系统。本文通过代表性端到端工作流和面向契约的验证矩阵，分析交互连续性、可追踪与恢复、授权与失败行为以及互操作性。结果表明，当前实现能够在不替换底层智能体运行时的前提下，把对话、计算、工件、论文交互和长期知识连接起来；本文不据此主张用户生产率提升或科学结论正确。
Scientific agents increasingly work across code, data, figures, manuscripts, and personal
knowledge, yet these activities are commonly split among chat interfaces, notebooks, file
browsers, and knowledge bases. This fragmentation makes it difficult to preserve continuity between
an agent action, its resulting artifact, later interpretation, and manuscript revision. SwarmX is
an open-source, local-first desktop environment that treats conversation as the coordination
surface for scientific work rather than as its sole source of truth. It extends an existing agent
runtime without replacing its session, permission, and agent-loop infrastructure, adding
non-destructive branching, serializable side-view inspection, structured references, a scientific
work layer, and a private personal knowledge base. The central design separates five truth domains:
the interaction log, mutable workspace files, an append-only scientific record, immutable artifacts,
and source-bearing durable knowledge. Scientific operations remain agent-mediated through
conversation and constrained by local Host authority and workspace boundaries; generated results
return as inspectable artifacts in the same conversational flow instead of opening a parallel
research workspace. Project provenance and durable knowledge can leave the system through open,
owner-readable representations. We analyze the implementation through a representative end-to-end
workflow and a contract-oriented validation matrix covering interaction continuity, traceability
and recovery, authorization and failure behavior, and interoperability. The analysis shows that the
implemented contracts connect conversation, computation, artifacts, manuscript interaction, and
durable knowledge without replacing the underlying agent runtime. It does not establish gains in
user productivity or the validity of scientific conclusions.

// 中文：关键词：智能体科研工作流；本地优先软件；对话中心交互；科研溯源；可复现研究；人机协作
#text(weight: "bold")[Keywords:] agentic scientific workflows; local-first software;
conversation-centered interaction; research provenance; reproducible research; human--AI
collaboration

#v(0.7em)

// 中文：代码元数据
= Code metadata

// 中文：表 1 汇总了本文固定软件快照的代码元数据。作者身份、支持邮箱与归档 DOI 等未知投稿信息均显式保留为待填写项，而不进行推测。
#figure(
  table(
    columns: (0.08fr, 0.34fr, 0.58fr),
    align: (left, left, left),
    inset: 4.4pt,
    stroke: 0.45pt,
    table.header(
      // 中文：编号
      [#strong[ID]],
      // 中文：元数据字段
      [#strong[Metadata field]],
      // 中文：本文软件快照的取值
      [#strong[Value for this software snapshot]],
    ),
    // 中文：C1，当前代码版本，0.1.0 开发快照。
    [C1], [Current code version], [`0.1.0` development snapshot],
    // 中文：C2，本文使用代码的永久链接，固定到提交 cbff41737b7e5280e0d43a808d21c0b87e95003f。
    [C2], [Permanent link to code used for this article],
    [#link("https://github.com/tcztzy/swarmx/tree/cbff41737b7e5280e0d43a808d21c0b87e95003f")[Commit `cbff41737b7e5280e0d43a808d21c0b87e95003f`]],
    // 中文：C3，可复现胶囊或归档的永久链接，待创建正式发布与归档 DOI。
    [C3], [Permanent link to a reproducible capsule], [TO BE COMPLETED: release archive and DOI not yet assigned],
    // 中文：C4，代码许可证，MIT License。
    [C4], [Legal code license], [MIT License],
    // 中文：C5，代码版本管理系统，Git。
    [C5], [Code versioning system], [Git],
    // 中文：C6，主要实现语言与运行边界，包括 TypeScript、Rust、Electron、SQLite、Markdown、RO-Crate 以及可选的本地 Notebook、论文预览和文献适配器。
    [C6], [Languages and principal runtime boundaries], [TypeScript, Rust, Electron, SQLite, Markdown, RO-Crate, and optional local notebook, paper-preview, and literature adapters],
    // 中文：C7，编译与运行要求为 Node.js 22.19 或 24 及以上和 pnpm 11.7；Notebook、文献与论文功能需要各自配置的本地依赖。
    [C7], [Compilation and operating requirements], [Node.js `^22.19.0` or `>=24.0.0`; pnpm `11.7.0`; locally configured dependencies for notebook, literature, and paper features],
    // 中文：C8，开发者文档位于仓库 README、SPEC、CODEBASE、docs 目录和包级文档。
    [C8], [Developer documentation], [Repository README, SPEC, CODEBASE, `docs/`, and package-level documentation],
    // 中文：C9，支持邮箱，投稿前待填写。
    [C9], [Support email], [TO BE COMPLETED BEFORE SUBMISSION],
  ),
  // 中文：本文所述 SwarmX 开发快照的代码元数据。软件版本与提交被固定，未知投稿信息不予推测。
  caption: [Code metadata for the SwarmX development snapshot described in this article. The software version and commit are fixed; unknown submission metadata is not inferred.],
)

// 中文：引言
= Introduction

// 中文：现代计算科研很少发生在单一文件或单一程序中。研究者可能在 Notebook 中探索数据，在工作流系统中组织依赖，在文件系统中保存图件和中间结果，在文献管理器中积累来源，并在独立写作工具中形成论文。Jupyter 将代码、叙述与输出组合为可交换文档，而科学工作流系统则把任务与数据依赖形式化；二者都显著改善了计算工作的可检查性与复用性 #cite(1) #cite(2) #cite(3)。然而，大规模 Notebook 研究也表明，交互式文档本身并不自动保证执行顺序、环境和输出可复现 #cite(4)。
Modern computational research rarely occurs in one file or one application. A researcher may
explore data in a notebook, organize dependencies in a workflow system, keep figures and
intermediate results in a file system, collect sources in a reference manager, and draft a paper in
a separate writing tool. Jupyter combines code, narrative, and output in an exchangeable document,
while scientific workflow systems formalize tasks and data dependencies; both improve the
inspectability and reuse of computational work #cite(1) #cite(2) #cite(3). Large-scale notebook
studies nevertheless show that an interactive document alone does not guarantee reproducible
execution order, environments, or outputs #cite(4).

// 中文：语言模型智能体在这些既有工具之上增加了新的协调层。一个智能体可以提出计划、调用工具、修改文件、运行分析并撰写解释；端到端科研智能体原型甚至试图从选题一路推进到实验和论文 #cite(5)。同时，ScienceAgentBench 等工作指出，在对完整科研自动化作出强主张之前，必须先严格评估工作流中的具体任务和执行结果 #cite(6)。这一张力揭示了一个系统问题：聊天可以降低工具切换成本，却也容易把“模型说过什么”“系统执行了什么”“当前文件是什么”“哪些结果应被长期记住”压缩成一条难以分辨的对话时间线。
Language-model agents add a new coordination layer above these tools. An agent can propose a plan,
invoke tools, modify files, execute analyses, and draft explanations; end-to-end agent prototypes
have even attempted to move from research ideas through experiments to papers #cite(5). At the same
time, work such as ScienceAgentBench argues that individual scientific tasks and their executed
outputs require rigorous assessment before strong claims about complete research automation are
made #cite(6). This tension exposes a systems problem: chat can reduce tool-switching costs, but it
can also collapse what the model said, what the system executed, what the current files contain, and
what should be remembered into one difficult-to-interpret conversational timeline.

// 中文：一种直接反应是为科研智能体再建一个完整工作台：单独的项目页、执行面板、工件浏览器、论文编辑器和知识图谱。然而，这会重新引入用户必须在多个并行界面之间维护上下文的问题，也容易复制智能体运行时已经拥有的会话、权限、工具、持久化和错误处理机制。另一种极端是让所有状态都留在聊天记录中，但聊天日志既不适合充当可变文件系统，也不适合充当结构化科学账本或经过整理的长期知识库。
One direct response is to build a complete second workbench for the scientific agent: a separate
project page, execution panel, artifact browser, manuscript editor, and knowledge graph. That design
reintroduces the burden of maintaining context across parallel interfaces and risks duplicating the
sessions, permissions, tools, persistence, and failure handling already owned by the agent runtime.
The opposite extreme is to keep every state transition in the chat history, but a chat log is neither
a suitable mutable file system nor a structured scientific ledger or curated long-term knowledge
base.

// 中文：SwarmX 研究一种不同的组合方式：保留对话作为人与智能体共同协调工作的主表面，同时明确拒绝把对话当作所有事实的唯一存储。它以内嵌的 DeepSeek Harness Web profile 作为智能体运行时，并通过已发布扩展接缝增加桌面安全边界、非破坏性分支操作、工件检查、科学域服务、论文交互和个人知识库。该方案的目标不是把各种工具塞进一个窗口，而是让每类状态由合适的所有者持有，并通过可检查定位符重新连接到同一对话流程。
SwarmX investigates a different composition: conversation remains the primary surface through
which a person and an agent coordinate work, while conversation is explicitly rejected as the sole
store of every fact. The system embeds the DeepSeek Harness Web profile as its agent runtime and adds
a desktop security boundary, non-destructive branching actions, artifact inspection, scientific
domain services, manuscript interaction, and a personal knowledge base through published extension
seams #cite(7) #cite(8). The goal is not to place every tool in one window, but to give each kind of
state an appropriate owner and reconnect it to the same conversational flow through inspectable
locators.

// 中文：本文提出并分析以下四项贡献。
The article makes and analyzes four contributions.

// 中文：第一，一种对话中心的加性架构：既有 Harness 继续拥有智能体循环、会话和权限；SwarmX 不创建替代渲染器或第二工作区，而是通过 Retry/Edit、注释、工件卡片和 Side View 扩展原对话。
- First, it presents an additive, conversation-centered architecture: the existing Harness retains the agent loop, sessions, and permissions; SwarmX creates neither a replacement renderer nor a second workspace, but extends the original conversation through Retry/Edit, annotations, artifact cards, and Side View.

// 中文：第二，一种分层事实模型：交互日志、工作区文件、科学 Journal、不可变工件和个人知识库具有不同的真相边界、更新规则和恢复方式。
- Second, it defines a layered truth model in which the interaction log, workspace files, Science Journal, immutable artifacts, and personal knowledge base have distinct authority, mutation, and recovery rules.

// 中文：第三，一条本地优先的端到端科研路径：计算、文献、实验、图件、写作、溯源导出与跨会话知识都在宿主授权下通过同一对话进入，而敏感路径和原始主机权限不会交给渲染器。
- Third, it implements a local-first end-to-end path in which computation, literature, experiments, figures, writing, provenance export, and cross-session knowledge enter through the same conversation under Host authorization, without transferring sensitive paths or raw Host authority to the renderer.

// 中文：第四，一种与主张相匹配的验证框架：代表性工作流、失败语义、测试类别和互操作边界被映射到四个研究问题，同时明确排除尚未进行的用户生产率、性能和科学有效性结论。
- Fourth, it provides a claim-aligned validation framework: a representative workflow, failure semantics, test categories, and interoperability boundaries are mapped to four research questions while explicitly excluding conclusions about user productivity, performance, and scientific validity that have not been evaluated.

// 中文：本文余下部分首先讨论相关研究与设计缺口，然后给出设计目标、系统架构、代表性工作流和面向契约的评估，最后讨论复用价值、局限与后续实证工作。
The remainder of the article discusses related research and the design gap, presents design goals and
system architecture, follows a representative workflow, evaluates the implementation at the
contract level, and concludes with reuse value, limitations, and required empirical work.

// 中文：背景与设计缺口
= Background and design gap

// 中文：计算文档、工作流与研究对象
== Computational documents, workflows, and research objects

// 中文：Notebook 与工作流系统分别优化了两类需求。Notebook 强调局部探索、叙述和可视输出；工作流强调可重复任务、显式依赖与批量执行。两者都可以成为科研智能体的工具，但两者都不能完整代表智能体与研究者之间的决策过程、论文修订和长期知识。更重要的是，“文件存在”并不等于“文件为何存在”已经得到记录。FAIR 原则把可发现、可访问、可互操作和可复用视为研究对象的高层目标，并特别强调来源与机器可操作性 #cite(9)。
Notebooks and workflow systems optimize different needs. Notebooks emphasize local exploration,
narrative, and visual output; workflows emphasize repeatable tasks, explicit dependencies, and
batch execution. Both can serve as tools for a scientific agent, but neither fully represents the
decision process between an agent and a researcher, manuscript revisions, or durable knowledge.
More importantly, the existence of a file does not establish why that file exists. The FAIR
principles frame findability, accessibility, interoperability, and reusability as high-level goals
for research objects, with particular attention to provenance and machine actionability #cite(9).

// 中文：RO-Crate 提供了一种轻量方式，用 JSON-LD 和 Schema.org 关系描述研究工件及其上下文 #cite(10)。它适合充当项目级交换边界，但不是交互界面、命令协议或可变工作区。因此，一个智能体科研环境仍需决定：何时记录动作，如何把动作与输出关联，怎样处理尚未完成或被否决的提议，以及如何让项目外的个人知识引用而非复制科学事实。
RO-Crate provides a lightweight way to describe research artifacts and their context through
JSON-LD and Schema.org relations #cite(10). It is suitable as a project-level exchange boundary, but
it is not an interaction surface, command protocol, or mutable workspace. An agentic research
environment must still decide when an action becomes a record, how actions are linked to outputs,
how unresolved or rejected proposals are represented, and how personal knowledge outside a project
should reference rather than duplicate scientific facts.

// 中文：人机交互与可恢复控制
== Human--AI interaction and recoverable control

// 中文：人机交互研究长期强调，AI 系统应使能力、状态、错误和修正路径对用户可见，并支持用户在系统出错时进行高效纠正 #cite(11)。对科研智能体而言，这不仅是界面礼貌问题。一次看似简单的“重试”若覆盖原始消息、隐藏失败或改变随后上下文，就会破坏审计线索；一次论文修改若没有精确 revision，则可能无意覆盖研究者刚刚完成的编辑。
Human--AI interaction research emphasizes that systems should expose capabilities, status, errors,
and correction paths, and should support efficient recovery when the system is wrong #cite(11). For
a scientific agent, this is more than an interface courtesy. A seemingly simple retry can destroy an
audit trail if it overwrites the original message, hides the failure, or changes subsequent context;
a manuscript edit without an exact revision can overwrite a change the researcher has just made.

// 中文：因此，对话中心并不意味着“所有控制都通过自由文本完成”。它意味着对话负责意图、讨论和后续编排，而关键对象仍需要结构化操作：分支而非覆盖，修订检查而非最后写入者获胜，定位符而非模糊文件名，显式批准而非静默跨工作区读取。SwarmX 的交互设计由这些可恢复原则驱动。
Conversation-centered design therefore does not mean that every control is expressed as free text.
Conversation carries intent, discussion, and subsequent orchestration, while consequential objects
still require structured operations: branching instead of overwriting, revision checks instead of
last-writer-wins, locators instead of ambiguous filenames, and explicit approval instead of silent
cross-workspace reads. These recoverability principles drive the SwarmX interaction design.

// 中文：本地优先、所有权与可移植知识
== Local-first authority and portable knowledge

// 中文：本地优先软件强调用户对数据和运行环境的所有权，即使云服务不可用，核心工作也不应因服务消失而失去可访问性 #cite(12)。SwarmX 采用其中与单用户科研桌面相关的部分：核心会话宿主、科学状态、工件和知识文件保存在本机；网络能力不会由科学服务隐式获得；工作区路径由宿主授权而不是浏览器直接访问。当前系统并不实现本地优先论文所讨论的多设备协作或 CRDT 同步，因此本文使用“local-first”描述本地权威和数据可持有性，而非完整协作模型。
Local-first software emphasizes user ownership of data and execution context, so that core work does
not become inaccessible merely because a cloud service disappears #cite(12). SwarmX adopts the
parts relevant to a single-user research desktop: the core session host, scientific state,
artifacts, and knowledge files remain local; the Science service acquires no implicit network
capability; and workspace paths are authorized by the Host rather than directly exposed to the
browser. The current system does not implement the multi-device collaboration or CRDT
synchronization discussed in the local-first literature, so this article uses “local-first” to mean
local authority and data possession rather than a complete collaborative model.

// 中文：持久知识还提出了另一个分层问题。聊天记录应保留原始交互证据，但不适合作为经过整理、可编辑且可跨会话复用的知识页面。相反，一个 Wiki 若把聊天片段重新表述为事实，又必须保留来源，否则会丢失其证据基础。SwarmX 因此把个人知识保存为普通 Markdown，并要求来自对话的主张引用确切会话片段；科学主张则保留在 Science Journal 中，个人知识仅保存定位符或综合说明。
Durable knowledge introduces another layering problem. A conversation log should retain exact
interaction evidence, but it is not an appropriate editable, curated, cross-session knowledge page.
Conversely, a wiki that restates conversational material as fact must preserve its sources or lose
its evidential basis. SwarmX therefore stores personal knowledge as ordinary Markdown and requires
conversation-derived claims to cite exact session excerpts; scientific claims remain in the Science
Journal, while personal knowledge stores only a locator or a synthesis.

// 中文：设计目标与范围
= Design goals and scope

// 中文：SwarmX 的设计由表 2 中五项目标约束。这些目标描述系统架构应保持的性质，而不是特定依赖项或界面样式。
#figure(
  table(
    columns: (0.11fr, 0.30fr, 0.59fr),
    align: (left, left, left),
    inset: 4.5pt,
    stroke: 0.45pt,
    table.header(
      // 中文：目标
      [#strong[Goal]],
      // 中文：名称
      [#strong[Name]],
      // 中文：设计要求
      [#strong[Requirement]],
    ),
    // 中文：G1，对话连续性。科研能力应进入既有 Chat 与 Trajectory，不建立需要用户同步维护的平行工作区。
    [G1], [Conversation continuity], [Scientific capability must enter the existing Chat and Trajectory rather than creating a parallel workspace that users must keep synchronized.],
    // 中文：G2，加性所有权。Harness 继续拥有会话、智能体循环、模型接入、工具基础设施与权限；SwarmX 仅通过公开接缝增加产品能力。
    [G2], [Additive ownership], [The Harness retains sessions, the agent loop, model access, tool infrastructure, and permissions; SwarmX adds product capability only through published seams.],
    // 中文：G3，分层事实。交互、文件、科学事实、工件与个人知识不得被压缩成一个可变存储。
    [G3], [Layered truth], [Interaction, files, scientific facts, artifacts, and personal knowledge must not be collapsed into one mutable store.],
    // 中文：G4，本地权威与可恢复失败。敏感资源由宿主授权；取消、过期 revision、拒绝或缺失依赖不得产生静默部分写入。
    [G4], [Local authority and recoverable failure], [Sensitive resources are authorized by the Host; cancellation, stale revisions, rejection, or missing dependencies must not produce silent partial writes.],
    // 中文：G5，可移植边界。项目、文件与知识应尽可能使用外部可读格式和标准定位，而不要求第三方理解 SwarmX 内部数据库。
    [G5], [Portable boundaries], [Projects, files, and knowledge should use externally readable formats and standard locators where possible, without requiring third parties to understand SwarmX-internal databases.],
  ),
  // 中文：SwarmX 的设计目标。目标描述应保持的系统性质，而非具体实现依赖。
  caption: [Design goals for SwarmX. The goals describe system properties to preserve rather than particular implementation dependencies.],
)

// 中文：这些目标同时限定了本文不主张的内容。SwarmX 不是特定学科的科学推理器，不判定一个实验是否足以支持某项结论，也不把模型输出自动视为知识。它不是云端协作平台，不为本地明文 PKB 提供静态加密或自动同步。它也不是新的智能体运行时；不同模型供应商、工具调度与会话压缩仍由 Harness 负责。
The goals also define what this article does not claim. SwarmX is not a domain-specific scientific
reasoner, does not decide whether an experiment is sufficient to support a conclusion, and does not
automatically treat model output as knowledge. It is not a cloud collaboration platform and does
not provide encryption at rest or automatic synchronization for its local plaintext PKB. It is also
not a new agent runtime; model-provider integration, tool scheduling, and session compaction remain
Harness responsibilities.

// 中文：系统架构
= System architecture

// 中文：加性桌面与运行时边界
== Additive desktop and runtime boundary

// 中文：SwarmX 是一个轻量 Electron 宿主及若干产品自有扩展。Electron 主进程在回环地址上启动已发布的 Harness Web profile，并把一个沙箱化 BrowserWindow 指向该地址。渲染器没有 preload 或 Node 集成；离开 Harness 原点的导航被拦截，浏览器权限默认拒绝。Harness 提供完整浏览器界面和 `/api` 传输，SwarmX 不维护替代渲染器、第二套会话存储或独立模型客户端。
SwarmX is a thin Electron Host plus product-owned extensions. The Electron main process boots the
published Harness Web profile on a loopback address and points one sandboxed BrowserWindow at that
address. The renderer has no preload or Node integration; navigation away from the Harness origin is
intercepted, and browser permissions are denied by default. The Harness supplies the complete
browser interface and `/api` transport, while SwarmX maintains no replacement renderer, second
session store, or independent model client.

// 中文：图 1 展示职责分层。最上层仍是一个对话与检查表面；其下是加性会话扩展和 Harness 自有运行时；宿主侧 Science 与 PKB 服务分别管理科研事实与长期知识；本地工作区、工件、Notebook、论文、文献库和 Markdown Vault 是受控资源；RO-Crate、普通文件与 Markdown 构成可移植输出。
#figure(
  [
    #stack(
      dir: ttb,
      spacing: 0.28em,
      // 中文：一个对话与检查表面：Chat、Trajectory、Composer、生成工件卡片与 Side View。
      architecture-box([
        #align(center)[#text(weight: "bold")[One conversation and inspection surface]]
        #align(center)[#text(size: 8.4pt)[Chat · Trajectory · Composer · generated artifact cards · Side View]]
      ], fill-color: rgb("#eaf3fb")),
      align(center)[#text(size: 12pt)[↓]],
      // 中文：加性会话扩展：只追加 Retry/Edit、结构化注释与可序列化的每会话标签页。
      architecture-box([
        #align(center)[#text(weight: "bold")[Additive conversation extensions]]
        #align(center)[#text(size: 8.4pt)[Append-only Retry/Edit · structured annotations · serializable per-session tabs]]
      ], fill-color: rgb("#f3f7fb")),
      align(center)[#text(size: 12pt)[↓]],
      // 中文：Harness 自有运行时：会话、智能体循环、模型访问、工具、权限、持久化和遥测。
      architecture-box([
        #align(center)[#text(weight: "bold")[Harness-owned agent runtime]]
        #align(center)[#text(size: 8.4pt)[Sessions · agent loop · model access · tools · permissions · persistence]]
      ], fill-color: rgb("#f6f0df")),
      align(center)[#text(size: 12pt)[↓]],
      grid(
        columns: (1fr, 1fr),
        gutter: 0.5em,
        // 中文：Science 服务：项目、计算、文献、图件、实验、论文与溯源。
        architecture-box([
          #align(center)[#text(weight: "bold")[Science service]]
          #align(center)[#text(size: 8.25pt)[Projects · computation · literature · figures · experiments · papers · provenance]]
        ], fill-color: rgb("#eef7ed")),
        // 中文：PKB 服务：带来源、修订与批准的全局和工作区知识。
        architecture-box([
          #align(center)[#text(weight: "bold")[Personal knowledge service]]
          #align(center)[#text(size: 8.25pt)[Global/workspace knowledge · sources · revisions · approvals]]
        ], fill-color: rgb("#eef7ed")),
      ),
      align(center)[#text(size: 12pt)[↓]],
      grid(
        columns: (1fr, 1fr),
        gutter: 0.5em,
        // 中文：受控本地资源：工作区文件、不可变工件、Notebook、论文运行时和本地文献库。
        architecture-box([
          #align(center)[#text(weight: "bold")[Authorized local research resources]]
          #align(center)[#text(size: 8.25pt)[Workspace files · immutable artifacts · notebooks · paper runtime · local literature]]
        ]),
        // 中文：所有者可读的 Markdown Vault：概念、索引、日志和确切会话摘录。
        architecture-box([
          #align(center)[#text(weight: "bold")[Owner-readable Markdown vault]]
          #align(center)[#text(size: 8.25pt)[Concepts · indexes · revision log · exact conversation excerpts]]
        ]),
      ),
      align(center)[#text(size: 12pt)[↓]],
      grid(
        columns: (1fr, 1fr, 1fr),
        gutter: 0.45em,
        // 中文：普通工作区文件与安全相对链接。
        architecture-box([
          #align(center)[#text(weight: "bold")[Workspace deliverables]]
          #align(center)[#text(size: 8.1pt)[Ordinary files · safe relative links]]
        ], fill-color: rgb("#f5eef8")),
        // 中文：项目级 RO-Crate 研究对象。
        architecture-box([
          #align(center)[#text(weight: "bold")[RO-Crate export]]
          #align(center)[#text(size: 8.1pt)[Project-level research object]]
        ], fill-color: rgb("#f5eef8")),
        // 中文：可移植 OKF/Markdown 个人知识。
        architecture-box([
          #align(center)[#text(weight: "bold")[OKF/Markdown knowledge]]
          #align(center)[#text(size: 8.1pt)[Portable personal synthesis]]
        ], fill-color: rgb("#f5eef8")),
      ),
    )
  ],
  // 中文：SwarmX 的加性架构。用户只维护一个对话与检查表面；Harness 继续拥有智能体运行时，SwarmX 宿主扩展分别管理科学工作和个人知识，并把本地资源转换为可检查、可移植的输出。
  caption: [The additive SwarmX architecture. Users maintain one conversation and inspection surface; the Harness retains the agent runtime, while SwarmX Host extensions manage scientific work and personal knowledge and connect local resources to inspectable, portable outputs.],
)

// 中文：这一边界带来两个重要结果。第一，标准、最小或其他 Harness preset 可以继续运行，而不必暴露科学模型工具；宿主服务与工件 UI 可以存在，但只有系统自有科学 preset 向模型注册科学能力。第二，产品扩展在 Harness 发布包之后、用户补丁之前组合，因此 SwarmX 能增加功能而不把上游界面复制为永久分叉。
This boundary has two important consequences. First, standard, minimal, and other Harness presets
can continue to run without exposing scientific model tools; Host services and artifact UI may be
present, but only the system-owned Science preset registers scientific capabilities for the model.
Second, product extensions are composed after the published Harness bundles and before user patches,
so SwarmX can add behavior without maintaining a permanent fork of the upstream interface.

// 中文：对话连续性：分支、注释与 Side View
== Conversation continuity: branching, annotation, and Side View

// 中文：Retry 与 Edit 被实现为只追加分支操作，而不是修改原会话。对一个已完成用户回合执行 Retry 时，系统准备一个从该回合之前开始的子会话，并重新发送原文本；Edit 则打开同样的子会话，并把原提示词放入草稿供用户修改。源会话中的用户消息、自动重试记录和终止错误保持不变，也不会被子会话投影到模型上下文中。对于无法对空前缀执行 fork 的首回合，系统在同一 Workspace 中创建新会话，而不是伪造历史。
Retry and Edit are implemented as append-only branch operations rather than mutations of the source
session. Retrying a completed user turn prepares a child session that begins before that turn and
sends the original text again; Edit opens the same kind of child session and places the original
prompt in the draft for revision. The user message, automatic-retry records, and terminal error in
the source session remain unchanged and are not projected into the child model context. For a first
turn whose empty prefix cannot be forked, the system creates a new session in the same Workspace
instead of fabricating history.

// 中文：Side View 复用 Harness 已有的可拖动右侧 details 区域，并按会话保存可序列化标签描述符。打开一个图件、表格或论文不会替换 Chat、Trajectory、Composer 或消息滚动状态。成功生成工件的科学调用在结束回答下方呈现 `GENERATED` 卡片；点击卡片只激活相应侧栏标签。该选择使工件检查与对话保持并列，而不是把用户带到另一个产品区域。
Side View reuses the Harness's draggable right-hand details region and stores serializable tab
descriptors per session. Opening a figure, table, or paper does not replace Chat, Trajectory, the
Composer, or message scroll state. Successful artifact-producing Science calls render `GENERATED`
cards beneath the closing answer; clicking a card activates the corresponding side-view tab. This
choice keeps artifact inspection adjacent to the conversation rather than navigating the user to a
second product area.

// 中文：结构化注释把侧栏检查重新连接到后续对话。用户可以选择一段既有消息、PDF 文本、论文图件区域或图像点，添加可选评论，并把引用插入当前 Composer。可见草稿文本保持原样，引用作为独立、可编辑和可删除的结构化项存在。模型若要访问受保护图像或论文内容，必须通过宿主查询重新授权，而不能从序列化引用推断主机路径。
Structured annotations reconnect side-view inspection to subsequent conversation. A user can select
part of an existing message, PDF text, a paper figure region, or an image point, add an optional
comment, and insert the reference into the current Composer. Visible draft text remains unchanged,
while the references exist as independent, editable, removable structured items. If the model needs
protected image or paper content, it must re-authorize access through a Host query rather than infer
a Host path from the serialized reference.

// 中文：科学工作层
== Scientific work layer

// 中文：Science 服务把科研活动组织为项目、Notebook、文档、图件、研究记录、实验/运行和导出等有界实体。模型不直接接触内部数据库或文件句柄，而是通过少量聚合工具提出任务。每个请求都携带会话与工作区上下文，并在宿主侧验证输入、取消状态、实体归属、revision 与幂等键。成功结果返回事实、推断或提议分类以及持久定位符，失败则返回稳定错误，不产生默默提交的半成品状态。
The Science service organizes work into bounded entities for projects, notebooks, documents,
figures, research records, experiments and runs, and exports. The model never receives an internal
database handle or file descriptor; it proposes tasks through a small set of aggregate tools. Each
request carries session and workspace context and is checked at the Host boundary for input shape,
cancellation, entity ownership, revision, and idempotency. Successful results return a fact,
inference, or proposal classification together with a durable locator; failures return stable
errors without silently committing partial state.

// 中文：计算通过工作区授权的 Notebook 控制器执行。当前适配器可以为每个 Notebook 维护一个持久本地 Jupyter 内核，并以有界标准 MIME 输出返回结果；其实现使用 JupyMCP，但该依赖是执行适配器而不是论文贡献本身 #cite(13)。输入工件在摘要和工作区归属检查后临时物化，并在成功、失败、取消或释放时清理。缺少配置的 Notebook 运行时会产生显式不可用错误，不会静默切换到语义不同的执行器。
Computation is performed by a workspace-authorized notebook controller. The current adapter can
maintain one persistent local Jupyter kernel per notebook and returns bounded standard MIME output;
it uses JupyMCP as an execution adapter rather than treating that dependency as the research
contribution #cite(13). Input artifacts are temporarily materialized after digest and workspace
ownership checks and are removed after success, failure, cancellation, or disposal. An unconfigured
notebook runtime produces an explicit unavailable result rather than silently switching to an
executor with different semantics.

// 中文：文献检索通过运行中的本地 Zotero 桌面实例完成。候选记录经所有者可读的 BibTeX 快照归一化，返回有界、可引用的元数据；服务不读取附件路径，也不把本地库隐式发送到云端。该功能体现了本地优先边界：SwarmX 负责受控交换与排序标签，Zotero 仍拥有原始文献库 #cite(14)。
Literature search uses a running local Zotero desktop instance. Candidate records are normalized
through an owner-readable BibTeX snapshot and returned as bounded, citation-ready metadata; the
service does not read attachment paths or implicitly send the local library to a cloud index. This
feature illustrates the local-first boundary: SwarmX owns the controlled exchange and ranking label,
while Zotero retains the source library #cite(14).

// 中文：论文与图件修改采用乐观并发控制。一个修改提议引用精确 source revision、选区和原文；只有 revision 仍匹配时才能接受。当前论文工作台以 Typst 为首个语义预览引擎，在同一编译快照中连接源码、PDF 与点击到源码定位；Typst 和语义 watcher 是当前实现选择，而不是摘要或关键词层面的研究对象 #cite(15) #cite(16)。这种边界可扩展到其他能够提供可验证源码—渲染映射的写作引擎。
Manuscript and figure modifications use optimistic concurrency control. A proposed change refers to
an exact source revision, selection, and original text and can be accepted only while that revision
still matches. The current paper workbench uses Typst as its first semantic preview engine,
connecting source, PDF, and click-to-source navigation within the same compiled snapshot; Typst and
the semantic watcher are implementation choices rather than the research object named in the
abstract or keywords #cite(15) #cite(16). The boundary can accommodate other writing engines that
provide a verifiable source-to-render mapping.

// 中文：分层事实模型
== Layered truth model

// 中文：SwarmX 的核心系统主张是：同一科研过程可以通过多个相互连接、但不可互相替代的事实域表达。表 3 给出每一层的所有者、更新方式、用途及明确不承担的职责。
#figure(
  table(
    columns: (0.18fr, 0.17fr, 0.19fr, 0.25fr, 0.21fr),
    align: (left, left, left, left, left),
    inset: 3.9pt,
    stroke: 0.42pt,
    table.header(
      // 中文：事实层
      [#strong[Truth layer]],
      // 中文：权威所有者
      [#strong[Authority]],
      // 中文：更新模型
      [#strong[Mutation model]],
      // 中文：承担的用途
      [#strong[What it establishes]],
      // 中文：明确不承担
      [#strong[What it does not establish]],
    ),
    // 中文：会话日志；Harness；只追加事件；精确交互、工具行与错误；不是科学状态或整理知识。
    [Session log], [Harness], [Append-only events], [Exact interaction, tool rows, and failures], [Not scientific state or curated knowledge],
    // 中文：工作区文件；用户与授权宿主操作；可变且受 revision 约束；当前源码、数据和交付物；不是动作历史或来源证明。
    [Workspace files], [User and authorized Host operations], [Mutable, revision-checked], [Current source, data, and deliverables], [Not an action history or provenance proof],
    // 中文：Science Journal；Science 服务；幂等只追加事件与可重建投影；科学实体、关系、实验和运行；不是聊天副本或原始文件存储。
    [Science Journal], [Science service], [Idempotent append-only events with replayed projections], [Scientific entities, relations, experiments, and runs], [Not a chat copy or raw file store],
    // 中文：Artifact Registry；Science 服务；按摘要不可变；已生成或导入字节的身份和完整性；不是工件的科学含义。
    [Artifact Registry], [Science service], [Digest-addressed immutable objects], [Identity and integrity of generated or imported bytes], [Not the scientific meaning of an artifact],
    // 中文：PKB Vault；所有者及经批准的智能体操作；带精确 revision 的 Markdown 更新；跨会话个人综合、概念与来源；不是原始会话或科学 Journal 的替代副本。
    [PKB Vault], [Owner and approved agent operations], [Exact-revision Markdown updates with history], [Cross-session personal synthesis, concepts, and sources], [Not a replacement copy of sessions or the Science Journal],
  ),
  // 中文：SwarmX 的五层事实模型。各层通过定位符和来源互连，但没有一层被允许冒充其他层的权威。
  caption: [The five-layer truth model in SwarmX. Layers are connected through locators and sources, but no layer is allowed to impersonate the authority of another.],
)

// 中文：这种分层避免了两种常见错误。第一，模型叙述不能单独证明某文件已经生成或某实验已经完成；相应主张必须指向工作区结果、Journal 事件或注册工件。第二，个人知识页不会因为重写得更流畅就取代原始证据；来自会话的陈述必须携带确切摘录来源，来自科学项目的陈述则引用 Science 实体。知识页可以被人直接编辑，但过期的智能体更新会因 revision 冲突而失败。
This layering prevents two common errors. First, model narration alone cannot establish that a file
was generated or an experiment completed; the claim must resolve to a workspace result, Journal
event, or registered artifact. Second, a polished personal knowledge page does not replace its
source evidence; conversation-derived statements carry exact excerpt sources, and project-derived
statements refer to Science entities. A person may edit a knowledge page directly, but a stale agent
update fails through revision conflict.

// 中文：Science Journal 使用版本化事件和确定性投影，使物化视图可以从历史重建；Artifact Registry 在提交元数据之前完成暂存、摘要计算和原子发布；PKB 在替换页面之前保留前一 revision 并重新生成索引与日志。具体存储技术属于实现层，但共同语义是相同的：成功边界必须明确，失败不能留下对用户不可见的半提交状态。
The Science Journal uses versioned events and deterministic projections so that materialized views
can be rebuilt from history; the Artifact Registry completes staging, digest calculation, and
atomic publication before metadata commit; and the PKB preserves the previous revision before page
replacement and index/log regeneration. The storage technologies are implementation details, but
the shared semantic rule is architectural: the success boundary must be explicit, and failure must
not leave invisible half-committed state.

// 中文：工件、溯源与可移植输出
== Artifacts, provenance, and portable outputs

// 中文：生成文件首先作为对话中的可见结果出现。Science 工具返回的工件定位符在同一完成回合中被恢复、去重和排序，并形成可点击卡片。侧栏预览只请求宿主授权、摘要校验且大小受限的文本、图像、表格或 PDF 表示。浏览器接收相对标识和有界内容，不接收绝对主机路径。最终回答使用 DSH alpha.4 原生解析的 Markdown 行内代码提及精确的工作区相对路径。
Generated files first appear as visible conversational results. Artifact locators returned by
Science tools are recovered, deduplicated, and ordered within the same completed turn and rendered
as clickable cards. A side-view preview requests only Host-authorized, digest-verified, size-bounded
text, image, table, or PDF representations. The browser receives relative identities and bounded
content, not absolute Host paths. Final answers mention exact workspace-relative paths as Markdown
inline code resolved by DSH alpha.4's native file-mention contract.

// 中文：项目级导出使用 RO-Crate 1.3 Metadata Document 表达根项目、文件、软件、问题、主张、证据、实验、运行及创建/更新动作。RO-Crate 是交换和读取模型，而不是命令协议；内部关系表不会被宣称为新的私有标准。导出使用标准 Schema.org 关系并避免泄露会话身份和绝对路径。该选择使外部工具能够检查研究对象，而不必加载 SwarmX 的运行数据库 #cite(10) #cite(17)。
Project-level export uses an RO-Crate 1.3 Metadata Document to describe the root project, files,
software, questions, claims, evidence, experiments, runs, and creation or update actions. RO-Crate
is the exchange and read model rather than a command protocol; internal relation tables are not
presented as a new private standard. The export uses standard Schema.org relations and avoids
exposing session identities or absolute paths. External tools can therefore inspect the research
object without loading the SwarmX runtime database #cite(10) #cite(17).

// 中文：对于单独传播的图件，系统还可以把有界生成记录写入标准文件元数据容器。该记录用于在文件脱离项目后恢复绘图代码、库和脱敏运行信息，但它不取代项目级来源，也不把摘要写回文件造成自引用循环。文件级元数据在本文中是支持可追踪性的一个实现机制，而不是 SwarmX 的整体定义。
For figures that travel separately, the system can also place a bounded generation record in a
standard file-metadata container. The record helps recover plotting code, library information, and
a redacted runtime after the file leaves its project, but it does not replace project-level
provenance or write the artifact digest back into the file and create a self-reference cycle.
File-level metadata is one implemented traceability mechanism, not the definition of SwarmX as a
whole.

// 中文：个人知识连续性
== Personal knowledge continuity

// 中文：PKB 将跨会话的个人综合保存为 Open Knowledge Format 0.2 兼容的 UTF-8 Markdown 树 #cite(18)。全局概念和当前工作区概念具有独立索引；模型只接收有界索引快照，页面正文通过显式工具读取。普通 YAML frontmatter、相对 Markdown 链接和命名脚注构成可移植子集，因此所有者可以用 Obsidian 或 MyST 等工具直接检查文件，而不必运行 SwarmX。
The PKB stores cross-session personal synthesis as an Open Knowledge Format 0.2-compatible UTF-8
Markdown tree #cite(18). Global concepts and current-workspace concepts have separate indexes; the
model receives only bounded index snapshots, while page bodies require explicit tool reads. Ordinary
YAML frontmatter, relative Markdown links, and named footnotes form the portable subset, allowing
the owner to inspect the files with tools such as Obsidian or MyST without running SwarmX.

// 中文：PKB 的写入规则刻意比普通聊天记忆严格。创建、更新、废弃或捕获会话摘录都需要当前工具调用的一次性批准；更新必须携带读取时返回的 SHA-256 revision。来自历史对话的来源页面记录确切会话事件范围，并被重新授权后作为“不受信任的历史证据”提供给模型，其中包含的旧指令或权限声明不能控制当前会话。格式畸形的手工编辑页面会被保留并排除出可信综合，而不是被静默修复或删除。
PKB writes are deliberately stricter than ordinary conversational memory. Creating, updating,
deprecating, or capturing a conversation excerpt requires a one-time approval for the current tool
call, and an update must carry the SHA-256 revision returned by the preceding read. A source page
derived from conversation records the exact session event range and is re-authorized before being
returned as untrusted historical evidence; old instructions or permission claims inside that text
cannot govern the current session. A malformed hand-edited page is preserved and excluded from
trusted synthesis rather than silently repaired or deleted.

// 中文：本地明文是可检查性与可移植性的选择，也带来明确限制。目录和文件使用仅所有者权限，但系统不声称提供静态加密、发布、云同步或设备丢失保护。用户若需要这些属性，必须依赖操作系统卷加密、备份或独立同步方案。
Local plaintext is a choice for inspectability and portability, and it carries a clear limitation.
Directories and files use owner-only permissions, but the system makes no claim of encryption at
rest, publication, cloud synchronization, or protection after device loss. Users who require those
properties must rely on operating-system volume encryption, backups, or a separate synchronization
system.

// 中文：代表性端到端工作流
= Representative end-to-end workflow

// 中文：为了说明架构层之间如何协同，表 4 与表 5 跟踪一个从问题形成到可移植输出和长期知识的代表性工作流。该工作流不是某一学科的科学有效性基准，而是用于检查系统能否保持交互连续性和证据关系。
Tables 4 and 5 trace a representative workflow from question formation to portable output and
durable knowledge. The workflow is not a domain benchmark for scientific validity; it is used to
inspect whether the system preserves interaction continuity and evidential relationships.

// 中文：表 4 展示工作流的探索和证据形成阶段。
#figure(
  table(
    columns: (0.07fr, 0.24fr, 0.27fr, 0.25fr, 0.17fr),
    align: (left, left, left, left, left),
    inset: 3.7pt,
    stroke: 0.42pt,
    table.header(
      // 中文：步骤
      [#strong[Step]],
      // 中文：对话中的动作
      [#strong[Conversational action]],
      // 中文：系统行为
      [#strong[System behavior]],
      // 中文：形成的持久证据
      [#strong[Durable evidence]],
      // 中文：用户控制点
      [#strong[User control]],
    ),
    // 中文：1，提出问题并创建项目；智能体建立项目和研究问题；会话事件、项目和 Question 实体；用户可修正问题或分支提示词。
    [1], [Frame a question and create a project], [The agent creates a project and an explicit research question], [Session event, project identity, and Question entity], [Revise the question or branch the prompt],
    // 中文：2，检索本地文献；系统查询 Zotero 并返回规范化记录；可引用文献记录和查询结果；来源库留在本地且不暴露附件。
    [2], [Search the local literature library], [The Host queries Zotero and returns normalized records], [Citation-ready records and query result], [The source library stays local; attachments remain hidden],
    // 中文：3，运行 Notebook 分析；控制器执行单元并捕获结果；执行记录、Notebook revision 和可选不可变工件；取消或缺失运行时显式失败。
    [3], [Run a notebook analysis], [The controller executes a cell and captures selected output], [Execution record, notebook revision, and optional immutable artifact], [Cancellation or missing runtime fails explicitly],
    // 中文：4，检查并评论图件；工件卡片打开 Side View，用户添加点或区域注释；工件摘要、预览定位符和结构化注释；模型读取受保护内容需重新授权。
    [4], [Inspect and comment on a figure], [An artifact card opens Side View and the user adds a point or region annotation], [Artifact digest, preview locator, and structured annotation], [Protected content requires re-authorization],
  ),
  // 中文：代表性工作流的探索与证据形成阶段。每一步留在同一对话流程中，但持久状态由相应事实层保存。
  caption: [Exploration and evidence formation in the representative workflow. Each step remains in one conversational flow, while durable state is held by the corresponding truth layer.],
)

// 中文：表 5 展示工作流的论证、写作、导出和知识沉淀阶段。
#figure(
  table(
    columns: (0.07fr, 0.24fr, 0.27fr, 0.25fr, 0.17fr),
    align: (left, left, left, left, left),
    inset: 3.7pt,
    stroke: 0.42pt,
    table.header(
      // 中文：步骤
      [#strong[Step]],
      // 中文：对话中的动作
      [#strong[Conversational action]],
      // 中文：系统行为
      [#strong[System behavior]],
      // 中文：形成的持久证据
      [#strong[Durable evidence]],
      // 中文：用户控制点
      [#strong[User control]],
    ),
    // 中文：5，记录主张与实验；智能体连接问题、假设、证据、实验和运行；Journal 事件及可重放关系；用户可拒绝提议或要求补充证据。
    [5], [Record a claim and experiment], [The agent links the question, hypothesis, evidence, experiment, and run], [Journal events and replayable entity relations], [Reject a proposal or request further evidence],
    // 中文：6，修订论文；提出精确选区补丁并检查 PDF；文档 revision、补丁状态和论文引用；过期修改被拒绝而不覆盖新文本。
    [6], [Revise the manuscript], [A precise source-selection patch is proposed and the rendered paper is inspected], [Document revision, patch resolution, and paper annotation], [A stale edit cannot overwrite newer text],
    // 中文：7，导出项目；系统投影并验证 RO-Crate；内容寻址导出和凭据；导出不暴露绝对路径或私有会话标识。
    [7], [Export the project], [The system projects and validates an RO-Crate Metadata Document], [Content-addressed export and export receipt], [Absolute paths and private session identifiers are excluded],
    // 中文：8，沉淀个人知识；经批准后写入带来源的 Markdown 概念；PKB 页面、revision 历史和确切脚注；用户可直接编辑，过期智能体写入冲突。
    [8], [Curate durable personal knowledge], [An approved write creates or updates a source-bearing Markdown concept], [PKB page, revision history, and exact source footnotes], [The owner may edit directly; stale agent writes conflict],
  ),
  // 中文：代表性工作流的论证、写作、导出和知识沉淀阶段。科学记录、交付物和个人综合保持连接但权威分离。
  caption: [Argumentation, writing, export, and knowledge curation in the representative workflow. Scientific records, deliverables, and personal synthesis remain connected but retain separate authority.],
)

// 中文：该工作流说明“对话中心”与“聊天即数据库”之间的区别。研究者始终在 Chat 中提出意图、查看总结并发起后续动作，但计算输出、论文源码、科学关系和知识页面并不被序列化为一大段自然语言。相反，回答中的卡片、文件链接和注释把用户带到被授权对象；对象完成修改后，再以定位符和简短结果回到对话。
The workflow illustrates the distinction between conversation-centered design and chat-as-database.
The researcher remains in Chat to express intent, inspect summaries, and initiate follow-up actions,
but computation output, manuscript source, scientific relations, and knowledge pages are not
serialized into one large natural-language transcript. Instead, cards, file links, and annotations
in the answer lead to authorized objects; after an object changes, its locator and bounded result
return to the conversation.

// 中文：它还说明了不同时间尺度的分工。会话日志记录“当时发生了什么”；Science Journal 记录“哪些科学实体和操作已提交”；Artifact Registry 记录“哪些字节已被固定”；论文和工作区记录“当前材料是什么”；PKB 记录“用户决定长期保留的综合是什么”。这种时间尺度分离使系统既能保留原始证据，又能允许研究材料和理解随时间修订。
It also illustrates a division across time scales. The session log records what happened at the
time; the Science Journal records which scientific entities and operations were committed; the
Artifact Registry records which bytes were fixed; the manuscript and workspace record the current
materials; and the PKB records the synthesis the user chose to retain. This temporal separation
preserves source evidence while allowing research materials and understanding to evolve.

// 中文：评估
= Evaluation

// 中文：研究问题与方法
== Research questions and method

// 中文：本文采用设计与契约验证，而不是生产率实验。评估固定到代码元数据表中的提交，并使用两类证据。第一类是对代表性工作流的端到端可追踪分析：每个用户动作必须解析到一个公开接口、明确的事实层、可恢复失败语义和可见返回路径。第二类是实现契约覆盖分析：仓库中的可执行测试和文档被映射到核心主张。由于本文环境未独立重跑并确认一个完整聚合测试计数，本文不报告未经重新验证的“全部测试通过”数字；存在测试只被视为可执行规范，不能替代运行记录或用户研究。
This article uses design-and-contract validation rather than a productivity experiment. The
evaluation is fixed to the commit in the code metadata table and uses two forms of evidence. First,
an end-to-end traceability analysis follows the representative workflow: every user action must
resolve to a public interface, an explicit truth layer, recoverable failure semantics, and a visible
return path. Second, an implementation-contract analysis maps executable tests and documentation in
the repository to the central claims. Because the manuscript environment did not independently
rerun and confirm an aggregate full-suite count, this article does not report an unverified “all
tests passed” number; the presence of tests is treated as executable specification, not as a
substitute for a run record or user study.

// 中文：评估回答以下四个研究问题。
The evaluation addresses four research questions.

// 中文：RQ1，交互连续性：科研任务是否能在不引入平行工作区的情况下，从对话进入计算、工件检查、论文修订并返回对话？
- *RQ1---Interaction continuity:* Can scientific tasks move from conversation into computation, artifact inspection, and manuscript revision and return to conversation without introducing a parallel workspace?

// 中文：RQ2，可追踪与恢复：能否从输出定位相应操作、输入和状态，并在进程重启或失败后恢复持久事实？
- *RQ2---Traceability and recovery:* Can an output be resolved to its operation, inputs, and state, and can durable facts be recovered after process restart or failure?

// 中文：RQ3，授权与失败行为：跨工作区访问、取消、拒绝、过期 revision、畸形文件和缺失本地依赖是否会安全且明确地失败？
- *RQ3---Authorization and failure behavior:* Do cross-workspace access, cancellation, rejection, stale revisions, malformed files, and missing local dependencies fail safely and explicitly?

// 中文：RQ4，互操作性：项目、交付文件和长期知识是否能以外部工具可理解的形式离开 SwarmX？
- *RQ4---Interoperability:* Can projects, deliverable files, and durable knowledge leave SwarmX in forms that external tools can understand?

// 中文：契约证据矩阵
== Contract evidence matrix

// 中文：表 6 汇总固定快照中支持每个研究问题的实现证据及其不能支持的结论。测试文件名称被按行为类别归纳，而不是把测试数量当作论文结果。
#figure(
  table(
    columns: (0.09fr, 0.29fr, 0.34fr, 0.28fr),
    align: (left, left, left, left),
    inset: 4.0pt,
    stroke: 0.43pt,
    table.header(
      // 中文：研究问题
      [#strong[RQ]],
      // 中文：实现与测试证据
      [#strong[Implementation and test evidence]],
      // 中文：得到支持的结论
      [#strong[Supported conclusion]],
      // 中文：未得到支持的结论
      [#strong[Not supported]],
    ),
    // 中文：RQ1；Retry/Edit 历史投影、Side View 生命周期、工件卡片、注释 Composer 与论文侧栏契约；所有能力通过既有 Chat/Trajectory/details 接缝进入；不证明界面更快或更易用。
    [RQ1], [Retry/Edit history projection, Side View lifecycle, artifact-card recovery, annotation Composer, and paper-side-view contracts], [Scientific actions and inspectable results are connected to the existing Chat, Trajectory, Composer, and details region without a peer scientific workspace], [No evidence of faster completion, lower cognitive load, or superior usability],
    // 中文：RQ2；Journal 迁移与重放、Artifact Registry 摘要与去重、Notebook/实验/导出、PKB revision 与会话来源契约；事实层具有明确恢复与定位路径；不证明科学结果可重复得到。
    [RQ2], [Journal migration and replay, artifact digest and deduplication, notebook/experiment/export, PKB revision, and conversation-source contracts], [Durable state has explicit replay, identity, revision, and source paths across sessions and processes], [No proof that a scientific conclusion or numerical result is reproducible in another environment],
    // 中文：RQ3；桌面导航与权限、工作区隔离、路径规范化、取消、过期 revision、批准拒绝、畸形 PKB 页面和缺失运行时契约；失败被拒绝、保留原状态或返回明确错误；不构成完整安全审计。
    [RQ3], [Desktop navigation and permission, workspace isolation, path normalization, cancellation, stale revision, approval denial, malformed PKB page, and missing-runtime contracts], [Specified boundary violations fail explicitly and do not silently publish the targeted state transition], [No complete threat model, penetration test, or guarantee for every third-party runtime],
    // 中文：RQ4；RO-Crate 投影、普通相对文件链接、标准 Markdown/OKF fixture、BibTeX 交换与可移植工件元数据契约；外部格式不要求加载内部数据库；不证明所有第三方工具的无损往返。
    [RQ4], [RO-Crate projection, ordinary relative file links, standard Markdown/OKF fixtures, BibTeX exchange, and portable artifact-metadata contracts], [Projects, files, citations, and personal synthesis have externally readable boundaries that do not require the internal database], [No claim of lossless round-trip through every external application],
  ),
  // 中文：研究问题、固定快照中的实现/测试证据、得到支持的结论与明确不支持结论之间的对应关系。
  caption: [Mapping from the research questions to implementation and test evidence in the fixed snapshot, together with the conclusions that evidence does and does not support.],
)

// 中文：RQ1 的结果是结构性的而非主观性的。代码所有权图显示 Science UI 不注册与 Chat 或 Trajectory 并列的 `conversation.view`；它只贡献完成回合工件、Side View 内容和注释引用。Retry/Edit 生成子会话而不改变源日志。代表性工作流中的每个阶段都通过对话中的工具结果、卡片、文件链接或注释返回。由此可以支持“一个协调面”的架构主张，但不能支持“用户更喜欢这个界面”或“任务更快”的经验主张。
The RQ1 result is structural rather than subjective. The ownership map shows that the Science UI
registers no `conversation.view` peer beside Chat or Trajectory; it contributes completed-turn
artifacts, Side View content, and annotation references. Retry/Edit creates child sessions without
mutating the source log. Every stage of the representative workflow returns through a tool result,
card, file link, or annotation in the conversation. This supports the architectural claim of one
coordination surface, but not empirical claims that users prefer the interface or complete work
faster.

// 中文：RQ2 的结果来自分层恢复路径。会话由 Harness 持久化；Science Journal 可从只追加事件重建物化视图；工件由摘要验证；文档和 PKB 以 revision 检查更新；RO-Crate 和对话摘录保存跨层定位。若某个模型回答没有附带定位符，它仍可作为交互记录存在，但不会自动获得科学事实或长期知识的权威。该负向规则与成功路径同样重要。
The RQ2 result follows from layered recovery paths. Sessions are persisted by the Harness; the
Science Journal rebuilds materialized views from append-only events; artifacts are verified by
digest; documents and PKB pages use revision-checked updates; and RO-Crate plus conversation
excerpts preserve cross-layer locators. If a model answer has no locator, it can remain an
interaction record but does not automatically acquire the authority of a scientific fact or durable
knowledge. This negative rule is as important as the successful path.

// 中文：RQ3 的结果表现为明确拒绝，而不是系统尝试“尽力而为”。跨工作区实体、路径逃逸、未授权跨会话搜索、过期文档或知识 revision、已取消操作、畸形或不受支持的输入以及缺失运行时都具有拒绝路径。失败应保留旧状态，且手工修改的畸形知识页不会被系统替换。需要强调，这些是已指定边界的契约，不是对 Electron、Node、Python、Typst、Zotero 或操作系统全部攻击面的形式化证明。
The RQ3 result is expressed through explicit rejection rather than a best-effort attempt. Cross-
workspace entities, path escape, unapproved cross-session search, stale document or knowledge
revisions, cancelled operations, malformed or unsupported inputs, and missing runtimes all
have refusal paths. Failure is expected to preserve prior state, and a malformed hand-edited
knowledge page is not replaced by the system. These are contracts for specified boundaries, not a
formal proof over every attack surface in Electron, Node, Python, Typst, Zotero, or the operating
system.

// 中文：RQ4 的结果是多种互补格式，而非一个万能容器。工作区文件使用普通相对链接；项目图使用 RO-Crate；引用交换使用 BibTeX；PKB 使用标准 Markdown/OKF；可选文件元数据帮助脱离项目的图件保持生成上下文。没有任何一种格式独占全部语义，也没有要求外部工具理解 SwarmX 会话 ID 或内部 SQLite 表。
The RQ4 result is a set of complementary formats rather than one universal container. Workspace
files use ordinary relative links; project graphs use RO-Crate; citation exchange uses BibTeX; the
PKB uses standard Markdown/OKF; and optional file metadata helps a figure retain generation context
after leaving its project. No single format is assigned every meaning, and external tools need not
understand SwarmX session identifiers or internal SQLite tables.

// 中文：失败场景与负向保证
== Failure cases and negative guarantees

// 中文：科研软件的可靠性不仅取决于成功路径，也取决于拒绝不安全或不一致动作时是否可预测。表 7 汇总当前架构最关键的负向保证。
#figure(
  table(
    columns: (0.28fr, 0.33fr, 0.39fr),
    align: (left, left, left),
    inset: 4.2pt,
    stroke: 0.44pt,
    table.header(
      // 中文：触发条件
      [#strong[Trigger]],
      // 中文：预期行为
      [#strong[Expected behavior]],
      // 中文：保留的证据或状态
      [#strong[Preserved evidence or state]],
    ),
    // 中文：用户重试失败回合；创建分支而不覆写；源消息、错误和重试事件仍留在原会话。
    [The user retries a failed turn], [Create a branch instead of overwriting], [The source message, failure, and retry records remain in the original session],
    // 中文：论文或知识 revision 已变化；拒绝过期修改；当前文件字节和此前 revision 历史不被覆盖。
    [A manuscript or knowledge revision has changed], [Reject the stale update], [Current file bytes and prior revision history remain intact],
    // 中文：模型请求跨工作区实体或路径；拒绝并不暴露主机路径；目标工作区和调用工作区均不发生状态更改。
    [The model requests a cross-workspace entity or escaping path], [Reject without exposing the Host path], [Neither the target nor calling workspace is mutated],
    // 中文：用户拒绝需要批准的 PKB 写入或跨工作区读取；返回拒绝结果；知识页面、索引、日志和模型上下文不获得未批准内容。
    [The user denies an approval-gated PKB write or cross-workspace read], [Return a denial result], [Knowledge pages, indexes, logs, and model context receive no unapproved content],
    // 中文：操作被取消或本地运行时缺失；结束操作并返回明确错误；不发布部分工件、Journal 事件或知识更新。
    [An operation is cancelled or a local runtime is unavailable], [Terminate with an explicit error], [No partial artifact, Journal event, or knowledge update is published],
    // 中文：手工编辑的 Markdown 页面格式畸形；保留原字节并从可信综合排除；所有者修改不被静默修复、隔离或删除。
    [A hand-edited Markdown page is malformed], [Preserve bytes and exclude it from trusted synthesis], [The owner's edit is not silently repaired, quarantined, or deleted],
  ),
  // 中文：关键失败场景及其负向保证。系统通过保留旧状态和原始证据来使失败可恢复。
  caption: [Key failure cases and their negative guarantees. Failure remains recoverable by preserving prior state and source evidence.],
)

// 中文：评估边界
== Evaluation boundary

// 中文：上述分析验证了架构和接口是否一致地表达了设计目标，但它没有回答三个经验问题。首先，没有受控用户研究比较 SwarmX 与独立聊天、Notebook 和写作工具的任务时间、错误率或认知负担。其次，没有延迟、内存、磁盘增长或大项目性能基准。第三，没有领域专家对智能体生成的假设、实验或论文结论进行科学有效性评分。因此，本文结果应被解读为“软件契约形成了一条可检查的工作路径”，而不是“SwarmX 已经提高科研生产率或科研质量”。
The analysis above validates whether the architecture and interfaces express the design goals
consistently, but it does not answer three empirical questions. First, no controlled user study
compares SwarmX with separate chat, notebook, and writing tools on task time, error rate, or cognitive
load. Second, there is no benchmark for latency, memory, disk growth, or large-project performance.
Third, no domain experts have scored the scientific validity of agent-generated hypotheses,
experiments, or manuscript conclusions. The result should therefore be read as “the software
contracts form an inspectable work path,” not “SwarmX has improved research productivity or quality.”

// 中文：后续评估应至少包括三类研究。可用性研究应让研究者在有无对话中心工件和注释的条件下完成同一任务；溯源研究应让独立评审者从论文图件追溯到代码、输入和运行，并测量成功率与时间；系统基准应在不同项目规模和操作系统上测量执行、预览、重放和导出开销。ScienceAgentBench 一类任务级评估还可以测试智能体能力，但必须与 SwarmX 的系统可靠性指标分开报告 #cite(6)。
Future evaluation should include at least three study types. A usability study should ask
researchers to complete the same tasks with and without conversation-centered artifacts and
annotations; a provenance study should ask independent reviewers to trace manuscript figures to
code, inputs, and runs and measure success and time; and a systems benchmark should measure
execution, preview, replay, and export overhead across project sizes and operating systems.
Task-level agent evaluations such as ScienceAgentBench can additionally test model capability, but
must be reported separately from SwarmX system-reliability measures #cite(6).

// 中文：影响与复用
= Impact and reuse

// 中文：SwarmX 的主要复用价值不是任何单个执行器，而是三种组合模式。第一，运行时所有权保持在已有 Harness 中，产品通过加性扩展加入，因此会话、权限和工具生命周期不被复制。第二，对话作为协调面，而工件、论文和知识使用结构化侧栏及定位符保持可检查。第三，不同事实层既保持独立权威，又通过来源连接，使系统能同时支持审计、修订与长期综合。
The principal reuse value of SwarmX is not any individual executor but three composition patterns.
First, runtime ownership stays in the existing Harness and product capability enters additively, so
sessions, permissions, and tool lifecycle are not duplicated. Second, conversation serves as the
coordination surface while artifacts, manuscripts, and knowledge remain inspectable through
structured side views and locators. Third, truth layers retain separate authority while being linked
by sources, enabling audit, revision, and long-term synthesis in the same system.

// 中文：研究软件工程团队可以复用这些模式，而不必采用同一学科工具。例如，一个领域 MCP 服务器可以继续拥有模拟器或数据库，SwarmX 只负责把工具结果注册为可检查工件、连接到项目记录并返回对话。另一个团队也可以替换当前 Notebook 或论文适配器，只要新适配器遵守工作区授权、有界输出、明确 revision 和可取消生命周期。
Research software teams can reuse these patterns without adopting the same disciplinary tools. A
domain MCP server can continue to own its simulator or database, while SwarmX registers resulting
artifacts, links them to the project record, and returns them to conversation. Another team can
replace the current notebook or paper adapter if the new adapter preserves workspace authorization,
bounded output, explicit revisions, and cancellable lifecycle.

// 中文：可移植输出降低了检查门槛。RO-Crate 使项目关系能够被通用研究对象工具读取；普通文件和相对链接避免把交付物锁在私有协议中；Markdown PKB 允许用户在熟悉的编辑器中直接查看和修改个人知识。开放源码与 MIT 许可证还允许研究者检查边界实现并提出替代设计。本文尚无外部采用或部署证据，因此这些是合理复用路径，而不是已测量影响。
Portable outputs reduce the inspection barrier. RO-Crate lets generic research-object tools read
project relations; ordinary files and relative links avoid locking deliverables into a private
protocol; and the Markdown PKB lets users view and modify personal knowledge in familiar editors.
Open source under the MIT License also permits researchers to inspect boundary implementations and
propose alternatives. The article contains no evidence of external adoption or deployment, so these
are plausible reuse paths rather than measured impact.

// 中文：局限性
= Limitations

// 中文：第一，本文描述的是一个快速演进的 0.1.0 开发快照。实现与指定的 DeepSeek Harness Web profile 及若干精确版本接缝耦合。加性架构原则可迁移到其他运行时，但当前快照没有第二个完整智能体运行时适配器，因此不能把“可迁移”误写为“已经证明跨运行时可移植”。
First, this article describes a rapidly evolving `0.1.0` development snapshot. The implementation is
coupled to a specified DeepSeek Harness Web profile and several exact-version seams. The additive
architecture principle could be adapted to another runtime, but the snapshot contains no second
complete agent-runtime adapter; potential portability must not be reported as demonstrated
cross-runtime portability.

// 中文：第二，本地优先能力依赖本机配置。Notebook、文献和论文工作台需要相应本地服务或运行时；系统不会自动安装缺失依赖。当前进程、存储和权限模型面向单用户桌面，不包括多用户协作、远程执行、分布式对象存储或跨设备冲突合并。
Second, local-first capability depends on local configuration. Notebook, literature, and paper
workbenches require their corresponding local services or runtimes, and missing dependencies are
not installed automatically. The current process, storage, and permission model targets a
single-user desktop and excludes multi-user collaboration, remote execution, distributed object
storage, and cross-device conflict merging.

// 中文：第三，PKB 选择所有者可读明文以支持直接检查和工具互操作，但这不是静态加密。目录权限无法防御已获得用户账户权限的进程或设备丢失。会话、科学 Journal 和 PKB 的分层减少语义混淆，却也要求用户理解哪些内容是原始证据、科学事实或个人综合。
Third, the PKB chooses owner-readable plaintext to support direct inspection and tool
interoperability, but this is not encryption at rest. Directory permissions do not protect against a
process that already has the user's account authority or against device loss. Separating sessions,
the Science Journal, and PKB reduces semantic confusion, but it also requires users to understand
which content is source evidence, scientific fact, or personal synthesis.

// 中文：第四，安全契约覆盖沙箱化渲染器、导航、权限、工作区隔离、路径规范化、批准与 revision 冲突，但不构成完整威胁模型。执行用户或模型编写的代码与文档仍调用本地第三方运行时，其安全性取决于操作系统账户、依赖版本和运行时本身。科学数据还可能具有额外合规要求，当前系统未声称满足医疗、个人身份或出口管制数据的专门法规。
Fourth, security contracts cover the sandboxed renderer, navigation, permissions, workspace
isolation, path normalization, approvals, and revision conflict, but do not constitute a complete
threat model. Executing user- or model-authored code and documents still invokes local third-party
runtimes and depends on operating-system accounts, dependency versions, and those runtimes. Research
data may also carry additional compliance obligations; the current system makes no claim of meeting
specialized regulation for medical, personally identifiable, or export-controlled data.

// 中文：第五，当前论文工作台以 Typst 为首个深度适配引擎，不能据此宣称所有写作格式都具有同等的源码—渲染映射。RO-Crate 导出是 Metadata Document 和内容寻址工件，不一定是包含所有外部载荷的完整归档包。PKB 的 OKF 兼容配置同样是一个受限可写子集，而不是对所有 Wiki 扩展语法的支持。
Fifth, the current paper workbench uses Typst as its first deeply integrated engine and therefore
does not establish equivalent source-to-render mapping for every writing format. The RO-Crate export
is a Metadata Document and content-addressed artifact, not necessarily a complete archival package
containing every external payload. The PKB's OKF-compatible profile is likewise a bounded writable
subset rather than support for every wiki extension syntax.

// 中文：最后，本文的评估是设计与契约分析。没有用户研究、性能基准、替代系统对照、外部采用数据或科学有效性评审。固定提交、作者列表、单位、支持邮箱、经费、利益冲突声明和归档 DOI 还需在正式投稿前完成最终核对。
Finally, the evaluation is a design-and-contract analysis. It includes no user study, performance
benchmark, comparison with alternative systems, external adoption data, or scientific-validity
review. The fixed commit, author list, affiliations, support email, funding, competing-interest
statement, and archival DOI require final verification before submission.

// 中文：结论
= Conclusion

// 中文：SwarmX 提出一种面向智能体科研桌面的系统组合：以对话作为统一协调面，但不把聊天记录当作全部事实；复用已有 Harness 作为智能体运行时，但不复制其会话和权限；把计算、工件、论文与知识带回同一交互流程，但让它们由不同事实层持有。非破坏性分支、结构化注释、Side View、Science Journal、不可变工件、RO-Crate 导出和 Markdown PKB 共同实现这一组合。
SwarmX presents a systems composition for an agentic research desktop: conversation is the unified
coordination surface but not the store of every fact; an existing Harness is reused as the agent
runtime without duplicating its sessions and permissions; and computation, artifacts, manuscripts,
and knowledge return to one interaction flow while remaining owned by distinct truth layers.
Non-destructive branching, structured annotations, Side View, the Science Journal, immutable
artifacts, RO-Crate export, and the Markdown PKB jointly implement this composition.

// 中文：代表性工作流和契约矩阵表明，当前快照为交互连续性、可追踪与恢复、明确失败以及外部可读输出提供了一致路径。更强的结论需要更强证据：用户研究应检验该组合是否减少上下文切换和错误，系统基准应测量开销，领域评审应单独检验智能体的科学工作质量。在这些工作完成之前，SwarmX 的最严谨定位是一个开源、本地优先、对话中心的科研环境及其可检查系统设计，而不是已经证明能够自动化科学发现的结果。
The representative workflow and contract matrix show that the current snapshot provides coherent
paths for interaction continuity, traceability and recovery, explicit failure, and externally
readable output. Stronger conclusions require stronger evidence: user studies should test whether
the composition reduces context switching and errors, systems benchmarks should measure overhead,
and domain review should separately assess the quality of the agent's scientific work. Until those
studies are completed, the rigorous claim for SwarmX is an open-source, local-first,
conversation-centered research environment and an inspectable systems design, not demonstrated
automation of scientific discovery.

#v(0.5em)
#set heading(numbering: none)

// 中文：代码可用性
= Code availability

// 中文：SwarmX 采用 MIT License 开源。本文分析的源代码固定在 GitHub 提交 cbff41737b7e5280e0d43a808d21c0b87e95003f。正式投稿前应从最终核验的提交创建公开发行版与归档 DOI，并同步更新代码元数据表。
SwarmX is open source under the MIT License. The source analyzed in this article is fixed to GitHub
commit `cbff41737b7e5280e0d43a808d21c0b87e95003f`. Before formal submission, a public release and
archival DOI should be created from the final verified commit and entered in the code metadata table.

// 中文：数据可用性
= Data availability

// 中文：本文未产生独立研究数据集。代表性工作流、接口契约、测试与文档均包含在固定代码快照中。未来用户研究或性能基准产生的数据应在不泄露私人会话、工作区路径或知识库内容的前提下单独归档。
This article produced no independent research dataset. The representative workflow, interface
contracts, tests, and documentation are contained in the fixed code snapshot. Data from future user
studies or performance benchmarks should be archived separately without exposing private sessions,
workspace paths, or knowledge-base content.

// 中文：CRediT 作者贡献声明
= CRediT authorship contribution statement

// 中文：待最终作者名单与贡献得到确认后，于投稿前填写。不得从代码提交记录推测正式作者贡献。
TO BE COMPLETED BEFORE SUBMISSION after the final author list and contributions are confirmed.
Formal authorship contributions are not inferred from commit history.

// 中文：经费声明
= Funding

// 中文：待作者于投稿前填写；不得推测经费来源。
TO BE COMPLETED BEFORE SUBMISSION; no funding source is inferred.

// 中文：利益冲突声明
= Declaration of competing interest

// 中文：待作者于投稿前填写；不得推测是否存在利益冲突。
TO BE COMPLETED BEFORE SUBMISSION; the presence or absence of competing interests is not inferred.

// 中文：致谢
= Acknowledgements

// 中文：待作者于投稿前填写。应仅列出获得同意且不符合作者资格的贡献者、基础设施或机构支持。
TO BE COMPLETED BEFORE SUBMISSION. Only consented contributions, infrastructure, or institutional
support that do not qualify for authorship should be listed.

// 中文：参考文献
= References

#set par(first-line-indent: 0em)
#set text(size: 8.55pt)

#enum(
  [T. Kluyver, B. Ragan-Kelley, F. Pérez, et al., “Jupyter Notebooks---a publishing format for reproducible computational workflows,” in _Positioning and Power in Academic Publishing: Players, Agents and Agendas_, IOS Press, 2016, pp. 87--90. https://doi.org/10.3233/978-1-61499-649-1-87.],
  [M. Atkinson, S. Gesing, J. Montagnat, and I. Taylor, “Scientific workflows: Past, present and future,” _Future Generation Computer Systems_, vol. 75, pp. 216--227, 2017. https://doi.org/10.1016/j.future.2017.05.041.],
  [M. R. Crusoe, S. Abeln, A. Iosup, et al., “Methods included: Standardizing computational reuse and portability with the Common Workflow Language,” _Communications of the ACM_, vol. 65, no. 6, pp. 54--63, 2022. https://doi.org/10.1145/3486897.],
  [J. F. Pimentel, L. Murta, V. Braganholo, and J. Freire, “A large-scale study about quality and reproducibility of Jupyter notebooks,” in _Proceedings of the 16th IEEE/ACM International Conference on Mining Software Repositories_, 2019, pp. 507--517. https://doi.org/10.1109/MSR.2019.00077.],
  [C. Lu, C. Lu, R. T. Lange, J. Foerster, J. Clune, and D. Ha, “The AI Scientist: Towards fully automated open-ended scientific discovery,” arXiv:2408.06292, 2024.],
  [Z. Chen, S. Chen, Y. Ning, et al., “ScienceAgentBench: Toward rigorous assessment of language agents for data-driven scientific discovery,” arXiv:2410.05080, 2024.],
  [DeepSeek AI, “DeepSeek Harness,” source repository. https://github.com/deepseek-ai/deepseek-harness. Accessed 25 August 2026.],
  [SwarmX Project, “SwarmX,” fixed source snapshot `cbff41737b7e5280e0d43a808d21c0b87e95003f`. https://github.com/tcztzy/swarmx/tree/cbff41737b7e5280e0d43a808d21c0b87e95003f. Accessed 25 August 2026.],
  [M. D. Wilkinson, M. Dumontier, I. J. Aalbersberg, et al., “The FAIR Guiding Principles for scientific data management and stewardship,” _Scientific Data_, vol. 3, 160018, 2016. https://doi.org/10.1038/sdata.2016.18.],
  [S. Soiland-Reyes, P. Sefton, M. Crosas, et al., “Packaging research artefacts with RO-Crate,” _Data Science_, vol. 5, no. 2, pp. 97--138, 2022. https://doi.org/10.3233/DS-210053.],
  [S. Amershi, D. Weld, M. Vorvoreanu, et al., “Guidelines for human--AI interaction,” in _Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems_, 2019, Paper 3, pp. 1--13. https://doi.org/10.1145/3290605.3300233.],
  [M. Kleppmann, A. Wiggins, P. van Hardenberg, and M. McGranaghan, “Local-first software: You own your data, in spite of the cloud,” in _Proceedings of the 2019 ACM SIGPLAN International Symposium on New Ideas, New Paradigms, and Reflections on Programming and Software_, 2019, pp. 154--178. https://doi.org/10.1145/3359591.3359737.],
  [JupyMCP contributors, “JupyMCP: A local-first Jupyter MCP server,” source repository. https://github.com/tcztzy/jupymcp. Accessed 25 August 2026.],
  [Zotero, “Local API,” Zotero Web API v3 developer documentation. https://www.zotero.org/support/dev/web_api/v3/local_api. Accessed 25 August 2026.],
  [Typst GmbH and contributors, “Typst,” source repository and documentation. https://github.com/typst/typst. Accessed 25 August 2026.],
  [Myriad-Dreamin and contributors, “Tinymist: An integrated language service for Typst,” source repository. https://github.com/Myriad-Dreamin/tinymist. Accessed 25 August 2026.],
  [RO-Crate Community, “RO-Crate 1.3 Specification.” https://www.researchobject.org/ro-crate/specification/1.3/. Accessed 25 August 2026.],
  [Google Cloud, “Open Knowledge Format Specification, version 0.2.” https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md. Accessed 25 August 2026.],
)
