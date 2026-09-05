import { useEffect, useState } from "react";
import { z } from "zod";
import { ConversationSurface } from "./chat.js";

const Bootstrap = z.strictObject({
  agents: z.array(z.string()),
  sessions: z.array(
    z.object({
      sessionId: z.string(),
      title: z.string().nullish(),
      updatedAt: z.string().nullish(),
    }),
  ),
  workspace: z.strictObject({ id: z.string(), label: z.string() }),
});
type Bootstrap = z.infer<typeof Bootstrap>;
interface Conversation {
  readonly id: string;
  readonly title: string;
}

export function App() {
  const [bootstrap, setBootstrap] = useState<Bootstrap>();
  const [error, setError] = useState<string>();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selected, setSelected] = useState("");
  const [agentId, setAgentId] = useState("swarm");

  useEffect(() => {
    void fetch("/api/v1/bootstrap")
      .then(async (response) => {
        if (!response.ok) throw new Error(await response.text());
        return Bootstrap.parse(await response.json());
      })
      .then(
        (value) => {
          setBootstrap(value);
          const sessions = value.sessions.map((session, index) => ({
            id: session.sessionId,
            title: session.title ?? `会话 ${String(index + 1)}`,
          }));
          setConversations(sessions);
          setSelected(sessions[0]?.id ?? "");
        },
        (cause: unknown) => setError(cause instanceof Error ? cause.message : String(cause)),
      );
  }, []);

  const create = async () => {
    const response = await fetch(`/api/v1/sessions?agent=${agentId}`, { method: "POST" });
    if (!response.ok) throw new Error(await response.text());
    const session = z.object({ sessionId: z.string() }).parse(await response.json());
    const next = { id: session.sessionId, title: `会话 ${String(conversations.length + 1)}` };
    setConversations((items) => [next, ...items]);
    setSelected(next.id);
  };

  const selectAgent = async (id: string) => {
    setAgentId(id);
    setSelected("");
    setConversations([]);
    setError(undefined);
    const response = await fetch(`/api/v1/sessions?agent=${id}`);
    if (!response.ok) throw new Error(await response.text());
    const sessions = Bootstrap.shape.sessions.parse(await response.json());
    setConversations(
      sessions.map((session) => ({
        id: session.sessionId,
        title: session.title ?? session.sessionId,
      })),
    );
    setSelected(sessions[0]?.sessionId ?? "");
  };

  if (bootstrap === undefined) {
    return (
      <main className="grid h-screen place-items-center bg-white text-neutral-950">
        {error ?? "正在连接 SwarmX Host…"}
      </main>
    );
  }

  return (
    <main className="grid h-screen grid-cols-1 bg-white text-neutral-950 md:grid-cols-[248px_minmax(0,1fr)]">
      <aside className="hidden flex-col gap-4 bg-black px-3.5 py-5 text-white md:flex">
        <header className="flex items-center gap-3 px-2 pt-1 pb-3">
          <span className="font-extrabold tracking-[0.12em]">SX</span>
          <div className="flex flex-col gap-0.5">
            <strong>SwarmX</strong>
            <small className="text-neutral-400">{bootstrap.workspace.label}</small>
          </div>
        </header>
        <select
          aria-label="Agent"
          className="rounded-lg border border-neutral-700 bg-black px-3 py-2"
          value={agentId}
          onChange={(event) =>
            void selectAgent(event.target.value).catch((cause: unknown) => setError(String(cause)))
          }
        >
          {bootstrap.agents.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
        <button
          className="w-full rounded-lg bg-white px-3 py-2.5 text-left font-bold text-black hover:bg-neutral-200"
          onClick={() => void create().catch((cause: unknown) => setError(String(cause)))}
          type="button"
        >
          ＋ 新会话
        </button>
        <nav aria-label="会话">
          {conversations.map((item) => (
            <button
              aria-current={item.id === selected ? "page" : undefined}
              className="mb-1 w-full rounded-lg bg-transparent px-3 py-2.5 text-left text-neutral-200 hover:bg-neutral-800 aria-[current=page]:bg-neutral-800"
              key={item.id}
              onClick={() => setSelected(item.id)}
              type="button"
            >
              {item.title}
            </button>
          ))}
        </nav>
        <footer className="mt-auto flex flex-col gap-0.5 border-neutral-800 border-t px-2 pt-3">
          <strong>{agentId}</strong>
          <small className="text-neutral-400">Native Agents · ACP / A2A ingress</small>
        </footer>
      </aside>
      <section className="h-full min-w-0">
        {selected === "" ? (
          <div className="grid h-full place-items-center">{error ?? "创建一个会话以开始。"}</div>
        ) : (
          <ConversationSurface
            key={`${agentId}:${selected}`}
            agentId={agentId}
            threadId={selected}
            title={conversations.find((item) => item.id === selected)?.title ?? "会话"}
          />
        )}
      </section>
    </main>
  );
}
