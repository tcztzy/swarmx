//! Agent module — ACP-backed agent.
//!
//! Each Agent delegates to an ACP agent process over stdio.
//! Supported backends: SwarmX (Python), Claude Code (Node.js), custom commands.

use crate::{hook::Hook, mcp::McpServer};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// How to reach the ACP agent process.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentBackend {
    /// Python SwarmX agent — `python3 -m swarmx.acp_server`
    #[default]
    SwarmX,
    /// Claude Code via Zed's official ACP agent — `claude-agent-acp`
    ClaudeCode,
    /// Any stdio ACP agent binary
    Custom {
        program: String,
        #[serde(default)]
        args: Vec<String>,
    },
}

impl AgentBackend {
    pub(crate) fn command(&self) -> (&str, Vec<&str>) {
        match self {
            AgentBackend::SwarmX => ("python3", vec!["-m", "swarmx.acp_server"]),
            AgentBackend::ClaudeCode => ("claude-agent-acp", vec![]),
            AgentBackend::Custom { program, args } => {
                // Leak-like but this is config-level, short-lived
                (program.as_str(), args.iter().map(|s| s.as_str()).collect())
            }
        }
    }
}

/// Agent in the agent graph.
///
/// Delegates LLM execution to an ACP agent via stdio.
/// The Python SwarmX side handles OpenAI calls, DSPy, MCP tools, hooks.
/// Claude Code handles Anthropic API, tool use, permission modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub returns: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client: Option<Value>,
    #[serde(default, rename = "mcpServers")]
    pub mcp_servers: HashMap<String, McpServer>,
    #[serde(default)]
    pub hooks: Vec<Hook>,
    /// ACP agent backend — how to reach the agent process.
    #[serde(default)]
    pub backend: AgentBackend,
}

impl Agent {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters: serde_json::json!({}),
            returns: None,
            model: None,
            instructions: None,
            client: None,
            mcp_servers: HashMap::new(),
            hooks: Vec::new(),
            backend: AgentBackend::default(),
        }
    }

    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_backend(mut self, backend: AgentBackend) -> Self {
        self.backend = backend;
        self
    }

    pub fn with_mcp_servers(mut self, servers: HashMap<String, McpServer>) -> Self {
        self.mcp_servers = servers;
        self
    }

    pub fn with_hook(mut self, hook: Hook) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Build a minimal single-agent swarm config for the Python ACP side.
    /// Only used by SwarmX backend — Claude Code ignores it.
    fn to_swarm_config(&self) -> Value {
        serde_json::json!({
            "name": self.name,
            "root": self.name,
            "nodes": {
                &self.name: {
                    "name": self.name,
                    "description": self.description,
                    "model": self.model,
                    "instructions": self.instructions,
                    "mcpServers": self.mcp_servers,
                    "hooks": self.hooks,
                }
            },
            "edges": [],
            "parameters": self.parameters,
        })
    }
}

impl Agent {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    pub fn hooks(&self) -> &[Hook] {
        &self.hooks
    }

    pub async fn call(
        &self,
        arguments: Value,
        _context: Option<HashMap<String, Value>>,
    ) -> anyhow::Result<Value> {
        let (program, args) = self.backend.command();
        delegate_via_acp(program, &args, arguments, &self.to_swarm_config(), None).await
    }

    /// Call this agent on an existing session (multi-turn).
    pub async fn call_with_session(
        &self,
        arguments: Value,
        session_id: &str,
        _context: Option<HashMap<String, Value>>,
    ) -> anyhow::Result<Value> {
        let (program, args) = self.backend.command();
        delegate_via_acp(
            program,
            &args,
            arguments,
            &self.to_swarm_config(),
            Some(session_id),
        )
        .await
    }

    /// List all sessions for this agent's backend.
    pub async fn list_sessions(
        &self,
        cwd: Option<&std::path::Path>,
    ) -> anyhow::Result<agent_client_protocol::schema::ListSessionsResponse> {
        let (program, args) = self.backend.command();
        acp_session_list(program, &args, cwd).await
    }

    /// Load and replay a session's conversation history.
    pub async fn load_session(
        &self,
        session_id: &str,
        cwd: &std::path::Path,
    ) -> anyhow::Result<(
        agent_client_protocol::schema::LoadSessionResponse,
        Vec<serde_json::Value>,
    )> {
        let (program, args) = self.backend.command();
        acp_session_load_with_messages(program, &args, session_id, cwd).await
    }

    /// Create a new session on this agent's backend.
    pub async fn new_session(&self, cwd: &std::path::Path) -> anyhow::Result<String> {
        let (program, args) = self.backend.command();
        acp_session_new(program, &args, cwd).await
    }
}

struct ChildGuard(tokio::process::Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.start_kill();
    }
}

fn spawn_acp_child(program: &str, args: &[&str]) -> anyhow::Result<ChildGuard> {
    let mut cmd = tokio::process::Command::new(program);
    cmd.args(args.iter().copied());
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::inherit());
    Ok(ChildGuard(cmd.spawn()?))
}

fn take_child_io(
    child: &mut ChildGuard,
) -> anyhow::Result<(tokio::process::ChildStdin, tokio::process::ChildStdout)> {
    let stdin = child
        .0
        .stdin
        .take()
        .ok_or_else(|| anyhow::anyhow!("stdin missing"))?;
    let stdout = child
        .0
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("stdout missing"))?;
    Ok((stdin, stdout))
}

/// Shared helper: spawn child, init ACP, run closure, kill child.
/// Each call is a fresh connection — suitable for stateless queries
/// like listing/loading sessions.
async fn connect_and_call<F, Fut, T>(program: &str, args: &[&str], f: F) -> anyhow::Result<T>
where
    F: FnOnce(agent_client_protocol::ConnectionTo<agent_client_protocol::Agent>) -> Fut,
    Fut: std::future::Future<Output = Result<T, agent_client_protocol::Error>>,
{
    use agent_client_protocol::Client;
    use agent_client_protocol::schema::{InitializeRequest, ProtocolVersion};
    use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

    let mut child = spawn_acp_child(program, args)?;
    let (stdin, stdout) = take_child_io(&mut child)?;
    let transport = agent_client_protocol::ByteStreams::new(stdin.compat_write(), stdout.compat());

    let result = Client
        .builder()
        .on_receive_notification(
            async move |_: agent_client_protocol::schema::SessionNotification, _| Ok(()),
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(
            transport,
            |conn: agent_client_protocol::ConnectionTo<agent_client_protocol::Agent>| async move {
                conn.send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                f(conn).await
            },
        )
        .await;

    drop(child);
    result.map_err(|e| anyhow::anyhow!(e))
}

fn extract_user_text(arguments: &Value) -> String {
    arguments
        .get("messages")
        .and_then(|m| m.as_array())
        .and_then(|msgs| msgs.last())
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string()
}

/// Spawn an ACP agent process, send prompt, collect response.
///
/// If `session_id` is Some, reuses the existing session (multi-turn).
/// Otherwise creates a new session (one-shot, killed after response).
pub(crate) async fn delegate_via_acp(
    program: &str,
    args: &[&str],
    arguments: Value,
    swarm_config: &Value,
    session_id: Option<&str>,
) -> anyhow::Result<Value> {
    use std::sync::Arc;

    use agent_client_protocol::schema::{
        ContentBlock, InitializeRequest, NewSessionRequest, PromptRequest, ProtocolVersion,
        SessionNotification, SessionUpdate, TextContent,
    };
    use agent_client_protocol::{Agent as AcpAgent, Client, ConnectionTo};
    use tokio::sync::Mutex;
    use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

    let user_text = extract_user_text(&arguments);

    let mut child = spawn_acp_child(program, args)?;
    let (stdin, stdout) = take_child_io(&mut child)?;
    let transport = agent_client_protocol::ByteStreams::new(stdin.compat_write(), stdout.compat());

    let responses: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let responses_c = responses.clone();

    let config = swarm_config.clone();
    Client
        .builder()
        .on_receive_notification(
            async move |notif: SessionNotification, _cx| {
                let update = &notif.update;
                let mut msg = match session_update_to_json(update) {
                    Some(m) => m,
                    None => return Ok(()),
                };
                // Enrich AgentMessageChunk with swarm metadata from text block meta
                if let SessionUpdate::AgentMessageChunk(chunk) = update {
                    if let ContentBlock::Text(text) = &chunk.content {
                        if let Some(meta) = &text.meta {
                            if let Some(role) = meta.get("role").and_then(|v| v.as_str()) {
                                msg["role"] = Value::String(role.to_string());
                            }
                            if let Some(event) = meta.get("swarm_event").and_then(|v| v.as_str()) {
                                msg["swarm_event"] = Value::String(event.to_string());
                            }
                            if let Some(agent) = meta.get("agent").and_then(|v| v.as_str()) {
                                msg["agent"] = Value::String(agent.to_string());
                            }
                        }
                    }
                }
                responses_c.lock().await.push(msg);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(transport, |connection: ConnectionTo<AcpAgent>| async move {
            connection
                .send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;

            let sid = if let Some(sid_str) = session_id {
                // Reuse existing session — skip session/new.
                use agent_client_protocol::schema::SessionId;
                SessionId::from(sid_str.to_string())
            } else {
                // Create a fresh session.
                let session_resp = connection
                    .send_request(NewSessionRequest::new(crate::utils::default_cwd()))
                    .block_task()
                    .await?;
                session_resp.session_id
            };

            let mut meta = serde_json::Map::new();
            meta.insert("swarm_config".to_string(), config);
            let text_block = TextContent::new(user_text).meta(meta);

            connection
                .send_request(PromptRequest::new(
                    sid,
                    vec![ContentBlock::Text(text_block)],
                ))
                .block_task()
                .await?;

            Ok(())
        })
        .await?;

    drop(child);

    let responses = responses.lock().await;
    let merged = merge_chunks(&responses);
    Ok(serde_json::json!({"messages": merged}))
}

/// Merge streaming text deltas into coherent messages.
///
/// ACP streams each token as a separate `AgentMessageChunk`. We accumulate
/// consecutive text deltas that share the same (role, agent, swarm_event) key
/// into a single message.
fn merge_chunks(chunks: &[Value]) -> Vec<Value> {
    fn key(c: &Value) -> (Option<&str>, Option<&str>, Option<&str>, Option<&str>) {
        (
            c.get("role").and_then(|v| v.as_str()),
            c.get("agent").and_then(|v| v.as_str()),
            c.get("swarm_event").and_then(|v| v.as_str()),
            c.get("kind").and_then(|v| v.as_str()),
        )
    }

    let mut merged: Vec<Value> = Vec::new();
    for chunk in chunks {
        let ck = key(chunk);
        let can_extend = merged.last().is_some_and(|last| key(last) == ck);

        if can_extend {
            if let Some(last) = merged.last_mut() {
                let tail = chunk.get("content").and_then(|v| v.as_str()).unwrap_or("");
                if tail.is_empty() {
                    continue;
                }
                if let Some(existing) = last
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                {
                    last["content"] = Value::String(format!("{}{}", existing, tail));
                }
            }
        } else {
            merged.push(chunk.clone());
        }
    }
    merged
}

// ── ACP Session Operations ─────────────────────────────────────────────────
//
// These use raw JSON-RPC to guarantee correct wire format, bypassing
// upstream serde quirks (e.g. skip_serializing_if on empty Vecs).

/// Create a new session on the agent process, return its session ID.
pub async fn acp_session_new(
    program: &str,
    args: &[&str],
    cwd: &std::path::Path,
) -> anyhow::Result<String> {
    let cwd = cwd.to_path_buf();
    connect_and_call(program, args, |conn| async move {
        let resp: agent_client_protocol::schema::NewSessionResponse = conn
            .send_request(agent_client_protocol::schema::NewSessionRequest::new(cwd))
            .block_task()
            .await?;
        Ok(resp.session_id.to_string())
    })
    .await
}

/// List all sessions known to the agent process.
pub async fn acp_session_list(
    program: &str,
    args: &[&str],
    cwd: Option<&std::path::Path>,
) -> anyhow::Result<agent_client_protocol::schema::ListSessionsResponse> {
    use agent_client_protocol::schema::ListSessionsRequest;
    let mut req = ListSessionsRequest::new();
    if let Some(cwd) = cwd {
        req = req.cwd(cwd.to_path_buf());
    }
    let req_json = serde_json::to_value(&req)?;
    tracing::debug!(?req_json, "session/list request");

    connect_and_call(program, args, |conn| async move {
        conn.send_request(req).block_task().await
    })
    .await
}

/// Load a session, replaying its conversation history.
///
/// Uses raw JSON-RPC to guarantee `mcpServers: []` in the request —
/// the upstream LoadSessionRequest may skip empty Vec fields during
/// serialization, which `claude-agent-acp` rejects.
pub async fn acp_session_load(
    program: &str,
    args: &[&str],
    session_id: &str,
    cwd: &std::path::Path,
) -> anyhow::Result<agent_client_protocol::schema::LoadSessionResponse> {
    use agent_client_protocol::schema::SessionId;
    let sid = SessionId::from(session_id.to_string());
    let cwd = cwd.to_path_buf();

    connect_and_call(program, args, |conn| async move {
        // Build request manually so mcpServers is always present.
        let req =
            agent_client_protocol::schema::LoadSessionRequest::new(sid, cwd).mcp_servers(vec![]);
        conn.send_request(req).block_task().await
    })
    .await
}

/// Load a session and collect all replayed messages as structured JSON Values.
/// Returns (LoadSessionResponse, Vec of message chunks).
pub async fn acp_session_load_with_messages(
    program: &str,
    args: &[&str],
    session_id: &str,
    cwd: &std::path::Path,
) -> anyhow::Result<(
    agent_client_protocol::schema::LoadSessionResponse,
    Vec<serde_json::Value>,
)> {
    use agent_client_protocol::Client;
    use agent_client_protocol::schema::{
        InitializeRequest, LoadSessionRequest, ProtocolVersion, SessionId, SessionNotification,
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

    let sid = SessionId::from(session_id.to_string());
    let cwd = cwd.to_path_buf();

    let mut child = spawn_acp_child(program, args)?;
    let (stdin, stdout) = take_child_io(&mut child)?;
    let transport = agent_client_protocol::ByteStreams::new(stdin.compat_write(), stdout.compat());

    let messages: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let messages_c = messages.clone();

    let result: anyhow::Result<agent_client_protocol::schema::LoadSessionResponse> = Client
        .builder()
        .on_receive_notification(
            async move |notif: SessionNotification, _cx| {
                let msg = session_update_to_json(&notif.update);
                if let Some(m) = msg {
                    messages_c.lock().await.push(m);
                }
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(
            transport,
            |conn: agent_client_protocol::ConnectionTo<agent_client_protocol::Agent>| async move {
                conn.send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                let req = LoadSessionRequest::new(sid, cwd).mcp_servers(vec![]);
                conn.send_request(req).block_task().await
            },
        )
        .await
        .map_err(Into::into);

    drop(child);

    let messages = messages.lock().await;
    Ok((result?, messages.clone()))
}

/// Convert a SessionUpdate notification to a simplified JSON Value.
fn session_update_to_json(
    update: &agent_client_protocol::schema::SessionUpdate,
) -> Option<serde_json::Value> {
    use agent_client_protocol::schema::{ContentBlock, SessionUpdate};
    match update {
        SessionUpdate::UserMessageChunk(chunk) => {
            if let ContentBlock::Text(t) = &chunk.content {
                if t.text.is_empty() {
                    return None;
                }
                Some(serde_json::json!({
                    "role": "user",
                    "content": t.text,
                    "kind": "message",
                }))
            } else {
                None
            }
        }
        SessionUpdate::AgentMessageChunk(chunk) => {
            if let ContentBlock::Text(t) = &chunk.content {
                if t.text.is_empty() {
                    return None;
                }
                Some(serde_json::json!({
                    "role": "assistant",
                    "content": t.text,
                    "kind": "message",
                }))
            } else {
                None
            }
        }
        SessionUpdate::AgentThoughtChunk(chunk) => {
            if let ContentBlock::Text(t) = &chunk.content {
                if t.text.is_empty() {
                    return None;
                }
                Some(serde_json::json!({
                    "role": "assistant",
                    "content": t.text,
                    "kind": "thinking",
                }))
            } else {
                None
            }
        }
        SessionUpdate::ToolCall(tc) => {
            let args = tc
                .raw_input
                .as_ref()
                .map(|v| serde_json::to_string(v).unwrap_or_default())
                .unwrap_or_default();
            Some(serde_json::json!({
                "role": "assistant",
                "content": args,
                "kind": "tool_call",
                "tool_name": tc.title,
            }))
        }
        SessionUpdate::ToolCallUpdate(tcu) => {
            let result = tcu
                .fields
                .raw_output
                .as_ref()
                .map(|v| serde_json::to_string(v).unwrap_or_default())
                .or_else(|| {
                    tcu.fields
                        .status
                        .as_ref()
                        .map(|s| serde_json::to_string(s).unwrap_or_default())
                })
                .unwrap_or_default();
            if result.is_empty() {
                return None;
            }
            let name = tcu.fields.title.as_deref().unwrap_or("tool");
            Some(serde_json::json!({
                "role": "assistant",
                "content": result,
                "kind": "tool_result",
                "tool_name": name,
            }))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests;
