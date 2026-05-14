//! Swarm orchestration module.

use crate::{
    agent::Agent,
    edge::Edge,
    hook::Hook,
    mcp::{McpManager, McpServer},
    tool::Tool,
};
use cel_interpreter::{Context, Program};
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

const MAX_STEPS: usize = 100;

type Graphs = (DiGraph<String, Edge>, HashMap<String, HashSet<String>>);

/// Orchestrator for agents, tools, and sub-swarms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Swarm {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub returns: Option<Value>,
    #[serde(default, rename = "mcpServers")]
    pub mcp_servers: HashMap<String, McpServer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queen: Option<Box<Agent>>,
    pub nodes: HashMap<String, SwarmNode>,
    pub edges: Vec<Edge>,
    pub root: String,
    #[serde(default)]
    pub hooks: Vec<Hook>,
}

/// A node inside a swarm can be an agent, tool, or nested swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SwarmNode {
    Agent(Agent),
    Tool(Tool),
    Swarm(Swarm),
}

impl SwarmNode {
    pub fn name(&self) -> &str {
        match self {
            SwarmNode::Agent(a) => &a.name,
            SwarmNode::Tool(t) => &t.name,
            SwarmNode::Swarm(s) => &s.name,
        }
    }

    pub fn description(&self) -> Option<&str> {
        match self {
            SwarmNode::Agent(a) => a.description.as_deref(),
            SwarmNode::Tool(t) => t.description.as_deref(),
            SwarmNode::Swarm(s) => s.description.as_deref(),
        }
    }
}

#[async_recursion::async_recursion]
async fn run_node(
    node: &SwarmNode,
    arguments: Value,
    context: HashMap<String, Value>,
) -> anyhow::Result<Vec<serde_json::Value>> {
    match node {
        SwarmNode::Agent(agent) => {
            let result = agent.call(arguments, Some(context)).await?;
            if let Some(msgs) = result.get("messages").and_then(|v| v.as_array()) {
                Ok(msgs.clone())
            } else {
                Ok(vec![])
            }
        }
        SwarmNode::Tool(tool) => {
            let result = tool.call(arguments, Some(context)).await?;
            Ok(vec![serde_json::json!({
                "role": "tool",
                "content": result.to_string(),
            })])
        }
        SwarmNode::Swarm(swarm) => {
            let result = swarm.execute(arguments, Some(context)).await?;
            Ok(result)
        }
    }
}

impl Swarm {
    pub fn new(name: impl Into<String>, root: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            name: name.clone(),
            description: None,
            parameters: serde_json::json!({}),
            returns: None,
            mcp_servers: HashMap::new(),
            queen: None,
            nodes: HashMap::new(),
            edges: Vec::new(),
            root: root.into(),
            hooks: Vec::new(),
        }
    }

    pub fn with_node(mut self, node: SwarmNode) -> Self {
        let name = node.name().to_string();
        self.nodes.insert(name, node);
        self
    }

    pub fn with_edge(mut self, edge: Edge) -> Self {
        self.edges.push(edge);
        self
    }

    pub fn rebuild_graphs(&self) -> anyhow::Result<Graphs> {
        let mut graph = DiGraph::new();
        let mut node_indices: HashMap<String, NodeIndex> = HashMap::new();

        for node_name in self.nodes.keys() {
            let idx = graph.add_node(node_name.clone());
            node_indices.insert(node_name.clone(), idx);
        }

        let mut unconditional_graph = DiGraph::new();
        let mut uncond_indices: HashMap<String, NodeIndex> = HashMap::new();
        for node_name in self.nodes.keys() {
            let idx = unconditional_graph.add_node(node_name.clone());
            uncond_indices.insert(node_name.clone(), idx);
        }

        let mut predecessors: HashMap<String, HashSet<String>> = HashMap::new();
        for name in self.nodes.keys() {
            predecessors.insert(name.clone(), HashSet::new());
        }

        for edge in &self.edges {
            if !self.nodes.contains_key(&edge.source) {
                return Err(anyhow::anyhow!(
                    "Unknown edge source {} in swarm",
                    edge.source
                ));
            }
            if self.nodes.contains_key(&edge.target) {
                if let (Some(&s), Some(&t)) = (
                    node_indices.get(&edge.source),
                    node_indices.get(&edge.target),
                ) {
                    graph.add_edge(s, t, edge.clone());
                }
            }
            if edge.condition.is_none() && self.nodes.contains_key(&edge.target) {
                if let (Some(&s), Some(&t)) = (
                    uncond_indices.get(&edge.source),
                    uncond_indices.get(&edge.target),
                ) {
                    unconditional_graph.add_edge(s, t, ());
                    predecessors
                        .entry(edge.target.clone())
                        .or_default()
                        .insert(edge.source.clone());
                }
            }
        }

        if petgraph::algo::is_cyclic_directed(&unconditional_graph) {
            return Err(anyhow::anyhow!("Swarm should be a DAG"));
        }

        Ok((graph, predecessors))
    }

    fn edge_condition_passes(
        &self,
        condition: Option<&str>,
        context: &HashMap<String, Value>,
    ) -> bool {
        let condition = condition.unwrap_or("true");
        match Program::compile(condition) {
            Ok(program) => {
                let mut ctx = Context::default();
                for (k, v) in context {
                    ctx.add_variable(k, v.clone()).ok();
                }
                program
                    .execute(&ctx)
                    .map(|v| cel_value_to_bool(&v))
                    .unwrap_or(false)
            }
            Err(_) => true,
        }
    }

    fn resolve_edge_targets(
        &self,
        edge: &Edge,
        context: &HashMap<String, Value>,
    ) -> anyhow::Result<Vec<String>> {
        let target = &edge.target;
        if self.nodes.contains_key(target) {
            return Ok(vec![target.clone()]);
        }
        if target.starts_with("mcp__") {
            return Ok(vec![]);
        }
        match Program::compile(target) {
            Ok(program) => {
                let mut ctx = Context::default();
                for (k, v) in context {
                    ctx.add_variable(k, v.clone()).ok();
                }
                match program.execute(&ctx) {
                    Ok(val) => Ok(normalize_destinations(&cel_value_to_json(&val))),
                    Err(_) => Ok(vec![]),
                }
            }
            Err(_) => Ok(vec![target.clone()]),
        }
    }

    #[async_recursion::async_recursion]
    pub async fn execute(
        &self,
        arguments: Value,
        context: Option<HashMap<String, Value>>,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let mut manager = McpManager::new();
        for (name, params) in &self.mcp_servers {
            manager.add_server(name, params.clone()).await?;
        }

        let mut new_messages: Vec<serde_json::Value> = Vec::new();
        let context = context.unwrap_or_default();

        let (_graph, predecessors) = self.rebuild_graphs()?;
        let mut visited: HashSet<String> = HashSet::new();
        let mut scheduled: HashSet<String> = HashSet::new();

        let root = self.root.clone();
        if !self.nodes.contains_key(&root) {
            return Err(anyhow::anyhow!("Root node {} not found in swarm", root));
        }

        let mut queue: Vec<String> = vec![root.clone()];
        scheduled.insert(root.clone());

        let mut steps = 0;
        while let Some(node_name) = queue.pop() {
            let node = self
                .nodes
                .get(&node_name)
                .ok_or_else(|| anyhow::anyhow!("Node {} not found", node_name))?;
            let node_messages = run_node(node, arguments.clone(), context.clone()).await?;
            visited.insert(node_name.clone());

            if !node_messages.is_empty() {
                new_messages.extend(node_messages);
            }

            for edge in &self.edges {
                if edge.source != node_name {
                    continue;
                }
                if !self.edge_condition_passes(edge.condition.as_deref(), &context) {
                    continue;
                }
                let targets = self.resolve_edge_targets(edge, &context)?;
                for target in targets {
                    if !self.nodes.contains_key(&target) {
                        return Err(anyhow::anyhow!("Unknown target {} in swarm", target));
                    }
                    if visited.contains(&target) || scheduled.contains(&target) {
                        continue;
                    }
                    let required = predecessors.get(&target).cloned().unwrap_or_default();
                    if !required.is_subset(&visited) {
                        continue;
                    }
                    queue.push(target.clone());
                    scheduled.insert(target);
                }
            }

            steps += 1;
            if steps >= MAX_STEPS {
                break;
            }
        }

        Ok(new_messages)
    }
}

// Session types are now ACP-native: agent_client_protocol::schema::SessionInfo
// and ListSessionsResponse. No custom wrappers needed.

impl Swarm {
    /// Collect sessions from every agent node in the swarm.
    ///
    /// Deduplicates by backend (program + args) so we only query each
    /// agent process once even if multiple nodes share the same backend.
    ///
    /// When `cwd` is `None`, returns sessions across all working directories.
    ///
    /// Returns `Vec<(agent_label, Vec<ACP SessionInfo>)>`.
    pub async fn list_all_sessions(
        &self,
        cwd: Option<&std::path::Path>,
    ) -> anyhow::Result<Vec<(String, Vec<agent_client_protocol::schema::SessionInfo>)>> {
        use std::collections::hash_map::Entry;
        // Map: backend process command -> Vec<agent_name>
        let mut backend_agents: std::collections::HashMap<crate::agent::AcpCommand, Vec<String>> =
            std::collections::HashMap::new();

        for (name, node) in &self.nodes {
            if let crate::swarm::SwarmNode::Agent(agent) = node {
                backend_agents
                    .entry(agent.acp_command())
                    .or_default()
                    .push(name.clone());
            }
        }

        // Also check the queen agent if present
        if let Some(ref queen) = self.queen {
            if let Entry::Vacant(e) = backend_agents.entry(queen.acp_command()) {
                e.insert(vec![queen.name.clone()]);
            }
        }

        let mut results = Vec::new();
        for (command, agent_names) in &backend_agents {
            let list_resp = match crate::agent::acp_session_list_with_command(command, cwd).await {
                Ok(resp) => resp,
                Err(e) => {
                    tracing::warn!(
                        "Failed to list sessions for backend {} (agents: {:?}): {}",
                        command.display_name(),
                        agent_names,
                        e
                    );
                    continue;
                }
            };

            for agent_name in agent_names {
                results.push((agent_name.clone(), list_resp.sessions.clone()));
            }
        }

        Ok(results)
    }
}

fn cel_value_to_bool(value: &cel_interpreter::Value) -> bool {
    match value {
        cel_interpreter::Value::Bool(b) => *b,
        cel_interpreter::Value::Int(i) => *i != 0,
        cel_interpreter::Value::UInt(u) => *u != 0,
        cel_interpreter::Value::Float(f) => *f != 0.0,
        cel_interpreter::Value::String(s) => !s.is_empty(),
        cel_interpreter::Value::Bytes(b) => !b.is_empty(),
        cel_interpreter::Value::List(l) => !l.is_empty(),
        cel_interpreter::Value::Map(_) => true,
        cel_interpreter::Value::Null => false,
        cel_interpreter::Value::Function(_, _) => true,
        cel_interpreter::Value::Duration(_) => true,
        cel_interpreter::Value::Timestamp(_) => true,
    }
}

fn cel_value_to_json(value: &cel_interpreter::Value) -> Value {
    match value {
        cel_interpreter::Value::Bool(b) => Value::Bool(*b),
        cel_interpreter::Value::Int(i) => Value::Number(serde_json::Number::from(*i)),
        cel_interpreter::Value::UInt(u) => Value::Number(serde_json::Number::from(*u)),
        cel_interpreter::Value::Float(f) => serde_json::json!(*f),
        cel_interpreter::Value::String(s) => Value::String((**s).clone()),
        cel_interpreter::Value::Bytes(b) => serde_json::json!(b),
        cel_interpreter::Value::List(l) => Value::Array(l.iter().map(cel_value_to_json).collect()),
        cel_interpreter::Value::Map(m) => {
            let mut map = serde_json::Map::new();
            for (k, v) in m.map.iter() {
                map.insert(k.to_string(), cel_value_to_json(v));
            }
            Value::Object(map)
        }
        cel_interpreter::Value::Null => Value::Null,
        cel_interpreter::Value::Function(_, _) => Value::Null,
        cel_interpreter::Value::Duration(d) => serde_json::json!(d.as_seconds_f64()),
        cel_interpreter::Value::Timestamp(t) => serde_json::json!(t.timestamp()),
    }
}

fn normalize_destinations(value: &Value) -> Vec<String> {
    match value {
        Value::Null => vec![],
        Value::String(s) => vec![s.clone()],
        Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        Value::Object(map) => {
            if let Some(dest) = map.get("destination").and_then(|v| v.as_str()) {
                vec![dest.to_string()]
            } else if let Some(dests) = map.get("destinations") {
                match dests {
                    Value::Array(arr) => arr
                        .iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect(),
                    Value::Null => vec![],
                    Value::String(s) => vec![s.clone()],
                    _ => vec![dests.to_string()],
                }
            } else {
                vec![]
            }
        }
        _ => vec![value.to_string()],
    }
}

#[cfg(test)]
mod tests;
