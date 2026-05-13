//! MCP client related abstractions.

use async_openai::types::ChatCompletionTool;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// SSE server configuration for MCP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseServer {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
}

/// MCP server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum McpServer {
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        env: Option<HashMap<String, String>>,
    },
    Sse(SseServer),
}

/// MCP Tool representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
}

/// MCP tool call result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallToolResult {
    pub content: Vec<ContentBlock>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_content: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

/// Content block in MCP result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text { text: String },
    Image { data: String, mime_type: String },
    Resource { resource: EmbeddedResource },
    Audio { data: String, mime_type: String },
    ResourceLink { name: String, uri: String },
}

/// Embedded resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedResource {
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

/// Registry for MCP tools and clients.
#[derive(Debug, Default, Clone)]
pub struct McpManager {
    pub servers: HashMap<String, McpServer>,
    pub tools: HashMap<String, Vec<McpTool>>,
}

impl McpManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tools_for_openai(
        &self,
        prefix: Option<&str>,
        exclude: Option<&std::collections::HashSet<String>>,
    ) -> Vec<ChatCompletionTool> {
        let mut openai_tools = Vec::new();
        for (server_name, tools) in &self.tools {
            for tool in tools {
                let prefixed_name = if let Some(p) = prefix {
                    format!("{}__{}__{}", p, server_name, tool.name)
                } else {
                    tool.name.clone()
                };
                if let Some(ex) = exclude {
                    if ex.contains(&tool.name) || ex.contains(&prefixed_name) {
                        continue;
                    }
                }
                openai_tools.push(ChatCompletionTool {
                    r#type: async_openai::types::ChatCompletionToolType::Function,
                    function: async_openai::types::FunctionObject {
                        name: prefixed_name,
                        description: tool.description.clone(),
                        parameters: Some(tool.input_schema.clone()),
                        strict: None,
                    },
                });
            }
        }
        openai_tools
    }

    pub async fn add_server(&mut self, name: &str, server_params: McpServer) -> anyhow::Result<()> {
        if self.servers.contains_key(name) {
            return Ok(());
        }
        self.servers.insert(name.to_string(), server_params);
        self.tools.insert(name.to_string(), Vec::new());
        Ok(())
    }

    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> anyhow::Result<CallToolResult> {
        let (server_name, tool_name) = self.parse_name(name)?;
        let _server = self
            .servers
            .get(&server_name)
            .ok_or_else(|| anyhow::anyhow!("MCP server not found: {}", server_name))?;
        tracing::warn!(
            "MCP tool call not yet implemented: {}.{}",
            server_name,
            tool_name
        );
        Ok(CallToolResult {
            content: vec![ContentBlock::Text {
                text: format!("Tool {} not implemented in Rust version yet", name),
            }],
            structured_content: arguments,
            is_error: Some(true),
        })
    }

    fn parse_name(&self, name: &str) -> anyhow::Result<(String, String)> {
        if name.starts_with("mcp__") {
            let parts: Vec<&str> = name.splitn(3, "__").collect();
            if parts.len() == 3 {
                return Ok((parts[1].to_string(), parts[2].to_string()));
            }
        }
        for (server_name, tools) in &self.tools {
            for tool in tools {
                if tool.name == name {
                    return Ok((server_name.clone(), tool.name.clone()));
                }
            }
        }
        Err(anyhow::anyhow!("Tool not found: {}", name))
    }
}

#[cfg(test)]
mod tests;
