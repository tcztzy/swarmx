//! Tool module.

use crate::mcp::{McpManager, McpTool};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Tool node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub returns: Option<Value>,
    #[serde(default)]
    pub hooks: Vec<crate::hook::Hook>,
    #[serde(skip)]
    pub mcp_manager: Option<McpManager>,
}

impl Tool {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters: serde_json::json!({}),
            returns: None,
            hooks: Vec::new(),
            mcp_manager: None,
        }
    }

    pub fn from_mcp(mcp_tool: &McpTool) -> Self {
        Self {
            name: mcp_tool.name.clone(),
            description: mcp_tool.description.clone(),
            parameters: mcp_tool.input_schema.clone(),
            returns: mcp_tool.output_schema.clone(),
            hooks: Vec::new(),
            mcp_manager: None,
        }
    }
}

impl Tool {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    pub async fn call(
        &self,
        arguments: Value,
        _context: Option<HashMap<String, Value>>,
    ) -> anyhow::Result<Value> {
        if let Some(ref manager) = self.mcp_manager {
            let result = manager.call_tool(&self.name, Some(arguments)).await?;
            Ok(serde_json::to_value(result)?)
        } else {
            Err(anyhow::anyhow!("Tool {} has no MCP manager", self.name))
        }
    }
}

#[cfg(test)]
mod tests;
