//! Edge definitions for the agent graph.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Edge in the agent graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Edge {
    pub source: String,
    pub target: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
}

impl Edge {
    /// Create a new edge.
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            condition: None,
        }
    }

    /// Set a condition on the edge.
    pub fn with_condition(mut self, condition: impl Into<String>) -> Self {
        self.condition = Some(condition.into());
        self
    }
}

#[cfg(test)]
mod tests;
