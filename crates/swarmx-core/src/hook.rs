//! Hook definitions for agent lifecycle events.

use serde::{Deserialize, Serialize};

/// Hook for agent lifecycle events.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hook {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_start: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_end: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_handoff: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_chunk: Option<String>,
}

#[cfg(test)]
mod tests;
