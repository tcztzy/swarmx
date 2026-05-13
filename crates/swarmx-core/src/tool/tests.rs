use crate::mcp::McpTool;
use crate::tool::Tool;
use serde_json::json;

#[test]
fn test_tool_creation() {
    let tool = Tool::new("test_tool");
    assert_eq!(tool.name(), "test_tool");
}

#[test]
fn test_tool_from_mcp() {
    let mcp_tool = McpTool {
        name: "mcp_tool".to_string(),
        description: Some("A test tool".to_string()),
        input_schema: json!({"type": "object"}),
        output_schema: None,
    };
    let tool = Tool::from_mcp(&mcp_tool);
    assert_eq!(tool.name(), "mcp_tool");
    assert_eq!(tool.description(), Some("A test tool"));
}

#[test]
fn test_tool_serde() {
    let tool = Tool::new("serde_tool");
    let json = serde_json::to_string(&tool).unwrap();
    assert!(json.contains("serde_tool"));
    let deserialized: Tool = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name(), "serde_tool");
}
