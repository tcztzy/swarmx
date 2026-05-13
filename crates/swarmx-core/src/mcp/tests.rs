use crate::mcp::{McpManager, McpServer, McpTool};
use serde_json::json;

#[test]
fn test_mcp_manager_new() {
    let manager = McpManager::new();
    assert!(manager.servers.is_empty());
    assert!(manager.tools.is_empty());
}

#[tokio::test]
async fn test_mcp_add_server() {
    let mut manager = McpManager::new();
    let server = McpServer::Stdio {
        command: "echo".to_string(),
        args: vec!["hello".to_string()],
        env: None,
    };
    manager.add_server("test_server", server).await.unwrap();
    assert!(manager.servers.contains_key("test_server"));
    assert!(manager.tools.contains_key("test_server"));
}

#[test]
fn test_tools_for_openai() {
    let mut manager = McpManager::new();
    manager.tools.insert(
        "test".to_string(),
        vec![McpTool {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }),
            output_schema: None,
        }],
    );
    let tools = manager.tools_for_openai(Some("mcp"), None);
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "mcp__test__get_weather");
}

#[test]
fn test_parse_name_prefixed() {
    let manager = McpManager::new();
    let result = manager.parse_name("mcp__server__tool");
    assert!(result.is_ok());
    let (server, tool) = result.unwrap();
    assert_eq!(server, "server");
    assert_eq!(tool, "tool");
}

#[test]
fn test_parse_name_not_found() {
    let manager = McpManager::new();
    assert!(manager.parse_name("unknown").is_err());
}
