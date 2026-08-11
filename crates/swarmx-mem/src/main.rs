use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Implementation, ListToolsResult, PaginatedRequestParams,
    ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::{ErrorData as McpError, ServerHandler, ServiceExt};
use serde_json::{Map, Value, json};
use swarmx_mem::{MemoryService, PROTOCOL_VERSION, RUNTIME_VERSION};

const SERVER_NAME: &str = "swarmx-mem";
const TOOL_NAME: &str = "swarmx_memory";

#[derive(Clone)]
struct MemoryMcpServer {
    service: Arc<Mutex<MemoryService>>,
}

impl ServerHandler for MemoryMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(SERVER_NAME, RUNTIME_VERSION))
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ListToolsResult, McpError>> + Send + '_ {
        std::future::ready(Ok(ListToolsResult {
            tools: vec![Tool::new(
                TOOL_NAME,
                "Private SwarmX Memory operation",
                tool_schema(),
            )],
            next_cursor: None,
            meta: None,
        }))
    }

    fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<CallToolResult, McpError>> + Send + '_ {
        let result = (|| -> Result<CallToolResult, McpError> {
            if request.name.as_ref() != TOOL_NAME {
                return Err(McpError::invalid_params(
                    "Unknown private Memory tool.",
                    None,
                ));
            }
            let input = Value::Object(request.arguments.unwrap_or_default());
            let response = self
                .service
                .lock()
                .map_err(|_| McpError::internal_error("Memory runtime is unavailable.", None))?
                .handle(input);
            let is_error = response.get("ok").and_then(Value::as_bool) == Some(false);
            Ok(if is_error {
                CallToolResult::structured_error(response)
            } else {
                CallToolResult::structured(response)
            })
        })();
        std::future::ready(result)
    }
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("swarmx-mem: {error}");
        std::process::exit(2);
    }
}

async fn run() -> Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.as_slice() == ["--version-json"] {
        println!(
            "{}",
            json!({
                "name": SERVER_NAME,
                "version": RUNTIME_VERSION,
                "protocolVersion": PROTOCOL_VERSION
            })
        );
        return Ok(());
    }

    let root = parse_serve_arguments(&arguments)?;
    let service = MemoryService::open(&root).context("failed to open Memory authority")?;
    let server = MemoryMcpServer {
        service: Arc::new(Mutex::new(service)),
    };
    server
        .serve(rmcp::transport::io::stdio())
        .await
        .map_err(|error| anyhow::anyhow!("failed to start private MCP transport: {error}"))?
        .waiting()
        .await
        .map_err(|error| anyhow::anyhow!("private MCP transport failed: {error}"))?;
    Ok(())
}

fn parse_serve_arguments(arguments: &[String]) -> Result<PathBuf> {
    if arguments.len() != 4
        || arguments[0] != "serve"
        || arguments[1] != "--root"
        || arguments[3] != "--stdio"
        || arguments[2].is_empty()
    {
        bail!("expected `serve --root <path> --stdio`");
    }
    Ok(PathBuf::from(&arguments[2]))
}

fn tool_schema() -> Arc<Map<String, Value>> {
    let properties: HashMap<&str, Value> = HashMap::from([
        (
            "protocolVersion",
            json!({ "type": "integer", "const": PROTOCOL_VERSION }),
        ),
        (
            "operation",
            json!({
                "type": "string",
                "enum": [
                    "list", "get", "search", "snapshot", "global_get", "global_save",
                    "global_forget", "create", "update", "delete", "history",
                    "get_version", "diff", "restore"
                ]
            }),
        ),
        (
            "target",
            json!({ "type": "string", "enum": ["user", "memory"] }),
        ),
        ("id", json!({ "type": "string" })),
        ("title", json!({ "type": "string" })),
        (
            "aliases",
            json!({ "type": "array", "items": { "type": "string" } }),
        ),
        ("content", json!({ "type": "string" })),
        ("query", json!({ "type": "string" })),
        ("limit", json!({ "type": "integer" })),
        ("expectedRevision", json!({ "type": "integer" })),
        ("version", json!({ "type": "string" })),
        ("fromVersion", json!({ "type": "string" })),
        ("toVersion", json!({ "type": "string" })),
    ]);
    let schema = json!({
        "type": "object",
        "properties": properties,
        "required": ["protocolVersion", "operation"],
        "additionalProperties": false
    });
    Arc::new(schema.as_object().expect("tool schema object").clone())
}
