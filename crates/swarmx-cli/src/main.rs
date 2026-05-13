//! SwarmX Command Line Interface.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use swarmx_core::{
    Agent, AgentBackend,
    server::create_server_app,
    swarm::{Swarm, SwarmNode},
};
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "swarmx")]
#[command(about = "SwarmX Command Line Interface")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Path to the swarmx configuration file
    #[arg(short, long, global = true)]
    file: Option<PathBuf>,

    /// Path to save the conversation output
    #[arg(short, long, global = true)]
    output: Option<PathBuf>,

    /// Print the data sent to the model
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Start SwarmX as an OpenAI-compatible API server
    Serve {
        /// Host to bind the server to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind the server to
        #[arg(long, default_value_t = 8000)]
        port: u16,

        /// Auto-execute tool calls
        #[arg(long, default_value_t = true)]
        auto_execute_tools: bool,
    },
    /// Run the agent as an MCP server over stdio
    Mcp {
        /// Path to the swarmx configuration file
        #[arg(short, long)]
        file: Option<PathBuf>,
    },
    /// List all sessions from all ACP agents in the swarm
    Sessions {
        /// Working directory filter for sessions
        #[arg(long, default_value = ".")]
        cwd: PathBuf,
    },
    /// Load and replay a specific session
    SessionLoad {
        /// Session ID to load
        session_id: String,
        /// Agent name (defaults to first agent in swarm)
        #[arg(long)]
        agent: Option<String>,
        /// Working directory
        #[arg(long, default_value = ".")]
        cwd: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Serve {
            host,
            port,
            auto_execute_tools,
        }) => {
            serve(host, port, cli.file, auto_execute_tools).await?;
        }
        Some(Commands::Mcp { file }) => {
            warn!("MCP server mode not yet fully implemented in Rust version");
            let _ = file.or(cli.file);
        }
        Some(Commands::Sessions { cwd }) => {
            list_all_sessions_command(cli.file, &cwd).await?;
        }
        Some(Commands::SessionLoad {
            session_id,
            agent,
            cwd,
        }) => {
            load_session_command(cli.file, &session_id, agent.as_deref(), &cwd).await?;
        }
        None => {
            repl(cli.file, cli.output, cli.verbose).await?;
        }
    }

    Ok(())
}

async fn serve(
    host: String,
    port: u16,
    file: Option<PathBuf>,
    auto_execute_tools: bool,
) -> Result<()> {
    let swarm = load_swarm(file).await?;
    let app = create_server_app(Arc::new(swarm), auto_execute_tools);

    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    info!("Starting SwarmX server on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn repl(file: Option<PathBuf>, output: Option<PathBuf>, verbose: bool) -> Result<()> {
    let swarm = load_swarm(file).await?;
    let mut messages: Vec<serde_json::Value> = Vec::new();

    println!("SwarmX REPL");
    println!("Type your messages below. Press Ctrl+C to exit.\n");

    loop {
        let mut input = String::new();
        print!(">>> ");
        std::io::Write::flush(&mut std::io::stdout())?;

        match std::io::stdin().read_line(&mut input) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        messages.push(serde_json::json!({
            "role": "user",
            "content": input
        }));

        if verbose {
            info!(
                "Request: {}",
                serde_json::to_string_pretty(&serde_json::json!({"messages": &messages}))?
            );
        }

        match swarm
            .execute(serde_json::json!({"messages": &messages}), None)
            .await
        {
            Ok(response_messages) => {
                for msg in &response_messages {
                    if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                        println!("{}", content);
                    } else {
                        println!("{}", serde_json::to_string(msg)?);
                    }
                    messages.push(msg.clone());
                }
            }
            Err(e) => {
                warn!("Error: {}", e);
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "refusal": e.to_string()
                }));
                break;
            }
        }
    }

    if let Some(output_path) = output {
        let content = serde_json::to_string_pretty(&messages)?;
        tokio::fs::write(&output_path, content)
            .await
            .with_context(|| format!("Failed to write output to {}", output_path.display()))?;
        info!("Conversation saved to {}", output_path.display());
    }

    Ok(())
}

async fn list_all_sessions_command(file: Option<PathBuf>, cwd: &std::path::Path) -> Result<()> {
    let swarm = load_swarm(file).await?;
    let all = swarm.list_all_sessions(Some(cwd)).await?;
    if all.is_empty() {
        println!("No sessions found.");
        return Ok(());
    }
    for (agent_name, sessions) in &all {
        println!("-- {} --", agent_name);
        for session in sessions {
            let updated_at = session.updated_at.as_deref().unwrap_or("-");
            let title = session.title.as_deref().unwrap_or("(untitled)");
            println!("  {}  {:20}  {}", session.session_id, updated_at, title,);
        }
        println!();
    }
    Ok(())
}

async fn load_session_command(
    file: Option<PathBuf>,
    session_id: &str,
    agent_name: Option<&str>,
    cwd: &std::path::Path,
) -> Result<()> {
    let swarm = load_swarm(file).await?;

    let agent: Agent = if let Some(name) = agent_name {
        match swarm.nodes.get(name) {
            Some(SwarmNode::Agent(a)) => a.clone(),
            _ => {
                anyhow::bail!("Agent '{}' not found in swarm", name);
            }
        }
    } else {
        // Pick first agent node
        match swarm.nodes.values().find_map(|n| {
            if let SwarmNode::Agent(a) = n {
                Some(a.clone())
            } else {
                None
            }
        }) {
            Some(a) => a,
            None => anyhow::bail!("No agents found in swarm"),
        }
    };

    let (_meta, messages) = agent.load_session(session_id, cwd).await?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "agent": agent.name,
            "session_id": session_id,
            "messages": messages,
        }))?
    );
    Ok(())
}

async fn load_swarm(file: Option<PathBuf>) -> Result<Swarm> {
    if let Some(path) = file {
        let content = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read {}", path.display()))?;
        let swarm: Swarm = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse swarm config from {}", path.display()))?;
        Ok(swarm)
    } else {
        Ok(
            Swarm::new("default", "agent").with_node(swarmx_core::swarm::SwarmNode::Agent(
                Agent::new("agent")
                    .with_instructions("You are a helpful agent.")
                    .with_backend(AgentBackend::ClaudeCode),
            )),
        )
    }
}
