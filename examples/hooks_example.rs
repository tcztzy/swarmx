//! Example demonstrating Hook functionality in SwarmX.
//!
//! This example shows how to use hooks to execute MCP tools at various points
//! in the agent lifecycle. Hook tools can merge structured output into context.

use swarmx_core::{Agent, Hook};

fn main() {
    // Create hooks that reference MCP tool names
    // These tools would need to be available in your MCP server
    let logging_hook = Hook {
        on_start: Some("log_agent_start".to_string()),
        on_chunk: Some("log_stream_chunk".to_string()),
        on_end: Some("log_agent_end".to_string()),
        ..Default::default()
    };

    // Create another hook for metrics collection
    let metrics_hook = Hook {
        on_start: Some("start_timer".to_string()),
        on_end: Some("end_timer_and_record_metrics".to_string()),
        ..Default::default()
    };

    // Create an agent with hooks
    let agent = Agent::new("HookedAgent")
        .with_instructions("You are a helpful assistant with lifecycle hooks.")
        .with_hook(logging_hook.clone())
        .with_hook(metrics_hook.clone());

    // Example of hook serialization
    println!("Hook serialization example:");
    println!("{}", serde_json::to_string_pretty(&logging_hook).unwrap());
    println!();

    // Example of agent with hooks serialization
    println!("Agent with hooks serialization:");
    let agent_json = serde_json::to_value(&agent).unwrap();
    println!(
        "Agent has {} hooks",
        agent_json["hooks"].as_array().unwrap().len()
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&agent_json["hooks"]).unwrap()
    );
    println!();

    // Example of deserializing agent with hooks
    println!("Deserializing agent with hooks:");
    let restored: Agent = serde_json::from_value(agent_json).unwrap();
    println!("Restored agent has {} hooks", restored.hooks.len());
    println!(
        "First hook on_start: {:?}",
        restored.hooks[0].on_start
    );
    println!();

    // Note: To actually run the agent, you would need:
    // 1. MCP servers configured with the hook tools that accept input format:
    //    {"messages": [...], "context": {...}, "agent": {...}}
    // 2. Hook tools that return structured output merged into context:
    //    {"system_info": {...}}
    // 3. Proper OpenAI API configuration in .env
    //
    // Example run:
    // #[tokio::main]
    // async fn main() -> anyhow::Result<()> {
    //     let messages = vec![
    //         swarmx_core::openai_types::user_message("Hello!"),
    //     ];
    //     let result = agent.call(
    //         serde_json::json!({"messages": messages}),
    //         None,
    //     ).await?;
    //     println!("Agent response: {}", result);
    //     Ok(())
    // }
}
