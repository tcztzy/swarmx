//! OpenAI-compatible server.

use crate::openai_types::{
    ChatChoice, ChatCompletionRequestMessage, ChatCompletionResponseMessage,
    CreateChatCompletionResponse, FinishReason, Role,
};
use crate::swarm::Swarm;
use crate::utils::{get_random_string, now};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use axum_streams::StreamBodyAs;
use futures::stream::{self};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Server state.
#[derive(Clone)]
pub struct AppState {
    pub swarm: Arc<Swarm>,
    pub auto_execute_tools: bool,
}

/// List models response.
#[derive(Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelData>,
}

#[derive(Serialize)]
struct ModelData {
    id: String,
    object: String,
    created: i64,
    owned_by: String,
}

/// Chat completion request.
#[derive(Deserialize)]
struct ChatCompletionRequest {
    messages: Vec<ChatCompletionRequestMessage>,
    model: String,
    #[serde(default)]
    stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

/// Error response.
#[derive(Serialize)]
struct ErrorResponse {
    error: ApiError,
}

#[derive(Serialize)]
struct ApiError {
    message: String,
    r#type: String,
}

fn error_response(message: impl Into<String>, status: StatusCode) -> Response {
    let body = Json(ErrorResponse {
        error: ApiError {
            message: message.into(),
            r#type: "invalid_request_error".to_string(),
        },
    });
    (status, body).into_response()
}

/// Create Axum app with OpenAI-compatible endpoints plus session management.
pub fn create_server_app(swarm: Arc<Swarm>, auto_execute_tools: bool) -> Router {
    let state = AppState {
        swarm,
        auto_execute_tools,
    };

    Router::new()
        .route("/models", get(list_models))
        .route("/chat/completions", post(create_chat_completions))
        .route("/sessions", get(list_sessions))
        .route("/sessions/{agent_name}/{session_id}", get(load_session))
        .with_state(state)
}

async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    let mut models = Vec::new();
    models.push(ModelData {
        id: state.swarm.name.clone(),
        object: "model".to_string(),
        created: now(),
        owned_by: "swarmx".to_string(),
    });

    for name in state.swarm.nodes.keys() {
        models.push(ModelData {
            id: name.clone(),
            object: "model".to_string(),
            created: now(),
            owned_by: "swarmx".to_string(),
        });
    }

    Json(ModelsResponse {
        object: "list".to_string(),
        data: models,
    })
}

fn build_chunk(id: &str, model: &str, delta_content: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": now(),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": delta_content},
            "finish_reason": "stop"
        }]
    })
}

#[allow(deprecated)]
async fn create_chat_completions(
    State(state): State<AppState>,
    Json(params): Json<ChatCompletionRequest>,
) -> Response {
    let model_name = params.model;
    let arguments = serde_json::json!({
        "messages": params.messages,
        "stream": params.stream,
        "auto_execute_tools": state.auto_execute_tools,
        "max_tokens": params.max_tokens,
    });

    if !params.stream {
        match state.swarm.execute(arguments, None).await {
            Ok(messages) => {
                let content = serde_json::to_string(&messages).unwrap_or_default();
                Json(CreateChatCompletionResponse {
                    id: format!("chatcmpl-{}", get_random_string(10)),
                    object: "chat.completion".to_string(),
                    created: now() as u32,
                    model: model_name,
                    system_fingerprint: None,
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatCompletionResponseMessage {
                            content: Some(content),
                            refusal: None,
                            role: Role::Assistant,
                            audio: None,
                            function_call: None,
                            tool_calls: None,
                        },
                        finish_reason: Some(FinishReason::Stop),
                        logprobs: None,
                    }],
                    usage: None,
                    service_tier: None,
                })
                .into_response()
            }
            Err(e) => error_response(e.to_string(), StatusCode::INTERNAL_SERVER_ERROR),
        }
    } else {
        // Streaming: execute once, emit result as SSE chunk + [DONE], then end.
        // True token-level streaming requires ACP integration — not yet wired.
        let msg_id = format!("chatcmpl-{}", get_random_string(10));
        let model = model_name.clone();
        let stream = stream::once(async move {
            match state.swarm.execute(arguments, None).await {
                Ok(messages) => {
                    let content = serde_json::to_string(&messages).unwrap_or_default();
                    let chunk = build_chunk(&msg_id, &model, &content);
                    format!("data: {}\n\ndata: [DONE]\n\n", chunk)
                }
                Err(e) => {
                    let chunk = build_chunk(&msg_id, &model, &e.to_string());
                    format!("data: {}\n\ndata: [DONE]\n\n", chunk)
                }
            }
        });
        StreamBodyAs::text(stream).into_response()
    }
}

/// GET /sessions?cwd=<path>
///
/// Lists all sessions from every ACP-backed agent in the swarm.
async fn list_sessions(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Response {
    let cwd: Option<std::path::PathBuf> = params.get("cwd").map(std::path::PathBuf::from);

    match state.swarm.list_all_sessions(cwd.as_deref()).await {
        Ok(sessions) => Json(sessions).into_response(),
        Err(e) => error_response(e.to_string(), StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// GET /sessions/:agent_name/:session_id?cwd=<path>
///
/// Loads and replays a specific session from the named agent.
async fn load_session(
    State(state): State<AppState>,
    axum::extract::Path((agent_name, session_id)): axum::extract::Path<(String, String)>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Response {
    let cwd = params
        .get("cwd")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(crate::utils::default_cwd);

    let agent = match state.swarm.nodes.get(&agent_name) {
        Some(crate::swarm::SwarmNode::Agent(a)) => a,
        _ => match &state.swarm.queen {
            Some(q) if q.name == agent_name => q.as_ref(),
            _ => {
                return error_response(
                    format!("Agent '{}' not found", agent_name),
                    StatusCode::NOT_FOUND,
                );
            }
        },
    };

    match agent.load_session(&session_id, &cwd).await {
        Ok((_meta, messages)) => Json(serde_json::json!({
            "agent": agent_name,
            "session_id": session_id,
            "messages": messages,
        }))
        .into_response(),
        Err(e) => error_response(e.to_string(), StatusCode::INTERNAL_SERVER_ERROR),
    }
}

#[cfg(test)]
mod tests;
