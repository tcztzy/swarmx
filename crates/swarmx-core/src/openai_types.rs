//! Re-exports and aliases for async-openai types to insulate from version changes.

pub use async_openai::types::{
    ChatChoice, ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestFunctionMessage, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestToolMessage,
    ChatCompletionRequestUserMessage, ChatCompletionResponseMessage, ChatCompletionTool,
    ChatCompletionToolType, CompletionUsage, CreateChatCompletionRequest,
    CreateChatCompletionResponse, FinishReason, FunctionCall, Role, Stop,
};

// Content enums
pub use async_openai::types::{
    ChatCompletionRequestAssistantMessageContent as AssistantMessageContent,
    ChatCompletionRequestSystemMessageContent as SystemMessageContent,
    ChatCompletionRequestToolMessageContent as ToolMessageContent,
    ChatCompletionRequestUserMessageContent as UserMessageContent,
};

// Message constructors
macro_rules! msg_ctor {
    ($name:ident, $variant:ident, $msg_ty:ident, $content_ty:ident) => {
        pub fn $name(content: impl Into<String>) -> ChatCompletionRequestMessage {
            ChatCompletionRequestMessage::$variant($msg_ty {
                content: $content_ty::Text(content.into()),
                ..Default::default()
            })
        }
    };
}

msg_ctor!(
    system_message,
    System,
    ChatCompletionRequestSystemMessage,
    SystemMessageContent
);
msg_ctor!(
    user_message,
    User,
    ChatCompletionRequestUserMessage,
    UserMessageContent
);

pub fn assistant_message(content: impl Into<String>) -> ChatCompletionRequestMessage {
    ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
        content: Some(AssistantMessageContent::Text(content.into())),
        ..Default::default()
    })
}

#[allow(clippy::needless_update)]
pub fn tool_message(
    tool_call_id: impl Into<String>,
    content: impl Into<String>,
) -> ChatCompletionRequestMessage {
    ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage {
        content: ToolMessageContent::Text(content.into()),
        tool_call_id: tool_call_id.into(),
        ..Default::default()
    })
}

/// Convert a response message back to a request message for conversation continuation.
pub fn response_to_request(msg: &ChatCompletionResponseMessage) -> ChatCompletionRequestMessage {
    let mut tool_calls = None;
    if let Some(ref calls) = msg.tool_calls {
        tool_calls = Some(calls.clone());
    }
    ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
        content: msg.content.clone().map(AssistantMessageContent::Text),
        refusal: msg.refusal.clone(),
        name: None,
        tool_calls,
        ..Default::default()
    })
}
