use super::*;
use crate::theme::ThemePreference;
use crate::tokens::{Density, DesignTokens};
use iced::widget::markdown::Content;

fn tokens() -> DesignTokens {
    DesignTokens::for_density(Density::Comfortable)
}

fn theme() -> iced_shadcn::Theme {
    ThemePreference::Dark.shadcn_theme()
}

fn user_msg() -> ChatMessage {
    ChatMessage::new_user("Hello, world!".into())
}

fn assistant_msg() -> ChatMessage {
    ChatMessage::new_assistant("Sure thing.".into())
}

fn thinking_msg() -> ChatMessage {
    ChatMessage::new_thinking("Let me think about this...".into())
}

fn tool_call_msg() -> ChatMessage {
    ChatMessage::new_tool_call("read_file".into(), r#"{"path": "src/main.rs"}"#.into())
}

fn tool_result_msg() -> ChatMessage {
    ChatMessage::new_tool_result("read_file".into(), "fn main() {}".into())
}

fn edit_tool_call_msg() -> ChatMessage {
    let args = serde_json::json!({
        "file_path": "/Users/tcztzy/swarmx/crates/swarmx-ui/src/main.rs",
        "old_string": "fn main() {\n    println!(\"old\");\n}\n",
        "new_string": "fn main() {\n    println!(\"new\");\n    println!(\"again\");\n}\n",
        "replace_all": false,
    });
    ChatMessage::new_tool_call("Edit crates/swarmx-ui/src/main.rs".into(), args.to_string())
}

fn bash_tool_call_msg(command: &str, description: Option<&str>) -> ChatMessage {
    let args = if let Some(description) = description {
        serde_json::json!({
            "command": command,
            "description": description,
        })
    } else {
        serde_json::json!({
            "command": command,
        })
    };
    ChatMessage::new_tool_call(command.into(), args.to_string())
}

// ── user message ───────────────────────────────────────────────────────────

#[test]
fn view_user_msg_does_not_panic_short() {
    let t = tokens();
    let th = theme();
    let _ = view_user_msg(&user_msg(), &t, &th);
}

#[test]
fn view_user_msg_does_not_panic_long() {
    let t = tokens();
    let th = theme();
    let msg = ChatMessage::new_user("a".repeat(500));
    let _ = view_user_msg(&msg, &t, &th);
}

// ── assistant message ──────────────────────────────────────────────────────

#[test]
fn view_assistant_msg_without_icon() {
    let t = tokens();
    let th = theme();
    let md = Content::parse("Hello **world**");
    let _ = view_assistant_msg(&assistant_msg(), &md, false, &[], &t, &th);
}

#[test]
fn view_assistant_msg_with_icon() {
    let t = tokens();
    let th = theme();
    let md = Content::parse("Hello **world**");
    let svg = b"<svg></svg>";
    let _ = view_assistant_msg(&assistant_msg(), &md, true, svg, &t, &th);
}

#[test]
fn view_assistant_msg_with_empty_svg_bytes() {
    let t = tokens();
    let th = theme();
    let md = Content::parse("text");
    let _ = view_assistant_msg(&assistant_msg(), &md, true, &[], &t, &th);
}

// ── thinking ───────────────────────────────────────────────────────────────

#[test]
fn view_thinking_collapsed() {
    let t = tokens();
    let th = theme();
    let md = Content::parse("inner thoughts");
    let _ = view_thinking_msg(0, &thinking_msg(), &md, false, false, &t, &th);
}

#[test]
fn view_thinking_expanded() {
    let t = tokens();
    let th = theme();
    let md = Content::parse("inner thoughts");
    let _ = view_thinking_msg(0, &thinking_msg(), &md, true, false, &t, &th);
}

#[test]
fn view_thinking_streaming() {
    let t = tokens();
    let th = theme();
    let md = Content::parse("...");
    let _ = view_thinking_msg(0, &thinking_msg(), &md, false, true, &t, &th);
}

#[test]
fn view_thinking_with_duration() {
    let t = tokens();
    let th = theme();
    let mut msg = thinking_msg();
    msg.duration_ms = Some(4200);
    let md = Content::parse("thoughts");
    let _ = view_thinking_msg(0, &msg, &md, false, false, &t, &th);
}

#[test]
fn view_thinking_with_subsecond_duration() {
    let t = tokens();
    let th = theme();
    let mut msg = thinking_msg();
    msg.duration_ms = Some(350);
    let md = Content::parse("thoughts");
    let _ = view_thinking_msg(0, &msg, &md, false, false, &t, &th);
}

// ── tool call ──────────────────────────────────────────────────────────────

#[test]
fn view_tool_call_running() {
    let t = tokens();
    let th = theme();
    let _ = view_tool_call_msg(&tool_call_msg(), ToolStatus::Running, &t, &th);
}

#[test]
fn view_tool_call_done() {
    let t = tokens();
    let th = theme();
    let _ = view_tool_call_msg(&tool_call_msg(), ToolStatus::Done, &t, &th);
}

#[test]
fn view_tool_call_error() {
    let t = tokens();
    let th = theme();
    let _ = view_tool_call_msg(&tool_call_msg(), ToolStatus::Error, &t, &th);
}

#[test]
fn view_tool_call_long_args_ignored() {
    let t = tokens();
    let th = theme();
    let long_args = ChatMessage::new_tool_call("bash".into(), "a".repeat(200));
    let _ = view_tool_call_msg(&long_args, ToolStatus::Done, &t, &th);
}

#[test]
fn view_tool_call_edit_delta() {
    let t = tokens();
    let th = theme();
    let _ = view_tool_call_msg(&edit_tool_call_msg(), ToolStatus::Done, &t, &th);
}

#[test]
fn edit_tool_summary_counts_changed_lines() {
    assert_eq!(
        edit_tool_summary(&edit_tool_call_msg()),
        Some(EditToolSummary {
            file_name: "main.rs".into(),
            additions: 2,
            deletions: 1,
        })
    );
}

#[test]
fn edit_tool_summary_uses_title_path_when_missing_file_path() {
    let args = serde_json::json!({
        "old_string": "old",
        "new_string": "new",
        "replace_all": false,
    });
    let msg = ChatMessage::new_tool_call("Edit src/lib.rs".into(), args.to_string());

    assert_eq!(
        edit_tool_summary(&msg),
        Some(EditToolSummary {
            file_name: "lib.rs".into(),
            additions: 1,
            deletions: 1,
        })
    );
}

#[test]
fn edit_tool_summary_ignores_non_edit_args() {
    assert_eq!(edit_tool_summary(&tool_call_msg()), None);
}

#[test]
fn tool_activity_item_summarizes_read_tool_by_file_name() {
    assert_eq!(
        tool_activity_item(
            &ChatMessage::new_tool_call(
                "Read /Users/tcztzy/project/src/main.rs".into(),
                serde_json::json!({
                    "file_path": "/Users/tcztzy/project/src/main.rs",
                })
                .to_string(),
            ),
            ToolStatus::Done,
        ),
        Some(ToolActivityItem::new(
            ToolActivityKind::Read,
            "Read main.rs".into()
        ))
    );
}

#[test]
fn tool_activity_item_treats_cat_as_read() {
    assert_eq!(
        tool_activity_item(
            &bash_tool_call_msg(
                "cat /Users/tcztzy/.claude/settings.json 2>/dev/null || echo missing",
                None,
            ),
            ToolStatus::Done,
        ),
        Some(ToolActivityItem::new(
            ToolActivityKind::Read,
            "Read settings.json".into()
        ))
    );
}

#[test]
fn tool_activity_item_summarizes_list_command() {
    assert_eq!(
        tool_activity_item(
            &bash_tool_call_msg(
                "ls -la /Users/tcztzy/swarmx/crates/swarmx-ui",
                Some("List UI crate"),
            ),
            ToolStatus::Done,
        ),
        Some(ToolActivityItem::new(
            ToolActivityKind::List,
            "Listed UI crate".into()
        ))
    );
}

#[test]
fn tool_activity_item_summarizes_edit_delta() {
    assert_eq!(
        tool_activity_item(&edit_tool_call_msg(), ToolStatus::Done),
        Some(ToolActivityItem {
            kind: ToolActivityKind::Edit,
            label: "Edited main.rs".into(),
            additions: Some(2),
            deletions: Some(1),
        })
    );
}

#[test]
fn tool_activity_summary_counts_exploration_and_commands() {
    let items = vec![
        ToolActivityItem::new(ToolActivityKind::Read, "Read a.rs".into()),
        ToolActivityItem::new(ToolActivityKind::Read, "Read b.rs".into()),
        ToolActivityItem::new(ToolActivityKind::List, "Listed files".into()),
        ToolActivityItem::new(ToolActivityKind::Hook, "PreToolUse hook".into()),
        ToolActivityItem::new(ToolActivityKind::Command, "Ran cargo test".into()),
    ];

    assert_eq!(
        tool_activity_summary(&items),
        "Explored 2 files, 1 list, ran 1 hook, ran 1 command"
    );
}

#[test]
fn view_tool_activity_group_does_not_panic() {
    let t = tokens();
    let th = theme();
    let items = vec![
        tool_activity_item(&edit_tool_call_msg(), ToolStatus::Done).unwrap(),
        ToolActivityItem::new(ToolActivityKind::Hook, "PostToolUse hook".into()),
    ];
    let _ = view_tool_activity_group(items, ToolStatus::Done, &t, &th);
}

// ── tool result ────────────────────────────────────────────────────────────

#[test]
fn view_tool_result_collapsed() {
    let t = tokens();
    let th = theme();
    let _ = view_tool_result_msg(0, &tool_result_msg(), false, &t, &th);
}

#[test]
fn view_tool_result_expanded() {
    let t = tokens();
    let th = theme();
    let _ = view_tool_result_msg(0, &tool_result_msg(), true, &t, &th);
}

#[test]
fn view_tool_result_long_content() {
    let t = tokens();
    let th = theme();
    let msg = ChatMessage::new_tool_result("bash".into(), "a".repeat(500));
    let _ = view_tool_result_msg(0, &msg, false, &t, &th);
}
