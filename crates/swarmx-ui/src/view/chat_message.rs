use iced::widget::{Space, column, container, row, text};
use iced::{Alignment, Element, Length};
use iced_shadcn::{
    BadgeProps, BadgeSize, BadgeVariant, ButtonProps, ButtonSize, ButtonVariant,
    CollapsibleContentProps, CollapsibleProps, Theme, TooltipProps, badge, collapsible,
    icon_button, spinner, tooltip,
};
use lucide_icons::Icon as LucideIcon;
use serde_json::Value;
use std::path::Path;

use crate::app::Message;
use crate::tokens::DesignTokens;

// ── Data Model ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum MessageKind {
    #[default]
    Message,
    Thinking,
    ToolCall,
    ToolResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolStatus {
    Running,
    Done,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolActivityKind {
    Read,
    List,
    Hook,
    Edit,
    Command,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolActivityItem {
    pub kind: ToolActivityKind,
    pub label: String,
    pub additions: Option<usize>,
    pub deletions: Option<usize>,
}

impl ToolActivityItem {
    fn new(kind: ToolActivityKind, label: String) -> Self {
        Self {
            kind,
            label,
            additions: None,
            deletions: None,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub is_user: bool,
    pub content: String,
    #[serde(default)]
    pub kind: MessageKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_result: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

impl ChatMessage {
    pub fn new(is_user: bool, content: String) -> Self {
        Self {
            is_user,
            content,
            kind: MessageKind::Message,
            tool_name: None,
            tool_result: None,
            duration_ms: None,
        }
    }

    pub fn new_user(content: String) -> Self {
        Self {
            is_user: true,
            content,
            kind: MessageKind::Message,
            tool_name: None,
            tool_result: None,
            duration_ms: None,
        }
    }

    pub fn new_assistant(content: String) -> Self {
        Self {
            is_user: false,
            content,
            kind: MessageKind::Message,
            tool_name: None,
            tool_result: None,
            duration_ms: None,
        }
    }

    pub fn new_thinking(content: String) -> Self {
        Self {
            is_user: false,
            content,
            kind: MessageKind::Thinking,
            tool_name: None,
            tool_result: None,
            duration_ms: None,
        }
    }

    pub fn new_tool_call(name: String, args: String) -> Self {
        Self {
            is_user: false,
            content: args,
            kind: MessageKind::ToolCall,
            tool_name: Some(name),
            tool_result: None,
            duration_ms: None,
        }
    }

    pub fn new_tool_result(name: String, result: String) -> Self {
        Self {
            is_user: false,
            content: result.clone(),
            kind: MessageKind::ToolResult,
            tool_name: Some(name),
            tool_result: Some(result),
            duration_ms: None,
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

pub fn lucide_icon(icon: LucideIcon, size: f32) -> iced::widget::Text<'static> {
    text(char::from(icon).to_string())
        .font(iced::Font::with_name("lucide"))
        .size(size)
}

pub fn copy_button<'a>(
    content: String,
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, Message> {
    let btn = icon_button(
        lucide_icon(LucideIcon::Copy, 14.0),
        Some(Message::CopyToClipboard(content)),
        ButtonProps::new()
            .variant(ButtonVariant::Ghost)
            .size(ButtonSize::Size1),
        t,
    );
    tooltip(
        btn,
        text("Copy").size(tokens.text_sm.size),
        TooltipProps::new(),
        t,
    )
    .into()
}

pub fn chevron_icon(open: bool, size: f32) -> iced::widget::Text<'static> {
    let icon = if open {
        LucideIcon::ChevronDown
    } else {
        LucideIcon::ChevronRight
    };
    lucide_icon(icon, size)
}

// ── Markdown ─────────────────────────────────────────────────────────────────

pub fn markdown_view<'a>(md: &'a iced::widget::markdown::Content) -> Element<'a, Message> {
    let md_settings = iced::widget::markdown::Settings::with_text_size(
        14.0,
        iced::widget::markdown::Style::from_palette(iced::theme::Palette::DARK),
    );
    iced::widget::markdown::view(md.items(), md_settings)
        .map(|uri| Message::LinkClicked(uri.to_string()))
}

// ── Per-Message Renderers ────────────────────────────────────────────────────

/// Full-width tinted card. No right-alignment, no copy button.
pub fn view_user_msg<'a>(
    msg: &'a ChatMessage,
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, Message> {
    container(
        text(&msg.content)
            .size(tokens.text_base.size)
            .color(t.palette.foreground),
    )
    .width(Length::Fill)
    .padding([tokens.space_3 as u16, tokens.space_4 as u16])
    .style(move |_theme| {
        iced::widget::container::Style::default()
            .background(iced::Color {
                a: 0.06,
                ..t.palette.foreground
            })
            .border(iced::border::rounded(tokens.radius_md))
    })
    .into()
}

/// Assistant text. When `with_agent_icon` is true, renders a 24px gutter with agent icon.
pub fn view_assistant_msg<'a>(
    msg: &'a ChatMessage,
    md: &'a iced::widget::markdown::Content,
    with_agent_icon: bool,
    agent_icon_svg: &[u8],
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, Message> {
    let md_elem = markdown_view(md);

    let copy_row = row![
        copy_button(msg.content.clone(), tokens, t),
        Space::new().width(Length::Fill),
    ]
    .width(Length::Fill);

    if with_agent_icon && !agent_icon_svg.is_empty() {
        let icon_handle = iced::widget::svg::Handle::from_memory(agent_icon_svg.to_vec());
        let icon_widget = iced::widget::svg::Svg::new(icon_handle)
            .width(Length::Fixed(16.0))
            .height(Length::Fixed(16.0));

        let gutter = container(icon_widget)
            .width(Length::Fixed(24.0))
            .align_x(Alignment::Center);

        let body = column![md_elem, Space::new().height(4.0), copy_row]
            .spacing(0)
            .width(Length::Fill);

        row![gutter, body]
            .spacing(tokens.space_2)
            .width(Length::Fill)
            .into()
    } else {
        column![md_elem, Space::new().height(4.0), copy_row]
            .spacing(0)
            .width(Length::Fill)
            .into()
    }
}

/// Thinking: inline single-row `▸ Thinking · 4s` or `▸ Thinking · ...` when streaming.
/// Default collapsed except when streaming or explicitly open.
pub fn view_thinking_msg<'a>(
    idx: usize,
    msg: &'a ChatMessage,
    md: &'a iced::widget::markdown::Content,
    is_open: bool,
    is_streaming: bool,
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, Message> {
    if is_streaming || is_open {
        let duration_text = if is_streaming {
            "...".to_string()
        } else if let Some(ms) = msg.duration_ms {
            if ms >= 1000 {
                format!("{}s", ms / 1000)
            } else {
                format!("{}ms", ms)
            }
        } else {
            String::new()
        };

        let label = if duration_text.is_empty() {
            "Thinking".to_string()
        } else {
            format!("Thinking · {}", duration_text)
        };

        let spinner_icon = if is_streaming {
            let sp = spinner(iced_shadcn::Spinner::new(t).size(iced_shadcn::SpinnerSize::Size1));
            Element::from(sp)
        } else {
            lucide_icon(LucideIcon::Brain, 14.0)
                .color(t.palette.muted_foreground)
                .into()
        };

        let trigger = row![
            chevron_icon(is_open, 12.0).color(t.palette.muted_foreground),
            Space::new().width(6.0),
            spinner_icon,
            Space::new().width(6.0),
            text(label)
                .size(tokens.text_sm.size)
                .color(t.palette.muted_foreground),
            Space::new().width(Length::Fill),
        ]
        .align_y(Alignment::Center);

        let content = container(markdown_view(md))
            .padding(iced::Padding {
                top: tokens.space_1,
                right: 0.0,
                bottom: tokens.space_1,
                left: 24.0,
            })
            .width(Length::Fill);

        column![collapsible(
            true,
            trigger,
            content,
            Some(move |new_open| Message::SetThinkingOpen(idx, new_open)),
            CollapsibleContentProps::new(),
            CollapsibleProps::new().compact(true),
            t,
        ),]
        .spacing(0)
        .width(Length::Fill)
        .into()
    } else {
        let duration_text = msg
            .duration_ms
            .map(|ms| {
                if ms >= 1000 {
                    format!("{}s", ms / 1000)
                } else {
                    format!("{}ms", ms)
                }
            })
            .unwrap_or_default();

        let label = if duration_text.is_empty() {
            "Thinking".to_string()
        } else {
            format!("Thinking · {}", duration_text)
        };

        let trigger = row![
            chevron_icon(false, 12.0).color(t.palette.muted_foreground),
            Space::new().width(6.0),
            lucide_icon(LucideIcon::Brain, 14.0).color(t.palette.muted_foreground),
            Space::new().width(6.0),
            text(label)
                .size(tokens.text_sm.size)
                .color(t.palette.muted_foreground),
            Space::new().width(Length::Fill),
        ]
        .align_y(Alignment::Center);

        iced::widget::button(trigger)
            .on_press(Message::SetThinkingOpen(idx, true))
            .style(move |_theme, status| {
                let base = iced::widget::button::Style::default();
                match status {
                    iced::widget::button::Status::Hovered => iced::widget::button::Style {
                        background: Some(iced::Background::Color(iced::Color {
                            a: 0.04,
                            ..t.palette.foreground
                        })),
                        ..base
                    },
                    _ => base,
                }
            })
            .width(Length::Fill)
            .into()
    }
}

fn view_status_badge<'a>(
    status: ToolStatus,
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, Message> {
    match status {
        ToolStatus::Running => {
            let sp = spinner(iced_shadcn::Spinner::new(t).size(iced_shadcn::SpinnerSize::Size1));
            row![
                sp,
                Space::new().width(4.0),
                text("running")
                    .size(tokens.text_xs.size)
                    .color(t.palette.muted_foreground),
            ]
            .align_y(Alignment::Center)
            .into()
        }
        ToolStatus::Done => badge(
            "done",
            BadgeProps::new()
                .variant(BadgeVariant::Default)
                .size(BadgeSize::Size1),
            t,
        ),
        ToolStatus::Error => badge(
            "error",
            BadgeProps::new()
                .variant(BadgeVariant::Destructive)
                .size(BadgeSize::Size1),
            t,
        ),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EditToolSummary {
    file_name: String,
    additions: usize,
    deletions: usize,
}

fn edit_tool_summary(msg: &ChatMessage) -> Option<EditToolSummary> {
    let raw = serde_json::from_str::<Value>(&msg.content).ok()?;
    let old = raw.get("old_string")?.as_str()?;
    let new = raw.get("new_string")?.as_str()?;
    let file_path = raw
        .get("file_path")
        .or_else(|| raw.get("path"))
        .and_then(Value::as_str)
        .or_else(|| {
            msg.tool_name
                .as_deref()
                .and_then(|name| name.strip_prefix("Edit "))
                .map(str::trim)
                .filter(|path| !path.is_empty())
        })?;

    let (additions, deletions) = changed_line_counts(old, new);
    Some(EditToolSummary {
        file_name: file_name_for_display(file_path),
        additions,
        deletions,
    })
}

fn file_name_for_display(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or(path)
        .to_string()
}

fn changed_line_counts(old: &str, new: &str) -> (usize, usize) {
    let old_lines: Vec<_> = old.lines().collect();
    let new_lines: Vec<_> = new.lines().collect();
    if old_lines.is_empty() || new_lines.is_empty() {
        return (new_lines.len(), old_lines.len());
    }

    if old_lines.len().saturating_mul(new_lines.len()) > 40_000 {
        return (new_lines.len(), old_lines.len());
    }

    let unchanged = lcs_line_count(&old_lines, &new_lines);
    (new_lines.len() - unchanged, old_lines.len() - unchanged)
}

fn lcs_line_count(left: &[&str], right: &[&str]) -> usize {
    let (outer, inner) = if left.len() > right.len() {
        (left, right)
    } else {
        (right, left)
    };
    let mut previous = vec![0; inner.len() + 1];
    let mut current = vec![0; inner.len() + 1];

    for outer_line in outer {
        for (index, inner_line) in inner.iter().enumerate() {
            current[index + 1] = if outer_line == inner_line {
                previous[index] + 1
            } else {
                previous[index + 1].max(current[index])
            };
        }
        std::mem::swap(&mut previous, &mut current);
        current.fill(0);
    }

    previous[inner.len()]
}

fn tool_input(msg: &ChatMessage) -> Option<Value> {
    serde_json::from_str::<Value>(&msg.content).ok()
}

fn input_string<'a>(input: &'a Value, key: &str) -> Option<&'a str> {
    input.get(key).and_then(Value::as_str)
}

fn trimmed_tool_title(msg: &ChatMessage) -> &str {
    msg.tool_name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("tool")
}

pub fn is_tool_activity_message(msg: &ChatMessage) -> bool {
    matches!(msg.kind, MessageKind::ToolCall | MessageKind::ToolResult)
}

pub fn tool_activity_item(msg: &ChatMessage, status: ToolStatus) -> Option<ToolActivityItem> {
    match msg.kind {
        MessageKind::ToolCall => Some(tool_call_activity_item(msg, status)),
        MessageKind::ToolResult => Some(ToolActivityItem::new(
            ToolActivityKind::Other,
            "Tool output".into(),
        )),
        _ => None,
    }
}

fn tool_call_activity_item(msg: &ChatMessage, status: ToolStatus) -> ToolActivityItem {
    if let Some(summary) = edit_tool_summary(msg) {
        let verb = match status {
            ToolStatus::Running => "Editing",
            ToolStatus::Done => "Edited",
            ToolStatus::Error => "Edit",
        };
        return ToolActivityItem {
            kind: ToolActivityKind::Edit,
            label: format!("{verb} {}", summary.file_name),
            additions: Some(summary.additions),
            deletions: Some(summary.deletions),
        };
    }

    let title = trimmed_tool_title(msg);
    if let Some(path) = read_tool_path(msg, title) {
        return ToolActivityItem::new(
            ToolActivityKind::Read,
            format!("Read {}", file_name_for_display(&path)),
        );
    }

    if is_hook_title(title) {
        return ToolActivityItem::new(ToolActivityKind::Hook, title.to_string());
    }

    if let Some(input) = tool_input(msg) {
        if let Some(command) = input_string(&input, "command") {
            if let Some(path) = shell_read_path(command) {
                return ToolActivityItem::new(
                    ToolActivityKind::Read,
                    format!("Read {}", file_name_for_display(path)),
                );
            }

            if is_list_command(command) {
                return ToolActivityItem::new(
                    ToolActivityKind::List,
                    list_command_label(command, input_string(&input, "description")),
                );
            }

            return ToolActivityItem::new(
                ToolActivityKind::Command,
                format!("Ran {}", truncate_middle(command, 120)),
            );
        }
    }

    ToolActivityItem::new(ToolActivityKind::Other, truncate_middle(title, 120))
}

fn read_tool_path(msg: &ChatMessage, title: &str) -> Option<String> {
    if let Some(path) = title.strip_prefix("Read ").map(str::trim) {
        if !path.is_empty() {
            return Some(path.to_string());
        }
    }

    tool_input(msg).and_then(|input| {
        input_string(&input, "file_path")
            .or_else(|| input_string(&input, "path"))
            .filter(|path| !path.is_empty())
            .map(str::to_string)
    })
}

fn is_hook_title(title: &str) -> bool {
    let lower = title.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "pretooluse hook"
            | "posttooluse hook"
            | "sessionstart hook"
            | "sessionend hook"
            | "userpromptsubmit hook"
            | "stop hook"
    ) || lower.ends_with(" hook")
}

fn shell_read_path(command: &str) -> Option<&str> {
    let command = command.trim_start();
    if !command.starts_with("cat ") {
        return None;
    }

    command
        .split_whitespace()
        .skip(1)
        .find(|part| {
            !part.starts_with('-')
                && !part.starts_with('>')
                && !part.starts_with('<')
                && *part != "||"
                && *part != "&&"
                && *part != "|"
        })
        .map(|part| part.trim_matches(['"', '\'']))
        .filter(|part| !part.is_empty())
}

fn is_list_command(command: &str) -> bool {
    let command = command.trim_start();
    command.starts_with("ls ")
        || command == "ls"
        || command.starts_with("find ")
        || command.starts_with("rg --files")
}

fn list_command_label(command: &str, description: Option<&str>) -> String {
    if let Some(description) = description {
        let trimmed = description.trim();
        if !trimmed.is_empty() {
            return trimmed
                .strip_prefix("List ")
                .map(|rest| format!("Listed {}", rest))
                .unwrap_or_else(|| truncate_middle(trimmed, 120));
        }
    }

    if let Some(path) = list_command_path(command) {
        return format!("Listed files in {}", file_name_for_display(path));
    }

    "Listed files".into()
}

fn list_command_path(command: &str) -> Option<&str> {
    command
        .split_whitespace()
        .skip(1)
        .find(|part| {
            !part.starts_with('-')
                && !part.contains('=')
                && *part != "&&"
                && *part != "||"
                && *part != "|"
        })
        .map(|part| part.trim_matches(['"', '\'']))
        .filter(|part| !part.is_empty())
}

fn truncate_middle(value: &str, limit: usize) -> String {
    let count = value.chars().count();
    if count <= limit {
        return value.to_string();
    }

    let keep = limit.saturating_sub(3);
    let head = keep / 2;
    let tail = keep - head;
    let prefix: String = value.chars().take(head).collect();
    let suffix: String = value
        .chars()
        .rev()
        .take(tail)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{prefix}...{suffix}")
}

pub fn tool_activity_summary(items: &[ToolActivityItem]) -> String {
    let mut reads = 0;
    let mut lists = 0;
    let mut hooks = 0;
    let mut edits = 0;
    let mut commands = 0;
    let mut others = 0;

    for item in items {
        match item.kind {
            ToolActivityKind::Read => reads += 1,
            ToolActivityKind::List => lists += 1,
            ToolActivityKind::Hook => hooks += 1,
            ToolActivityKind::Edit => edits += 1,
            ToolActivityKind::Command => commands += 1,
            ToolActivityKind::Other => others += 1,
        }
    }

    let mut parts = Vec::new();
    if edits > 0 {
        parts.push(format!("Edited {} {}", edits, plural("file", edits)));
    }
    if reads > 0 || lists > 0 {
        let mut explored = Vec::new();
        if reads > 0 {
            explored.push(format!("{} {}", reads, plural("file", reads)));
        }
        if lists > 0 {
            explored.push(format!("{} {}", lists, plural("list", lists)));
        }
        parts.push(format!("Explored {}", explored.join(", ")));
    }
    if hooks > 0 {
        parts.push(format!("ran {} {}", hooks, plural("hook", hooks)));
    }
    if commands > 0 {
        parts.push(format!("ran {} {}", commands, plural("command", commands)));
    }
    if others > 0 && parts.is_empty() {
        parts.push(format!("used {} {}", others, plural("tool", others)));
    }

    if parts.is_empty() {
        "Used tools".into()
    } else {
        sentence_case(&parts.join(", "))
    }
}

fn plural(word: &str, count: usize) -> &str {
    if count == 1 {
        word
    } else {
        match word {
            "list" => "lists",
            "hook" => "hooks",
            "command" => "commands",
            "tool" => "tools",
            _ => "files",
        }
    }
}

fn sentence_case(value: &str) -> String {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    format!("{}{}", first.to_ascii_uppercase(), chars.as_str())
}

pub fn view_tool_activity_group<'a>(
    items: Vec<ToolActivityItem>,
    status: ToolStatus,
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, Message> {
    let summary = tool_activity_summary(&items);
    let trigger = row![
        chevron_icon(true, 12.0).color(t.palette.muted_foreground),
        Space::new().width(6.0),
        lucide_icon(LucideIcon::FolderSearch, 14.0).color(t.palette.muted_foreground),
        Space::new().width(8.0),
        text(summary)
            .size(tokens.text_sm.size)
            .color(t.palette.muted_foreground),
        Space::new().width(8.0),
        view_status_badge(status, tokens, t),
        Space::new().width(Length::Fill),
    ]
    .align_y(Alignment::Center);

    let detail_rows: Vec<Element<'a, Message>> = items
        .into_iter()
        .filter(|item| item.kind != ToolActivityKind::Other)
        .map(|item| view_tool_activity_item(item, tokens, t))
        .collect();

    let details = container(column(detail_rows).spacing(tokens.space_2))
        .padding(iced::Padding {
            top: tokens.space_2,
            right: 0.0,
            bottom: tokens.space_1,
            left: 24.0,
        })
        .width(Length::Fill);

    collapsible(
        true,
        trigger,
        details,
        None::<fn(bool) -> Message>,
        CollapsibleContentProps::new(),
        CollapsibleProps::new().compact(true),
        t,
    )
}

fn view_tool_activity_item<'a>(
    item: ToolActivityItem,
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, Message> {
    let icon = match item.kind {
        ToolActivityKind::Read => LucideIcon::FileSearch,
        ToolActivityKind::List => LucideIcon::List,
        ToolActivityKind::Hook => LucideIcon::Wrench,
        ToolActivityKind::Edit => LucideIcon::FilePenLine,
        ToolActivityKind::Command => LucideIcon::SquareTerminal,
        ToolActivityKind::Other => LucideIcon::FileText,
    };

    let base = row![
        lucide_icon(icon, 12.0).color(t.palette.muted_foreground),
        Space::new().width(8.0),
        text(item.label)
            .size(tokens.text_sm.size)
            .color(t.palette.muted_foreground),
    ]
    .align_y(Alignment::Center);

    if let (Some(additions), Some(deletions)) = (item.additions, item.deletions) {
        let edit_green = iced::Color::from_rgb(0.48, 0.78, 0.23);
        base.push(Space::new().width(6.0))
            .push(
                text(format!("+{additions}"))
                    .size(tokens.text_sm.size)
                    .color(edit_green),
            )
            .push(Space::new().width(4.0))
            .push(
                text(format!("-{deletions}"))
                    .size(tokens.text_sm.size)
                    .color(t.palette.destructive),
            )
            .into()
    } else {
        base.into()
    }
}

/// Tool call: compact non-interactive label. Tool args are internal noise, except
/// edit calls where the UI can show a small line delta summary.
pub fn view_tool_call_msg<'a>(
    msg: &'a ChatMessage,
    status: ToolStatus,
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, Message> {
    let tool_name = msg.tool_name.as_deref().unwrap_or("tool");
    let edit_summary = edit_tool_summary(msg);
    let edit_green = iced::Color::from_rgb(0.48, 0.78, 0.23);

    let label: Element<'_, Message> = if let Some(summary) = edit_summary {
        let verb = match status {
            ToolStatus::Running => "Editing",
            ToolStatus::Done => "Edited",
            ToolStatus::Error => "Edit",
        };
        row![
            text(format!("{verb} {}", summary.file_name))
                .size(tokens.text_xs.size)
                .color(t.palette.muted_foreground),
            Space::new().width(6.0),
            text(format!("+{}", summary.additions))
                .size(tokens.text_xs.size)
                .color(edit_green),
            Space::new().width(4.0),
            text(format!("-{}", summary.deletions))
                .size(tokens.text_xs.size)
                .color(t.palette.destructive),
        ]
        .align_y(Alignment::Center)
        .into()
    } else {
        text(tool_name)
            .size(tokens.text_xs.size)
            .color(t.palette.muted_foreground)
            .into()
    };

    container(
        row![
            lucide_icon(LucideIcon::Wrench, 12.0).color(t.palette.muted_foreground),
            Space::new().width(6.0),
            label,
            Space::new().width(8.0),
            view_status_badge(status, tokens, t),
            Space::new().width(Length::Fill),
        ]
        .align_y(Alignment::Center),
    )
    .padding([2, 0])
    .width(Length::Fill)
    .into()
}

/// Tool result: collapsible monospace code block, default collapsed.
pub fn view_tool_result_msg<'a>(
    idx: usize,
    msg: &'a ChatMessage,
    is_open: bool,
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, Message> {
    let tool_name = msg.tool_name.as_deref().unwrap_or("tool");

    let trigger = row![
        chevron_icon(is_open, 12.0).color(t.palette.muted_foreground),
        Space::new().width(6.0),
        lucide_icon(LucideIcon::FileText, 12.0).color(t.palette.muted_foreground),
        Space::new().width(6.0),
        text(format!("Output: {tool_name}"))
            .size(tokens.text_xs.size)
            .color(t.palette.muted_foreground),
        Space::new().width(Length::Fill),
    ]
    .align_y(Alignment::Center);

    if is_open {
        let display = if msg.content.len() > 2000 {
            format!(
                "{}\n\n[truncated, {} bytes total]",
                &msg.content[..2000],
                msg.content.len()
            )
        } else {
            msg.content.clone()
        };

        let body = container(
            text(display)
                .font(iced::Font::MONOSPACE)
                .size(12.0)
                .color(t.palette.foreground),
        )
        .style(move |_theme| {
            iced::widget::container::Style::default()
                .background(iced::Color {
                    a: 0.05,
                    ..t.palette.muted_foreground
                })
                .border(iced::border::rounded(tokens.radius_sm))
        })
        .padding([6, 10]);

        collapsible(
            true,
            trigger,
            body,
            Some(move |new_open| Message::SetToolOpen(idx, new_open)),
            CollapsibleContentProps::new(),
            CollapsibleProps::new().compact(true),
            t,
        )
    } else {
        iced::widget::button(trigger)
            .on_press(Message::SetToolOpen(idx, true))
            .style(move |_theme, status| {
                let base = iced::widget::button::Style::default();
                match status {
                    iced::widget::button::Status::Hovered => iced::widget::button::Style {
                        background: Some(iced::Background::Color(iced::Color {
                            a: 0.04,
                            ..t.palette.foreground
                        })),
                        ..base
                    },
                    _ => base,
                }
            })
            .width(Length::Fill)
            .into()
    }
}

#[cfg(test)]
mod tests;
