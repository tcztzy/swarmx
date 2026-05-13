use std::collections::HashSet;

use iced::widget::{Space, column, container, row, svg, text};
use iced::{Alignment, Element, Length};
use iced_shadcn::{
    ButtonProps, ButtonSize, ButtonVariant, ScrollAreaProps, ScrollAreaScrollbars, ScrollAreaSize,
    Spinner, SpinnerSize, Theme, button, scroll_area, spinner,
};

use crate::instance::{AgentInstance, ModelProvider};
use crate::tokens::DesignTokens;
use lucide_icons::Icon as LucideIcon;

use crate::view::chat_message::{
    ChatMessage, MessageKind, ToolStatus, is_tool_activity_message, lucide_icon,
    tool_activity_item, view_assistant_msg, view_thinking_msg, view_tool_activity_group,
    view_user_msg,
};
use crate::view::composer;

// ── Message Sub-Enum ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Message {
    UseSuggestion(String),
    LinkClicked(String),
    CopyToClipboard(String),
    SetThinkingOpen(usize, bool),
    SetToolOpen(usize, bool),
}

// ── Public Entry ─────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn view<'a>(
    instance: Option<&'a AgentInstance>,
    provider: Option<&'a ModelProvider>,
    instance_idx: Option<usize>,
    cwd: Option<&'a str>,
    messages: &'a [ChatMessage],
    md_contents: &'a [iced::widget::markdown::Content],
    input_text: &'a str,
    loading: bool,
    error: Option<&'a str>,
    thinking_expanded: &'a HashSet<usize>,
    _tool_expanded: &'a HashSet<usize>,
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, crate::app::Message> {
    let header = if let (Some(inst), Some(prov), Some(cwd)) = (instance, provider, cwd) {
        view_conversation_header(inst, prov, cwd, tokens, t)
    } else {
        Space::new().width(0).height(0).into()
    };

    let messages_view = if messages.is_empty() && !loading && error.is_none() {
        view_empty_chat(tokens, t)
    } else {
        view_message_list(
            messages,
            MessageListContext {
                md_contents,
                loading,
                error,
                instance,
                thinking_expanded,
                tokens,
                t,
            },
        )
    };

    let available = provider
        .map(|p| p.available_models.as_slice())
        .unwrap_or(&[]);
    let current = instance.map(|i| i.model.as_str()).unwrap_or("");
    let idx = instance_idx.unwrap_or(0);

    let input_row = composer::view(input_text, loading, available, current, idx, cwd, tokens, t);

    column![header, messages_view, input_row]
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

// ── Empty Chat State ─────────────────────────────────────────────────────────

use iced_shadcn::{ButtonProps as Bp, ButtonSize as Bs, ButtonVariant as Bv, button as btn};

fn view_empty_chat<'a>(tokens: &'a DesignTokens, t: &'a Theme) -> Element<'a, crate::app::Message> {
    let suggestions = [
        "What is SwarmX?",
        "Write a Rust hello world",
        "Explain quantum computing",
        "Help me debug my code",
    ];
    let chips: Vec<Element<_>> = suggestions
        .iter()
        .map(|label| {
            let prompt = label.to_string();
            btn(
                *label,
                Some(crate::app::Message::UseSuggestion(prompt)),
                Bp::new().variant(Bv::Outline).size(Bs::Size1),
                t,
            )
            .into()
        })
        .collect();

    container(
        column![
            Space::new().height(Length::Fill),
            text("Start a conversation")
                .size(tokens.text_2xl.size)
                .color(t.palette.foreground),
            Space::new().height(tokens.space_1),
            text("Pick an agent from the sidebar, then type a message.")
                .size(tokens.text_sm.size)
                .color(t.palette.muted_foreground),
            Space::new().height(tokens.space_5),
            row(chips).spacing(tokens.space_2).wrap(),
            Space::new().height(Length::Fill),
        ]
        .align_x(Alignment::Center),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .center_x(Length::Fill)
    .center_y(Length::Fill)
    .into()
}

// ── Conversation Header ──────────────────────────────────────────────────────

fn view_conversation_header<'a>(
    instance: &'a AgentInstance,
    _provider: &'a ModelProvider,
    cwd: &'a str,
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, crate::app::Message> {
    let dark = crate::theme::is_dark_theme(t);
    let icon_bytes = instance.harness.icon().svg_bytes(dark).into_owned();
    let icon_handle = svg::Handle::from_memory(icon_bytes);
    let icon_widget = svg::Svg::new(icon_handle)
        .width(Length::Fixed(18.0))
        .height(Length::Fixed(18.0));

    let model_name = instance.model.as_str();

    let top_row = row![
        icon_widget,
        Space::new().width(tokens.space_2),
        text(&instance.label)
            .size(tokens.text_sm.size)
            .color(t.palette.foreground),
        Space::new().width(tokens.space_2),
        text("·")
            .size(tokens.text_sm.size)
            .color(t.palette.muted_foreground),
        Space::new().width(tokens.space_2),
        text(model_name)
            .size(tokens.text_sm.size)
            .color(t.palette.muted_foreground),
        Space::new().width(Length::Fill),
        button(
            "\u{22EF}",
            Some(crate::app::Message::ToggleSettings),
            ButtonProps::new()
                .variant(ButtonVariant::Ghost)
                .size(ButtonSize::Size1),
            t,
        ),
    ]
    .align_y(Alignment::Center);

    // Truncate cwd path for display
    let cwd_display = if cwd.len() > 60 {
        format!("...{}", &cwd[cwd.len() - 57..])
    } else {
        cwd.to_string()
    };

    let bottom_row = row![
        text(cwd_display)
            .size(tokens.text_xs.size)
            .color(t.palette.muted_foreground),
        Space::new().width(Length::Fill),
    ]
    .align_y(Alignment::Center);

    let accent = crate::theme::agent_color(instance.harness, t);

    container(
        column![
            top_row,
            Space::new().height(2.0),
            bottom_row,
            Space::new().height(2.0),
            container(Space::new().width(Length::Fill).height(2.0))
                .style(move |_theme| {
                    iced::widget::container::Style::default().background(accent)
                })
                .width(Length::Fill)
                .height(Length::Fixed(2.0)),
        ]
        .spacing(0),
    )
    .padding([tokens.space_3 as u16, tokens.space_4 as u16])
    .width(Length::Fill)
    .into()
}

// ── Message List ─────────────────────────────────────────────────────────────

struct MessageListContext<'a> {
    md_contents: &'a [iced::widget::markdown::Content],
    loading: bool,
    error: Option<&'a str>,
    instance: Option<&'a AgentInstance>,
    thinking_expanded: &'a HashSet<usize>,
    tokens: &'a DesignTokens,
    t: &'a Theme,
}

fn view_message_list<'a>(
    messages: &'a [ChatMessage],
    ctx: MessageListContext<'a>,
) -> Element<'a, crate::app::Message> {
    let MessageListContext {
        md_contents,
        loading,
        error,
        instance,
        thinking_expanded,
        tokens,
        t,
    } = ctx;
    let mut md_idx = 0;

    // Pre-compute agent icon once
    let dark = crate::theme::is_dark_theme(t);
    let agent_icon_svg = instance
        .map(|inst| inst.harness.icon().svg_bytes(dark).into_owned())
        .unwrap_or_default();

    // Find last thinking index for streaming determination
    let last_thinking_idx = messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, m)| m.kind == MessageKind::Thinking)
        .map(|(i, _)| i);

    // Build tool status map: tool call i is "Done" if followed by a ToolResult
    let mut tool_statuses: Vec<ToolStatus> = vec![ToolStatus::Done; messages.len()];
    for i in 0..messages.len() {
        if messages[i].kind == MessageKind::ToolCall {
            let has_result = messages[i + 1..]
                .iter()
                .take_while(|m| !m.is_user)
                .any(|m| m.kind == MessageKind::ToolResult);
            if has_result {
                tool_statuses[i] = ToolStatus::Done;
            } else if loading && i == messages.len() - 1 {
                tool_statuses[i] = ToolStatus::Running;
            } else {
                tool_statuses[i] = ToolStatus::Done;
            }
        }
    }

    let mut prev_was_user = true; // first assistant msg gets icon gutter

    let mut msgs: Vec<Element<_>> = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        let msg = &messages[i];
        let elem: Element<'_, crate::app::Message> = if msg.is_user {
            prev_was_user = true;
            i += 1;
            view_user_msg(msg, tokens, t)
        } else if is_tool_activity_message(msg) {
            let mut items = Vec::new();
            let mut status = ToolStatus::Done;

            while i < messages.len()
                && !messages[i].is_user
                && is_tool_activity_message(&messages[i])
            {
                md_idx += 1;
                let item_status = tool_statuses[i];
                if item_status == ToolStatus::Error {
                    status = ToolStatus::Error;
                } else if item_status == ToolStatus::Running && status != ToolStatus::Error {
                    status = ToolStatus::Running;
                }
                if let Some(item) = tool_activity_item(&messages[i], item_status) {
                    items.push(item);
                }
                i += 1;
            }

            view_tool_activity_group(items, status, tokens, t)
        } else {
            let result: Element<'_, crate::app::Message> = match msg.kind {
                MessageKind::Thinking => {
                    let md = &md_contents[md_idx];
                    md_idx += 1;
                    let is_open = thinking_expanded.contains(&i);
                    let is_streaming = Some(i) == last_thinking_idx && loading;
                    let idx = i;
                    i += 1;
                    view_thinking_msg(idx, msg, md, is_open, is_streaming, tokens, t)
                }
                _ => {
                    let md = &md_contents[md_idx];
                    md_idx += 1;
                    let with_icon = prev_was_user;
                    prev_was_user = false;
                    i += 1;
                    view_assistant_msg(msg, md, with_icon, &agent_icon_svg, tokens, t)
                }
            };
            result
        };

        // Full-width container, no special alignment
        msgs.push(container(elem).width(Length::Fill).into());
    }

    let mut col = column![
        scroll_area(
            column(msgs)
                .spacing(tokens.space_4)
                .padding([tokens.space_4 as u16, tokens.space_4 as u16]),
            ScrollAreaProps::new()
                .id(iced::widget::Id::new("chat-messages"))
                .scrollbars(ScrollAreaScrollbars::Vertical)
                .size(ScrollAreaSize::Size1),
            t,
        )
        .height(Length::Fill)
    ];

    if loading {
        let sp = spinner(Spinner::new(t).size(SpinnerSize::Size2));
        col = col.push(
            row![
                sp,
                Space::new().width(tokens.space_2),
                text("Thinking...")
                    .size(tokens.text_sm.size)
                    .color(t.palette.muted_foreground)
            ]
            .align_y(Alignment::Center),
        );
    }
    if let Some(e) = error {
        col = col.push(
            container(
                row![
                    lucide_icon(LucideIcon::TriangleAlert, 14.0).color(t.palette.destructive),
                    Space::new().width(tokens.space_2),
                    text(e)
                        .size(tokens.text_sm.size)
                        .color(t.palette.destructive),
                ]
                .align_y(Alignment::Center),
            )
            .style(move |_theme| {
                iced::widget::container::Style::default()
                    .background(iced::Color {
                        a: 0.08,
                        ..t.palette.destructive
                    })
                    .border(iced::border::rounded(tokens.radius_sm))
            })
            .padding([tokens.space_2 as u16, tokens.space_3 as u16]),
        );
    }

    col.width(Length::Fill).height(Length::Fill).into()
}
