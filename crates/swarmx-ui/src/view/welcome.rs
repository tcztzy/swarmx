//! Welcome gallery — agent instance cards + runtime dep cards + suggestions (spec §5).
//!
//! Replaces the old popup + env banner. Shown when `Surface::Welcome` is active.

use iced::widget::{Space, column, container, row, scrollable, svg, text};
use iced::{Alignment, Element, Length};
use iced_shadcn::{ButtonProps, ButtonSize, ButtonVariant, Theme, button};

use crate::app::Message;
use crate::environment::{AgentRuntime, AgentRuntimeStatus, DepStatus, RuntimeDep};
use crate::instance::AgentInstance;
use crate::instance::ModelProvider;
use crate::tokens::DesignTokens;

// ── Gallery ────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn gallery<'a>(
    instances: &'a [AgentInstance],
    providers: &'a [ModelProvider],
    env_checks: &'a [DepStatus],
    env_installing: Option<RuntimeDep>,
    agent_statuses: &'a [AgentRuntimeStatus],
    _home_dir: &'a str,
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, Message> {
    let heading = text("Start a new conversation")
        .size(tokens.text_2xl.size)
        .color(t.palette.foreground);
    let subtitle = text("Pick an agent. Configure once, reuse forever.")
        .size(tokens.text_sm.size)
        .color(t.palette.muted_foreground);

    // -- Ready instances --
    let ready: Vec<&AgentInstance> = instances
        .iter()
        .filter(|inst| instance_is_ready(inst, agent_statuses))
        .collect();
    let missing: Vec<&AgentInstance> = instances
        .iter()
        .filter(|inst| !instance_is_ready(inst, agent_statuses))
        .collect();

    // -- Missing runtime deps --
    let missing_deps: Vec<&DepStatus> = env_checks.iter().filter(|s| !s.installed).collect();

    let mut body = column![
        heading,
        Space::new().height(tokens.space_1),
        subtitle,
        Space::new().height(tokens.space_5),
    ]
    .spacing(0);

    // Ready section
    if !ready.is_empty() {
        let section = view_section_header("Ready", tokens, t);
        let cards: Vec<Element<_>> = ready
            .iter()
            .map(|inst| view_instance_card(inst, providers, tokens, t))
            .collect();
        body = body
            .push(section)
            .push(Space::new().height(tokens.space_2))
            .push(row(cards).spacing(tokens.space_3).wrap())
            .push(Space::new().height(tokens.space_5));
    }

    // Install to enable section (missing instances + missing deps)
    if !missing.is_empty() || !missing_deps.is_empty() {
        let section = view_section_header("Install to enable", tokens, t);
        body = body.push(section).push(Space::new().height(tokens.space_2));

        let mut install_cards: Vec<Element<_>> = Vec::new();

        for inst in &missing {
            install_cards.push(view_missing_instance_card(inst, agent_statuses, tokens, t));
        }
        for dep in &missing_deps {
            install_cards.push(view_dep_card(dep, env_installing, tokens, t));
        }

        body = body
            .push(row(install_cards).spacing(tokens.space_3).wrap())
            .push(Space::new().height(tokens.space_5));
    }

    // Suggestions
    body = body
        .push(view_suggestions(tokens, t))
        .push(Space::new().height(tokens.space_5));

    // Footer
    let add_instance_btn = button(
        "+ Add custom agent instance",
        Some(Message::GoToSettingsTab(
            crate::view::settings::Tab::General,
        )),
        ButtonProps::new()
            .variant(ButtonVariant::Ghost)
            .size(ButtonSize::Size1),
        t,
    );
    body = body.push(add_instance_btn);

    scrollable(
        container(body)
            .padding([tokens.space_6 as u16, tokens.space_7 as u16])
            .width(Length::Fill),
    )
    .height(Length::Fill)
    .into()
}

// ── Card helpers ───────────────────────────────────────────────────────────────

fn view_section_header<'a>(
    label: &'static str,
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, Message> {
    text(label)
        .size(tokens.text_xs.size)
        .color(t.palette.muted_foreground)
        .into()
}

fn view_instance_card<'a>(
    inst: &'a AgentInstance,
    providers: &'a [ModelProvider],
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, Message> {
    let dark = crate::theme::is_dark_theme(t);
    let icon_bytes = inst.harness.icon().svg_bytes(dark).into_owned();
    let icon_handle = svg::Handle::from_memory(icon_bytes);
    let icon_widget = svg::Svg::new(icon_handle)
        .width(Length::Fixed(24.0))
        .height(Length::Fixed(24.0));

    let provider = providers.iter().find(|p| p.id == inst.provider_id);
    let model_line = provider
        .map(|p| format!("({})", p.default_model))
        .unwrap_or_default();
    let subtitle = inst
        .instructions
        .as_deref()
        .unwrap_or(&model_line)
        .to_string();

    let id = inst.id.clone();
    let start_btn = button(
        "Start",
        Some(Message::CreateSession(id)),
        ButtonProps::new()
            .variant(ButtonVariant::Solid)
            .size(ButtonSize::Size1),
        t,
    );

    container(
        column![
            icon_widget,
            Space::new().height(tokens.space_1),
            text(&inst.label)
                .size(tokens.text_sm.size)
                .color(t.palette.foreground),
            text(subtitle)
                .size(tokens.text_xs.size)
                .color(t.palette.muted_foreground),
            Space::new().height(tokens.space_2),
            start_btn.width(Length::Fill),
        ]
        .spacing(0)
        .width(Length::Fill)
        .align_x(Alignment::Center),
    )
    .padding(tokens.space_3 as u16)
    .style(move |_theme| {
        iced::widget::container::Style::default()
            .border(iced::border::rounded(tokens.radius_md))
            .background(iced::Color {
                a: 0.04,
                ..t.palette.foreground
            })
    })
    .width(Length::Fixed(160.0))
    .into()
}

fn view_missing_instance_card<'a>(
    inst: &'a AgentInstance,
    agent_statuses: &'a [AgentRuntimeStatus],
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, Message> {
    let dark = crate::theme::is_dark_theme(t);
    let icon_bytes = inst.harness.icon().svg_bytes(dark).into_owned();
    let icon_handle = svg::Handle::from_memory(icon_bytes);
    let icon_widget = svg::Svg::new(icon_handle)
        .width(Length::Fixed(24.0))
        .height(Length::Fixed(24.0));

    let runtime = harness_to_runtime(inst.harness);
    let needs_bun = runtime.is_some_and(|r| r.needs_bun());
    let bin_name = runtime.map(|r| r.binary()).unwrap_or("?");

    let pill = if needs_bun {
        format!("needs: bun + {}", bin_name)
    } else {
        format!("needs: {}", bin_name)
    };

    let action_btn: Element<_> = if needs_bun {
        button(
            "Install Bun",
            Some(Message::InstallTool(RuntimeDep::Bun)),
            ButtonProps::new()
                .variant(ButtonVariant::Outline)
                .size(ButtonSize::Size1),
            t,
        )
        .into()
    } else {
        let cmd = runtime
            .and_then(|r| {
                let status = agent_statuses.iter().find(|s| s.agent == r)?;
                status.agent.install_label().map(|s| s.to_string())
            })
            .unwrap_or_default();
        if cmd.is_empty() {
            text("Manual install")
                .size(tokens.text_xs.size)
                .color(t.palette.muted_foreground)
                .into()
        } else {
            button(
                cmd,
                Some(Message::InstallTool(RuntimeDep::Bun)),
                ButtonProps::new()
                    .variant(ButtonVariant::Outline)
                    .size(ButtonSize::Size1),
                t,
            )
            .into()
        }
    };

    container(
        column![
            icon_widget,
            Space::new().height(tokens.space_1),
            text(&inst.label)
                .size(tokens.text_sm.size)
                .color(t.palette.muted_foreground),
            text(pill)
                .size(tokens.text_xs.size)
                .color(t.palette.muted_foreground),
            Space::new().height(tokens.space_2),
            action_btn,
        ]
        .spacing(0)
        .width(Length::Fill)
        .align_x(Alignment::Center),
    )
    .padding(tokens.space_3 as u16)
    .style(move |_theme| {
        iced::widget::container::Style::default()
            .border(iced::border::rounded(tokens.radius_md))
            .background(iced::Color {
                a: 0.03,
                ..t.palette.muted_foreground
            })
    })
    .width(Length::Fixed(160.0))
    .into()
}

fn view_dep_card<'a>(
    dep: &'a DepStatus,
    installing: Option<RuntimeDep>,
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, Message> {
    let busy = installing == Some(dep.dep);
    let btn_label = if busy { "Installing..." } else { "Install" };

    let action = button(
        btn_label,
        if busy {
            None
        } else {
            Some(Message::InstallTool(dep.dep))
        },
        ButtonProps::new()
            .variant(ButtonVariant::Outline)
            .size(ButtonSize::Size1),
        t,
    );

    container(
        column![
            text(dep.dep.label())
                .size(tokens.text_sm.size)
                .color(t.palette.muted_foreground),
            text(dep.dep.why())
                .size(tokens.text_xs.size)
                .color(t.palette.muted_foreground),
            Space::new().height(tokens.space_2),
            action,
        ]
        .spacing(0)
        .width(Length::Fill)
        .align_x(Alignment::Center),
    )
    .padding(tokens.space_3 as u16)
    .style(move |_theme| {
        iced::widget::container::Style::default()
            .border(iced::border::rounded(tokens.radius_md))
            .background(iced::Color {
                a: 0.03,
                ..t.palette.muted_foreground
            })
    })
    .width(Length::Fixed(160.0))
    .into()
}

fn view_suggestions<'a>(tokens: &'a DesignTokens, t: &'a Theme) -> Element<'a, Message> {
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
            button(
                *label,
                Some(Message::UseSuggestion(prompt)),
                ButtonProps::new()
                    .variant(ButtonVariant::Outline)
                    .size(ButtonSize::Size1),
                t,
            )
            .into()
        })
        .collect();

    row(chips).spacing(tokens.space_2).wrap().into()
}

// ── Helpers ────────────────────────────────────────────────────────────────────

fn harness_to_runtime(harness: crate::harness::Harness) -> Option<AgentRuntime> {
    use crate::harness::Harness;
    match harness {
        Harness::ClaudeCode => Some(AgentRuntime::ClaudeAgentAcp),
        Harness::Codex => Some(AgentRuntime::CodexAcp),
        Harness::OpenCode => Some(AgentRuntime::OpenCode),
        Harness::Hermes => Some(AgentRuntime::Hermes),
        Harness::OpenClaw => Some(AgentRuntime::OpenClaw),
        Harness::SwarmX => None,
    }
}

fn instance_is_ready(inst: &AgentInstance, agent_statuses: &[AgentRuntimeStatus]) -> bool {
    match harness_to_runtime(inst.harness) {
        Some(rt) => agent_statuses.iter().any(|s| s.agent == rt && s.available),
        None => true, // SwarmX always ready
    }
}
