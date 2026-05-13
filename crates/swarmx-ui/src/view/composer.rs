use iced::widget::{Space, column, container, row, text};
use iced::{Alignment, Element, Length};
use iced_shadcn::{
    BadgeProps, BadgeSize, BadgeVariant, ButtonProps, ButtonSize, ButtonVariant, InputProps,
    InputSize, SelectProps, Theme, badge, icon_button, input, select,
};
use lucide_icons::Icon as LucideIcon;

use crate::tokens::DesignTokens;
use crate::view::chat_message::lucide_icon;

// ── View ──────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn view<'a>(
    input_text: &'a str,
    loading: bool,
    available_models: &'a [String],
    current_model: &'a str,
    instance_idx: usize,
    cwd: Option<&'a str>,
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, crate::app::Message> {
    // ── Top row: model selector + cwd hint ─────────────────────────────────
    let model_select = select(
        available_models,
        Some(current_model.to_string()),
        "Model...",
        move |m| crate::app::Message::ModelChanged(instance_idx, m),
        SelectProps::new(),
        t,
    );

    let model_badge: Element<_> = if !current_model.is_empty() {
        badge(
            current_model,
            BadgeProps::<crate::app::Message>::new()
                .variant(BadgeVariant::Secondary)
                .size(BadgeSize::Size1),
            t,
        )
    } else {
        Space::new().width(0).height(0).into()
    };

    let cwd_text = cwd.map(|c| {
        let truncated = if c.len() > 50 {
            format!("...{}", &c[c.len() - 47..])
        } else {
            c.to_string()
        };
        text(truncated)
            .size(tokens.text_xs.size)
            .color(t.palette.muted_foreground)
    });

    let top_row = row![
        model_select,
        Space::new().width(tokens.space_2),
        model_badge,
        Space::new().width(Length::Fill),
        if let Some(c) = cwd_text {
            Element::from(c)
        } else {
            Space::new().width(0).height(0).into()
        },
    ]
    .align_y(Alignment::Center);

    // ── Bottom row: input + action button ──────────────────────────────────
    let send_enabled = !input_text.trim().is_empty() && !loading;

    let action_btn: Element<_> = if loading {
        let stop_icon = lucide_icon(LucideIcon::Square, 16.0);
        icon_button(
            stop_icon,
            Some(crate::app::Message::StopGeneration),
            ButtonProps::new()
                .variant(ButtonVariant::Destructive)
                .size(ButtonSize::Size2),
            t,
        )
        .into()
    } else {
        let send_icon = lucide_icon(LucideIcon::ArrowUp, 16.0);
        icon_button(
            send_icon,
            if send_enabled {
                Some(crate::app::Message::SendMessage)
            } else {
                None
            },
            ButtonProps::new()
                .variant(ButtonVariant::Solid)
                .size(ButtonSize::Size2),
            t,
        )
        .into()
    };

    let text_input = input(
        input_text,
        "Send a message...",
        Some(crate::app::Message::InputChanged),
        InputProps::new().size(InputSize::Size2),
        t,
    )
    .on_submit(if send_enabled {
        crate::app::Message::SendMessage
    } else {
        crate::app::Message::InputChanged(input_text.to_string())
    });

    container(column![
        top_row,
        Space::new().height(tokens.space_2),
        row![
            text_input.width(Length::Fill),
            Space::new().width(tokens.space_2),
            action_btn,
        ]
        .align_y(Alignment::Center),
    ])
    .padding([tokens.space_3 as u16, tokens.space_4 as u16])
    .style(move |_theme| {
        iced::widget::container::Style::default().border(
            iced::border::rounded(tokens.radius_md)
                .color(iced::Color {
                    a: 0.5,
                    ..t.palette.border
                })
                .width(1.0),
        )
    })
    .width(Length::Fill)
    .into()
}

#[cfg(test)]
mod tests;
