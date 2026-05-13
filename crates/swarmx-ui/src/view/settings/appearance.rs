use iced::widget::{Space, column, container, row, scrollable, text};
use iced::{Element, Length};
use iced_shadcn::{ButtonProps, ButtonSize, ButtonVariant, Theme, button};

use crate::app::App;
use crate::theme::ThemePreference;
use crate::tokens::Density;

// ── Appearance Tab ──────────────────────────────────────────────────────────

pub fn view<'a>(app: &'a App, t: &'a Theme) -> Element<'a, crate::app::Message> {
    let heading = text("Appearance").size(16).color(t.palette.foreground);

    let desc = text("Choose a theme for the application.")
        .size(12)
        .color(t.palette.muted_foreground);

    let themes: Vec<Element<_>> = ThemePreference::all()
        .iter()
        .map(|pref| {
            let is_active = app.theme_preference == *pref;
            button(
                pref.label(),
                Some(crate::app::Message::SetTheme(*pref)),
                ButtonProps::new()
                    .variant(if is_active {
                        ButtonVariant::Solid
                    } else {
                        ButtonVariant::Outline
                    })
                    .size(ButtonSize::Size2),
                t,
            )
            .into()
        })
        .collect();

    let densities: Vec<Element<_>> = [Density::Comfortable, Density::Compact]
        .iter()
        .map(|d| {
            let is_active = app.density == *d;
            let label = match d {
                Density::Comfortable => "Comfortable",
                Density::Compact => "Compact",
            };
            button(
                label,
                Some(crate::app::Message::SetDensity(*d)),
                ButtonProps::new()
                    .variant(if is_active {
                        ButtonVariant::Solid
                    } else {
                        ButtonVariant::Outline
                    })
                    .size(ButtonSize::Size1),
                t,
            )
            .into()
        })
        .collect();

    container(scrollable(
        column![
            heading,
            Space::new().height(4.0),
            desc,
            Space::new().height(16.0),
            text("Theme").size(13).color(t.palette.foreground),
            Space::new().height(8.0),
            column(themes).spacing(8).width(Length::Fill),
            Space::new().height(20.0),
            text("Density").size(13).color(t.palette.foreground),
            Space::new().height(8.0),
            row(densities).spacing(8),
        ]
        .spacing(0),
    ))
    .into()
}
