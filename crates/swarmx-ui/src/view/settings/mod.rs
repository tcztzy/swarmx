pub mod about;
pub mod agents;
pub mod appearance;

use iced::widget::{Space, column, container, row, text};
use iced::{Element, Length};
use iced_shadcn::{ButtonProps, ButtonSize, ButtonVariant, Theme, button};

use crate::app::{App, Surface};
use crate::theme::ThemePreference;
use crate::tokens::Density;

// ── Settings Tab ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tab {
    Appearance,
    Agents,
    General,
}

impl Tab {
    fn label(&self) -> &str {
        match self {
            Self::Appearance => "Appearance",
            Self::Agents => "Agents",
            Self::General => "General",
        }
    }
}

// ── Message Sub-Enum ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Message {
    GoToSettingsTab(Tab),
    SetTheme(ThemePreference),
    SetDensity(Density),
}

// ── View ────────────────────────────────────────────────────────────────────

pub fn view<'a>(app: &'a App, t: &'a Theme) -> Element<'a, crate::app::Message> {
    let tab = match app.surface {
        Surface::Settings { tab } => tab,
        _ => Tab::Appearance,
    };
    let sidebar = view_sidebar(tab, t);
    let content = match tab {
        Tab::Appearance => appearance::view(app, t),
        Tab::Agents => agents::view(app, t),
        Tab::General => about::view(t),
    };

    row![sidebar, Space::new().width(24.0), content]
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(16)
        .into()
}

fn view_sidebar<'a>(active_tab: Tab, t: &'a Theme) -> Element<'a, crate::app::Message> {
    let tabs: Vec<Element<_>> = [Tab::Appearance, Tab::General]
        .iter()
        .map(|tab| {
            let is_active = active_tab == *tab;
            button(
                tab.label(),
                Some(crate::app::Message::GoToSettingsTab(*tab)),
                ButtonProps::new()
                    .variant(if is_active {
                        ButtonVariant::Solid
                    } else {
                        ButtonVariant::Ghost
                    })
                    .size(ButtonSize::Size2),
                t,
            )
            .width(Length::Fill)
            .into()
        })
        .collect();

    let heading = text("Settings").size(18).color(t.palette.foreground);

    container(column![heading, Space::new().height(16.0), column(tabs).spacing(4),].width(160.0))
        .into()
}
