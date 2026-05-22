pub mod chat;
pub mod chat_message;
pub mod command_palette;
pub mod composer;
pub mod settings;
pub mod sidebar;
pub mod welcome;

use iced::widget::{Space, column, container, row, text};
use iced::{Alignment, Element, Length};
use iced_shadcn::{ButtonProps, ButtonSize, ButtonVariant, Theme, button};

use crate::app::{App, Message, Surface};
use crate::tokens::DesignTokens;

// ── Main View ────────────────────────────────────────────────────────────────

pub fn main_view(app: &App) -> Element<'_, Message> {
    let t = &app.theme;

    if app.startup_loading {
        return view_loading_screen(&app.tokens, t, app.spinner_progress);
    }

    let sidebar_view = sidebar::view(
        &app.sessions,
        app.active_session,
        app.session_grouping,
        app.session_sort_by,
        &app.sidebar_search,
        &app.group_collapsed,
        app.renaming_session,
        app.renaming_remote_session.as_ref(),
        &app.rename_buffer,
        &app.instances,
        &app.remote_sessions,
        &app.remote_title_overrides,
        &app.tokens,
        t,
    );

    let main: Element<_> = match app.surface {
        Surface::Welcome => welcome::gallery(
            &app.instances,
            &app.providers,
            &app.env_checks,
            app.env_installing,
            &app.agent_statuses,
            &app.home_dir,
            &app.tokens,
            t,
        ),
        Surface::Conversation => {
            let (instance, provider, inst_idx, cwd, msgs, md) = app
                .active_session
                .and_then(|i| app.sessions.get(i))
                .map(|s| {
                    let inst = app
                        .instances
                        .iter()
                        .find(|inst| inst.id == s.agent_instance_id);
                    let prov =
                        inst.and_then(|i| app.providers.iter().find(|p| p.id == i.provider_id));
                    let idx = app
                        .instances
                        .iter()
                        .position(|inst| inst.id == s.agent_instance_id);
                    let cwd = s.working_dir.as_str();
                    let md = app.md_cache.get(&s.id).map(|c| c.as_slice()).unwrap_or(&[]);
                    (inst, prov, idx, cwd, s.messages.as_slice(), md)
                })
                .unwrap_or((None, None, None, "", &[], &[]));
            chat::view(
                instance,
                provider,
                inst_idx,
                Some(cwd),
                msgs,
                md,
                &app.input,
                app.loading,
                app.error.as_deref(),
                &app.thinking_expanded,
                &app.tool_expanded,
                &app.tokens,
                t,
            )
        }
        Surface::Settings { .. } => settings::view(app, t),
    };

    let sidebar_element: Element<_> = if app.sidebar_collapsed {
        let expand_btn = button(
            "\u{25B6}",
            Some(Message::ToggleSidebar),
            ButtonProps::new()
                .variant(ButtonVariant::Ghost)
                .size(ButtonSize::Size1),
            t,
        );
        container(column![expand_btn].align_x(Alignment::Center).spacing(8))
            .width(Length::Fixed(24.0))
            .height(Length::Fill)
            .into()
    } else {
        container(sidebar_view)
            .width(Length::Fixed(app.sidebar_width))
            .height(Length::Fill)
            .into()
    };

    let drag_handle: Element<_> = if app.sidebar_collapsed {
        Space::new().width(0).height(0).into()
    } else {
        let handle_color = if app.sidebar_dragging {
            t.palette.primary
        } else {
            iced::Color {
                a: 0.08,
                ..t.palette.border
            }
        };
        iced::widget::mouse_area(
            container(Space::new().width(4.0).height(Length::Fill)).style(move |_theme| {
                iced::widget::container::Style::default().background(handle_color)
            }),
        )
        .on_press(Message::StartSidebarDrag)
        .into()
    };

    container(
        row![sidebar_element, drag_handle, main]
            .width(Length::Fill)
            .height(Length::Fill),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}

// ── Loading Screen ───────────────────────────────────────────────────────────

fn view_loading_screen<'a>(
    tokens: &'a DesignTokens,
    t: &'a Theme,
    progress: f32,
) -> Element<'a, Message> {
    let sp = iced_shadcn::spinner(
        iced_shadcn::Spinner::new(t)
            .size(iced_shadcn::SpinnerSize::Size2)
            .progress(progress),
    );
    container(
        column![
            Space::new().height(Length::Fill),
            text("SwarmX")
                .size(tokens.text_2xl.size)
                .color(t.palette.foreground),
            Space::new().height(tokens.space_4),
            sp,
            Space::new().height(tokens.space_3),
            text("Loading...")
                .size(tokens.text_sm.size)
                .color(t.palette.muted_foreground),
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
