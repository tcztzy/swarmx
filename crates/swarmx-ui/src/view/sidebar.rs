use iced::widget::{Space, column, container, row, scrollable, stack, svg, text, text_input};
use iced::{Alignment, Color, Element, Length, Shadow, Vector};
use iced_shadcn::{
    ButtonProps, ButtonRadius, ButtonSize, ButtonVariant, ContextMenuContentSize, ContextMenuEntry,
    ContextMenuItem, ContextMenuItemProps, ContextMenuProps, DropdownMenuCheckboxItem,
    DropdownMenuContentSize, DropdownMenuEntry, DropdownMenuProps, Theme, button, button_content,
    context_menu, dropdown_menu, icon_button,
};
use lucide_icons::Icon as LucideIcon;

use crate::app::FilterChange;
use crate::data::{AgentIcon, RemoteAgentSessions, RemoteSessionSource, Session};
use crate::environment::{AgentRuntime, RemoteAgentRef};
use crate::instance::AgentInstance;
use crate::persistence::{SessionGrouping, SessionSortBy};
use crate::theme::{agent_color, is_dark_theme};
use crate::tokens::DesignTokens;

// ── Group ─────────────────────────────────────────────────────────────────────

const UNKNOWN_PROJECT_LABEL: &str = "Unknown project";

pub(crate) struct Group {
    pub name: String,
    pub session_indices: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SessionAction {
    Select(usize),
    LoadRemote {
        agent_ref: RemoteAgentRef,
        source: RemoteSessionSource,
        session_id: String,
        cwd: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SidebarSessionEntry {
    session_id: String,
    copy_session_id: String,
    agent_instance_id: String,
    agent_runtime: Option<AgentRuntime>,
    working_dir: String,
    title: Option<String>,
    first_message: Option<String>,
    created_at: String,
    updated_at: String,
    pinned: bool,
    unread: bool,
    action: SessionAction,
}

pub(crate) trait SessionEntryData {
    fn agent_instance_id(&self) -> &str;
    fn agent_runtime(&self) -> Option<AgentRuntime>;
    fn working_dir(&self) -> &str;
    fn title(&self) -> Option<&str>;
    fn first_message(&self) -> Option<&str>;
    fn created_at(&self) -> &str;
    fn updated_at(&self) -> &str;
    fn pinned(&self) -> bool;
}

impl SessionEntryData for Session {
    fn agent_instance_id(&self) -> &str {
        &self.agent_instance_id
    }

    fn agent_runtime(&self) -> Option<AgentRuntime> {
        self.agent_runtime
    }

    fn working_dir(&self) -> &str {
        &self.working_dir
    }

    fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    fn first_message(&self) -> Option<&str> {
        self.messages.first().map(|m| m.content.as_str())
    }

    fn created_at(&self) -> &str {
        &self.created_at
    }

    fn updated_at(&self) -> &str {
        &self.updated_at
    }

    fn pinned(&self) -> bool {
        self.pinned
    }
}

impl SessionEntryData for SidebarSessionEntry {
    fn agent_instance_id(&self) -> &str {
        &self.agent_instance_id
    }

    fn agent_runtime(&self) -> Option<AgentRuntime> {
        self.agent_runtime
    }

    fn working_dir(&self) -> &str {
        &self.working_dir
    }

    fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    fn first_message(&self) -> Option<&str> {
        self.first_message.as_deref()
    }

    fn created_at(&self) -> &str {
        &self.created_at
    }

    fn updated_at(&self) -> &str {
        &self.updated_at
    }

    fn pinned(&self) -> bool {
        self.pinned
    }
}

// ── View ──────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn view<'a>(
    sessions: &'a [Session],
    active_index: Option<usize>,
    grouping: SessionGrouping,
    sort_by: SessionSortBy,
    search_query: &'a str,
    collapsed_groups: &'a std::collections::HashSet<String>,
    renaming_session: Option<usize>,
    rename_buffer: &'a str,
    instances: &'a [AgentInstance],
    remote_sessions: &'a [RemoteAgentSessions],
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, crate::app::Message> {
    let session_entries = collect_session_entries(sessions, remote_sessions);
    let header = view_header(tokens, t);
    let search = view_search_input(search_query, grouping, sort_by, tokens, t);

    let filtered = (0..session_entries.len()).collect::<Vec<_>>();
    let mut groups = group_indices(&filtered, &session_entries, grouping, instances);
    for group in &mut groups {
        sort_within_group(group, &session_entries, sort_by);
    }
    let groups_section = view_groups(
        &groups,
        &session_entries,
        active_index,
        grouping,
        search_query,
        collapsed_groups,
        renaming_session,
        rename_buffer,
        instances,
        tokens,
        t,
    );

    let mut session_list = column![groups_section].spacing(0).width(Length::Fill);
    session_list = session_list.push(Space::new().height(sidebar_footer_reserved_height(tokens)));

    let session_scroll = container(
        scrollable(session_list)
            .id(iced::widget::Id::new("sidebar-session-list"))
            .direction(iced::widget::scrollable::Direction::Vertical(
                iced::widget::scrollable::Scrollbar::new()
                    .width(8.0)
                    .scroller_width(4.0)
                    .margin(2.0),
            ))
            .width(Length::Fill)
            .height(Length::Fill),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .clip(true);

    let footer = view_sidebar_footer(instances, tokens, t);
    let body = stack![session_scroll, view_sidebar_footer_overlay(footer, t),]
        .width(Length::Fill)
        .height(Length::Fill);

    let col = column![
        header,
        Space::new().height(tokens.space_2),
        search,
        Space::new().height(tokens.space_2),
        body
    ]
    .spacing(0)
    .width(Length::Fill)
    .height(Length::Fill);

    container(col)
        .width(Length::Fixed(tokens.sidebar_default_width))
        .height(Length::Fill)
        .into()
}

// ── Header ────────────────────────────────────────────────────────────────────

fn view_header<'a>(tokens: &'a DesignTokens, t: &'a Theme) -> Element<'a, crate::app::Message> {
    let new_btn = button(
        "+ New conversation",
        Some(crate::app::Message::NewSession),
        ButtonProps::new()
            .variant(ButtonVariant::Solid)
            .size(ButtonSize::Size2),
        t,
    );

    container(new_btn.width(Length::Fill))
        .padding([tokens.space_4 as u16, tokens.space_3 as u16])
        .width(Length::Fill)
        .into()
}

// ── Search ────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn view_search_input<'a>(
    query: &'a str,
    grouping: SessionGrouping,
    sort_by: SessionSortBy,
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, crate::app::Message> {
    let icon = text(char::from(LucideIcon::Search).to_string())
        .font(iced::Font::with_name("lucide"))
        .size(14)
        .color(t.palette.muted_foreground);

    let input = text_input("Search sessions...", query)
        .on_input(crate::app::Message::SearchSidebar)
        .size(tokens.text_sm.size)
        .padding([tokens.space_1 as u16, tokens.space_2 as u16]);

    let filter_icon = text(char::from(LucideIcon::SlidersHorizontal).to_string())
        .font(iced::Font::with_name("lucide"))
        .size(16)
        .color(t.palette.foreground);
    let filter_trigger = icon_button(
        filter_icon,
        None::<crate::app::Message>,
        ButtonProps::new()
            .variant(ButtonVariant::Ghost)
            .size(ButtonSize::Size1),
        t,
    );
    let filter_menu = dropdown_menu(
        filter_trigger,
        build_filter_entries(grouping, sort_by),
        DropdownMenuProps::new()
            .size(DropdownMenuContentSize::Size2)
            .width(220),
        t,
    );

    container(
        row![
            icon,
            Space::new().width(tokens.space_2),
            input.width(Length::Fill),
            Space::new().width(tokens.space_2),
            filter_menu,
        ]
        .align_y(Alignment::Center),
    )
    .padding([0u16, tokens.space_3 as u16])
    .width(Length::Fill)
    .into()
}

// ── Filter Menu ───────────────────────────────────────────────────────────────

fn build_filter_entries<'a>(
    grouping: SessionGrouping,
    sort_by: SessionSortBy,
) -> Vec<DropdownMenuEntry<'a, crate::app::Message>> {
    vec![
        DropdownMenuEntry::Label("Organize".into()),
        checked_menu_entry(
            SessionGrouping::Project.organize_label(),
            grouping == SessionGrouping::Project,
            crate::app::Message::FilterChanged(FilterChange::Grouping(SessionGrouping::Project)),
        ),
        checked_menu_entry(
            SessionGrouping::Harness.organize_label(),
            grouping == SessionGrouping::Harness,
            crate::app::Message::FilterChanged(FilterChange::Grouping(SessionGrouping::Harness)),
        ),
        checked_menu_entry(
            SessionGrouping::Date.organize_label(),
            grouping == SessionGrouping::Date,
            crate::app::Message::FilterChanged(FilterChange::Grouping(SessionGrouping::Date)),
        ),
        checked_menu_entry(
            SessionGrouping::None.organize_label(),
            grouping == SessionGrouping::None,
            crate::app::Message::FilterChanged(FilterChange::Grouping(SessionGrouping::None)),
        ),
        DropdownMenuEntry::Separator,
        DropdownMenuEntry::Label("Sort by".into()),
        checked_menu_entry(
            "Created",
            sort_by == SessionSortBy::Created,
            crate::app::Message::FilterChanged(FilterChange::Sort(SessionSortBy::Created)),
        ),
        checked_menu_entry(
            "Updated",
            sort_by == SessionSortBy::Recency,
            crate::app::Message::FilterChanged(FilterChange::Sort(SessionSortBy::Recency)),
        ),
        checked_menu_entry(
            "Title",
            sort_by == SessionSortBy::Alphabetically,
            crate::app::Message::FilterChanged(FilterChange::Sort(SessionSortBy::Alphabetically)),
        ),
    ]
}

fn checked_menu_entry<'a>(
    label: &'static str,
    checked: bool,
    message: crate::app::Message,
) -> DropdownMenuEntry<'a, crate::app::Message> {
    DropdownMenuEntry::CheckboxItem(DropdownMenuCheckboxItem::new(label, checked, Some(message)))
}

fn project_name(working_dir: &str) -> String {
    if is_unknown_working_dir(working_dir) {
        return UNKNOWN_PROJECT_LABEL.to_string();
    }

    std::path::Path::new(working_dir)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| working_dir.to_string())
}

fn project_path_key(working_dir: &str) -> String {
    if is_unknown_working_dir(working_dir) {
        return String::new();
    }

    working_dir.trim_end_matches('/').to_string()
}

fn display_project_path(working_dir: &str) -> String {
    if is_unknown_working_dir(working_dir) {
        return UNKNOWN_PROJECT_LABEL.to_string();
    }

    let path = std::path::Path::new(working_dir);
    if let Some(home) = std::env::var_os("HOME").filter(|home| !home.is_empty()) {
        let home = std::path::PathBuf::from(home);
        if let Ok(stripped) = path.strip_prefix(&home) {
            return if stripped.as_os_str().is_empty() {
                "~".to_string()
            } else {
                format!("~/{}", stripped.display())
            };
        }
    }
    working_dir.to_string()
}

fn is_unknown_working_dir(working_dir: &str) -> bool {
    let trimmed = working_dir.trim();
    trimmed.is_empty() || trimmed == "."
}

fn duplicate_project_names<T: SessionEntryData>(
    indices: &[usize],
    sessions: &[T],
) -> std::collections::HashSet<String> {
    let mut paths_by_name: std::collections::HashMap<String, std::collections::HashSet<String>> =
        std::collections::HashMap::new();

    for &i in indices {
        let working_dir = sessions[i].working_dir();
        paths_by_name
            .entry(project_name(working_dir))
            .or_default()
            .insert(project_path_key(working_dir));
    }

    paths_by_name
        .into_iter()
        .filter_map(|(name, paths)| (paths.len() > 1).then_some(name))
        .collect()
}

fn project_label(working_dir: &str, duplicate_names: &std::collections::HashSet<String>) -> String {
    let name = project_name(working_dir);
    if duplicate_names.contains(&name) {
        format!("{} ({})", name, display_project_path(working_dir))
    } else {
        name
    }
}

fn runtime_icon(runtime: AgentRuntime) -> AgentIcon {
    match runtime {
        AgentRuntime::ClaudeAgentAcp => AgentIcon::ClaudeCode,
        AgentRuntime::CodexAcp => AgentIcon::Codex,
        AgentRuntime::OpenCode => AgentIcon::OpenCode,
        AgentRuntime::Hermes => AgentIcon::Hermes,
        AgentRuntime::OpenClaw => AgentIcon::OpenClaw,
    }
}

fn runtime_harness(runtime: AgentRuntime) -> crate::harness::Harness {
    match runtime {
        AgentRuntime::ClaudeAgentAcp => crate::harness::Harness::ClaudeCode,
        AgentRuntime::CodexAcp => crate::harness::Harness::Codex,
        AgentRuntime::OpenCode => crate::harness::Harness::OpenCode,
        AgentRuntime::Hermes => crate::harness::Harness::Hermes,
        AgentRuntime::OpenClaw => crate::harness::Harness::OpenClaw,
    }
}

fn session_harness_label<T: SessionEntryData>(session: &T, instances: &[AgentInstance]) -> String {
    if let Some(runtime) = session.agent_runtime() {
        return runtime.session_label().to_string();
    }

    instances
        .iter()
        .find(|inst| inst.id == session.agent_instance_id())
        .map(|inst| inst.harness.label().to_string())
        .unwrap_or_else(|| "Unknown Harness".to_string())
}

fn session_icon_bytes<T: SessionEntryData>(
    session: &T,
    instances: &[AgentInstance],
    is_dark: bool,
) -> Vec<u8> {
    if let Some(runtime) = session.agent_runtime() {
        return runtime_icon(runtime).svg_bytes(is_dark).into_owned();
    }

    instances
        .iter()
        .find(|inst| inst.id == session.agent_instance_id())
        .map(|inst| inst.harness.icon().svg_bytes(is_dark).into_owned())
        .unwrap_or_default()
}

fn collect_session_entries(
    sessions: &[Session],
    remote_sessions: &[RemoteAgentSessions],
) -> Vec<SidebarSessionEntry> {
    let mut entries: Vec<SidebarSessionEntry> = sessions
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.archived)
        .map(|(index, s)| SidebarSessionEntry {
            session_id: s.id.clone(),
            copy_session_id: s.acp_session_id.clone().unwrap_or_else(|| s.id.clone()),
            agent_instance_id: s.agent_instance_id.clone(),
            agent_runtime: s.agent_runtime,
            working_dir: s.working_dir.clone(),
            title: s.title.clone(),
            first_message: s.messages.first().map(|m| m.content.clone()),
            created_at: s.created_at.clone(),
            updated_at: s.updated_at.clone(),
            pinned: s.pinned,
            unread: s.unread,
            action: SessionAction::Select(index),
        })
        .collect();

    for agent_sessions in remote_sessions {
        for session in &agent_sessions.sessions {
            let cwd = session.cwd.display().to_string();
            let title = remote_session_title(session.title.as_deref(), &cwd);
            let updated_at = session.updated_at.clone().unwrap_or_default();
            let session_id = session.session_id.to_string();
            let (agent_instance_id, agent_runtime) = match &agent_sessions.agent_ref {
                RemoteAgentRef::Instance(instance_id) => (instance_id.clone(), None),
                RemoteAgentRef::Runtime(runtime) => (runtime.id().to_string(), Some(*runtime)),
            };
            entries.push(SidebarSessionEntry {
                session_id: session_id.clone(),
                copy_session_id: session_id.clone(),
                agent_instance_id,
                agent_runtime,
                working_dir: cwd.clone(),
                title: Some(title),
                first_message: None,
                created_at: updated_at.clone(),
                updated_at,
                pinned: false,
                unread: false,
                action: SessionAction::LoadRemote {
                    agent_ref: agent_sessions.agent_ref.clone(),
                    source: agent_sessions.source,
                    session_id,
                    cwd,
                },
            });
        }
    }

    entries
}

fn remote_session_title(title: Option<&str>, cwd: &str) -> String {
    title
        .filter(|title| !title.trim().is_empty())
        .map(ToString::to_string)
        .or_else(|| {
            if is_unknown_working_dir(cwd) {
                return None;
            }

            std::path::Path::new(cwd)
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| !name.is_empty())
                .map(ToString::to_string)
        })
        .unwrap_or_else(|| {
            if is_unknown_working_dir(cwd) {
                "Untitled session".to_string()
            } else {
                cwd.rsplit('/').next().unwrap_or(cwd).to_string()
            }
        })
}

// ── Group / Sort Pipeline ─────────────────────────────────────────────────────

pub(crate) fn group_indices<T: SessionEntryData>(
    indices: &[usize],
    sessions: &[T],
    grouping: SessionGrouping,
    instances: &[AgentInstance],
) -> Vec<Group> {
    let mut groups: Vec<Group> = Vec::new();
    let duplicate_names = if grouping == SessionGrouping::Project {
        duplicate_project_names(indices, sessions)
    } else {
        std::collections::HashSet::new()
    };

    for &i in indices {
        let s = &sessions[i];
        let key = match grouping {
            SessionGrouping::Project => project_label(s.working_dir(), &duplicate_names),
            SessionGrouping::Harness => session_harness_label(s, instances),
            SessionGrouping::Date => time_bucket(s.updated_at()),
            SessionGrouping::None => String::new(),
        };

        if let Some(group) = groups.iter_mut().find(|g| g.name == key) {
            group.session_indices.push(i);
        } else {
            groups.push(Group {
                name: key,
                session_indices: vec![i],
            });
        }
    }

    match grouping {
        SessionGrouping::Date => groups.sort_by_key(|a| time_bucket_order(&a.name)),
        SessionGrouping::None => {}
        _ => groups.sort_by(|a, b| a.name.cmp(&b.name)),
    }

    groups
}

pub(crate) fn sort_within_group<T: SessionEntryData>(
    group: &mut Group,
    sessions: &[T],
    sort_by: SessionSortBy,
) {
    match sort_by {
        SessionSortBy::Alphabetically => {
            group.session_indices.sort_by(|&a, &b| {
                let pinned = sessions[b].pinned().cmp(&sessions[a].pinned());
                if pinned != std::cmp::Ordering::Equal {
                    return pinned;
                }
                session_display_title(&sessions[a])
                    .to_lowercase()
                    .cmp(&session_display_title(&sessions[b]).to_lowercase())
            });
        }
        SessionSortBy::Created => {
            group.session_indices.sort_by(|&a, &b| {
                let pinned = sessions[b].pinned().cmp(&sessions[a].pinned());
                if pinned != std::cmp::Ordering::Equal {
                    return pinned;
                }
                parse_ts(sessions[b].created_at()).cmp(&parse_ts(sessions[a].created_at()))
            });
        }
        SessionSortBy::Recency => {
            group.session_indices.sort_by(|&a, &b| {
                let pinned = sessions[b].pinned().cmp(&sessions[a].pinned());
                if pinned != std::cmp::Ordering::Equal {
                    return pinned;
                }
                parse_ts(sessions[b].updated_at()).cmp(&parse_ts(sessions[a].updated_at()))
            });
        }
    }
}

fn parse_ts(raw: &str) -> i64 {
    raw.parse()
        .or_else(|_| chrono::DateTime::parse_from_rfc3339(raw).map(|dt| dt.timestamp()))
        .unwrap_or(0)
}

fn session_display_title<T: SessionEntryData>(s: &T) -> String {
    s.title()
        .filter(|title| !title.trim().is_empty())
        .map(ToString::to_string)
        .or_else(|| s.first_message().map(ToString::to_string))
        .unwrap_or_else(|| "New session".to_string())
}

fn session_item_label<T: SessionEntryData>(s: &T) -> String {
    session_display_title(s)
}

#[allow(dead_code)]
pub(crate) fn relative_time(updated_at: &str) -> String {
    let ts = parse_ts(updated_at);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let diff = (now - ts).max(0);

    if diff < 60 {
        "1m".into()
    } else if diff < 3600 {
        format!("{}m", diff / 60)
    } else if diff < 86400 {
        format!("{}h", diff / 3600)
    } else if diff < 604800 {
        format!("{}d", diff / 86400)
    } else {
        let weeks = diff / 604800;
        if weeks > 99 {
            "99w+".into()
        } else {
            format!("{}w", weeks)
        }
    }
}

pub(crate) fn time_bucket(updated_at: &str) -> String {
    let ts = parse_ts(updated_at);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    let diff_secs = (now - ts).max(0);
    let diff_days = diff_secs / 86400;

    if diff_days == 0 {
        "Today".into()
    } else if diff_days == 1 {
        "Yesterday".into()
    } else if diff_days < 7 {
        "This Week".into()
    } else if diff_days < 30 {
        "This Month".into()
    } else {
        "Older".into()
    }
}

fn time_bucket_order(name: &str) -> u8 {
    match name {
        "Today" => 0,
        "Yesterday" => 1,
        "This Week" => 2,
        "This Month" => 3,
        _ => 4,
    }
}

// ── Groups View ───────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn view_groups<'a>(
    groups: &[Group],
    sessions: &[SidebarSessionEntry],
    active_index: Option<usize>,
    grouping: SessionGrouping,
    search_query: &str,
    collapsed_groups: &'a std::collections::HashSet<String>,
    renaming_session: Option<usize>,
    rename_buffer: &'a str,
    instances: &'a [AgentInstance],
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, crate::app::Message> {
    if groups.is_empty() && search_query.is_empty() {
        return container(
            text("No sessions yet")
                .size(tokens.text_sm.size)
                .color(t.palette.muted_foreground),
        )
        .padding([tokens.space_2 as u16, tokens.space_3 as u16])
        .width(Length::Fill)
        .into();
    }

    let mut col = column![].spacing(0);

    for group in groups {
        let total = group.session_indices.len();
        let q_lower = search_query.to_lowercase();

        // Filter indices by search query
        let filtered: Vec<usize> = if search_query.is_empty() {
            group.session_indices.clone()
        } else {
            group
                .session_indices
                .iter()
                .filter(|&&i| session_matches_search(&sessions[i], instances, &q_lower))
                .copied()
                .collect()
        };

        let visible = filtered.len();
        let is_collapsed = collapsed_groups.contains(&group.name);

        if !group.name.is_empty() {
            let count_text = if visible < total {
                format!("({}/{})", visible, total)
            } else {
                format!("({})", total)
            };
            let header_content = row![
                group_header_icon(grouping, group, sessions, instances, is_collapsed, t),
                Space::new().width(tokens.space_2),
                text(group.name.clone())
                    .size(tokens.text_sm.size)
                    .color(t.palette.foreground)
                    .wrapping(iced::widget::text::Wrapping::None),
                Space::new().width(4.0),
                text(count_text)
                    .size(tokens.text_sm.size)
                    .color(t.palette.foreground),
            ]
            .align_y(Alignment::Center)
            .width(Length::Fill);

            let header_btn = button_content(
                header_content,
                Some(crate::app::Message::ToggleGroup(group.name.clone())),
                ButtonProps::new()
                    .variant(ButtonVariant::Ghost)
                    .size(ButtonSize::Size1),
                t,
            );

            col = col.push(header_btn);
        }

        let show_items = group.name.is_empty() || !is_collapsed;
        if show_items && !filtered.is_empty() {
            let item_context = SessionItemContext {
                active_index,
                grouping,
                renaming_session,
                rename_buffer,
                instances,
                tokens,
                t,
            };
            let items: Vec<Element<_>> = filtered
                .iter()
                .map(|&i| view_session_item(i, &sessions[i], item_context))
                .collect();
            col = col.push(column(items).spacing(2));
        }
    }

    container(col.padding([tokens.space_2 as u16, tokens.space_3 as u16]))
        .width(Length::Fill)
        .into()
}

fn group_header_icon<'a>(
    grouping: SessionGrouping,
    group: &Group,
    sessions: &[SidebarSessionEntry],
    instances: &[AgentInstance],
    is_collapsed: bool,
    t: &Theme,
) -> Element<'a, crate::app::Message> {
    match grouping {
        SessionGrouping::Project => lucide_group_icon(
            if is_collapsed {
                LucideIcon::FolderClosed
            } else {
                LucideIcon::FolderOpen
            },
            t,
        ),
        SessionGrouping::Harness => harness_group_icon(group, sessions, instances, t),
        SessionGrouping::Date => lucide_group_icon(
            if is_collapsed {
                LucideIcon::ChevronRight
            } else {
                LucideIcon::ChevronDown
            },
            t,
        ),
        SessionGrouping::None => Space::new().width(0).height(0).into(),
    }
}

fn lucide_group_icon<'a>(icon: LucideIcon, t: &Theme) -> Element<'a, crate::app::Message> {
    text(char::from(icon).to_string())
        .font(iced::Font::with_name("lucide"))
        .size(16)
        .color(t.palette.foreground)
        .into()
}

fn harness_group_icon<'a>(
    group: &Group,
    sessions: &[SidebarSessionEntry],
    instances: &[AgentInstance],
    t: &Theme,
) -> Element<'a, crate::app::Message> {
    let dark = is_dark_theme(t);
    let icon_bytes = group
        .session_indices
        .first()
        .and_then(|&i| sessions.get(i))
        .map(|session| session_icon_bytes(session, instances, dark))
        .filter(|bytes| !bytes.is_empty());

    if let Some(icon_bytes) = icon_bytes {
        let icon_handle = svg::Handle::from_memory(icon_bytes);
        return container(
            svg::Svg::new(icon_handle)
                .width(Length::Fixed(16.0))
                .height(Length::Fixed(16.0)),
        )
        .width(Length::Fixed(16.0))
        .height(Length::Fixed(16.0))
        .into();
    }

    lucide_group_icon(LucideIcon::Wrench, t)
}

fn session_matches_search(
    session: &SidebarSessionEntry,
    instances: &[AgentInstance],
    query: &str,
) -> bool {
    let title_match = session_display_title(session)
        .to_lowercase()
        .contains(query);
    let content_match = session
        .first_message()
        .map(|content| content.to_lowercase().contains(query))
        .unwrap_or(false);
    let cwd_match = !is_unknown_working_dir(session.working_dir())
        && session.working_dir().to_lowercase().contains(query);
    let agent_match = session
        .agent_runtime()
        .map(|runtime| runtime.session_label().to_lowercase().contains(query))
        .unwrap_or_else(|| {
            instances
                .iter()
                .find(|inst| inst.id == session.agent_instance_id())
                .map(|inst| inst.label.to_lowercase().contains(query))
                .unwrap_or(false)
        });

    title_match || content_match || cwd_match || agent_match
}

fn session_deeplink(session_id: &str, working_dir: &str) -> String {
    format!(
        "swarmx://session/{}?cwd={}",
        percent_encode(session_id),
        percent_encode(working_dir)
    )
}

fn percent_encode(raw: &str) -> String {
    let mut encoded = String::new();
    for byte in raw.bytes() {
        if matches!(
            byte,
            b'A'..=b'Z'
                | b'a'..=b'z'
                | b'0'..=b'9'
                | b'-'
                | b'.'
                | b'_'
                | b'~'
        ) {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

fn session_menu_item<'a>(
    label: impl Into<std::borrow::Cow<'a, str>>,
    message: Option<crate::app::Message>,
    disabled: bool,
) -> ContextMenuEntry<'a, crate::app::Message> {
    let on_select = if disabled { None } else { message };
    ContextMenuEntry::Item(
        ContextMenuItem::new(label, on_select)
            .props(ContextMenuItemProps::new().disabled(disabled)),
    )
}

fn build_session_menu_entries<'a>(
    session: &SidebarSessionEntry,
) -> Vec<ContextMenuEntry<'a, crate::app::Message>> {
    let local_index = match &session.action {
        SessionAction::Select(index) => Some(*index),
        SessionAction::LoadRemote { .. } => None,
    };
    let local_only = local_index.is_none();
    let pin_label = if session.pinned {
        "Unpin chat"
    } else {
        "Pin chat"
    };
    let unread_label = if session.unread {
        "Mark as read"
    } else {
        "Mark as unread"
    };

    let cwd_unknown = is_unknown_working_dir(&session.working_dir);

    vec![
        session_menu_item(
            pin_label,
            local_index.map(crate::app::Message::TogglePinSession),
            local_only,
        ),
        session_menu_item(
            "Rename chat",
            local_index.map(crate::app::Message::StartRenameSession),
            local_only,
        ),
        session_menu_item(
            "Archive chat",
            local_index.map(crate::app::Message::ArchiveSession),
            local_only,
        ),
        session_menu_item(
            unread_label,
            local_index.map(crate::app::Message::ToggleUnreadSession),
            local_only,
        ),
        ContextMenuEntry::Separator,
        session_menu_item(
            "Open in Finder",
            Some(crate::app::Message::OpenWorkingDirectory(
                session.working_dir.clone(),
            )),
            cwd_unknown,
        ),
        session_menu_item(
            "Copy working directory",
            Some(crate::app::Message::CopyToClipboard(
                session.working_dir.clone(),
            )),
            cwd_unknown,
        ),
        session_menu_item(
            "Copy session ID",
            Some(crate::app::Message::CopyToClipboard(
                session.copy_session_id.clone(),
            )),
            session.copy_session_id.is_empty(),
        ),
        session_menu_item(
            "Copy deeplink",
            Some(crate::app::Message::CopyToClipboard(session_deeplink(
                &session.session_id,
                &session.working_dir,
            ))),
            session.session_id.is_empty(),
        ),
        ContextMenuEntry::Separator,
        session_menu_item("Fork into local", None, true),
        session_menu_item("Fork into new worktree", None, true),
        ContextMenuEntry::Separator,
        session_menu_item(
            "Open in new window",
            Some(crate::app::Message::OpenSessionInNewWindow(
                session.session_id.clone(),
            )),
            session.session_id.is_empty(),
        ),
    ]
}

// ── Session Item ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct SessionItemContext<'a> {
    active_index: Option<usize>,
    grouping: SessionGrouping,
    renaming_session: Option<usize>,
    rename_buffer: &'a str,
    instances: &'a [AgentInstance],
    tokens: &'a DesignTokens,
    t: &'a Theme,
}

fn view_session_item<'a>(
    list_index: usize,
    session: &SidebarSessionEntry,
    ctx: SessionItemContext<'a>,
) -> Element<'a, crate::app::Message> {
    let SessionItemContext {
        active_index,
        grouping,
        renaming_session,
        rename_buffer,
        instances,
        tokens,
        t,
    } = ctx;
    let is_active = match &session.action {
        SessionAction::Select(index) => active_index == Some(*index),
        SessionAction::LoadRemote { .. } => false,
    };

    let dark = is_dark_theme(t);
    let instance = instances
        .iter()
        .find(|i| i.id == session.agent_instance_id());
    let icon_widget: Option<Element<_>> = if grouping == SessionGrouping::Harness {
        None
    } else {
        let icon_bytes = session_icon_bytes(session, instances, dark);
        let icon_handle = svg::Handle::from_memory(icon_bytes);
        Some(
            svg::Svg::new(icon_handle)
                .width(Length::Fixed(14.0))
                .height(Length::Fixed(14.0))
                .into(),
        )
    };

    let label = session_item_label(session);

    let menu_trigger = button(
        "\u{22EF}",
        None::<crate::app::Message>,
        ButtonProps::new()
            .variant(ButtonVariant::Ghost)
            .size(ButtonSize::Size1),
        t,
    );
    let menu_btn = dropdown_menu(
        menu_trigger,
        build_session_menu_entries(session),
        DropdownMenuProps::new()
            .size(DropdownMenuContentSize::Size2)
            .width(250),
        t,
    );

    // Accent strip color for active session
    let accent = if is_active {
        session
            .agent_runtime()
            .map(|runtime| agent_color(runtime_harness(runtime), t))
            .or_else(|| instance.map(|inst| agent_color(inst.harness, t)))
            .unwrap_or(t.palette.primary)
    } else {
        iced::Color::TRANSPARENT
    };
    let text_color = t.palette.foreground;
    let active_bg = iced::Color {
        a: 0.12,
        ..t.palette.primary
    };
    let hover_bg = iced::Color {
        a: 0.08,
        ..t.palette.foreground
    };
    let pressed_bg = iced::Color {
        a: 0.14,
        ..t.palette.foreground
    };
    let row_bg = if is_active {
        Some(iced::Background::Color(active_bg))
    } else {
        None
    };
    let on_press = match session.action.clone() {
        SessionAction::Select(index) => crate::app::Message::SelectSession(index),
        SessionAction::LoadRemote {
            agent_ref,
            source,
            session_id,
            cwd,
        } => crate::app::Message::LoadRemoteSession(agent_ref, source, session_id, cwd),
    };

    let is_renaming =
        matches!(&session.action, SessionAction::Select(index) if renaming_session == Some(*index));
    let title_control: Element<_> = if is_renaming {
        text_input("Session title", rename_buffer)
            .on_input(crate::app::Message::SessionRenameChanged)
            .on_submit(crate::app::Message::CommitSessionRename)
            .size(tokens.text_xs.size)
            .padding([2, 4])
            .width(Length::Fill)
            .into()
    } else {
        iced::widget::button(
            text(label)
                .color(text_color)
                .size(tokens.text_xs.size)
                .wrapping(iced::widget::text::Wrapping::None),
        )
        .on_press(on_press)
        .padding([2, 4])
        .width(Length::Fill)
        .style(move |_theme, status| {
            let background = match status {
                iced::widget::button::Status::Hovered => Some(iced::Background::Color(hover_bg)),
                iced::widget::button::Status::Pressed => Some(iced::Background::Color(pressed_bg)),
                _ => row_bg,
            };
            iced::widget::button::Style {
                background,
                text_color,
                ..iced::widget::button::Style::default()
            }
        })
        .into()
    };

    let pin_indicator: Element<_> = if session.pinned {
        text(char::from(LucideIcon::Pin).to_string())
            .font(iced::Font::with_name("lucide"))
            .size(11)
            .color(t.palette.muted_foreground)
            .into()
    } else {
        Space::new().width(0).height(0).into()
    };

    let unread_indicator: Element<_> = if session.unread {
        container(Space::new().width(6.0).height(6.0))
            .style({
                let color = t.palette.primary;
                move |_theme| {
                    iced::widget::container::Style::default()
                        .background(color)
                        .border(iced::border::rounded(6.0))
                }
            })
            .width(Length::Fixed(6.0))
            .height(Length::Fixed(6.0))
            .into()
    } else {
        Space::new().width(0).height(0).into()
    };

    let mut row_content = row![
        container(Space::new().width(3.0).height(Length::Fill))
            .style(move |_theme| { iced::widget::container::Style::default().background(accent) })
            .width(Length::Fixed(3.0))
            .height(Length::Fill),
        Space::new().width(6.0),
    ]
    .align_y(Alignment::Center);

    if let Some(icon_widget) = icon_widget {
        row_content = row_content.push(icon_widget).push(Space::new().width(6.0));
    }

    let row_content = row_content
        .push(
            container(title_control)
                .width(Length::Fill)
                .height(Length::Fixed(20.0))
                .clip(true),
        )
        .push(pin_indicator)
        .push(unread_indicator)
        .push(menu_btn);

    let row = container(row_content)
        .padding([2u16, tokens.space_1 as u16])
        .width(Length::Fill)
        .height(Length::Fixed(28.0))
        .clip(true);

    let _ = list_index;
    context_menu(
        row,
        build_session_menu_entries(session),
        ContextMenuProps::new()
            .size(ContextMenuContentSize::Size2)
            .width(250),
        t,
    )
}

// ── Footer ────────────────────────────────────────────────────────────────────

const SIDEBAR_FOOTER_BUTTON_HEIGHT: f32 = 36.0;
const SIDEBAR_FOOTER_FADE_HEIGHT: f32 = 18.0;

fn sidebar_footer_reserved_height(tokens: &DesignTokens) -> f32 {
    SIDEBAR_FOOTER_BUTTON_HEIGHT + tokens.space_1 * 2.0 + SIDEBAR_FOOTER_FADE_HEIGHT
}

fn opacity(mut color: Color, alpha: f32) -> Color {
    color.a = alpha;
    color
}

fn view_sidebar_footer_overlay<'a>(
    footer: Element<'a, crate::app::Message>,
    t: &'a Theme,
) -> Element<'a, crate::app::Message> {
    let glass_alpha = if is_dark_theme(t) { 0.76 } else { 0.84 };
    let glass = opacity(t.palette.background, glass_alpha);
    let glass_edge = opacity(t.palette.background, 0.0);
    let border = opacity(t.palette.border, if is_dark_theme(t) { 0.24 } else { 0.38 });

    let fade = iced::gradient::Linear::new(0.0)
        .add_stop(0.0, glass)
        .add_stop(1.0, glass_edge);

    let fade_band = container(Space::new().height(SIDEBAR_FOOTER_FADE_HEIGHT))
        .width(Length::Fill)
        .height(Length::Fixed(SIDEBAR_FOOTER_FADE_HEIGHT))
        .style(move |_theme| iced::widget::container::Style::from(fade));

    let panel = container(footer).width(Length::Fill).style(move |_theme| {
        iced::widget::container::Style::default()
            .background(glass)
            .border(iced::Border {
                color: border,
                width: 1.0,
                ..Default::default()
            })
            .shadow(Shadow {
                color: opacity(Color::BLACK, 0.28),
                offset: Vector::new(0.0, -8.0),
                blur_radius: 22.0,
            })
    });

    container(column![Space::new().height(Length::Fill), fade_band, panel].spacing(0))
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn view_sidebar_footer<'a>(
    instances: &'a [AgentInstance],
    tokens: &'a DesignTokens,
    t: &'a Theme,
) -> Element<'a, crate::app::Message> {
    let settings_icon = text(char::from(LucideIcon::Settings).to_string())
        .font(iced::Font::with_name("lucide"))
        .color(t.palette.foreground)
        .size(18);
    let label = text(sidebar_footer_label(instances))
        .size(14)
        .color(t.palette.foreground);
    let content = row![settings_icon, Space::new().width(tokens.space_2), label]
        .align_y(Alignment::Center)
        .width(Length::Fill);
    let settings_btn = button_content(
        content,
        Some(crate::app::Message::ToggleSettings),
        ButtonProps::new()
            .variant(ButtonVariant::Ghost)
            .size(ButtonSize::Size2)
            .radius(ButtonRadius::Large),
        t,
    )
    .width(Length::Fill);

    container(settings_btn)
        .padding([tokens.space_1 as u16, tokens.space_3 as u16])
        .width(Length::Fill)
        .into()
}

fn sidebar_footer_label(_instances: &[AgentInstance]) -> &'static str {
    "Settings"
}

#[cfg(test)]
mod tests;
