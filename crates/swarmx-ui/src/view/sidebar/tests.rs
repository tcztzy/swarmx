use crate::data::{RemoteAgentSessions, RemoteSessionSource, Session};
use crate::environment::{AgentRuntime, RemoteAgentRef};
use crate::harness::Harness;
use crate::instance::AgentInstance;
use crate::persistence::{SessionGrouping, SessionSortBy};

use super::{
    SessionAction, SidebarSessionEntry, collect_session_entries, group_indices,
    is_unknown_working_dir, relative_time, remote_session_title, session_deeplink,
    session_item_label, sidebar_footer_label, sort_within_group, time_bucket,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_session(id: &str, instance_id: &str, cwd: &str) -> Session {
    Session::new(id, instance_id, cwd, None)
}

fn make_instance(id: &str, label: &str) -> AgentInstance {
    let mut inst = AgentInstance::claude_seed("provider-1");
    inst.id = id.to_string();
    inst.label = label.to_string();
    inst
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

// ── relative_time ─────────────────────────────────────────────────────────────

#[test]
fn relative_time_seconds() {
    let ts = (now_secs() - 30).to_string();
    assert_eq!(relative_time(&ts), "1m");
}

#[test]
fn relative_time_hours() {
    let ts = (now_secs() - 7200).to_string();
    assert_eq!(relative_time(&ts), "2h");
}

#[test]
fn relative_time_days() {
    let ts = (now_secs() - 3 * 86400).to_string();
    assert_eq!(relative_time(&ts), "3d");
}

#[test]
fn relative_time_weeks() {
    let ts = (now_secs() - 14 * 86400).to_string();
    assert_eq!(relative_time(&ts), "2w");
}

#[test]
fn relative_time_99w_cap() {
    let ts = (now_secs() - 200 * 7 * 86400).to_string();
    assert_eq!(relative_time(&ts), "99w+");
}

// ── time_bucket ───────────────────────────────────────────────────────────────

#[test]
fn time_bucket_today() {
    assert_eq!(time_bucket(&now_secs().to_string()), "Today");
}

#[test]
fn time_bucket_yesterday() {
    let ts = (now_secs() - 86400).to_string();
    assert_eq!(time_bucket(&ts), "Yesterday");
}

#[test]
fn time_bucket_this_week() {
    let ts = (now_secs() - 3 * 86400).to_string();
    assert_eq!(time_bucket(&ts), "This Week");
}

#[test]
fn time_bucket_this_month() {
    let ts = (now_secs() - 14 * 86400).to_string();
    assert_eq!(time_bucket(&ts), "This Month");
}

#[test]
fn time_bucket_older() {
    let ts = (now_secs() - 40 * 86400).to_string();
    assert_eq!(time_bucket(&ts), "Older");
}

fn sidebar_entry(title: &str, cwd: &str, updated_at: &str) -> SidebarSessionEntry {
    SidebarSessionEntry {
        session_id: "sid".to_string(),
        copy_session_id: "sid".to_string(),
        agent_instance_id: "inst-1".to_string(),
        agent_runtime: None,
        working_dir: cwd.to_string(),
        title: Some(title.to_string()),
        first_message: None,
        created_at: updated_at.to_string(),
        updated_at: updated_at.to_string(),
        pinned: false,
        unread: false,
        action: SessionAction::Select(0),
    }
}

#[test]
fn sort_within_group_recency_parses_rfc3339() {
    let entries = vec![
        sidebar_entry("old", "/a", "100"),
        sidebar_entry("new", "/b", "1970-01-01T00:05:00Z"),
    ];
    let mut group = super::Group {
        name: "g".into(),
        session_indices: vec![0, 1],
    };
    sort_within_group(&mut group, &entries, SessionSortBy::Recency);
    assert_eq!(group.session_indices, vec![1, 0]);
}

#[test]
fn sort_within_group_keeps_pinned_first() {
    let unpinned_new = sidebar_entry("new", "/a", "300");
    let mut pinned_old = sidebar_entry("old pinned", "/b", "100");
    pinned_old.pinned = true;
    let entries = vec![unpinned_new, pinned_old];
    let mut group = super::Group {
        name: "g".into(),
        session_indices: vec![0, 1],
    };
    sort_within_group(&mut group, &entries, SessionSortBy::Recency);
    assert_eq!(group.session_indices, vec![1, 0]);
}

#[test]
fn collect_session_entries_omits_archived_local_sessions() {
    let mut visible = make_session("s1", "inst-1", "/a");
    let mut archived = make_session("s2", "inst-1", "/b");
    visible.title = Some("visible".into());
    archived.archived = true;
    let entries = collect_session_entries(&[visible, archived], &[]);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].session_id, "s1");
}

#[test]
fn collect_session_entries_keeps_discovered_runtime_source() {
    let remote = RemoteAgentSessions {
        agent_name: "Codex".into(),
        agent_ref: RemoteAgentRef::Runtime(AgentRuntime::CodexAcp),
        source: RemoteSessionSource::Acp,
        sessions: vec![
            agent_client_protocol::schema::SessionInfo::new("codex-session", "/work/swarmx")
                .title("Review diff")
                .updated_at("123"),
        ],
    };

    let entries = collect_session_entries(&[], &[remote]);

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].agent_runtime, Some(AgentRuntime::CodexAcp));
    assert_eq!(entries[0].agent_instance_id, AgentRuntime::CodexAcp.id());
    assert_eq!(
        entries[0].action,
        SessionAction::LoadRemote {
            agent_ref: RemoteAgentRef::Runtime(AgentRuntime::CodexAcp),
            source: RemoteSessionSource::Acp,
            session_id: "codex-session".into(),
            cwd: "/work/swarmx".into(),
        }
    );
}

#[test]
fn collect_session_entries_preserves_unknown_hermes_native_cwd() {
    let remote = RemoteAgentSessions {
        agent_name: "Hermes".into(),
        agent_ref: RemoteAgentRef::Runtime(AgentRuntime::Hermes),
        source: RemoteSessionSource::HermesNative,
        sessions: vec![
            agent_client_protocol::schema::SessionInfo::new(
                "hermes-session",
                std::path::PathBuf::new(),
            )
            .title("Recovered Hermes chat")
            .updated_at("123"),
        ],
    };

    let entries = collect_session_entries(&[], &[remote]);

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].working_dir, "");
    assert_eq!(
        entries[0].action,
        SessionAction::LoadRemote {
            agent_ref: RemoteAgentRef::Runtime(AgentRuntime::Hermes),
            source: RemoteSessionSource::HermesNative,
            session_id: "hermes-session".into(),
            cwd: "".into(),
        }
    );
}

#[test]
fn session_deeplink_percent_encodes_cwd() {
    assert_eq!(
        session_deeplink("abc-123", "/work/Swarm X"),
        "swarmx://session/abc-123?cwd=%2Fwork%2FSwarm%20X"
    );
}

// ── group_indices ─────────────────────────────────────────────────────────────

#[test]
fn group_indices_project_uses_leaf_directory_name() {
    let instances = vec![make_instance("inst-1", "Claude")];
    let sessions = vec![
        make_session("s1", "inst-1", "/home/user/projects/swarmx"),
        make_session("s2", "inst-1", "/home/user/projects/other"),
    ];
    let indices = (0..sessions.len()).collect::<Vec<_>>();
    let groups = group_indices(&indices, &sessions, SessionGrouping::Project, &instances);
    assert_eq!(groups.len(), 2);
    assert!(groups.iter().any(|g| g.name == "swarmx"));
    assert!(groups.iter().any(|g| g.name == "other"));
}

#[test]
fn group_indices_project_reuses_leaf_name_for_same_path() {
    let instances = vec![make_instance("inst-1", "Claude")];
    let sessions = vec![
        make_session("s1", "inst-1", "/home/user/GitHub/Panno"),
        make_session("s2", "inst-1", "/home/user/GitHub/Panno/"),
    ];
    let indices = (0..sessions.len()).collect::<Vec<_>>();
    let groups = group_indices(&indices, &sessions, SessionGrouping::Project, &instances);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].name, "Panno");
}

#[test]
fn group_indices_project_uses_unknown_project_for_missing_cwd() {
    let instances = vec![make_instance("inst-1", "Claude")];
    let sessions = vec![
        make_session("s1", "inst-1", ""),
        make_session("s2", "inst-1", "."),
    ];
    let indices = (0..sessions.len()).collect::<Vec<_>>();
    let groups = group_indices(&indices, &sessions, SessionGrouping::Project, &instances);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].name, "Unknown project");
}

#[test]
fn group_indices_project_disambiguates_duplicate_leaf_names() {
    let instances = vec![make_instance("inst-1", "Claude")];
    let sessions = vec![
        make_session("s1", "inst-1", "/work/github/Panno"),
        make_session("s2", "inst-1", "/work/archive/Panno"),
        make_session("s3", "inst-1", "/work/github/SwarmX"),
    ];
    let indices = (0..sessions.len()).collect::<Vec<_>>();
    let groups = group_indices(&indices, &sessions, SessionGrouping::Project, &instances);
    assert_eq!(groups.len(), 3);
    assert!(
        groups
            .iter()
            .any(|g| g.name == "Panno (/work/github/Panno)")
    );
    assert!(
        groups
            .iter()
            .any(|g| g.name == "Panno (/work/archive/Panno)")
    );
    assert!(groups.iter().any(|g| g.name == "SwarmX"));
}

#[test]
fn group_indices_harness_uses_instance_harness_labels() {
    let mut claude = make_instance("claude", "Claude");
    claude.harness = Harness::ClaudeCode;
    let mut codex = make_instance("codex", "Codex");
    codex.harness = Harness::Codex;
    let instances = vec![claude, codex];
    let sessions = vec![
        make_session("s1", "claude", "/a"),
        make_session("s2", "codex", "/b"),
        make_session("s3", "missing", "/c"),
    ];
    let indices = (0..sessions.len()).collect::<Vec<_>>();
    let groups = group_indices(&indices, &sessions, SessionGrouping::Harness, &instances);
    assert_eq!(groups.len(), 3);
    assert!(groups.iter().any(|g| g.name == "Claude Code"));
    assert!(groups.iter().any(|g| g.name == "Codex"));
    assert!(groups.iter().any(|g| g.name == "Unknown Harness"));
}

#[test]
fn group_indices_harness_uses_discovered_runtime_label() {
    let entry = SidebarSessionEntry {
        session_id: "sid".to_string(),
        copy_session_id: "sid".to_string(),
        agent_instance_id: AgentRuntime::CodexAcp.id().to_string(),
        agent_runtime: Some(AgentRuntime::CodexAcp),
        working_dir: "/work/swarmx".to_string(),
        title: Some("Codex task".to_string()),
        first_message: None,
        created_at: "100".to_string(),
        updated_at: "100".to_string(),
        pinned: false,
        unread: false,
        action: SessionAction::LoadRemote {
            agent_ref: RemoteAgentRef::Runtime(AgentRuntime::CodexAcp),
            source: RemoteSessionSource::Acp,
            session_id: "sid".to_string(),
            cwd: "/work/swarmx".to_string(),
        },
    };
    let groups = group_indices(&[0], &[entry], SessionGrouping::Harness, &[]);

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].name, "Codex");
}

#[test]
fn group_indices_none_yields_single_unnamed_group() {
    let instances = vec![make_instance("inst-1", "Claude")];
    let sessions = vec![
        make_session("s1", "inst-1", "/a"),
        make_session("s2", "inst-1", "/b"),
    ];
    let indices = (0..sessions.len()).collect::<Vec<_>>();
    let groups = group_indices(&indices, &sessions, SessionGrouping::None, &instances);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].name, "");
    assert_eq!(groups[0].session_indices.len(), 2);
}

// ── sort_within_group ─────────────────────────────────────────────────────────

#[test]
fn sort_within_group_alphabetically_uses_title() {
    let mut s1 = make_session("s1", "inst-1", "/a");
    s1.title = Some("Zebra".into());
    let mut s2 = make_session("s2", "inst-1", "/b");
    s2.title = Some("Aardvark".into());
    let mut s3 = make_session("s3", "inst-1", "/c");
    s3.title = Some("Monkey".into());
    let sessions = vec![s1, s2, s3];
    let mut group = super::Group {
        name: "g".into(),
        session_indices: vec![0, 1, 2],
    };
    sort_within_group(&mut group, &sessions, SessionSortBy::Alphabetically);
    assert_eq!(group.session_indices, vec![1, 2, 0]);
}

#[test]
fn session_item_label_uses_session_title_without_agent_name() {
    let entry = sidebar_entry("Review API changes", "/work/swarmx", "100");
    assert_eq!(session_item_label(&entry), "Review API changes");
}

#[test]
fn sidebar_footer_label_is_settings() {
    let instances = vec![make_instance("inst-1", "I")];
    assert_eq!(sidebar_footer_label(&instances), "Settings");
}

#[test]
fn remote_session_title_falls_back_for_unknown_cwd() {
    assert!(is_unknown_working_dir(""));
    assert!(is_unknown_working_dir("."));
    assert_eq!(remote_session_title(None, ""), "Untitled session");
}

#[test]
fn sort_within_group_recency_puts_newest_first() {
    let mut s1 = make_session("s1", "inst-1", "/a");
    s1.updated_at = "100".into();
    let mut s2 = make_session("s2", "inst-1", "/b");
    s2.updated_at = "300".into();
    let mut s3 = make_session("s3", "inst-1", "/c");
    s3.updated_at = "200".into();
    let sessions = vec![s1, s2, s3];
    let mut group = super::Group {
        name: "g".into(),
        session_indices: vec![0, 1, 2],
    };
    sort_within_group(&mut group, &sessions, SessionSortBy::Recency);
    assert_eq!(group.session_indices, vec![1, 2, 0]);
}

#[test]
fn sort_within_group_created_puts_newest_first() {
    let mut s1 = make_session("s1", "inst-1", "/a");
    s1.created_at = "100".into();
    let mut s2 = make_session("s2", "inst-1", "/b");
    s2.created_at = "300".into();
    let sessions = vec![s1, s2];
    let mut group = super::Group {
        name: "g".into(),
        session_indices: vec![0, 1],
    };
    sort_within_group(&mut group, &sessions, SessionSortBy::Created);
    assert_eq!(group.session_indices, vec![1, 0]);
}
