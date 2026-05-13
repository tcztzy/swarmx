use super::*;
use crate::theme::ThemePreference;
use crate::tokens::Density;

#[test]
fn app_settings_default() {
    let s = AppSettings::default();
    assert_eq!(s.theme, ThemePreference::Dark);
    assert_eq!(s.density, Density::Comfortable);
    assert!(s.default_cwd.is_none());
    assert_eq!(s.session_grouping, SessionGrouping::Date);
    assert_eq!(s.session_sort_by, SessionSortBy::Recency);
    assert!(s.project_filter.is_none());
}

#[test]
fn app_settings_serde_round_trip() {
    let s = AppSettings {
        theme: ThemePreference::CatppuccinMocha,
        density: Density::Compact,
        default_cwd: Some("/tmp/work".to_string()),
        session_grouping: SessionGrouping::Project,
        session_sort_by: SessionSortBy::Alphabetically,
        project_filter: Some("/home/me/swarmx".to_string()),
    };
    let json = serde_json::to_string(&s).unwrap();
    let back: AppSettings = serde_json::from_str(&json).unwrap();
    assert_eq!(s, back);
}

#[test]
fn app_settings_load_returns_default_when_missing() {
    let dir = tempfile_dir();
    let path = dir.join("missing.json");
    let loaded = load_settings_from(&path);
    assert_eq!(loaded, AppSettings::default());
}

#[test]
fn app_settings_save_then_load() {
    let dir = tempfile_dir();
    let path = dir.join("settings.json");
    let s = AppSettings {
        theme: ThemePreference::TokyoNight,
        density: Density::Compact,
        default_cwd: Some("/home/me".to_string()),
        session_grouping: SessionGrouping::Harness,
        session_sort_by: SessionSortBy::Created,
        project_filter: None,
    };
    save_settings_to(&path, &s).unwrap();
    let back = load_settings_from(&path);
    assert_eq!(back, s);
}

#[test]
fn app_settings_ignores_unknown_fields() {
    // Old settings.json fields should load cleanly and legacy environment grouping maps to Date.
    let legacy = r#"{
        "theme":"dark",
        "density":"comfortable",
        "sidebar_grouping":"agent",
        "session_grouping":"environment",
        "last_activity":"three_days",
        "environment":"all"
    }"#;
    let s: AppSettings = serde_json::from_str(legacy).unwrap();
    assert_eq!(s.session_grouping, SessionGrouping::Date);
}

fn tempfile_dir() -> std::path::PathBuf {
    let base = std::env::temp_dir().join(format!(
        "swarmx-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&base).unwrap();
    base
}
