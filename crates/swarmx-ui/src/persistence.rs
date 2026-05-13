use crate::theme::ThemePreference;
use crate::tokens::Density;
use std::path::{Path, PathBuf};

// ── Session Grouping ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionGrouping {
    Project,
    Harness,
    #[serde(alias = "environment")]
    #[default]
    Date,
    None,
}

impl SessionGrouping {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Project => "Project",
            Self::Harness => "Harness",
            Self::Date => "Date",
            Self::None => "None",
        }
    }

    pub fn organize_label(&self) -> &'static str {
        match self {
            Self::Project => "By project",
            Self::Harness => "Harness",
            Self::Date => "Recent projects",
            Self::None => "Chronological list",
        }
    }
}

// ── Session Sort ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionSortBy {
    Alphabetically,
    Created,
    #[default]
    Recency,
}

impl SessionSortBy {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Alphabetically => "Alphabetically",
            Self::Created => "Created time",
            Self::Recency => "Recency",
        }
    }
}

// ── App Settings ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AppSettings {
    #[serde(default)]
    pub theme: ThemePreference,
    #[serde(default)]
    pub density: Density,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_cwd: Option<String>,
    #[serde(default)]
    pub session_grouping: SessionGrouping,
    #[serde(default)]
    pub session_sort_by: SessionSortBy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_filter: Option<String>,
}

pub fn settings_path() -> PathBuf {
    let home = dirs_next::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".swarmx").join("settings.json")
}

pub fn load_settings() -> AppSettings {
    load_settings_from(&settings_path())
}

pub fn save_settings(s: &AppSettings) -> std::io::Result<()> {
    save_settings_to(&settings_path(), s)
}

pub fn load_settings_from(path: &Path) -> AppSettings {
    match std::fs::read_to_string(path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
        Err(_) => AppSettings::default(),
    }
}

pub fn save_settings_to(path: &Path, s: &AppSettings) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(s)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

#[cfg(test)]
mod tests;
