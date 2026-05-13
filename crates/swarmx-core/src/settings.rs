//! Settings and configuration.

use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};

fn parse_env_file(path: &Path) -> HashMap<String, String> {
    let mut vars = HashMap::new();
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return vars,
    };
    for raw_line in contents.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = if let Some(stripped) = line.strip_prefix("export ") {
            stripped.trim()
        } else {
            line
        };
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim();
            let value = value.trim().trim_matches('"').trim_matches('\'');
            if !key.is_empty() {
                vars.insert(key.to_string(), value.to_string());
            }
        }
    }
    vars
}

/// Application settings.
#[derive(Debug, Clone)]
pub struct Settings {
    pub openai_base_url: String,
    pub openai_api_key: Option<String>,
    pub openai_model: String,
    pub agents_md: Vec<PathBuf>,
}

impl Default for Settings {
    fn default() -> Self {
        let env_vars = parse_env_file(Path::new(".env"));
        let env_or = |key: &str, default: &str| -> String {
            env::var(key)
                .ok()
                .or_else(|| env_vars.get(key).cloned())
                .unwrap_or_else(|| default.to_string())
        };
        Self {
            openai_base_url: env_or("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_api_key: env::var("OPENAI_API_KEY")
                .ok()
                .or_else(|| env_vars.get("OPENAI_API_KEY").cloned()),
            openai_model: env_or("OPENAI_MODEL", "gpt-oss:20b"),
            agents_md: vec![PathBuf::from("AGENTS.md")],
        }
    }
}

impl Settings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_agents_md_content(&self) -> String {
        let mut contents = Vec::new();
        for path in &self.agents_md {
            if path.exists() {
                if let Ok(text) = std::fs::read_to_string(path) {
                    contents.push(format!(
                        "```markdown title=\"{}\"\n{}\n```",
                        path.display(),
                        text.trim()
                    ));
                }
            }
        }
        contents.join("\n\n")
    }
}

pub static SETTINGS: std::sync::OnceLock<Settings> = std::sync::OnceLock::new();

pub fn get_settings() -> &'static Settings {
    SETTINGS.get_or_init(Settings::new)
}

#[cfg(test)]
mod tests;
