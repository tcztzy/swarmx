use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, SecondsFormat, Utc};
use git2::{Oid, Repository, Sort};
use llm_wiki::engine::WikiEngine as MemoryEngine;
use llm_wiki::{git, ops, spaces};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use similar::TextDiff;
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;
use walkdir::WalkDir;

pub const PROTOCOL_VERSION: u32 = 1;
pub const RUNTIME_VERSION: &str = env!("CARGO_PKG_VERSION");
const MEMORY_SPACE: &str = "swarmx";
const PAGES_DIRECTORY: &str = "pages";
const MAX_PAGES: usize = 2_048;
const MAX_PAGE_CHARS: usize = 64_000;
const MAX_TOTAL_CHARS: usize = 8_000_000;
const MAX_ALIASES: usize = 32;
const MAX_LINK_MARKERS_PER_PAGE: usize = 2_048;
const MAX_LINK_MARKERS_TOTAL: usize = 10_000;
const MAX_SEARCH_RESULTS: usize = 50;
const MAX_HISTORY_RESULTS: usize = 100;
const MAX_DIFF_CHARS: usize = 128_000;
const MAX_GLOBAL_MEMORY_CHARS: usize = 4_000;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum GlobalMemoryTarget {
    User,
    Memory,
}

impl GlobalMemoryTarget {
    fn file_name(self) -> &'static str {
        match self {
            Self::User => "USER.md",
            Self::Memory => "MEMORY.md",
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct GlobalMemoryFile {
    target: GlobalMemoryTarget,
    file_name: &'static str,
    content: Option<String>,
    revision: u64,
    updated_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Page {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub content: String,
    pub revision: u64,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PageFile {
    id: String,
    title: String,
    #[serde(default)]
    aliases: Vec<String>,
    revision: u64,
    created_at: String,
    updated_at: String,
    #[serde(default)]
    deleted: bool,
    #[serde(rename = "type")]
    page_type: String,
    status: String,
    last_updated: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct PageSummary {
    id: String,
    title: String,
    aliases: Vec<String>,
    revision: u64,
    created_at: String,
    updated_at: String,
}

#[derive(Debug, Clone)]
struct StoredPage {
    page: Page,
    deleted: bool,
}

#[derive(Debug, Deserialize)]
#[serde(
    tag = "operation",
    rename_all = "snake_case",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
enum Request {
    List {
        protocol_version: u32,
    },
    Get {
        protocol_version: u32,
        id: String,
    },
    Search {
        protocol_version: u32,
        query: String,
        #[serde(default = "default_search_limit")]
        limit: usize,
    },
    Snapshot {
        protocol_version: u32,
    },
    GlobalGet {
        protocol_version: u32,
    },
    GlobalSave {
        protocol_version: u32,
        target: GlobalMemoryTarget,
        expected_revision: u64,
        content: String,
    },
    GlobalForget {
        protocol_version: u32,
        target: GlobalMemoryTarget,
        expected_revision: u64,
    },
    Create {
        protocol_version: u32,
        title: String,
        #[serde(default)]
        aliases: Vec<String>,
        content: String,
    },
    Update {
        protocol_version: u32,
        id: String,
        expected_revision: u64,
        title: Option<String>,
        aliases: Option<Vec<String>>,
        content: Option<String>,
    },
    Delete {
        protocol_version: u32,
        id: String,
        expected_revision: u64,
    },
    History {
        protocol_version: u32,
        id: String,
        #[serde(default = "default_history_limit")]
        limit: usize,
    },
    GetVersion {
        protocol_version: u32,
        id: String,
        version: String,
    },
    Diff {
        protocol_version: u32,
        id: String,
        from_version: String,
        to_version: String,
    },
    Restore {
        protocol_version: u32,
        id: String,
        expected_revision: u64,
        version: String,
    },
}

impl Request {
    fn protocol_version(&self) -> u32 {
        match self {
            Self::List { protocol_version }
            | Self::Get {
                protocol_version, ..
            }
            | Self::Search {
                protocol_version, ..
            }
            | Self::Snapshot {
                protocol_version, ..
            }
            | Self::GlobalGet {
                protocol_version, ..
            }
            | Self::GlobalSave {
                protocol_version, ..
            }
            | Self::GlobalForget {
                protocol_version, ..
            }
            | Self::Create {
                protocol_version, ..
            }
            | Self::Update {
                protocol_version, ..
            }
            | Self::Delete {
                protocol_version, ..
            }
            | Self::History {
                protocol_version, ..
            }
            | Self::GetVersion {
                protocol_version, ..
            }
            | Self::Diff {
                protocol_version, ..
            }
            | Self::Restore {
                protocol_version, ..
            } => *protocol_version,
        }
    }

    fn operation(&self) -> &'static str {
        match self {
            Self::List { .. } => "list",
            Self::Get { .. } => "get",
            Self::Search { .. } => "search",
            Self::Snapshot { .. } => "snapshot",
            Self::GlobalGet { .. } => "global_get",
            Self::GlobalSave { .. } => "global_save",
            Self::GlobalForget { .. } => "global_forget",
            Self::Create { .. } => "create",
            Self::Update { .. } => "update",
            Self::Delete { .. } => "delete",
            Self::History { .. } => "history",
            Self::GetVersion { .. } => "get_version",
            Self::Diff { .. } => "diff",
            Self::Restore { .. } => "restore",
        }
    }
}

#[derive(Debug)]
struct ServiceError {
    code: &'static str,
    safe_message: &'static str,
}

impl ServiceError {
    fn invalid() -> Self {
        Self {
            code: "invalid_input",
            safe_message: "Memory runtime input is invalid.",
        }
    }

    fn not_found() -> Self {
        Self {
            code: "not_found",
            safe_message: "Memory page or version was not found.",
        }
    }

    fn conflict() -> Self {
        Self {
            code: "conflict",
            safe_message: "Memory revision conflict.",
        }
    }

    fn corrupt() -> Self {
        Self {
            code: "corrupt",
            safe_message: "Memory authority is corrupt.",
        }
    }

    fn internal() -> Self {
        Self {
            code: "internal",
            safe_message: "Memory runtime internal failure.",
        }
    }
}

pub struct MemoryService {
    root: PathBuf,
    pages_root: PathBuf,
    engine: MemoryEngine,
}

impl MemoryService {
    pub fn open(root: &Path) -> Result<Self> {
        let root = root.to_path_buf();
        fs::create_dir_all(&root)
            .with_context(|| format!("failed to create Memory root {}", root.display()))?;
        set_directory_permissions(&root)?;
        ensure_runtime_gitignore(&root)?;

        let state_root = root.join(".runtime");
        fs::create_dir_all(&state_root)?;
        set_directory_permissions(&state_root)?;
        let config_path = state_root.join("config.toml");
        spaces::create(
            &root,
            MEMORY_SPACE,
            Some("SwarmX Memory"),
            false,
            true,
            &config_path,
            Some(PAGES_DIRECTORY),
        )?;
        let engine = MemoryEngine::build(&config_path)?;
        engine.rebuild_index(MEMORY_SPACE)?;
        let pages_root = root.join(PAGES_DIRECTORY);
        set_directory_permissions(&pages_root)?;
        Ok(Self {
            root,
            pages_root,
            engine,
        })
    }

    pub fn handle(&mut self, input: Value) -> Value {
        let fallback_operation = input
            .get("operation")
            .and_then(Value::as_str)
            .unwrap_or("list")
            .to_owned();
        let request: Request = match serde_json::from_value(input) {
            Ok(request) => request,
            Err(_) => return error_response(&fallback_operation, ServiceError::invalid()),
        };
        let operation = request.operation();
        if request.protocol_version() != PROTOCOL_VERSION {
            return error_response(operation, ServiceError::invalid());
        }
        match self.execute(request) {
            Ok(result) => success_response(operation, result),
            Err(error) => error_response(operation, error),
        }
    }

    fn execute(&mut self, request: Request) -> std::result::Result<Value, ServiceError> {
        match request {
            Request::List { .. } => self.list_result(),
            Request::Get { id, .. } => self.get_result(&id),
            Request::Search { query, limit, .. } => self.search_result(&query, limit),
            Request::Snapshot { .. } => self.snapshot_result(),
            Request::GlobalGet { .. } => self.global_get_result(),
            Request::GlobalSave {
                target,
                expected_revision,
                content,
                ..
            } => self.global_save_result(target, expected_revision, content),
            Request::GlobalForget {
                target,
                expected_revision,
                ..
            } => self.global_forget_result(target, expected_revision),
            Request::Create {
                title,
                aliases,
                content,
                ..
            } => self.create_result(title, aliases, content),
            Request::Update {
                id,
                expected_revision,
                title,
                aliases,
                content,
                ..
            } => self.update_result(&id, expected_revision, title, aliases, content),
            Request::Delete {
                id,
                expected_revision,
                ..
            } => self.delete_result(&id, expected_revision),
            Request::History { id, limit, .. } => self.history_result(&id, limit),
            Request::GetVersion { id, version, .. } => self.get_version_result(&id, &version),
            Request::Diff {
                id,
                from_version,
                to_version,
                ..
            } => self.diff_result(&id, &from_version, &to_version),
            Request::Restore {
                id,
                expected_revision,
                version,
                ..
            } => self.restore_result(&id, expected_revision, &version),
        }
    }

    fn list_result(&self) -> std::result::Result<Value, ServiceError> {
        let pages = self.active_pages()?;
        let summaries: Vec<PageSummary> = pages.iter().map(page_summary).collect();
        Ok(json!({ "pages": summaries }))
    }

    fn get_result(&self, id: &str) -> std::result::Result<Value, ServiceError> {
        validate_id(id)?;
        let page = self.read_current(id)?;
        Ok(json!({
            "page": page.filter(|stored| !stored.deleted).map(|stored| stored.page)
        }))
    }

    fn snapshot_result(&self) -> std::result::Result<Value, ServiceError> {
        let pages = self.active_pages()?;
        Ok(json!({ "generation": self.generation()?, "pages": pages }))
    }

    fn global_get_result(&mut self) -> std::result::Result<Value, ServiceError> {
        self.reconcile_global_file(GlobalMemoryTarget::User)?;
        self.reconcile_global_file(GlobalMemoryTarget::Memory)?;
        Ok(json!({
            "user": self.global_file(GlobalMemoryTarget::User)?,
            "memory": self.global_file(GlobalMemoryTarget::Memory)?
        }))
    }

    fn global_save_result(
        &mut self,
        target: GlobalMemoryTarget,
        expected_revision: u64,
        content: String,
    ) -> std::result::Result<Value, ServiceError> {
        validate_global_content(&content)?;
        self.reconcile_global_file(target)?;
        let current = self.global_file(target)?;
        if current.revision != expected_revision {
            return Err(ServiceError::conflict());
        }
        if current.content.as_deref() == Some(content.as_str()) {
            return Err(ServiceError::conflict());
        }
        self.write_global_file(target, &content)?;
        let version = self.commit_global_file(target, "global_save")?;
        Ok(json!({ "file": self.global_file(target)?, "version": version }))
    }

    fn global_forget_result(
        &mut self,
        target: GlobalMemoryTarget,
        expected_revision: u64,
    ) -> std::result::Result<Value, ServiceError> {
        self.reconcile_global_file(target)?;
        let current = self.global_file(target)?;
        if current.revision != expected_revision {
            return Err(ServiceError::conflict());
        }
        if current.content.is_none() {
            return Err(ServiceError::not_found());
        }
        fs::remove_file(self.global_path(target)).map_err(|_| ServiceError::internal())?;
        let version = self.commit_global_file(target, "global_forget")?;
        Ok(json!({ "file": self.global_file(target)?, "version": version }))
    }

    fn create_result(
        &mut self,
        title: String,
        aliases: Vec<String>,
        content: String,
    ) -> std::result::Result<Value, ServiceError> {
        let id = format!("mem_{}", Uuid::new_v4().simple());
        let timestamp = now();
        let page = Page {
            id,
            title,
            aliases,
            content,
            revision: 1,
            created_at: timestamp.clone(),
            updated_at: timestamp,
        };
        self.validate_candidate(&page, None)?;
        self.write_stored(&StoredPage {
            page: page.clone(),
            deleted: false,
        })?;
        let version = self.commit_page(&page.id, "create")?;
        Ok(json!({ "page": page, "version": version }))
    }

    fn update_result(
        &mut self,
        id: &str,
        expected_revision: u64,
        title: Option<String>,
        aliases: Option<Vec<String>>,
        content: Option<String>,
    ) -> std::result::Result<Value, ServiceError> {
        validate_id(id)?;
        if title.is_none() && aliases.is_none() && content.is_none() {
            return Err(ServiceError::invalid());
        }
        let current = self
            .read_current(id)?
            .filter(|page| !page.deleted)
            .ok_or_else(ServiceError::not_found)?;
        if current.page.revision != expected_revision {
            return Err(ServiceError::conflict());
        }
        let page = Page {
            id: current.page.id.clone(),
            title: title.unwrap_or(current.page.title.clone()),
            aliases: aliases.unwrap_or(current.page.aliases.clone()),
            content: content.unwrap_or(current.page.content.clone()),
            revision: current.page.revision + 1,
            created_at: current.page.created_at.clone(),
            updated_at: now(),
        };
        if page.title == current.page.title
            && page.aliases == current.page.aliases
            && page.content == current.page.content
        {
            return Err(ServiceError::conflict());
        }
        self.validate_candidate(&page, Some(id))?;
        self.write_stored(&StoredPage {
            page: page.clone(),
            deleted: false,
        })?;
        let version = self.commit_page(id, "update")?;
        Ok(json!({ "page": page, "version": version }))
    }

    fn delete_result(
        &mut self,
        id: &str,
        expected_revision: u64,
    ) -> std::result::Result<Value, ServiceError> {
        validate_id(id)?;
        let current = self
            .read_current(id)?
            .filter(|page| !page.deleted)
            .ok_or_else(ServiceError::not_found)?;
        if current.page.revision != expected_revision {
            return Err(ServiceError::conflict());
        }
        let page = Page {
            content: String::new(),
            revision: current.page.revision + 1,
            updated_at: now(),
            ..current.page
        };
        self.write_stored(&StoredPage {
            page: page.clone(),
            deleted: true,
        })?;
        let version = self.commit_page(id, "delete")?;
        Ok(json!({ "page": page, "version": version }))
    }

    fn search_result(&self, query: &str, limit: usize) -> std::result::Result<Value, ServiceError> {
        let query = query.trim();
        if query.is_empty()
            || query.chars().count() > 256
            || !(1..=MAX_SEARCH_RESULTS).contains(&limit)
        {
            return Err(ServiceError::invalid());
        }
        let active = self.active_pages()?;
        let by_id: BTreeMap<&str, &Page> =
            active.iter().map(|page| (page.id.as_str(), page)).collect();
        let engine = self
            .engine
            .state
            .read()
            .map_err(|_| ServiceError::internal())?;
        let result = ops::search(
            &engine,
            MEMORY_SPACE,
            &ops::SearchParams {
                query,
                type_filter: Some("swarmx_memory"),
                no_excerpt: true,
                top_k: Some(limit),
                include_sections: false,
                cross_wiki: false,
            },
        )
        .map_err(|_| ServiceError::invalid())?;

        let mut pages = Vec::new();
        let mut seen = HashSet::new();
        for result in result.results {
            let id = result.slug.rsplit('/').next().unwrap_or(&result.slug);
            if let Some(page) = by_id.get(id) {
                seen.insert(page.id.clone());
                pages.push((*page).clone());
            }
        }
        let normalized_query = normalize_name(query);
        for page in &active {
            if pages.len() >= limit {
                break;
            }
            if seen.contains(&page.id) {
                continue;
            }
            if normalize_name(&page.title).contains(&normalized_query)
                || page
                    .aliases
                    .iter()
                    .any(|alias| normalize_name(alias).contains(&normalized_query))
                || normalize_name(&page.content).contains(&normalized_query)
            {
                seen.insert(page.id.clone());
                pages.push(page.clone());
            }
        }
        pages.truncate(limit);
        Ok(json!({ "pages": pages }))
    }

    fn history_result(&self, id: &str, limit: usize) -> std::result::Result<Value, ServiceError> {
        validate_id(id)?;
        if !(1..=MAX_HISTORY_RESULTS).contains(&limit) {
            return Err(ServiceError::invalid());
        }
        let versions = self.history(id, limit)?;
        Ok(json!({ "versions": versions }))
    }

    fn get_version_result(
        &self,
        id: &str,
        version: &str,
    ) -> std::result::Result<Value, ServiceError> {
        validate_id(id)?;
        let oid = parse_version(version)?;
        let repo = self.repository()?;
        let commit = repo
            .find_commit(oid)
            .map_err(|_| ServiceError::not_found())?;
        let stored = page_at_commit(&repo, &commit, &page_relative_path(id))?
            .ok_or_else(ServiceError::not_found)?;
        if stored.page.id != id {
            return Err(ServiceError::corrupt());
        }
        Ok(json!({
            "version": {
                "version": commit.id().to_string(),
                "revision": stored.page.revision,
                "operation": commit_operation(commit.message().unwrap_or_default()),
                "committedAt": commit_timestamp(&commit)?,
                "page": stored.page,
                "deleted": stored.deleted
            }
        }))
    }

    fn diff_result(
        &self,
        id: &str,
        from_version: &str,
        to_version: &str,
    ) -> std::result::Result<Value, ServiceError> {
        validate_id(id)?;
        if from_version == to_version {
            return Err(ServiceError::invalid());
        }
        let repo = self.repository()?;
        let from_commit = repo
            .find_commit(parse_version(from_version)?)
            .map_err(|_| ServiceError::not_found())?;
        let to_commit = repo
            .find_commit(parse_version(to_version)?)
            .map_err(|_| ServiceError::not_found())?;
        let relative = page_relative_path(id);
        let from = raw_page_at_commit(&repo, &from_commit, &relative)?
            .ok_or_else(ServiceError::not_found)?;
        let to = raw_page_at_commit(&repo, &to_commit, &relative)?
            .ok_or_else(ServiceError::not_found)?;
        let raw = TextDiff::from_lines(&from, &to)
            .unified_diff()
            .header(from_version, to_version)
            .to_string();
        let (unified_diff, truncated) = truncate_chars(raw, MAX_DIFF_CHARS);
        Ok(json!({
            "diff": {
                "id": id,
                "fromVersion": from_version,
                "toVersion": to_version,
                "unifiedDiff": unified_diff,
                "truncated": truncated
            }
        }))
    }

    fn restore_result(
        &mut self,
        id: &str,
        expected_revision: u64,
        version: &str,
    ) -> std::result::Result<Value, ServiceError> {
        validate_id(id)?;
        let current = self.read_current(id)?.ok_or_else(ServiceError::not_found)?;
        if current.page.revision != expected_revision {
            return Err(ServiceError::conflict());
        }
        let target = {
            let repo = self.repository()?;
            let commit = repo
                .find_commit(parse_version(version)?)
                .map_err(|_| ServiceError::not_found())?;
            page_at_commit(&repo, &commit, &page_relative_path(id))?
                .filter(|page| !page.deleted)
                .ok_or_else(ServiceError::not_found)?
        };
        let page = Page {
            id: id.to_owned(),
            title: target.page.title,
            aliases: target.page.aliases,
            content: target.page.content,
            revision: current.page.revision + 1,
            created_at: current.page.created_at,
            updated_at: now(),
        };
        self.validate_candidate(&page, Some(id))?;
        self.write_stored(&StoredPage {
            page: page.clone(),
            deleted: false,
        })?;
        let version = self.commit_page(id, "restore")?;
        Ok(json!({ "page": page, "version": version }))
    }

    fn active_pages(&self) -> std::result::Result<Vec<Page>, ServiceError> {
        let mut pages: Vec<Page> = self
            .stored_pages()?
            .into_iter()
            .filter(|stored| !stored.deleted)
            .map(|stored| stored.page)
            .collect();
        pages.sort_by(|left, right| left.title.cmp(&right.title).then(left.id.cmp(&right.id)));
        Ok(pages)
    }

    fn stored_pages(&self) -> std::result::Result<Vec<StoredPage>, ServiceError> {
        let mut pages = Vec::new();
        for entry in WalkDir::new(&self.pages_root)
            .min_depth(1)
            .max_depth(1)
            .into_iter()
        {
            let entry = entry.map_err(|_| ServiceError::corrupt())?;
            if !entry.file_type().is_file()
                || entry.path().extension().and_then(|value| value.to_str()) != Some("md")
            {
                continue;
            }
            let raw = fs::read_to_string(entry.path()).map_err(|_| ServiceError::corrupt())?;
            pages.push(parse_stored_page(&raw)?);
        }
        if pages.len() > MAX_PAGES {
            return Err(ServiceError::corrupt());
        }
        let active: Vec<Page> = pages
            .iter()
            .filter(|stored| !stored.deleted)
            .map(|stored| stored.page.clone())
            .collect();
        validate_page_set(&active)?;
        Ok(pages)
    }

    fn read_current(&self, id: &str) -> std::result::Result<Option<StoredPage>, ServiceError> {
        let path = self.page_path(id);
        let raw = match fs::read_to_string(path) {
            Ok(raw) => raw,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(_) => return Err(ServiceError::corrupt()),
        };
        parse_stored_page(&raw).map(Some)
    }

    fn validate_candidate(
        &self,
        page: &Page,
        excluding_id: Option<&str>,
    ) -> std::result::Result<(), ServiceError> {
        validate_page(page)?;
        let mut pages = self.active_pages()?;
        pages.retain(|candidate| Some(candidate.id.as_str()) != excluding_id);
        pages.push(page.clone());
        validate_page_set(&pages)
    }

    fn write_stored(&self, stored: &StoredPage) -> std::result::Result<(), ServiceError> {
        validate_page(&stored.page)?;
        let rendered = render_stored_page(stored)?;
        let destination = self.page_path(&stored.page.id);
        let temporary = self.pages_root.join(format!(
            ".{}.{}.tmp",
            stored.page.id,
            Uuid::new_v4().simple()
        ));
        fs::write(&temporary, rendered).map_err(|_| ServiceError::internal())?;
        set_file_permissions(&temporary).map_err(|_| ServiceError::internal())?;
        fs::rename(&temporary, &destination).map_err(|_| ServiceError::internal())?;
        set_file_permissions(&destination).map_err(|_| ServiceError::internal())?;
        Ok(())
    }

    fn commit_page(
        &mut self,
        id: &str,
        operation: &str,
    ) -> std::result::Result<String, ServiceError> {
        let page_path = self.page_path(id);
        let version = git::commit_paths(
            &self.root,
            &[page_path.as_path()],
            &format!("memory:{operation}:{id}"),
        )
        .map_err(|_| ServiceError::internal())?;
        if version.is_empty() {
            return Err(ServiceError::internal());
        }
        self.engine
            .rebuild_index(MEMORY_SPACE)
            .map_err(|_| ServiceError::internal())?;
        Ok(version)
    }

    fn global_file(
        &self,
        target: GlobalMemoryTarget,
    ) -> std::result::Result<GlobalMemoryFile, ServiceError> {
        let content = match fs::read_to_string(self.global_path(target)) {
            Ok(content) => {
                validate_global_content(&content).map_err(|_| ServiceError::corrupt())?;
                Some(content)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(_) => return Err(ServiceError::corrupt()),
        };
        let (revision, last_changed_at) = self.global_history(target)?;
        Ok(GlobalMemoryFile {
            target,
            file_name: target.file_name(),
            updated_at: content.as_ref().and(last_changed_at),
            content,
            revision,
        })
    }

    fn global_history(
        &self,
        target: GlobalMemoryTarget,
    ) -> std::result::Result<(u64, Option<String>), ServiceError> {
        let repo = self.repository()?;
        let mut revwalk = repo.revwalk().map_err(|_| ServiceError::corrupt())?;
        if revwalk.push_head().is_err() {
            return Ok((0, None));
        }
        revwalk
            .set_sorting(Sort::TIME | Sort::TOPOLOGICAL)
            .map_err(|_| ServiceError::corrupt())?;
        let relative = Path::new(target.file_name());
        let mut revision = 0_u64;
        let mut last_changed_at = None;
        for oid in revwalk {
            let commit = repo
                .find_commit(oid.map_err(|_| ServiceError::corrupt())?)
                .map_err(|_| ServiceError::corrupt())?;
            let current = raw_page_at_commit(&repo, &commit, relative)?;
            let parent = if commit.parent_count() > 0 {
                raw_page_at_commit(
                    &repo,
                    &commit.parent(0).map_err(|_| ServiceError::corrupt())?,
                    relative,
                )?
            } else {
                None
            };
            if current == parent {
                continue;
            }
            revision = revision.saturating_add(1);
            if last_changed_at.is_none() {
                last_changed_at = Some(commit_timestamp(&commit)?);
            }
        }
        Ok((revision, last_changed_at))
    }

    fn reconcile_global_file(
        &self,
        target: GlobalMemoryTarget,
    ) -> std::result::Result<(), ServiceError> {
        let path = self.global_path(target);
        let worktree = match fs::read_to_string(&path) {
            Ok(content) => {
                validate_global_content(&content).map_err(|_| ServiceError::corrupt())?;
                Some(content)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(_) => return Err(ServiceError::corrupt()),
        };
        let repo = self.repository()?;
        let committed = match repo.head().and_then(|head| head.peel_to_commit()) {
            Ok(commit) => raw_page_at_commit(&repo, &commit, Path::new(target.file_name()))?,
            Err(_) => None,
        };
        if worktree != committed {
            let version = commit_memory_path(
                &self.root,
                &path,
                &format!("memory:external_edit:{}", target.file_name()),
            )?;
            if version.is_empty() {
                return Err(ServiceError::internal());
            }
        }
        Ok(())
    }

    fn write_global_file(
        &self,
        target: GlobalMemoryTarget,
        content: &str,
    ) -> std::result::Result<(), ServiceError> {
        validate_global_content(content)?;
        let destination = self.global_path(target);
        let temporary = self.root.join(format!(
            ".{}.{}.tmp",
            target.file_name(),
            Uuid::new_v4().simple()
        ));
        fs::write(&temporary, content).map_err(|_| ServiceError::internal())?;
        set_file_permissions(&temporary).map_err(|_| ServiceError::internal())?;
        fs::rename(&temporary, &destination).map_err(|_| ServiceError::internal())?;
        set_file_permissions(&destination).map_err(|_| ServiceError::internal())?;
        Ok(())
    }

    fn commit_global_file(
        &self,
        target: GlobalMemoryTarget,
        operation: &str,
    ) -> std::result::Result<String, ServiceError> {
        let version = commit_memory_path(
            &self.root,
            &self.global_path(target),
            &format!("memory:{operation}:{}", target.file_name()),
        )?;
        if version.is_empty() {
            return Err(ServiceError::internal());
        }
        Ok(version)
    }

    fn history(&self, id: &str, limit: usize) -> std::result::Result<Vec<Value>, ServiceError> {
        let repo = self.repository()?;
        let mut revwalk = repo.revwalk().map_err(|_| ServiceError::corrupt())?;
        revwalk.push_head().map_err(|_| ServiceError::corrupt())?;
        revwalk
            .set_sorting(Sort::TIME | Sort::TOPOLOGICAL)
            .map_err(|_| ServiceError::corrupt())?;
        let relative = page_relative_path(id);
        let mut versions = Vec::new();
        for oid in revwalk {
            let oid = oid.map_err(|_| ServiceError::corrupt())?;
            let commit = repo.find_commit(oid).map_err(|_| ServiceError::corrupt())?;
            let current = raw_page_at_commit(&repo, &commit, &relative)?;
            let parent = if commit.parent_count() > 0 {
                raw_page_at_commit(
                    &repo,
                    &commit.parent(0).map_err(|_| ServiceError::corrupt())?,
                    &relative,
                )?
            } else {
                None
            };
            if current == parent {
                continue;
            }
            let Some(stored) = current.as_deref().map(parse_stored_page).transpose()? else {
                continue;
            };
            let operation = commit_operation(commit.message().unwrap_or_default())
                .ok_or_else(ServiceError::corrupt)?;
            versions.push(json!({
                "version": commit.id().to_string(),
                "revision": stored.page.revision,
                "operation": operation,
                "committedAt": commit_timestamp(&commit)?
            }));
            if versions.len() >= limit {
                break;
            }
        }
        Ok(versions)
    }

    fn generation(&self) -> std::result::Result<u64, ServiceError> {
        let repo = self.repository()?;
        let mut revwalk = repo.revwalk().map_err(|_| ServiceError::corrupt())?;
        revwalk.push_head().map_err(|_| ServiceError::corrupt())?;
        let mut count = 0_u64;
        for oid in revwalk {
            let commit = repo
                .find_commit(oid.map_err(|_| ServiceError::corrupt())?)
                .map_err(|_| ServiceError::corrupt())?;
            if commit.message().unwrap_or_default().starts_with("memory:") {
                count = count.saturating_add(1);
            }
        }
        Ok(count)
    }

    fn repository(&self) -> std::result::Result<Repository, ServiceError> {
        Repository::open(&self.root).map_err(|_| ServiceError::corrupt())
    }

    fn page_path(&self, id: &str) -> PathBuf {
        self.pages_root.join(format!("{id}.md"))
    }

    fn global_path(&self, target: GlobalMemoryTarget) -> PathBuf {
        self.root.join(target.file_name())
    }
}

fn success_response(operation: &str, result: Value) -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": operation,
        "ok": true,
        "result": result
    })
}

fn error_response(operation: &str, error: ServiceError) -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": operation,
        "ok": false,
        "error": { "code": error.code, "message": error.safe_message }
    })
}

fn commit_memory_path(
    root: &Path,
    path: &Path,
    message: &str,
) -> std::result::Result<String, ServiceError> {
    let repo = Repository::open(root).map_err(|_| ServiceError::corrupt())?;
    let signature = repo
        .signature()
        .or_else(|_| git2::Signature::now("swarmx-mem", "swarmx-mem@localhost"))
        .map_err(|_| ServiceError::internal())?;
    let relative = path
        .strip_prefix(root)
        .map_err(|_| ServiceError::internal())?;
    let mut index = repo.index().map_err(|_| ServiceError::internal())?;
    if path.is_file() {
        index
            .add_path(relative)
            .map_err(|_| ServiceError::internal())?;
    } else {
        index
            .remove_path(relative)
            .map_err(|_| ServiceError::internal())?;
    }
    index.write().map_err(|_| ServiceError::internal())?;
    let tree_id = index.write_tree().map_err(|_| ServiceError::internal())?;
    let tree = repo
        .find_tree(tree_id)
        .map_err(|_| ServiceError::internal())?;
    let parent = repo.head().ok().and_then(|head| head.peel_to_commit().ok());
    if parent
        .as_ref()
        .is_some_and(|parent| parent.tree_id() == tree_id)
    {
        return Ok(String::new());
    }
    let parents: Vec<&git2::Commit<'_>> = parent.iter().collect();
    repo.commit(
        Some("HEAD"),
        &signature,
        &signature,
        message,
        &tree,
        &parents,
    )
    .map(|oid| oid.to_string())
    .map_err(|_| ServiceError::internal())
}

fn validate_global_content(content: &str) -> std::result::Result<(), ServiceError> {
    if content.trim().is_empty()
        || content.encode_utf16().count() > MAX_GLOBAL_MEMORY_CHARS
        || content.chars().any(|character| {
            let code = character as u32;
            (code < 32 && !matches!(character, '\t' | '\n' | '\r')) || code == 127
        })
    {
        return Err(ServiceError::invalid());
    }
    Ok(())
}

fn validate_page(page: &Page) -> std::result::Result<(), ServiceError> {
    validate_id(&page.id)?;
    validate_name(&page.title)?;
    if page.aliases.len() > MAX_ALIASES
        || page.revision == 0
        || page.content.chars().count() > MAX_PAGE_CHARS
        || page.content.contains('\0')
        || !is_timestamp(&page.created_at)
        || !is_timestamp(&page.updated_at)
        || count_link_markers(&page.content) > MAX_LINK_MARKERS_PER_PAGE
    {
        return Err(ServiceError::invalid());
    }
    for alias in &page.aliases {
        validate_name(alias)?;
    }
    Ok(())
}

fn validate_page_set(pages: &[Page]) -> std::result::Result<(), ServiceError> {
    if pages.len() > MAX_PAGES {
        return Err(ServiceError::invalid());
    }
    let mut ids = HashSet::new();
    let mut names = HashSet::new();
    let mut total_chars = 0_usize;
    let mut total_links = 0_usize;
    for page in pages {
        validate_page(page)?;
        if !ids.insert(page.id.clone()) {
            return Err(ServiceError::invalid());
        }
        for name in std::iter::once(&page.title).chain(&page.aliases) {
            if !names.insert(normalize_name(name)) {
                return Err(ServiceError::conflict());
            }
        }
        total_chars = total_chars.saturating_add(page.content.chars().count());
        total_links = total_links.saturating_add(count_link_markers(&page.content));
    }
    if total_chars > MAX_TOTAL_CHARS || total_links > MAX_LINK_MARKERS_TOTAL {
        return Err(ServiceError::invalid());
    }
    Ok(())
}

fn validate_id(id: &str) -> std::result::Result<(), ServiceError> {
    if id.is_empty()
        || id.len() > 256
        || !id.starts_with("mem_")
        || !id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
    {
        return Err(ServiceError::invalid());
    }
    Ok(())
}

fn validate_name(name: &str) -> std::result::Result<(), ServiceError> {
    let trimmed = name.trim();
    if trimmed.is_empty()
        || trimmed != name
        || name.chars().count() > 256
        || name
            .chars()
            .any(|character| character.is_control() || matches!(character, '[' | ']' | '|' | '#'))
    {
        return Err(ServiceError::invalid());
    }
    Ok(())
}

fn normalize_name(value: &str) -> String {
    value.nfc().flat_map(char::to_lowercase).collect()
}

fn count_link_markers(value: &str) -> usize {
    value.match_indices("[[").count()
}

fn is_timestamp(value: &str) -> bool {
    DateTime::parse_from_rfc3339(value).is_ok()
}

fn page_summary(page: &Page) -> PageSummary {
    PageSummary {
        id: page.id.clone(),
        title: page.title.clone(),
        aliases: page.aliases.clone(),
        revision: page.revision,
        created_at: page.created_at.clone(),
        updated_at: page.updated_at.clone(),
    }
}

fn render_stored_page(stored: &StoredPage) -> std::result::Result<String, ServiceError> {
    let frontmatter = PageFile {
        id: stored.page.id.clone(),
        title: stored.page.title.clone(),
        aliases: stored.page.aliases.clone(),
        revision: stored.page.revision,
        created_at: stored.page.created_at.clone(),
        updated_at: stored.page.updated_at.clone(),
        deleted: stored.deleted,
        page_type: if stored.deleted {
            "swarmx_deleted".to_owned()
        } else {
            "swarmx_memory".to_owned()
        },
        status: if stored.deleted {
            "draft".to_owned()
        } else {
            "active".to_owned()
        },
        last_updated: stored.page.updated_at.clone(),
    };
    let yaml = serde_yaml::to_string(&frontmatter).map_err(|_| ServiceError::internal())?;
    Ok(format!("---\n{yaml}---\n\n{}", stored.page.content))
}

fn parse_stored_page(raw: &str) -> std::result::Result<StoredPage, ServiceError> {
    let rest = raw
        .strip_prefix("---\n")
        .ok_or_else(ServiceError::corrupt)?;
    let (yaml, body) = rest
        .split_once("\n---\n")
        .ok_or_else(ServiceError::corrupt)?;
    let frontmatter: PageFile = serde_yaml::from_str(yaml).map_err(|_| ServiceError::corrupt())?;
    if !matches!(
        frontmatter.page_type.as_str(),
        "swarmx_memory" | "swarmx_deleted"
    ) || frontmatter.deleted != (frontmatter.page_type == "swarmx_deleted")
    {
        return Err(ServiceError::corrupt());
    }
    let page = Page {
        id: frontmatter.id,
        title: frontmatter.title,
        aliases: frontmatter.aliases,
        content: body.strip_prefix('\n').unwrap_or(body).to_owned(),
        revision: frontmatter.revision,
        created_at: frontmatter.created_at,
        updated_at: frontmatter.updated_at,
    };
    validate_page(&page).map_err(|_| ServiceError::corrupt())?;
    Ok(StoredPage {
        page,
        deleted: frontmatter.deleted,
    })
}

fn page_relative_path(id: &str) -> PathBuf {
    PathBuf::from(PAGES_DIRECTORY).join(format!("{id}.md"))
}

fn raw_page_at_commit(
    repo: &Repository,
    commit: &git2::Commit<'_>,
    relative: &Path,
) -> std::result::Result<Option<String>, ServiceError> {
    let tree = commit.tree().map_err(|_| ServiceError::corrupt())?;
    let entry = match tree.get_path(relative) {
        Ok(entry) => entry,
        Err(error) if error.code() == git2::ErrorCode::NotFound => return Ok(None),
        Err(_) => return Err(ServiceError::corrupt()),
    };
    let blob = entry
        .to_object(repo)
        .and_then(|object| object.peel_to_blob())
        .map_err(|_| ServiceError::corrupt())?;
    let raw = std::str::from_utf8(blob.content()).map_err(|_| ServiceError::corrupt())?;
    Ok(Some(raw.to_owned()))
}

fn page_at_commit(
    repo: &Repository,
    commit: &git2::Commit<'_>,
    relative: &Path,
) -> std::result::Result<Option<StoredPage>, ServiceError> {
    raw_page_at_commit(repo, commit, relative)?
        .as_deref()
        .map(parse_stored_page)
        .transpose()
}

fn parse_version(version: &str) -> std::result::Result<Oid, ServiceError> {
    if version.len() != 40
        || !version
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    {
        return Err(ServiceError::invalid());
    }
    Oid::from_str(version).map_err(|_| ServiceError::invalid())
}

fn commit_operation(message: &str) -> Option<&'static str> {
    if message.starts_with("memory:create:") {
        Some("create")
    } else if message.starts_with("memory:update:") {
        Some("update")
    } else if message.starts_with("memory:delete:") {
        Some("delete")
    } else if message.starts_with("memory:restore:") {
        Some("restore")
    } else {
        None
    }
}

fn commit_timestamp(commit: &git2::Commit<'_>) -> std::result::Result<String, ServiceError> {
    DateTime::<Utc>::from_timestamp(commit.time().seconds(), 0)
        .map(|timestamp| timestamp.to_rfc3339_opts(SecondsFormat::Millis, true))
        .ok_or_else(ServiceError::corrupt)
}

fn now() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn truncate_chars(value: String, limit: usize) -> (String, bool) {
    if value.chars().count() <= limit {
        return (value, false);
    }
    (value.chars().take(limit).collect(), true)
}

fn default_search_limit() -> usize {
    20
}

fn default_history_limit() -> usize {
    20
}

fn ensure_runtime_gitignore(root: &Path) -> Result<()> {
    let path = root.join(".gitignore");
    let mut content = fs::read_to_string(&path).unwrap_or_default();
    if !content.lines().any(|line| line.trim() == ".runtime/") {
        if !content.is_empty() && !content.ends_with('\n') {
            content.push('\n');
        }
        content.push_str(".runtime/\n");
        fs::write(&path, content)?;
        set_file_permissions(&path)?;
    }
    Ok(())
}

#[cfg(unix)]
fn set_directory_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_directory_permissions(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn set_file_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_file_permissions(_path: &Path) -> Result<()> {
    Ok(())
}
