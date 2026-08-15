use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, SecondsFormat, Utc};
use git2::{IndexEntry, IndexTime, Oid, Repository, Sort};
use llm_wiki::engine::WikiEngine as MemoryEngine;
use llm_wiki::{ops, spaces};
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
const TOMBSTONES_DIRECTORY: &str = ".swarmx/tombstones";
const INDEX_FILE: &str = "INDEX.md";
const DISAMBIGUATION_FILE: &str = "DISAMBIGUATION.md";
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
const TRANSACTIONS_DIRECTORY: &str = ".runtime/transactions";
const TRANSACTION_SCHEMA_VERSION: u32 = 1;
const HEAD_REF_FALLBACK: &str = "refs/heads/main";
const LEGACY_VAULT_README: &str = "# swarmx\n\nSwarmX Memory\n\nManaged by [llm-wiki](https://github.com/geronimo-iia/llm-wiki). Run `llm-wiki serve` to start the MCP server.\n";
const VAULT_README: &str = "# SwarmX Memory\n\nThis folder is your local-first Memory vault. Open it directly in Obsidian or any Markdown editor.\n\n- `USER.md` contains stable personal preferences and working habits.\n- `MEMORY.md` contains compact cross-project experience.\n- `pages/` contains the linked knowledge wiki in human-readable files and folders.\n\nYou can create, edit, move, rename, and delete Markdown notes under `pages/`. SwarmX keeps the generated `id`, revision, and timestamps in YAML frontmatter synchronized with Git history. Other frontmatter fields are preserved.\n";

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
    #[serde(default)]
    pub kind: PageKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(default)]
    pub sources: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disambiguation: Option<String>,
    pub content: String,
    pub revision: u64,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PageKind {
    Project,
    Person,
    Organization,
    Technology,
    Decision,
    Concept,
    #[default]
    Note,
}

impl PageKind {
    fn label(self) -> &'static str {
        match self {
            Self::Project => "project",
            Self::Person => "person",
            Self::Organization => "organization",
            Self::Technology => "technology",
            Self::Decision => "decision",
            Self::Concept => "concept",
            Self::Note => "note",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct PageFile {
    id: String,
    title: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    kind: PageKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
    #[serde(default)]
    sources: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scope: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    disambiguation: Option<String>,
    revision: u64,
    created_at: String,
    updated_at: String,
    #[serde(default)]
    deleted: bool,
    #[serde(rename = "type")]
    page_type: String,
    status: String,
    last_updated: String,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_yaml::Value>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HumanPageFile {
    id: Option<String>,
    title: Option<String>,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    kind: PageKind,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    sources: Vec<String>,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    disambiguation: Option<String>,
    revision: Option<u64>,
    created_at: Option<String>,
    updated_at: Option<String>,
    #[serde(default)]
    deleted: bool,
    #[serde(rename = "type")]
    _page_type: Option<String>,
    #[serde(rename = "status")]
    _status: Option<String>,
    last_updated: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_yaml::Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct PageSummary {
    id: String,
    title: String,
    aliases: Vec<String>,
    kind: PageKind,
    summary: Option<String>,
    sources: Vec<String>,
    scope: Option<String>,
    disambiguation: Option<String>,
    revision: u64,
    created_at: String,
    updated_at: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchHit {
    title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
    kind: PageKind,
    sources: Vec<String>,
    related_pages: Vec<String>,
    id: String,
}

#[derive(Debug, Clone, PartialEq)]
struct StoredPage {
    page: Page,
    deleted: bool,
    extra: BTreeMap<String, serde_yaml::Value>,
}

#[derive(Debug, Clone, PartialEq)]
struct PageRecord {
    stored: StoredPage,
    relative_path: PathBuf,
}

struct CreatePageInput {
    title: String,
    aliases: Vec<String>,
    kind: PageKind,
    summary: Option<String>,
    sources: Vec<String>,
    scope: Option<String>,
    content: String,
}

struct UpdatePageInput {
    id: String,
    expected_revision: u64,
    title: Option<String>,
    aliases: Option<Vec<String>>,
    kind: Option<PageKind>,
    summary: Option<String>,
    sources: Option<Vec<String>>,
    scope: Option<String>,
    content: Option<String>,
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
        #[serde(default)]
        kind: PageKind,
        summary: Option<String>,
        #[serde(default)]
        sources: Vec<String>,
        scope: Option<String>,
        content: String,
    },
    Update {
        protocol_version: u32,
        id: String,
        expected_revision: u64,
        title: Option<String>,
        aliases: Option<Vec<String>>,
        kind: Option<PageKind>,
        summary: Option<String>,
        sources: Option<Vec<String>>,
        scope: Option<String>,
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
    _writer_lock: MemoryWriterLock,
    transaction: Option<MemoryTransaction>,
    faulted: bool,
}

#[derive(Debug)]
struct MemoryWriterLock {
    _file: File,
}

impl MemoryWriterLock {
    fn acquire(root: &Path) -> std::result::Result<Self, ServiceError> {
        let path = root.join(".runtime/writer.lock");
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)
            .map_err(|_| ServiceError::internal())?;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if result != 0 {
                return Err(ServiceError::conflict());
            }
        }
        Ok(Self { _file: file })
    }
}

#[derive(Debug)]
struct MemoryTransaction {
    root: PathBuf,
    txid: String,
    message: String,
    base_head: Option<Oid>,
    head_ref: String,
    before: BTreeMap<PathBuf, Option<Vec<u8>>>,
    patches: BTreeMap<PathBuf, Option<Vec<u8>>>,
    intended_commit: Option<Oid>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TransactionManifest {
    schema_version: u32,
    txid: String,
    base_head: Option<String>,
    intended_commit: String,
    head_ref: String,
    message: String,
    files: Vec<TransactionFile>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TransactionFile {
    path: String,
    before_present: bool,
    after_present: bool,
}

impl MemoryTransaction {
    fn new(root: &Path, message: impl Into<String>) -> std::result::Result<Self, ServiceError> {
        let repo = Repository::open(root).map_err(|_| ServiceError::corrupt())?;
        let head = repo.head().map_err(|_| ServiceError::corrupt())?;
        let head_ref = head
            .name()
            .ok()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| HEAD_REF_FALLBACK.to_owned());
        Ok(Self {
            root: root.to_path_buf(),
            txid: format!("tx_{}", Uuid::new_v4().simple()),
            message: message.into(),
            base_head: head.target(),
            head_ref,
            before: BTreeMap::new(),
            patches: BTreeMap::new(),
            intended_commit: None,
        })
    }

    fn set_message(&mut self, message: impl Into<String>) {
        self.message = message.into();
    }

    fn capture_before(&mut self, relative: &Path) -> std::result::Result<(), ServiceError> {
        validate_transaction_relative_path(relative)?;
        if self.before.contains_key(relative) {
            return Ok(());
        }
        let content = match fs::read(self.root.join(relative)) {
            Ok(content) => Some(content),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(_) => return Err(ServiceError::internal()),
        };
        self.before.insert(relative.to_path_buf(), content);
        Ok(())
    }

    fn read(&self, relative: &Path) -> std::result::Result<Option<Vec<u8>>, ServiceError> {
        validate_transaction_relative_path(relative)?;
        if let Some(content) = self.patches.get(relative) {
            return Ok(content.clone());
        }
        match fs::read(self.root.join(relative)) {
            Ok(content) => Ok(Some(content)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(_) => Err(ServiceError::internal()),
        }
    }

    fn stage_write(
        &mut self,
        relative: &Path,
        content: Vec<u8>,
    ) -> std::result::Result<(), ServiceError> {
        self.capture_before(relative)?;
        self.patches.insert(relative.to_path_buf(), Some(content));
        Ok(())
    }

    fn stage_remove(&mut self, relative: &Path) -> std::result::Result<(), ServiceError> {
        self.capture_before(relative)?;
        self.patches.insert(relative.to_path_buf(), None);
        Ok(())
    }

    fn wal_dir(&self) -> PathBuf {
        self.root.join(TRANSACTIONS_DIRECTORY).join(&self.txid)
    }

    fn prepare(&mut self) -> std::result::Result<Option<Oid>, ServiceError> {
        if self.patches.is_empty() {
            return Ok(None);
        }
        let repo = Repository::open(&self.root).map_err(|_| ServiceError::corrupt())?;
        if current_head(&repo) != self.base_head {
            return Err(ServiceError::conflict());
        }
        let mut index = repo.index().map_err(|_| ServiceError::internal())?;
        let parent = if let Some(base_head) = self.base_head {
            let commit = repo
                .find_commit(base_head)
                .map_err(|_| ServiceError::corrupt())?;
            let tree = commit.tree().map_err(|_| ServiceError::corrupt())?;
            index
                .read_tree(&tree)
                .map_err(|_| ServiceError::internal())?;
            Some(commit)
        } else {
            None
        };
        for (relative, content) in &self.patches {
            validate_transaction_relative_path(relative)?;
            match content {
                Some(content) => {
                    let mut entry = index.get_path(relative, 0).unwrap_or_else(new_index_entry);
                    entry.mode = 0o100644;
                    entry.path = relative.to_string_lossy().replace('\\', "/").into_bytes();
                    index
                        .add_frombuffer(&entry, content)
                        .map_err(|_| ServiceError::internal())?;
                }
                None => {
                    if index.get_path(relative, 0).is_some() {
                        index
                            .remove_path(relative)
                            .map_err(|_| ServiceError::internal())?;
                    }
                }
            }
        }
        let tree_id = index
            .write_tree_to(&repo)
            .map_err(|_| ServiceError::internal())?;
        if parent
            .as_ref()
            .is_some_and(|commit| commit.tree_id() == tree_id)
        {
            self.intended_commit = self.base_head;
            return Ok(self.base_head);
        }
        let tree = repo
            .find_tree(tree_id)
            .map_err(|_| ServiceError::internal())?;
        let signature = repo
            .signature()
            .or_else(|_| git2::Signature::now("swarmx-mem", "swarmx-mem@localhost"))
            .map_err(|_| ServiceError::internal())?;
        let parents: Vec<&git2::Commit<'_>> = parent.iter().collect();
        let intended = repo
            .commit(None, &signature, &signature, &self.message, &tree, &parents)
            .map_err(|_| ServiceError::internal())?;
        self.intended_commit = Some(intended);
        self.write_wal(intended)?;
        Ok(Some(intended))
    }

    fn write_wal(&self, intended: Oid) -> std::result::Result<(), ServiceError> {
        let transaction_dir = self.wal_dir();
        fs::create_dir_all(transaction_dir.join("before")).map_err(|_| ServiceError::internal())?;
        set_directory_permissions(&transaction_dir).map_err(|_| ServiceError::internal())?;
        set_directory_permissions(&transaction_dir.join("before"))
            .map_err(|_| ServiceError::internal())?;
        let mut files = Vec::with_capacity(self.patches.len());
        for (relative, after) in &self.patches {
            let before = self.before.get(relative).cloned().flatten();
            if let Some(before) = &before {
                let path = transaction_dir.join("before").join(relative);
                write_durable(&path, before)?;
            }
            files.push(TransactionFile {
                path: relative.to_string_lossy().replace('\\', "/"),
                before_present: before.is_some(),
                after_present: after.is_some(),
            });
        }
        let summary = json!({
            "schemaVersion": TRANSACTION_SCHEMA_VERSION,
            "txid": self.txid,
            "fileCount": files.len(),
            "message": self.message,
        });
        write_durable(
            &transaction_dir.join("summary.json"),
            serde_json::to_vec(&summary)
                .map_err(|_| ServiceError::internal())?
                .as_slice(),
        )?;
        let manifest = TransactionManifest {
            schema_version: TRANSACTION_SCHEMA_VERSION,
            txid: self.txid.clone(),
            base_head: self.base_head.map(|head| head.to_string()),
            intended_commit: intended.to_string(),
            head_ref: self.head_ref.clone(),
            message: self.message.clone(),
            files,
        };
        write_durable(
            &transaction_dir.join("manifest.json"),
            &serde_json::to_vec_pretty(&manifest).map_err(|_| ServiceError::internal())?,
        )?;
        sync_directory(
            &self
                .root
                .join(TRANSACTIONS_DIRECTORY)
                .canonicalize()
                .unwrap_or_else(|_| self.root.join(TRANSACTIONS_DIRECTORY)),
        )?;
        Ok(())
    }

    fn verify_live_state(&self, expected_after: bool) -> std::result::Result<(), ServiceError> {
        for relative in self.patches.keys() {
            let current = match fs::read(self.root.join(relative)) {
                Ok(content) => Some(content),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(_) => return Err(ServiceError::internal()),
            };
            let before = self.before.get(relative).cloned().flatten();
            let after = self.patches.get(relative).cloned().flatten();
            let valid = if expected_after {
                current == after
            } else {
                current == before || current == after
            };
            if !valid {
                return Err(ServiceError::conflict());
            }
        }
        Ok(())
    }

    fn apply_files(&self) -> std::result::Result<(), ServiceError> {
        for (relative, content) in &self.patches {
            let destination = self.root.join(relative);
            match content {
                Some(content) => {
                    durable_replace(&destination, content)?;
                    failpoint(&format!("write:{}", path_key(relative)));
                }
                None => {
                    match fs::remove_file(&destination) {
                        Ok(()) => sync_directory(destination.parent().unwrap_or(&self.root))?,
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                        Err(_) => return Err(ServiceError::internal()),
                    }
                    failpoint(&format!("delete:{}", path_key(relative)));
                }
            }
        }
        Ok(())
    }

    fn publish(&self, intended: Oid) -> std::result::Result<(), ServiceError> {
        let repo = Repository::open(&self.root).map_err(|_| ServiceError::corrupt())?;
        let mut transaction = repo.transaction().map_err(|_| ServiceError::internal())?;
        transaction
            .lock_ref(&self.head_ref)
            .map_err(|_| ServiceError::conflict())?;
        let locked_head = repo
            .find_reference(&self.head_ref)
            .ok()
            .and_then(|reference| reference.target());
        if locked_head != self.base_head {
            return Err(ServiceError::conflict());
        }
        failpoint("git_publish");
        let signature = repo
            .signature()
            .or_else(|_| git2::Signature::now("swarmx-mem", "swarmx-mem@localhost"))
            .map_err(|_| ServiceError::internal())?;
        transaction
            .set_target(
                &self.head_ref,
                intended,
                Some(&signature),
                "swarmx-mem:publish",
            )
            .map_err(|_| ServiceError::internal())?;
        transaction.commit().map_err(|_| ServiceError::internal())?;
        failpoint("git_published");
        Ok(())
    }

    fn cleanup(&self) -> std::result::Result<(), ServiceError> {
        let transaction_dir = self.wal_dir();
        if transaction_dir.exists() {
            fs::remove_dir_all(&transaction_dir).map_err(|_| ServiceError::internal())?;
            sync_directory(&self.root.join(TRANSACTIONS_DIRECTORY))?;
        }
        failpoint("wal_cleanup");
        Ok(())
    }
}

fn new_index_entry() -> IndexEntry {
    IndexEntry {
        ctime: IndexTime::new(0, 0),
        mtime: IndexTime::new(0, 0),
        dev: 0,
        ino: 0,
        mode: 0o100644,
        uid: 0,
        gid: 0,
        file_size: 0,
        id: Oid::ZERO_SHA1,
        flags: 0,
        flags_extended: 0,
        path: Vec::new(),
    }
}

fn path_key(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn current_head(repo: &Repository) -> Option<Oid> {
    repo.head().ok().and_then(|head| head.target())
}

fn write_durable(path: &Path, content: &[u8]) -> std::result::Result<(), ServiceError> {
    let parent = path.parent().ok_or_else(ServiceError::internal)?;
    fs::create_dir_all(parent).map_err(|_| ServiceError::internal())?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        Uuid::new_v4().simple()
    ));
    let mut file = File::create(&temporary).map_err(|_| ServiceError::internal())?;
    file.write_all(content)
        .map_err(|_| ServiceError::internal())?;
    file.sync_all().map_err(|_| ServiceError::internal())?;
    set_file_permissions(&temporary).map_err(|_| ServiceError::internal())?;
    fs::rename(&temporary, path).map_err(|_| ServiceError::internal())?;
    set_file_permissions(path).map_err(|_| ServiceError::internal())?;
    sync_directory(parent)?;
    Ok(())
}

fn durable_replace(path: &Path, content: &[u8]) -> std::result::Result<(), ServiceError> {
    write_durable(path, content)
}

fn sync_directory(path: &Path) -> std::result::Result<(), ServiceError> {
    #[cfg(unix)]
    {
        File::open(path)
            .map_err(|_| ServiceError::internal())?
            .sync_all()
            .map_err(|_| ServiceError::internal())?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

fn failpoint(name: &str) {
    let configured = std::env::var("SWARMX_MEM_FAILPOINT").ok();
    if configured.as_deref().is_some_and(|value| {
        value == "*" || value == name || name.starts_with(&format!("{value}:"))
    }) {
        std::process::abort();
    }
}

fn recover_transactions(root: &Path) -> Result<()> {
    let transactions_root = root.join(TRANSACTIONS_DIRECTORY);
    fs::create_dir_all(&transactions_root)?;
    set_directory_permissions(&transactions_root)?;
    let mut entries = fs::read_dir(&transactions_root)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_dir())
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        recover_transaction(root, &entry.path())?;
    }
    Ok(())
}

fn recover_transaction(root: &Path, transaction_dir: &Path) -> Result<()> {
    let manifest_path = transaction_dir.join("manifest.json");
    let manifest_bytes = fs::read(&manifest_path).with_context(|| {
        format!(
            "Memory transaction manifest is unreadable: {}",
            manifest_path.display()
        )
    })?;
    let manifest: TransactionManifest = serde_json::from_slice(&manifest_bytes)
        .context("Memory transaction manifest is invalid")?;
    if manifest.schema_version != TRANSACTION_SCHEMA_VERSION
        || transaction_dir.file_name().and_then(|name| name.to_str()) != Some(&manifest.txid)
    {
        anyhow::bail!("Memory transaction manifest is unsupported");
    }
    let repo = Repository::open(root)?;
    let base_head = manifest
        .base_head
        .as_deref()
        .map(Oid::from_str)
        .transpose()?;
    let intended = Oid::from_str(&manifest.intended_commit)?;
    let head = current_head(&repo);
    let target_commit = repo.find_commit(intended)?;
    let target_tree = target_commit.tree()?;
    let mut paths = Vec::with_capacity(manifest.files.len());
    for file in &manifest.files {
        let relative = PathBuf::from(&file.path);
        validate_transaction_relative_path(&relative)
            .map_err(|_| anyhow::anyhow!("Memory transaction path is invalid"))?;
        let before = if file.before_present {
            Some(
                fs::read(transaction_dir.join("before").join(&relative)).with_context(|| {
                    format!("Memory transaction before-image is missing: {}", file.path)
                })?,
            )
        } else {
            None
        };
        let after = raw_blob_at_tree(&repo, &target_tree, &relative)?;
        if file.after_present != after.is_some() {
            anyhow::bail!("Memory transaction after-image does not match Git");
        }
        paths.push((relative, before, after));
    }
    match head {
        value if value == base_head => {
            verify_recovery_paths(root, &paths)?;
            for (relative, before, _) in &paths {
                apply_recovery_path(root, relative, before.as_deref())?;
            }
            sync_index_to_commit(&repo, base_head)?;
        }
        Some(value) if value == intended => {
            verify_recovery_paths(root, &paths)?;
            for (relative, _, after) in &paths {
                apply_recovery_path(root, relative, after.as_deref())?;
            }
            sync_index_to_commit(&repo, Some(intended))?;
        }
        _ => anyhow::bail!(
            "Memory transaction has an unknown HEAD state; refusing to overwrite external edits"
        ),
    }
    fs::remove_dir_all(transaction_dir)?;
    sync_directory(&root.join(TRANSACTIONS_DIRECTORY))
        .map_err(|_| anyhow::anyhow!("failed to sync Memory transaction cleanup"))?;
    Ok(())
}

type RecoveryPath = (PathBuf, Option<Vec<u8>>, Option<Vec<u8>>);

fn verify_recovery_paths(root: &Path, paths: &[RecoveryPath]) -> Result<()> {
    for (relative, before, after) in paths {
        let current = match fs::read(root.join(relative)) {
            Ok(content) => Some(content),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        if current != *before && current != *after {
            anyhow::bail!(
                "Memory transaction path has an unknown external state: {}",
                relative.display()
            );
        }
    }
    Ok(())
}

fn apply_recovery_path(root: &Path, relative: &Path, content: Option<&[u8]>) -> Result<()> {
    let path = root.join(relative);
    match content {
        Some(content) => durable_replace(&path, content)
            .map_err(|_| anyhow::anyhow!("failed to restore Memory transaction path")),
        None => match fs::remove_file(&path) {
            Ok(()) => sync_directory(path.parent().unwrap_or(root))
                .map_err(|_| anyhow::anyhow!("failed to sync Memory transaction deletion")),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        },
    }
}

fn raw_blob_at_tree(
    repo: &Repository,
    tree: &git2::Tree<'_>,
    relative: &Path,
) -> Result<Option<Vec<u8>>> {
    let entry = match tree.get_path(relative) {
        Ok(entry) => entry,
        Err(error) if error.code() == git2::ErrorCode::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let object = entry.to_object(repo)?;
    let blob = object.peel_to_blob()?;
    Ok(Some(blob.content().to_vec()))
}

fn sync_index_to_commit(repo: &Repository, commit: Option<Oid>) -> Result<()> {
    let mut index = repo.index()?;
    if let Some(commit) = commit {
        let commit = repo.find_commit(commit)?;
        let tree = commit.tree()?;
        index.read_tree(&tree)?;
    } else {
        let paths = (0..index.len())
            .filter_map(|position| {
                index
                    .get(position)
                    .map(|entry| PathBuf::from(String::from_utf8_lossy(&entry.path).into_owned()))
            })
            .collect::<Vec<_>>();
        for path in paths {
            let _ = index.remove_path(&path);
        }
    }
    index.write()?;
    sync_directory(&repo.path().join("index"))
        .map_err(|_| anyhow::anyhow!("failed to sync Memory Git index"))?;
    Ok(())
}

impl MemoryService {
    pub fn open(root: &Path) -> Result<Self> {
        let root = root.to_path_buf();
        fs::create_dir_all(&root)
            .with_context(|| format!("failed to create Memory root {}", root.display()))?;
        set_directory_permissions(&root)?;

        let state_root = root.join(".runtime");
        fs::create_dir_all(&state_root)?;
        set_directory_permissions(&state_root)?;
        let writer_lock = MemoryWriterLock::acquire(&root)
            .map_err(|_| anyhow::anyhow!("Memory root is already open for writing"))?;
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
        recover_transactions(&root)?;
        let pages_root = root.join(PAGES_DIRECTORY);
        set_directory_permissions(&pages_root)?;
        let tombstones_root = root.join(TOMBSTONES_DIRECTORY);
        fs::create_dir_all(&tombstones_root)?;
        set_directory_permissions(&tombstones_root)?;
        let mut service = Self {
            root,
            pages_root,
            engine: MemoryEngine::build(&config_path)?,
            _writer_lock: writer_lock,
            transaction: None,
            faulted: false,
        };
        service.ensure_runtime_gitignore()?;
        service.ensure_vault_readme()?;
        service.migrate_legacy_paths()?;
        service
            .reconcile_pages()
            .map_err(|error| anyhow::anyhow!(error.safe_message))?;
        service.engine.rebuild_index(MEMORY_SPACE)?;
        Ok(service)
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
        if self.faulted {
            return error_response(operation, ServiceError::internal());
        }
        if let Err(error) = self.reconcile_pages() {
            return error_response(operation, error);
        }
        let result = self.execute(request);
        if result.is_err() && !self.faulted {
            self.transaction = None;
        }
        match result {
            Ok(result) => success_response(operation, result),
            Err(error) => error_response(operation, error),
        }
    }

    fn ensure_transaction(&mut self) -> std::result::Result<&mut MemoryTransaction, ServiceError> {
        if self.transaction.is_none() {
            self.transaction = Some(MemoryTransaction::new(&self.root, "memory:pending")?);
        }
        self.transaction.as_mut().ok_or_else(ServiceError::internal)
    }

    fn set_transaction_message(&mut self, message: impl Into<String>) {
        if let Some(transaction) = self.transaction.as_mut() {
            transaction.set_message(message);
        }
    }

    fn transaction_read(
        &self,
        relative: &Path,
    ) -> std::result::Result<Option<Vec<u8>>, ServiceError> {
        match &self.transaction {
            Some(transaction) => transaction.read(relative),
            None => match fs::read(self.root.join(relative)) {
                Ok(content) => Ok(Some(content)),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(_) => Err(ServiceError::internal()),
            },
        }
    }

    fn stage_write(
        &mut self,
        relative: &Path,
        content: Vec<u8>,
    ) -> std::result::Result<(), ServiceError> {
        validate_transaction_relative_path(relative)?;
        self.ensure_transaction()?.stage_write(relative, content)
    }

    fn stage_remove(&mut self, relative: &Path) -> std::result::Result<(), ServiceError> {
        validate_transaction_relative_path(relative)?;
        self.ensure_transaction()?.stage_remove(relative)
    }

    fn stage_if_changed(
        &mut self,
        relative: &Path,
        content: Vec<u8>,
    ) -> std::result::Result<bool, ServiceError> {
        if self.transaction_read(relative)?.as_deref() == Some(content.as_slice()) {
            return Ok(false);
        }
        self.stage_write(relative, content)?;
        Ok(true)
    }

    fn transaction_paths_under(&self, directory: &Path) -> BTreeSet<PathBuf> {
        let mut paths = BTreeSet::new();
        if let Ok(entries) = WalkDir::new(directory)
            .min_depth(1)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
        {
            for entry in entries {
                if entry.file_type().is_file()
                    && let Ok(relative) = entry.path().strip_prefix(&self.root)
                {
                    paths.insert(relative.to_path_buf());
                }
            }
        }
        if let Some(transaction) = &self.transaction {
            let prefix = directory.strip_prefix(&self.root).unwrap_or(directory);
            for relative in transaction.patches.keys() {
                if relative.starts_with(prefix) {
                    paths.insert(relative.clone());
                }
            }
        }
        paths
    }

    fn finish_transaction(&mut self) -> std::result::Result<String, ServiceError> {
        let Some(mut transaction) = self.transaction.take() else {
            return Ok(String::new());
        };
        let intended = match transaction.prepare() {
            Ok(intended) => intended,
            Err(error) => {
                if transaction.wal_dir().exists() {
                    self.transaction = Some(transaction);
                    self.faulted = true;
                }
                return Err(error);
            }
        };
        let Some(intended) = intended else {
            return Ok(String::new());
        };
        if transaction.base_head == Some(intended) {
            return Ok(String::new());
        }
        let failed = |service: &mut Self, transaction: MemoryTransaction, error: ServiceError| {
            service.transaction = Some(transaction);
            service.faulted = true;
            error
        };
        if let Err(error) = transaction.verify_live_state(false) {
            return Err(failed(self, transaction, error));
        }
        if let Err(error) = transaction.apply_files() {
            return Err(failed(self, transaction, error));
        }
        if let Err(error) = transaction.verify_live_state(true) {
            return Err(failed(self, transaction, error));
        }
        if let Err(error) = transaction.publish(intended) {
            return Err(failed(self, transaction, error));
        }
        let repo = match Repository::open(&self.root) {
            Ok(repo) => repo,
            Err(_) => return Err(failed(self, transaction, ServiceError::corrupt())),
        };
        if let Err(error) =
            sync_index_to_commit(&repo, Some(intended)).map_err(|_| ServiceError::internal())
        {
            return Err(failed(self, transaction, error));
        }
        failpoint("git_index");
        if self.engine.rebuild_index(MEMORY_SPACE).is_err() {
            return Err(failed(self, transaction, ServiceError::internal()));
        }
        failpoint("search_rebuild");
        if let Err(error) = transaction.cleanup() {
            return Err(failed(self, transaction, error));
        }
        Ok(intended.to_string())
    }

    fn ensure_runtime_gitignore(&mut self) -> Result<()> {
        let relative = Path::new(".gitignore");
        let mut content = self
            .transaction_read(relative)
            .map_err(|_| anyhow::anyhow!("failed to read Memory .gitignore"))?
            .and_then(|bytes| String::from_utf8(bytes).ok())
            .unwrap_or_default();
        if content.lines().any(|line| line.trim() == ".runtime/") {
            return Ok(());
        }
        if !content.is_empty() && !content.ends_with('\n') {
            content.push('\n');
        }
        content.push_str(".runtime/\n");
        self.stage_write(relative, content.into_bytes())
            .map_err(|_| anyhow::anyhow!("failed to stage Memory .gitignore"))?;
        self.set_transaction_message("vault:gitignore");
        self.finish_transaction()
            .map(|_| ())
            .map_err(|_| anyhow::anyhow!("failed to publish Memory .gitignore"))
    }

    fn ensure_vault_readme(&mut self) -> Result<()> {
        let relative = Path::new("README.md");
        let current = self
            .transaction_read(relative)
            .map_err(|_| anyhow::anyhow!("failed to read Memory README"))?
            .and_then(|bytes| String::from_utf8(bytes).ok());
        if current
            .as_deref()
            .is_some_and(|value| value != LEGACY_VAULT_README)
            || current.as_deref() == Some(VAULT_README)
        {
            return Ok(());
        }
        self.stage_write(relative, VAULT_README.as_bytes().to_vec())
            .map_err(|_| anyhow::anyhow!("failed to stage Memory README"))?;
        self.set_transaction_message("vault:readme");
        self.finish_transaction()
            .map(|_| ())
            .map_err(|_| anyhow::anyhow!("failed to publish Memory README"))
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
                kind,
                summary,
                sources,
                scope,
                content,
                ..
            } => self.create_result(CreatePageInput {
                title,
                aliases,
                kind,
                summary,
                sources,
                scope,
                content,
            }),
            Request::Update {
                id,
                expected_revision,
                title,
                aliases,
                kind,
                summary,
                sources,
                scope,
                content,
                ..
            } => self.update_result(UpdatePageInput {
                id,
                expected_revision,
                title,
                aliases,
                kind,
                summary,
                sources,
                scope,
                content,
            }),
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
            "page": page
                .filter(|record| !record.stored.deleted)
                .map(|record| record.stored.page)
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
        self.stage_remove(Path::new(target.file_name()))?;
        let version = self.commit_global_file(target, "global_forget")?;
        Ok(json!({ "file": self.global_file(target)?, "version": version }))
    }

    fn create_result(
        &mut self,
        input: CreatePageInput,
    ) -> std::result::Result<Value, ServiceError> {
        let CreatePageInput {
            title,
            aliases,
            kind,
            summary,
            sources,
            scope,
            content,
        } = input;
        let id = format!("mem_{}", Uuid::new_v4().simple());
        let timestamp = now();
        let page = Page {
            id,
            title,
            aliases,
            kind,
            summary,
            sources,
            scope,
            disambiguation: None,
            content,
            revision: 1,
            created_at: timestamp.clone(),
            updated_at: timestamp,
        };
        let records = self.active_records()?;
        let ambiguity_group: Vec<PageRecord> = records
            .iter()
            .filter(|record| {
                normalize_name(&record.stored.page.title) == normalize_name(&page.title)
                    || record
                        .stored
                        .page
                        .disambiguation
                        .as_ref()
                        .is_some_and(|base| normalize_name(base) == normalize_name(&page.title))
            })
            .cloned()
            .collect();
        if !ambiguity_group.is_empty() {
            return self.create_disambiguated_page(page, records, ambiguity_group);
        }
        self.validate_candidate(&page, None)?;
        let relative_path = self.unique_page_relative_path(&page.title, None)?;
        let stored = StoredPage {
            page: page.clone(),
            deleted: false,
            extra: BTreeMap::new(),
        };
        self.write_stored(&stored, &relative_path)?;
        let version = self.commit_page(&page.id, "create", &[relative_path])?;
        Ok(json!({ "page": page, "version": version }))
    }

    fn create_disambiguated_page(
        &mut self,
        mut page: Page,
        records: Vec<PageRecord>,
        ambiguity_group: Vec<PageRecord>,
    ) -> std::result::Result<Value, ServiceError> {
        let base_title = page.title.clone();
        let timestamp = page.updated_at.clone();
        let mut occupied: BTreeSet<String> = records
            .iter()
            .map(|record| normalize_name(&record.stored.page.title))
            .collect();
        let mut replacements = BTreeMap::new();
        for record in &ambiguity_group {
            if normalize_name(&record.stored.page.title) != normalize_name(&base_title) {
                continue;
            }
            occupied.remove(&normalize_name(&record.stored.page.title));
            let qualified = qualified_page_title(&base_title, &record.stored.page, &occupied);
            occupied.insert(normalize_name(&qualified));
            replacements.insert(record.stored.page.id.clone(), qualified);
        }
        page.title = qualified_page_title(&base_title, &page, &occupied);
        page.disambiguation = Some(base_title.clone());
        if page
            .aliases
            .iter()
            .any(|alias| normalize_name(alias) == normalize_name(&base_title))
        {
            return Err(ServiceError::conflict());
        }

        let primary_title = replacements
            .values()
            .next()
            .cloned()
            .or_else(|| {
                ambiguity_group
                    .iter()
                    .map(|record| record.stored.page.title.clone())
                    .min()
            })
            .ok_or_else(ServiceError::conflict)?;
        let mut prospective = Vec::new();
        let mut writes = Vec::new();
        let mut paths = BTreeSet::new();
        for record in records {
            let replacement = replacements.get(&record.stored.page.id);
            let rewritten =
                rewrite_wiki_link_target(&record.stored.page.content, &base_title, &primary_title);
            let changed = replacement.is_some() || rewritten != record.stored.page.content;
            let mut stored = record.stored;
            let mut destination = record.relative_path.clone();
            if let Some(title) = replacement {
                stored.page.title = title.clone();
                stored.page.disambiguation = Some(base_title.clone());
                stored
                    .page
                    .aliases
                    .retain(|alias| normalize_name(alias) != normalize_name(&base_title));
                destination = self.renamed_page_relative_path(title, &record.relative_path)?;
            }
            if changed {
                stored.page.content = rewritten;
                stored.page.revision += 1;
                stored.page.updated_at = timestamp.clone();
                paths.insert(record.relative_path.clone());
                paths.insert(destination.clone());
                writes.push((stored.clone(), record.relative_path, destination));
            }
            prospective.push(stored.page);
        }
        prospective.push(page.clone());
        validate_page_set(&prospective)?;

        let destination = self.unique_page_relative_path(&page.title, None)?;
        paths.insert(destination.clone());
        for (stored, previous, destination) in writes {
            self.write_stored(&stored, &destination)?;
            if previous != destination {
                self.stage_remove(&previous)?;
            }
        }
        self.write_stored(
            &StoredPage {
                page: page.clone(),
                deleted: false,
                extra: BTreeMap::new(),
            },
            &destination,
        )?;
        let paths: Vec<PathBuf> = paths.into_iter().collect();
        let version = self.commit_page(&page.id, "create", &paths)?;
        Ok(json!({ "page": page, "version": version }))
    }

    fn update_result(
        &mut self,
        input: UpdatePageInput,
    ) -> std::result::Result<Value, ServiceError> {
        let UpdatePageInput {
            id,
            expected_revision,
            title,
            aliases,
            kind,
            summary,
            sources,
            scope,
            content,
        } = input;
        validate_id(&id)?;
        if title.is_none()
            && aliases.is_none()
            && kind.is_none()
            && summary.is_none()
            && sources.is_none()
            && scope.is_none()
            && content.is_none()
        {
            return Err(ServiceError::invalid());
        }
        let current = self
            .read_current(&id)?
            .filter(|record| !record.stored.deleted)
            .ok_or_else(ServiceError::not_found)?;
        if current.stored.page.revision != expected_revision {
            return Err(ServiceError::conflict());
        }
        let old_title = current.stored.page.title.clone();
        let title_changed = title
            .as_ref()
            .is_some_and(|candidate| normalize_name(candidate) != normalize_name(&old_title));
        let mut next_aliases = aliases.unwrap_or(current.stored.page.aliases.clone());
        if title_changed
            && !next_aliases
                .iter()
                .any(|alias| normalize_name(alias) == normalize_name(&old_title))
        {
            next_aliases.push(old_title.clone());
        }
        let page = Page {
            id: current.stored.page.id.clone(),
            title: title.unwrap_or(current.stored.page.title.clone()),
            aliases: next_aliases,
            kind: kind.unwrap_or(current.stored.page.kind),
            summary: summary.or(current.stored.page.summary.clone()),
            sources: sources.unwrap_or(current.stored.page.sources.clone()),
            scope: scope.or(current.stored.page.scope.clone()),
            disambiguation: current.stored.page.disambiguation.clone(),
            content: content.unwrap_or(current.stored.page.content.clone()),
            revision: current.stored.page.revision + 1,
            created_at: current.stored.page.created_at.clone(),
            updated_at: now(),
        };
        let page = if title_changed {
            Page {
                content: rewrite_wiki_link_target(&page.content, &old_title, &page.title),
                ..page
            }
        } else {
            page
        };
        if page.title == current.stored.page.title
            && page.aliases == current.stored.page.aliases
            && page.kind == current.stored.page.kind
            && page.summary == current.stored.page.summary
            && page.sources == current.stored.page.sources
            && page.scope == current.stored.page.scope
            && page.content == current.stored.page.content
        {
            return Err(ServiceError::conflict());
        }
        self.validate_candidate(&page, Some(&id))?;
        let destination = if title_changed {
            self.renamed_page_relative_path(&page.title, &current.relative_path)?
        } else {
            current.relative_path.clone()
        };
        let mut paths = BTreeSet::from([current.relative_path.clone(), destination.clone()]);
        let mut linked_writes = Vec::new();
        if title_changed {
            for record in self.active_records()? {
                if record.stored.page.id == id {
                    continue;
                }
                let rewritten =
                    rewrite_wiki_link_target(&record.stored.page.content, &old_title, &page.title);
                if rewritten == record.stored.page.content {
                    continue;
                }
                let linked = StoredPage {
                    page: Page {
                        content: rewritten,
                        revision: record.stored.page.revision + 1,
                        updated_at: page.updated_at.clone(),
                        ..record.stored.page
                    },
                    deleted: false,
                    extra: record.stored.extra,
                };
                paths.insert(record.relative_path.clone());
                linked_writes.push((linked, record.relative_path));
            }
        }
        self.write_stored(
            &StoredPage {
                page: page.clone(),
                deleted: false,
                extra: current.stored.extra,
            },
            &destination,
        )?;
        if destination != current.relative_path {
            self.stage_remove(&current.relative_path)?;
        }
        for (linked, path) in linked_writes {
            self.write_stored(&linked, &path)?;
        }
        let paths: Vec<PathBuf> = paths.into_iter().collect();
        let version = self.commit_page(&id, "update", &paths)?;
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
            .filter(|record| !record.stored.deleted)
            .ok_or_else(ServiceError::not_found)?;
        if current.stored.page.revision != expected_revision {
            return Err(ServiceError::conflict());
        }
        let page = Page {
            content: String::new(),
            revision: current.stored.page.revision + 1,
            updated_at: now(),
            ..current.stored.page
        };
        let tombstone_path = tombstone_relative_path(id);
        self.write_stored(
            &StoredPage {
                page: page.clone(),
                deleted: true,
                extra: current.stored.extra,
            },
            &tombstone_path,
        )?;
        self.stage_remove(&current.relative_path)?;
        let version = self.commit_page(id, "delete", &[current.relative_path, tombstone_path])?;
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
        let active_records = self.active_records()?;
        let active: Vec<Page> = active_records
            .iter()
            .map(|record| record.stored.page.clone())
            .collect();
        let by_slug: BTreeMap<String, &Page> = active_records
            .iter()
            .map(|record| (page_slug(&record.relative_path), &record.stored.page))
            .collect();
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
            if let Some(page) = by_slug.get(&result.slug) {
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
                || page
                    .summary
                    .as_ref()
                    .is_some_and(|summary| normalize_name(summary).contains(&normalized_query))
                || normalize_name(page.kind.label()).contains(&normalized_query)
                || page
                    .scope
                    .as_ref()
                    .is_some_and(|scope| normalize_name(scope).contains(&normalized_query))
                || page
                    .sources
                    .iter()
                    .any(|source| normalize_name(source).contains(&normalized_query))
                || normalize_name(&page.content).contains(&normalized_query)
            {
                seen.insert(page.id.clone());
                pages.push(page.clone());
            }
        }
        pages.truncate(limit);
        let records_by_id: BTreeMap<String, PageRecord> = active_records
            .into_iter()
            .map(|record| (record.stored.page.id.clone(), record))
            .collect();
        let results: Vec<SearchHit> = pages
            .iter()
            .map(|page| search_hit(page, &records_by_id))
            .collect();
        Ok(json!({ "pages": pages, "results": results }))
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
        let record =
            page_at_commit_by_id(&repo, &commit, id)?.ok_or_else(ServiceError::not_found)?;
        let operation =
            page_operation_at_commit(&repo, &commit, id)?.ok_or_else(ServiceError::not_found)?;
        Ok(json!({
            "version": {
                "version": commit.id().to_string(),
                "revision": record.stored.page.revision,
                "operation": operation,
                "committedAt": commit_timestamp(&commit)?,
                "page": record.stored.page,
                "deleted": record.stored.deleted
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
        let from =
            page_at_commit_by_id(&repo, &from_commit, id)?.ok_or_else(ServiceError::not_found)?;
        let to =
            page_at_commit_by_id(&repo, &to_commit, id)?.ok_or_else(ServiceError::not_found)?;
        let from = render_stored_page(&from.stored)?;
        let to = render_stored_page(&to.stored)?;
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
        if current.stored.page.revision != expected_revision {
            return Err(ServiceError::conflict());
        }
        let target = {
            let repo = self.repository()?;
            let commit = repo
                .find_commit(parse_version(version)?)
                .map_err(|_| ServiceError::not_found())?;
            page_at_commit_by_id(&repo, &commit, id)?
                .filter(|record| !record.stored.deleted)
                .ok_or_else(ServiceError::not_found)?
        };
        let page = Page {
            id: id.to_owned(),
            title: target.stored.page.title,
            aliases: target.stored.page.aliases,
            kind: target.stored.page.kind,
            summary: target.stored.page.summary,
            sources: target.stored.page.sources,
            scope: target.stored.page.scope,
            disambiguation: target.stored.page.disambiguation,
            content: target.stored.page.content,
            revision: current.stored.page.revision + 1,
            created_at: current.stored.page.created_at,
            updated_at: now(),
        };
        self.validate_candidate(&page, Some(id))?;
        let destination = if target.relative_path.starts_with(PAGES_DIRECTORY) {
            self.unique_page_relative_path(&page.title, Some(&target.relative_path))?
        } else {
            self.unique_page_relative_path(&page.title, None)?
        };
        let tombstone_path = current.relative_path;
        self.write_stored(
            &StoredPage {
                page: page.clone(),
                deleted: false,
                extra: target.stored.extra,
            },
            &destination,
        )?;
        if self.transaction_read(&tombstone_path)?.is_some() {
            self.stage_remove(&tombstone_path)?;
        }
        let version = self.commit_page(id, "restore", &[tombstone_path, destination])?;
        Ok(json!({ "page": page, "version": version }))
    }

    fn active_pages(&self) -> std::result::Result<Vec<Page>, ServiceError> {
        let mut pages: Vec<Page> = self
            .active_records()?
            .into_iter()
            .map(|record| record.stored.page)
            .collect();
        pages.sort_by(|left, right| left.title.cmp(&right.title).then(left.id.cmp(&right.id)));
        Ok(pages)
    }

    fn active_records(&self) -> std::result::Result<Vec<PageRecord>, ServiceError> {
        let mut records: Vec<PageRecord> = self
            .stored_pages()?
            .into_iter()
            .filter(|record| !record.stored.deleted)
            .collect();
        records.sort_by(|left, right| {
            left.stored
                .page
                .title
                .cmp(&right.stored.page.title)
                .then(left.stored.page.id.cmp(&right.stored.page.id))
        });
        Ok(records)
    }

    fn stored_pages(&self) -> std::result::Result<Vec<PageRecord>, ServiceError> {
        let mut records = self.scan_page_directory(&self.pages_root, false)?;
        records.extend(self.scan_page_directory(&self.root.join(TOMBSTONES_DIRECTORY), true)?);
        if records.len() > MAX_PAGES {
            return Err(ServiceError::corrupt());
        }
        let mut ids = HashSet::new();
        for record in &records {
            if !ids.insert(record.stored.page.id.clone()) {
                return Err(ServiceError::corrupt());
            }
        }
        let active: Vec<Page> = records
            .iter()
            .filter(|record| !record.stored.deleted)
            .map(|record| record.stored.page.clone())
            .collect();
        validate_page_set(&active).map_err(|error| {
            if error.code == "conflict" {
                error
            } else {
                ServiceError::corrupt()
            }
        })?;
        Ok(records)
    }

    fn scan_page_directory(
        &self,
        directory: &Path,
        expect_deleted: bool,
    ) -> std::result::Result<Vec<PageRecord>, ServiceError> {
        let mut records = Vec::new();
        for relative in self.transaction_paths_under(directory) {
            if relative.extension().and_then(|value| value.to_str()) != Some("md") {
                continue;
            }
            if relative.components().any(|component| {
                component
                    .as_os_str()
                    .to_str()
                    .is_some_and(|value| value.starts_with('.') && value != ".swarmx")
            }) {
                continue;
            }
            let Some(raw) = self
                .transaction_read(&relative)?
                .and_then(|bytes| String::from_utf8(bytes).ok())
            else {
                continue;
            };
            let stored = parse_stored_page(&raw)?;
            if stored.deleted != expect_deleted {
                return Err(ServiceError::corrupt());
            }
            records.push(PageRecord {
                stored,
                relative_path: relative,
            });
        }
        Ok(records)
    }

    fn read_current(&self, id: &str) -> std::result::Result<Option<PageRecord>, ServiceError> {
        Ok(self
            .stored_pages()?
            .into_iter()
            .find(|record| record.stored.page.id == id))
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

    fn write_stored(
        &mut self,
        stored: &StoredPage,
        relative_path: &Path,
    ) -> std::result::Result<(), ServiceError> {
        validate_page(&stored.page)?;
        validate_page_relative_path(relative_path, stored.deleted)?;
        let rendered = render_stored_page(stored)?;
        self.stage_write(relative_path, rendered.into_bytes())
    }

    fn commit_page(
        &mut self,
        id: &str,
        operation: &str,
        relative_paths: &[PathBuf],
    ) -> std::result::Result<String, ServiceError> {
        let mut paths = relative_paths.to_vec();
        paths.extend(self.write_vault_views()?);
        paths.sort();
        paths.dedup();
        self.set_transaction_message(format!("memory:{operation}:{id}"));
        let version = self.finish_transaction()?;
        if version.is_empty() {
            return Err(ServiceError::internal());
        }
        Ok(version)
    }

    fn write_vault_views(&mut self) -> std::result::Result<Vec<PathBuf>, ServiceError> {
        let records = self.active_records()?;
        let (index, disambiguation) = render_vault_views(&records);
        let mut changed = Vec::new();
        for (relative, content) in [
            (PathBuf::from(INDEX_FILE), index),
            (PathBuf::from(DISAMBIGUATION_FILE), disambiguation),
        ] {
            if self.stage_if_changed(&relative, content.into_bytes())? {
                changed.push(relative);
            }
        }
        Ok(changed)
    }

    fn unique_page_relative_path(
        &self,
        title: &str,
        preferred: Option<&Path>,
    ) -> std::result::Result<PathBuf, ServiceError> {
        if let Some(preferred) = preferred
            && preferred.starts_with(PAGES_DIRECTORY)
            && preferred.extension().and_then(|value| value.to_str()) == Some("md")
            && self.transaction_read(preferred)?.is_none()
        {
            validate_page_relative_path(preferred, false)?;
            return Ok(preferred.to_path_buf());
        }
        let stem = safe_file_stem(title);
        for suffix in 1..=MAX_PAGES + 1 {
            let filename = if suffix == 1 {
                format!("{stem}.md")
            } else {
                format!("{stem} ({suffix}).md")
            };
            let relative = PathBuf::from(PAGES_DIRECTORY).join(filename);
            if self.transaction_read(&relative)?.is_none() {
                return Ok(relative);
            }
        }
        Err(ServiceError::conflict())
    }

    fn renamed_page_relative_path(
        &self,
        title: &str,
        current: &Path,
    ) -> std::result::Result<PathBuf, ServiceError> {
        let parent = current.parent().ok_or_else(ServiceError::internal)?;
        let stem = safe_file_stem(title);
        for suffix in 1..=MAX_PAGES + 1 {
            let filename = if suffix == 1 {
                format!("{stem}.md")
            } else {
                format!("{stem} ({suffix}).md")
            };
            let candidate = parent.join(filename);
            if candidate == current || self.transaction_read(&candidate)?.is_none() {
                validate_page_relative_path(&candidate, false)?;
                return Ok(candidate);
            }
        }
        Err(ServiceError::conflict())
    }

    fn migrate_legacy_paths(&mut self) -> Result<()> {
        let records = self
            .scan_page_directory(&self.pages_root, false)
            .map_err(|error| anyhow::anyhow!(error.safe_message))?;
        let mut reserved: BTreeSet<PathBuf> = records
            .iter()
            .map(|record| record.relative_path.clone())
            .collect();
        let mut moves = Vec::new();
        let mut changed = BTreeSet::new();
        for record in records {
            let legacy =
                PathBuf::from(PAGES_DIRECTORY).join(format!("{}.md", record.stored.page.id));
            if record.relative_path != legacy {
                continue;
            }
            reserved.remove(&record.relative_path);
            let stem = safe_file_stem(&record.stored.page.title);
            let destination = (1..=MAX_PAGES + 1)
                .map(|suffix| {
                    let filename = if suffix == 1 {
                        format!("{stem}.md")
                    } else {
                        format!("{stem} ({suffix}).md")
                    };
                    PathBuf::from(PAGES_DIRECTORY).join(filename)
                })
                .find(|candidate| {
                    !reserved.contains(candidate)
                        && self.transaction_read(candidate).ok().flatten().is_none()
                })
                .ok_or_else(|| anyhow::anyhow!(ServiceError::conflict().safe_message))?;
            reserved.insert(destination.clone());
            changed.insert(record.relative_path.clone());
            changed.insert(destination.clone());
            moves.push((record.relative_path, destination));
        }
        if !changed.is_empty() {
            for (source, destination) in moves {
                let content = self
                    .transaction_read(&source)
                    .map_err(|error| anyhow::anyhow!(error.safe_message))?
                    .ok_or_else(|| anyhow::anyhow!("legacy Memory page disappeared"))?;
                self.stage_write(&destination, content)
                    .map_err(|error| anyhow::anyhow!(error.safe_message))?;
                self.stage_remove(&source)
                    .map_err(|error| anyhow::anyhow!(error.safe_message))?;
            }
            changed.extend(
                self.write_vault_views()
                    .map_err(|error| anyhow::anyhow!(error.safe_message))?,
            );
            self.set_transaction_message("memory:migrate:human-vault");
            self.finish_transaction()
                .map_err(|error| anyhow::anyhow!(error.safe_message))?;
        }
        Ok(())
    }

    fn reconcile_pages(&mut self) -> std::result::Result<(), ServiceError> {
        let repo = self.repository()?;
        let head = match repo.head().and_then(|head| head.peel_to_commit()) {
            Ok(commit) => commit,
            Err(_) => return Ok(()),
        };
        let committed = page_records_at_commit(&repo, &head)?;
        let committed_by_path: BTreeMap<PathBuf, String> = committed
            .iter()
            .map(|record| (record.relative_path.clone(), record.stored.page.id.clone()))
            .collect();
        let committed_by_title: BTreeMap<String, String> = committed
            .iter()
            .filter(|record| !record.stored.deleted)
            .map(|record| {
                (
                    normalize_name(&record.stored.page.title),
                    record.stored.page.id.clone(),
                )
            })
            .collect();
        let mut committed_by_id: BTreeMap<String, PageRecord> = committed
            .into_iter()
            .map(|record| (record.stored.page.id.clone(), record))
            .collect();
        let mut working = self.scan_active_worktree_relaxed()?;
        working.extend(self.scan_page_directory(&self.root.join(TOMBSTONES_DIRECTORY), true)?);
        if working.len() > MAX_PAGES {
            return Err(ServiceError::corrupt());
        }

        let timestamp = now();
        let mut seen = HashSet::new();
        let mut normalized = Vec::new();
        let mut writes = Vec::new();
        let mut paths = BTreeSet::new();
        let mut labels = Vec::new();
        for mut record in working {
            let requested_id = record.stored.page.id.clone();
            let previous_id = committed_by_path
                .get(&record.relative_path)
                .cloned()
                .or_else(|| {
                    committed_by_id
                        .contains_key(&requested_id)
                        .then_some(requested_id)
                })
                .or_else(|| {
                    committed_by_title
                        .get(&normalize_name(&record.stored.page.title))
                        .cloned()
                });
            let previous = previous_id
                .as_deref()
                .and_then(|id| committed_by_id.remove(id));
            if let Some(previous) = &previous {
                record.stored.page.id = previous.stored.page.id.clone();
            }
            let id = record.stored.page.id.clone();
            if !seen.insert(id.clone()) {
                return Err(ServiceError::corrupt());
            }
            match previous {
                Some(previous) if record.stored.deleted => {
                    if record != previous {
                        return Err(ServiceError::corrupt());
                    }
                }
                Some(previous) => {
                    if previous.stored.deleted || record.stored != previous.stored {
                        record.stored.page.revision = previous.stored.page.revision + 1;
                        record.stored.page.created_at = previous.stored.page.created_at.clone();
                        record.stored.page.updated_at = timestamp.clone();
                        record.stored.deleted = false;
                        writes.push((record.stored.clone(), record.relative_path.clone()));
                        labels.push(("update", id.clone()));
                    } else if record.relative_path != previous.relative_path {
                        labels.push(("move", id.clone()));
                    }
                    if record.relative_path != previous.relative_path
                        || record.stored != previous.stored
                    {
                        if record.relative_path != previous.relative_path {
                            self.stage_remove(&previous.relative_path)?;
                        }
                        paths.insert(previous.relative_path);
                        paths.insert(record.relative_path.clone());
                    }
                }
                None if record.stored.deleted => {
                    return Err(ServiceError::corrupt());
                }
                None => {
                    record.stored.page.revision = 1;
                    record.stored.page.created_at = timestamp.clone();
                    record.stored.page.updated_at = timestamp.clone();
                    writes.push((record.stored.clone(), record.relative_path.clone()));
                    paths.insert(record.relative_path.clone());
                    labels.push(("create", id));
                }
            }
            normalized.push(record);
        }

        for (_, previous) in committed_by_id {
            if previous.stored.deleted {
                return Err(ServiceError::corrupt());
            }
            let tombstone_path = tombstone_relative_path(&previous.stored.page.id);
            let tombstone = StoredPage {
                page: Page {
                    content: String::new(),
                    revision: previous.stored.page.revision + 1,
                    updated_at: timestamp.clone(),
                    ..previous.stored.page
                },
                deleted: true,
                extra: previous.stored.extra,
            };
            writes.push((tombstone.clone(), tombstone_path.clone()));
            self.stage_remove(&previous.relative_path)?;
            paths.insert(previous.relative_path);
            paths.insert(tombstone_path.clone());
            labels.push(("delete", tombstone.page.id.clone()));
            normalized.push(PageRecord {
                stored: tombstone,
                relative_path: tombstone_path,
            });
        }

        let active: Vec<Page> = normalized
            .iter()
            .filter(|record| !record.stored.deleted)
            .map(|record| record.stored.page.clone())
            .collect();
        validate_page_set(&active)?;
        if paths.is_empty() {
            paths.extend(self.write_vault_views()?);
            if paths.is_empty() {
                return Ok(());
            }
        } else {
            for (stored, path) in writes {
                self.write_stored(&stored, &path)?;
            }
            paths.extend(self.write_vault_views()?);
        }
        let paths: Vec<PathBuf> = paths.into_iter().collect();
        let message = if labels.len() == 1 {
            format!("memory:{}:{}", labels[0].0, labels[0].1)
        } else {
            "memory:reconcile:vault".to_owned()
        };
        self.set_transaction_message(message);
        let version = self.finish_transaction()?;
        if version.is_empty() && !paths.is_empty() {
            return Err(ServiceError::internal());
        }
        Ok(())
    }

    fn scan_active_worktree_relaxed(&self) -> std::result::Result<Vec<PageRecord>, ServiceError> {
        let mut records = Vec::new();
        for relative_path in self.transaction_paths_under(&self.pages_root) {
            if relative_path.extension().and_then(|value| value.to_str()) != Some("md") {
                continue;
            }
            if relative_path.components().any(|component| {
                component
                    .as_os_str()
                    .to_str()
                    .is_some_and(|value| value.starts_with('.'))
            }) {
                continue;
            }
            validate_page_relative_path(&relative_path, false)?;
            let raw = String::from_utf8(
                self.transaction_read(&relative_path)?
                    .ok_or_else(ServiceError::corrupt)?,
            )
            .map_err(|_| ServiceError::corrupt())?;
            let stored = match parse_stored_page(&raw) {
                Ok(stored) if !stored.deleted => stored,
                Ok(_) => return Err(ServiceError::corrupt()),
                Err(_) => parse_human_page(&raw, &relative_path)?,
            };
            records.push(PageRecord {
                stored,
                relative_path,
            });
        }
        Ok(records)
    }

    fn global_file(
        &self,
        target: GlobalMemoryTarget,
    ) -> std::result::Result<GlobalMemoryFile, ServiceError> {
        let content = match self.transaction_read(Path::new(target.file_name()))? {
            Some(bytes) => {
                let content = String::from_utf8(bytes).map_err(|_| ServiceError::corrupt())?;
                validate_global_content(&content).map_err(|_| ServiceError::corrupt())?;
                Some(content)
            }
            None => None,
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
        &mut self,
        target: GlobalMemoryTarget,
    ) -> std::result::Result<(), ServiceError> {
        let relative = Path::new(target.file_name());
        let worktree = match self.transaction_read(relative)? {
            Some(bytes) => {
                let content = String::from_utf8(bytes).map_err(|_| ServiceError::corrupt())?;
                validate_global_content(&content).map_err(|_| ServiceError::corrupt())?;
                Some(content)
            }
            None => None,
        };
        let repo = self.repository()?;
        let committed = match repo.head().and_then(|head| head.peel_to_commit()) {
            Ok(commit) => raw_page_at_commit(&repo, &commit, Path::new(target.file_name()))?,
            Err(_) => None,
        };
        if worktree != committed {
            match worktree {
                Some(content) => self.stage_write(relative, content.into_bytes())?,
                None => self.stage_remove(relative)?,
            }
            self.set_transaction_message(format!("memory:external_edit:{}", target.file_name()));
            let version = self.finish_transaction()?;
            if version.is_empty() {
                return Err(ServiceError::internal());
            }
        }
        Ok(())
    }

    fn write_global_file(
        &mut self,
        target: GlobalMemoryTarget,
        content: &str,
    ) -> std::result::Result<(), ServiceError> {
        validate_global_content(content)?;
        self.stage_write(Path::new(target.file_name()), content.as_bytes().to_vec())
    }

    fn commit_global_file(
        &mut self,
        target: GlobalMemoryTarget,
        operation: &str,
    ) -> std::result::Result<String, ServiceError> {
        self.set_transaction_message(format!("memory:{operation}:{}", target.file_name()));
        let version = self.finish_transaction()?;
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
        let mut versions = Vec::new();
        for oid in revwalk {
            let oid = oid.map_err(|_| ServiceError::corrupt())?;
            let commit = repo.find_commit(oid).map_err(|_| ServiceError::corrupt())?;
            let current = page_at_commit_by_id(&repo, &commit, id)?;
            let parent = if commit.parent_count() > 0 {
                page_at_commit_by_id(
                    &repo,
                    &commit.parent(0).map_err(|_| ServiceError::corrupt())?,
                    id,
                )?
            } else {
                None
            };
            if current.as_ref().map(|record| &record.stored)
                == parent.as_ref().map(|record| &record.stored)
            {
                continue;
            }
            let Some(record) = current else {
                continue;
            };
            let operation =
                page_operation(&record.stored, parent.as_ref().map(|item| &item.stored))
                    .ok_or_else(ServiceError::corrupt)?;
            versions.push(json!({
                "version": commit.id().to_string(),
                "revision": record.stored.page.revision,
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

fn validate_transaction_relative_path(relative: &Path) -> std::result::Result<(), ServiceError> {
    if relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        return Err(ServiceError::internal());
    }
    Ok(())
}

fn validate_global_content(content: &str) -> std::result::Result<(), ServiceError> {
    if content.trim().is_empty()
        || content.encode_utf16().count() > MAX_GLOBAL_MEMORY_CHARS
        || contains_secret_material(content)
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
        || page
            .summary
            .as_ref()
            .is_some_and(|summary| summary.trim() != summary || summary.chars().count() > 2_000)
        || page.sources.len() > 32
        || page.sources.iter().any(|source| {
            source.trim() != source
                || source.is_empty()
                || source.chars().count() > 4_096
                || source.chars().any(char::is_control)
        })
        || page
            .scope
            .as_ref()
            .is_some_and(|scope| validate_name(scope).is_err())
        || page
            .disambiguation
            .as_ref()
            .is_some_and(|name| validate_name(name).is_err())
        || std::iter::once(page.title.as_str())
            .chain(page.aliases.iter().map(String::as_str))
            .chain(page.summary.iter().map(String::as_str))
            .chain(page.sources.iter().map(String::as_str))
            .chain(page.scope.iter().map(String::as_str))
            .chain(page.disambiguation.iter().map(String::as_str))
            .chain(std::iter::once(page.content.as_str()))
            .any(contains_secret_material)
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

fn contains_secret_material(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    if lower.contains("-----begin private key-----")
        || lower.contains("-----begin rsa private key-----")
        || lower.contains("authorization: bearer ")
    {
        return true;
    }
    for token in lower.split_whitespace() {
        let token = token.trim_matches(|character: char| {
            matches!(
                character,
                '"' | '\'' | '`' | '<' | '>' | '(' | ')' | ',' | ';'
            )
        });
        if [
            "sk-",
            "rk-",
            "pk-",
            "ghp-",
            "gho-",
            "ghu-",
            "ghs-",
            "github_pat-",
            "github_pat_",
            "xoxb-",
            "xoxp-",
        ]
        .iter()
        .any(|prefix| token.starts_with(prefix) && token.len() >= prefix.len() + 4)
        {
            return true;
        }
        if let Some((_, remainder)) = token.split_once("://") {
            let authority = remainder.split('/').next().unwrap_or_default();
            if let Some((credentials, _)) = authority.rsplit_once('@')
                && credentials.contains(':')
            {
                return true;
            }
        }
    }
    value.lines().any(|line| {
        let Some((key, raw_value)) = line.split_once([':', '=']) else {
            return false;
        };
        let key: String = key
            .trim()
            .chars()
            .filter(|character| !matches!(character, '_' | '-' | ' '))
            .flat_map(char::to_lowercase)
            .collect();
        if !matches!(
            key.as_str(),
            "apikey"
                | "accesstoken"
                | "authtoken"
                | "clientsecret"
                | "password"
                | "privatekey"
                | "refreshtoken"
                | "secret"
        ) {
            return false;
        }
        let candidate = raw_value.trim();
        !candidate.is_empty()
            && !matches!(
                candidate.to_ascii_lowercase().as_str(),
                "[redacted]" | "<redacted>" | "redacted" | "none" | "null" | "unset"
            )
            && !candidate.starts_with("${")
    })
}

fn is_timestamp(value: &str) -> bool {
    DateTime::parse_from_rfc3339(value).is_ok()
}

fn page_summary(page: &Page) -> PageSummary {
    PageSummary {
        id: page.id.clone(),
        title: page.title.clone(),
        aliases: page.aliases.clone(),
        kind: page.kind,
        summary: page.summary.clone(),
        sources: page.sources.clone(),
        scope: page.scope.clone(),
        disambiguation: page.disambiguation.clone(),
        revision: page.revision,
        created_at: page.created_at.clone(),
        updated_at: page.updated_at.clone(),
    }
}

fn search_hit(page: &Page, records: &BTreeMap<String, PageRecord>) -> SearchHit {
    let mut name_owners = BTreeMap::new();
    for record in records.values() {
        for name in std::iter::once(&record.stored.page.title).chain(&record.stored.page.aliases) {
            name_owners.insert(normalize_name(name), record.stored.page.title.clone());
        }
    }
    let page_names: BTreeSet<String> = std::iter::once(&page.title)
        .chain(&page.aliases)
        .map(|name| normalize_name(name))
        .collect();
    let mut related = BTreeSet::new();
    for target in wiki_link_targets(&page.content) {
        if let Some(title) = name_owners.get(&normalize_name(&target))
            && normalize_name(title) != normalize_name(&page.title)
        {
            related.insert(title.clone());
        }
    }
    for record in records.values() {
        if record.stored.page.id == page.id {
            continue;
        }
        if wiki_link_targets(&record.stored.page.content)
            .iter()
            .any(|target| page_names.contains(&normalize_name(target)))
        {
            related.insert(record.stored.page.title.clone());
        }
    }
    SearchHit {
        title: page.title.clone(),
        summary: page.summary.clone(),
        kind: page.kind,
        sources: page.sources.clone(),
        related_pages: related.into_iter().collect(),
        id: page.id.clone(),
    }
}

fn render_stored_page(stored: &StoredPage) -> std::result::Result<String, ServiceError> {
    let frontmatter = PageFile {
        id: stored.page.id.clone(),
        title: stored.page.title.clone(),
        aliases: stored.page.aliases.clone(),
        kind: stored.page.kind,
        summary: stored.page.summary.clone(),
        sources: stored.page.sources.clone(),
        scope: stored.page.scope.clone(),
        disambiguation: stored.page.disambiguation.clone(),
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
        extra: stored.extra.clone(),
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
        kind: frontmatter.kind,
        summary: frontmatter.summary,
        sources: frontmatter.sources,
        scope: frontmatter.scope,
        disambiguation: frontmatter.disambiguation,
        content: body.strip_prefix('\n').unwrap_or(body).to_owned(),
        revision: frontmatter.revision,
        created_at: frontmatter.created_at,
        updated_at: frontmatter.updated_at,
    };
    validate_page(&page).map_err(|_| ServiceError::corrupt())?;
    Ok(StoredPage {
        page,
        deleted: frontmatter.deleted,
        extra: frontmatter.extra,
    })
}

fn parse_human_page(
    raw: &str,
    relative_path: &Path,
) -> std::result::Result<StoredPage, ServiceError> {
    let (frontmatter, body) = if let Some(rest) = raw.strip_prefix("---\n") {
        let (yaml, body) = rest
            .split_once("\n---\n")
            .ok_or_else(ServiceError::corrupt)?;
        let frontmatter: HumanPageFile =
            serde_yaml::from_str(yaml).map_err(|_| ServiceError::corrupt())?;
        (frontmatter, body.strip_prefix('\n').unwrap_or(body))
    } else {
        (HumanPageFile::default(), raw)
    };
    if frontmatter.deleted {
        return Err(ServiceError::corrupt());
    }
    let timestamp = now();
    let title = frontmatter.title.unwrap_or_else(|| {
        relative_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("Untitled")
            .to_owned()
    });
    let page = Page {
        id: frontmatter
            .id
            .unwrap_or_else(|| format!("mem_{}", Uuid::new_v4().simple())),
        title,
        aliases: frontmatter.aliases,
        kind: frontmatter.kind,
        summary: frontmatter.summary,
        sources: frontmatter.sources,
        scope: frontmatter.scope,
        disambiguation: frontmatter.disambiguation,
        content: body.to_owned(),
        revision: frontmatter.revision.unwrap_or(1),
        created_at: frontmatter.created_at.unwrap_or_else(|| timestamp.clone()),
        updated_at: frontmatter
            .updated_at
            .or(frontmatter.last_updated)
            .unwrap_or(timestamp),
    };
    validate_page(&page).map_err(|_| ServiceError::corrupt())?;
    Ok(StoredPage {
        page,
        deleted: false,
        extra: frontmatter.extra,
    })
}

fn tombstone_relative_path(id: &str) -> PathBuf {
    PathBuf::from(TOMBSTONES_DIRECTORY).join(format!("{id}.md"))
}

fn page_slug(relative_path: &Path) -> String {
    relative_path
        .strip_prefix(PAGES_DIRECTORY)
        .unwrap_or(relative_path)
        .with_extension("")
        .to_string_lossy()
        .replace('\\', "/")
}

fn safe_file_stem(title: &str) -> String {
    let mut value = String::new();
    let mut replacing = false;
    for character in title.chars() {
        let replace = character.is_control()
            || matches!(
                character,
                '/' | '\\' | '<' | '>' | ':' | '"' | '|' | '?' | '*'
            );
        if replace {
            if !replacing && !value.ends_with('-') {
                value.push('-');
            }
            replacing = true;
        } else {
            value.push(character);
            replacing = false;
        }
    }
    let mut value = value
        .trim_matches(|character: char| character == '-' || character == '.' || character == ' ')
        .to_owned();
    while value.len() > 180 {
        value.pop();
    }
    value = value.trim_end_matches(['.', ' ']).to_owned();
    if value.is_empty() {
        value = "Untitled".to_owned();
    }
    let upper = value.to_ascii_uppercase();
    let reserved = matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (upper.len() == 4
            && (upper.starts_with("COM") || upper.starts_with("LPT"))
            && upper.as_bytes()[3].is_ascii_digit());
    if reserved {
        value.push_str(" note");
    }
    value
}

fn qualified_page_title(base: &str, page: &Page, occupied: &BTreeSet<String>) -> String {
    let descriptor = page
        .scope
        .as_ref()
        .map(|scope| format!("{}, {scope}", page.kind.label()))
        .unwrap_or_else(|| page.kind.label().to_owned());
    for suffix in 1..=MAX_PAGES + 1 {
        let qualifier = if suffix == 1 {
            descriptor.clone()
        } else {
            format!("{descriptor}, {suffix}")
        };
        let trailer = format!(" ({qualifier})");
        let available = 256_usize.saturating_sub(trailer.chars().count());
        let prefix: String = base.chars().take(available).collect();
        let candidate = format!("{prefix}{trailer}");
        if !occupied.contains(&normalize_name(&candidate)) {
            return candidate;
        }
    }
    format!("Untitled ({})", page.kind.label())
}

fn render_vault_views(records: &[PageRecord]) -> (String, String) {
    let mut name_owners = BTreeMap::new();
    for record in records {
        let page = &record.stored.page;
        for name in std::iter::once(&page.title).chain(&page.aliases) {
            name_owners.insert(normalize_name(name), page.title.clone());
        }
    }
    let mut backlinks: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for record in records {
        for target in wiki_link_targets(&record.stored.page.content) {
            if let Some(title) = name_owners.get(&normalize_name(&target))
                && *title != record.stored.page.title
            {
                backlinks
                    .entry(title.clone())
                    .or_default()
                    .insert(record.stored.page.title.clone());
            }
        }
    }

    let mut index = String::from(
        "# Memory Index\n\n> Generated from the Markdown pages in this vault. Edit the pages, not this view.\n\n## Pages\n\n",
    );
    if records.is_empty() {
        index.push_str("_No pages yet._\n");
    }
    for record in records {
        let page = &record.stored.page;
        index.push_str(&format!("- [[{}]] — {}", page.title, page.kind.label()));
        if let Some(scope) = &page.scope {
            index.push_str(&format!(" · {}", markdown_inline(scope)));
        }
        if let Some(summary) = &page.summary {
            index.push_str(&format!(" — {}", markdown_inline(summary)));
        }
        if let Some(source) = page.sources.first() {
            index.push_str(&format!(" · source: {}", markdown_inline(source)));
        }
        index.push('\n');
    }
    index.push_str("\n## Backlinks\n\n");
    if backlinks.is_empty() {
        index.push_str("_No backlinks yet._\n");
    }
    for (target, sources) in backlinks {
        let sources = sources
            .into_iter()
            .map(|source| format!("[[{source}]]"))
            .collect::<Vec<_>>()
            .join(", ");
        index.push_str(&format!("- [[{target}]] ← {sources}\n"));
    }

    let mut groups: BTreeMap<String, Vec<&Page>> = BTreeMap::new();
    for record in records {
        if let Some(base) = &record.stored.page.disambiguation {
            groups
                .entry(base.clone())
                .or_default()
                .push(&record.stored.page);
        }
    }
    let mut disambiguation = String::from(
        "# Memory Disambiguation\n\n> Generated from pages that share a human name.\n",
    );
    if groups.is_empty() {
        disambiguation.push_str("\n_No ambiguous names._\n");
    }
    for (base, mut pages) in groups {
        pages.sort_by(|left, right| left.title.cmp(&right.title));
        disambiguation.push_str(&format!("\n## {}\n\n", markdown_inline(&base)));
        for page in pages {
            disambiguation.push_str(&format!("- [[{}]] — {}", page.title, page.kind.label()));
            if let Some(scope) = &page.scope {
                disambiguation.push_str(&format!(" · {}", markdown_inline(scope)));
            }
            if let Some(summary) = &page.summary {
                disambiguation.push_str(&format!(" — {}", markdown_inline(summary)));
            }
            disambiguation.push('\n');
        }
    }
    (index, disambiguation)
}

fn wiki_link_targets(markdown: &str) -> Vec<String> {
    let mut targets = Vec::new();
    let mut fenced = false;
    for line in markdown.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        let mut offset = 0;
        while let Some(start) = line[offset..].find("[[") {
            let start = offset + start;
            if line[..start].matches('`').count() % 2 == 1 {
                offset = start + 2;
                continue;
            }
            let Some(end) = line[start + 2..].find("]]") else {
                break;
            };
            let end = start + 2 + end;
            let inner = &line[start + 2..end];
            let destination = inner.split_once('|').map_or(inner, |(value, _)| value);
            let target = destination
                .split_once('#')
                .map_or(destination, |(value, _)| value)
                .trim()
                .strip_suffix(".md")
                .unwrap_or_else(|| {
                    destination
                        .split_once('#')
                        .map_or(destination, |(value, _)| value)
                        .trim()
                })
                .trim();
            if !target.is_empty() {
                targets.push(target.to_owned());
            }
            offset = end + 2;
        }
    }
    targets
}

fn rewrite_wiki_link_target(markdown: &str, old_title: &str, new_title: &str) -> String {
    let mut rendered = String::with_capacity(markdown.len());
    let mut fenced = false;
    for segment in markdown.split_inclusive('\n') {
        let line = segment.strip_suffix('\n').unwrap_or(segment);
        let newline = segment.ends_with('\n');
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            fenced = !fenced;
            rendered.push_str(line);
            if newline {
                rendered.push('\n');
            }
            continue;
        }
        if fenced {
            rendered.push_str(line);
            if newline {
                rendered.push('\n');
            }
            continue;
        }
        let mut offset = 0;
        while let Some(relative_start) = line[offset..].find("[[") {
            let start = offset + relative_start;
            let Some(relative_end) = line[start + 2..].find("]]") else {
                break;
            };
            let end = start + 2 + relative_end;
            rendered.push_str(&line[offset..start]);
            let inner = &line[start + 2..end];
            if line[..start].matches('`').count() % 2 == 1 {
                rendered.push_str("[[");
                rendered.push_str(inner);
                rendered.push_str("]]");
                offset = end + 2;
                continue;
            }
            let (destination, alias) = inner
                .split_once('|')
                .map_or((inner, None), |(left, right)| (left, Some(right)));
            let (target, heading) = destination
                .split_once('#')
                .map_or((destination, None), |(left, right)| (left, Some(right)));
            let normalized = target.trim().strip_suffix(".md").unwrap_or(target.trim());
            rendered.push_str("[[");
            if normalize_name(normalized) == normalize_name(old_title) {
                rendered.push_str(new_title);
                if let Some(heading) = heading {
                    rendered.push('#');
                    rendered.push_str(heading);
                }
                if let Some(alias) = alias {
                    rendered.push('|');
                    rendered.push_str(alias);
                }
            } else {
                rendered.push_str(inner);
            }
            rendered.push_str("]]");
            offset = end + 2;
        }
        rendered.push_str(&line[offset..]);
        if newline {
            rendered.push('\n');
        }
    }
    rendered
}

fn markdown_inline(value: &str) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut escaped = String::with_capacity(compact.len());
    for character in compact.chars() {
        if matches!(character, '[' | ']' | '|' | '\\') {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

fn validate_page_relative_path(
    relative_path: &Path,
    deleted: bool,
) -> std::result::Result<(), ServiceError> {
    let expected_root = if deleted {
        Path::new(TOMBSTONES_DIRECTORY)
    } else {
        Path::new(PAGES_DIRECTORY)
    };
    if !relative_path.starts_with(expected_root)
        || relative_path.extension().and_then(|value| value.to_str()) != Some("md")
        || relative_path.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        return Err(ServiceError::invalid());
    }
    Ok(())
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

fn page_records_at_commit(
    repo: &Repository,
    commit: &git2::Commit<'_>,
) -> std::result::Result<Vec<PageRecord>, ServiceError> {
    let tree = commit.tree().map_err(|_| ServiceError::corrupt())?;
    let mut records = Vec::new();
    let mut failure = None;
    tree.walk(git2::TreeWalkMode::PreOrder, |directory, entry| {
        if failure.is_some() || entry.kind() != Some(git2::ObjectType::Blob) {
            return git2::TreeWalkResult::Ok;
        }
        let name = match entry.name() {
            Ok(name) => name,
            Err(_) => {
                failure = Some(ServiceError::corrupt());
                return git2::TreeWalkResult::Abort;
            }
        };
        let relative_path = PathBuf::from(format!("{directory}{name}"));
        let active = relative_path.starts_with(PAGES_DIRECTORY);
        let deleted = relative_path.starts_with(TOMBSTONES_DIRECTORY);
        if (!active && !deleted)
            || relative_path.extension().and_then(|value| value.to_str()) != Some("md")
        {
            return git2::TreeWalkResult::Ok;
        }
        if active
            && relative_path.components().any(|component| {
                component
                    .as_os_str()
                    .to_str()
                    .is_some_and(|value| value.starts_with('.'))
            })
        {
            return git2::TreeWalkResult::Ok;
        }
        let parsed = entry
            .to_object(repo)
            .and_then(|object| object.peel_to_blob())
            .map_err(|_| ServiceError::corrupt())
            .and_then(|blob| {
                std::str::from_utf8(blob.content())
                    .map_err(|_| ServiceError::corrupt())
                    .and_then(parse_stored_page)
            });
        match parsed {
            Ok(stored) if stored.deleted == deleted => records.push(PageRecord {
                stored,
                relative_path,
            }),
            _ => {
                failure = Some(ServiceError::corrupt());
                return git2::TreeWalkResult::Abort;
            }
        }
        git2::TreeWalkResult::Ok
    })
    .map_err(|_| ServiceError::corrupt())?;
    if let Some(error) = failure {
        return Err(error);
    }
    if records.len() > MAX_PAGES {
        return Err(ServiceError::corrupt());
    }
    let mut ids = HashSet::new();
    for record in &records {
        if !ids.insert(record.stored.page.id.clone()) {
            return Err(ServiceError::corrupt());
        }
    }
    Ok(records)
}

fn page_at_commit_by_id(
    repo: &Repository,
    commit: &git2::Commit<'_>,
    id: &str,
) -> std::result::Result<Option<PageRecord>, ServiceError> {
    Ok(page_records_at_commit(repo, commit)?
        .into_iter()
        .find(|record| record.stored.page.id == id))
}

fn page_operation_at_commit(
    repo: &Repository,
    commit: &git2::Commit<'_>,
    id: &str,
) -> std::result::Result<Option<&'static str>, ServiceError> {
    let current = page_at_commit_by_id(repo, commit, id)?;
    let parent = if commit.parent_count() > 0 {
        page_at_commit_by_id(
            repo,
            &commit.parent(0).map_err(|_| ServiceError::corrupt())?,
            id,
        )?
    } else {
        None
    };
    Ok(current
        .and_then(|current| page_operation(&current.stored, parent.as_ref().map(|p| &p.stored))))
}

fn page_operation(current: &StoredPage, parent: Option<&StoredPage>) -> Option<&'static str> {
    match (parent.map(|page| page.deleted), current.deleted) {
        (None, false) => Some("create"),
        (Some(false), false) => Some("update"),
        (Some(false), true) => Some("delete"),
        (Some(true), false) => Some("restore"),
        _ => None,
    }
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
