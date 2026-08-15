use serde_json::{Value, json};
use std::path::Path;
use std::process::Command;
use swarmx_mem::{MemoryService, PROTOCOL_VERSION};
use tempfile::tempdir;

#[test]
fn persists_global_user_and_memory_files_with_revision_conflicts() {
    let root = tempdir().expect("temporary memory");
    let mut service = MemoryService::open(root.path()).expect("open memory");

    let initial = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "global_get"
    })));
    assert_eq!(initial["user"]["content"], Value::Null);
    assert_eq!(initial["memory"]["revision"], 0);

    let saved_user = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "global_save",
        "target": "user",
        "expectedRevision": 0,
        "content": "Prefers concise answers."
    })));
    assert_eq!(saved_user["file"]["fileName"], "USER.md");
    assert_eq!(saved_user["file"]["revision"], 1);
    assert_eq!(
        std::fs::read_to_string(root.path().join("USER.md")).expect("USER.md"),
        "Prefers concise answers."
    );

    let stale = service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "global_save",
        "target": "user",
        "expectedRevision": 0,
        "content": "Stale overwrite"
    }));
    assert_eq!(stale["error"]["code"], "conflict");

    std::fs::write(root.path().join("MEMORY.md"), "SwarmX uses ACP.").expect("manual MEMORY.md");
    let reconciled = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "global_get"
    })));
    assert_eq!(reconciled["memory"]["content"], "SwarmX uses ACP.");
    assert_eq!(reconciled["memory"]["revision"], 1);

    drop(service);
    let mut service = MemoryService::open(root.path()).expect("reopen persisted memory");
    let reopened = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "global_get"
    })));
    assert_eq!(reopened["user"]["content"], "Prefers concise answers.");
    assert_eq!(reopened["memory"]["content"], "SwarmX uses ACP.");

    let forgotten = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "global_forget",
        "target": "memory",
        "expectedRevision": 1
    })));
    assert_eq!(forgotten["file"]["content"], Value::Null);
    assert_eq!(forgotten["file"]["revision"], 2);
    assert!(!root.path().join("MEMORY.md").exists());
}

#[test]
fn persists_crud_search_and_recoverable_versions_in_markdown_git() {
    let root = tempdir().expect("temporary memory");
    let mut service = MemoryService::open(root.path()).expect("open memory");

    let created = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "create",
        "title": "SwarmX",
        "aliases": ["Swarm X"],
        "content": "Uses durable [[Memory]]."
    })));
    let page = created["page"].clone();
    let id = page["id"].as_str().expect("page id").to_owned();
    let create_version = created["version"]
        .as_str()
        .expect("create version")
        .to_owned();
    assert_eq!(page["revision"], 1);

    assert!(root.path().join(".git").is_dir());
    assert!(
        std::fs::read_to_string(root.path().join("README.md"))
            .expect("vault README")
            .contains("Open it directly in Obsidian")
    );
    let page_path = root.path().join("pages/SwarmX.md");
    assert!(page_path.is_file());
    assert!(
        std::fs::read_to_string(&page_path)
            .expect("human-readable page")
            .contains(&format!("id: {id}"))
    );

    let listed = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "list"
    })));
    assert_eq!(listed["pages"].as_array().expect("pages").len(), 1);

    let searched = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "search",
        "query": "durable",
        "limit": 10
    })));
    assert_eq!(searched["pages"][0]["id"], id);

    let updated = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "update",
        "id": id,
        "expectedRevision": 1,
        "content": "Uses versioned [[Memory]]."
    })));
    assert_eq!(updated["page"]["revision"], 2);
    let update_version = updated["version"]
        .as_str()
        .expect("update version")
        .to_owned();

    drop(service);
    let mut service = MemoryService::open(root.path()).expect("reopen persisted memory");
    let reopened = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "get",
        "id": id
    })));
    assert_eq!(reopened["page"]["revision"], 2);
    assert_eq!(reopened["page"]["content"], "Uses versioned [[Memory]].");
    let reopened_search = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "search",
        "query": "versioned",
        "limit": 10
    })));
    assert_eq!(reopened_search["pages"][0]["id"], id);
    let reopened_history = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "history",
        "id": id,
        "limit": 20
    })));
    assert_eq!(
        reopened_history["versions"]
            .as_array()
            .expect("history")
            .len(),
        2
    );

    let conflict = service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "update",
        "id": id,
        "expectedRevision": 1,
        "content": "stale"
    }));
    assert_eq!(conflict["ok"], false);
    assert_eq!(conflict["error"]["code"], "conflict");

    let diff = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "diff",
        "id": id,
        "fromVersion": create_version,
        "toVersion": update_version
    })));
    assert!(
        diff["diff"]["unifiedDiff"]
            .as_str()
            .expect("diff")
            .contains("versioned")
    );

    let deleted = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "delete",
        "id": id,
        "expectedRevision": 2
    })));
    assert_eq!(deleted["page"]["revision"], 3);
    assert_eq!(deleted["page"]["content"], "");
    assert_eq!(
        ok(service.handle(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "operation": "get",
            "id": id
        })))["page"],
        Value::Null
    );

    let history = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "history",
        "id": id,
        "limit": 20
    })));
    assert_eq!(history["versions"].as_array().expect("history").len(), 3);

    let historical = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "get_version",
        "id": id,
        "version": create_version
    })));
    assert_eq!(
        historical["version"]["page"]["content"],
        "Uses durable [[Memory]]."
    );
    assert_eq!(historical["version"]["deleted"], false);

    let restored = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "restore",
        "id": id,
        "expectedRevision": 3,
        "version": create_version
    })));
    assert_eq!(restored["page"]["revision"], 4);
    assert_eq!(restored["page"]["content"], "Uses durable [[Memory]].");
}

#[test]
fn reconciles_an_obsidian_vault_without_using_ids_as_paths() {
    let root = tempdir().expect("temporary memory");
    let mut service = MemoryService::open(root.path()).expect("open memory");

    let created = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "create",
        "title": "Herdr / SwarmX",
        "content": "Initial note."
    })));
    let id = created["page"]["id"].as_str().expect("page id").to_owned();
    let generated_path = root.path().join("pages/Herdr - SwarmX.md");
    assert!(generated_path.is_file());
    assert!(!root.path().join("pages").join(format!("{id}.md")).exists());

    let moved_path = root.path().join("pages/Projects/Herdr.md");
    std::fs::create_dir_all(moved_path.parent().expect("page parent")).expect("nested folder");
    std::fs::rename(&generated_path, &moved_path).expect("move in Obsidian");
    let edited = std::fs::read_to_string(&moved_path)
        .expect("moved page")
        .replace(&format!("id: {id}\n"), "")
        .replace(
            "status: active\n",
            "status: active\ntags: [agent, runtime]\n",
        )
        .replace("Initial note.", "Edited directly in Obsidian.");
    std::fs::write(&moved_path, edited).expect("edit in Obsidian");

    let reconciled = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "get",
        "id": id
    })));
    assert_eq!(reconciled["page"]["revision"], 2);
    assert_eq!(
        reconciled["page"]["content"],
        "Edited directly in Obsidian."
    );
    assert!(
        std::fs::read_to_string(&moved_path)
            .expect("reconciled page")
            .contains("tags:")
    );
    let stale = service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "update",
        "id": id,
        "expectedRevision": 1,
        "content": "Overwrite the human edit."
    }));
    assert_eq!(stale["error"]["code"], "conflict");

    let moved_again = root.path().join("pages/Tools/Herdr.md");
    std::fs::create_dir_all(moved_again.parent().expect("page parent")).expect("nested folder");
    std::fs::rename(&moved_path, &moved_again).expect("second move");
    let listed = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "list"
    })));
    assert_eq!(listed["pages"][0]["revision"], 2);
    assert!(moved_again.is_file());

    let human_path = root.path().join("pages/Research/Human Note.md");
    std::fs::create_dir_all(human_path.parent().expect("human note parent"))
        .expect("human note folder");
    std::fs::write(
        &human_path,
        "---\ntags: [human]\n---\n\n# Human Note\n\nLinks to [[Herdr / SwarmX]].",
    )
    .expect("human-created note");
    let adopted = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "search",
        "query": "Human Note",
        "limit": 10
    })));
    let adopted_page = adopted["pages"]
        .as_array()
        .expect("search pages")
        .iter()
        .find(|page| page["title"] == "Human Note")
        .expect("adopted human page");
    let adopted_id = adopted_page["id"].as_str().expect("adopted id").to_owned();
    let normalized = std::fs::read_to_string(&human_path).expect("normalized human note");
    assert!(normalized.contains(&format!("id: {adopted_id}")));
    assert!(normalized.contains("tags:"));

    std::fs::remove_file(&human_path).expect("delete in Obsidian");
    let deleted = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "get",
        "id": adopted_id
    })));
    assert_eq!(deleted["page"], Value::Null);
    assert!(!human_path.exists());
    assert!(
        root.path()
            .join(".swarmx/tombstones")
            .join(format!("{adopted_id}.md"))
            .is_file()
    );
}

#[test]
fn resolves_sanitized_filename_collisions_with_readable_suffixes() {
    let root = tempdir().expect("temporary memory");
    let mut service = MemoryService::open(root.path()).expect("open memory");
    for title in ["A/B", "A\\B"] {
        ok(service.handle(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "operation": "create",
            "title": title,
            "content": "Collision-safe note."
        })));
    }
    assert!(root.path().join("pages/A-B.md").is_file());
    assert!(root.path().join("pages/A-B (2).md").is_file());
}

#[test]
fn renames_pages_with_inbound_links_and_rebuilds_human_views_in_one_version() {
    let root = tempdir().expect("temporary memory");
    let mut service = MemoryService::open(root.path()).expect("open memory");
    let target = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "create",
        "title": "Old Name",
        "kind": "technology",
        "summary": "A runtime component.",
        "sources": ["https://example.test/runtime"],
        "content": "Target page."
    })));
    let target_id = target["page"]["id"].as_str().expect("target id").to_owned();
    let source = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "create",
        "title": "Source Page",
        "content": "Uses [[Old Name#Details|the old component]]."
    })));
    let source_id = source["page"]["id"].as_str().expect("source id").to_owned();

    let renamed = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "update",
        "id": target_id,
        "expectedRevision": 1,
        "title": "新名称"
    })));
    let version = renamed["version"].as_str().expect("rename version");
    assert_eq!(renamed["page"]["aliases"], json!(["Old Name"]));
    assert!(!root.path().join("pages/Old Name.md").exists());
    assert!(root.path().join("pages/新名称.md").is_file());

    let linked = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "get",
        "id": source_id
    })));
    assert_eq!(linked["page"]["revision"], 2);
    assert_eq!(
        linked["page"]["content"],
        "Uses [[新名称#Details|the old component]]."
    );
    let source_history = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "history",
        "id": source_id,
        "limit": 10
    })));
    assert_eq!(source_history["versions"][0]["version"], version);
    let index = std::fs::read_to_string(root.path().join("INDEX.md")).expect("Memory index");
    assert!(index.contains("[[新名称]] — technology"));
    assert!(index.contains("[[新名称]] ← [[Source Page]]"));
    let searched = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "search",
        "query": "runtime component",
        "limit": 10
    })));
    assert_eq!(searched["results"][0]["title"], "新名称");
    assert_eq!(searched["results"][0]["kind"], "technology");
    assert_eq!(
        searched["results"][0]["relatedPages"],
        json!(["Source Page"])
    );
    assert_eq!(searched["results"][0]["id"], target_id);
}

#[test]
fn disambiguates_same_named_entities_by_type_and_scope_without_overwrite() {
    let root = tempdir().expect("temporary memory");
    let mut service = MemoryService::open(root.path()).expect("open memory");
    let organization = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "create",
        "title": "Mercury",
        "kind": "organization",
        "scope": "Project Atlas",
        "summary": "A company.",
        "content": "Organization page."
    })));
    let organization_id = organization["page"]["id"]
        .as_str()
        .expect("organization id")
        .to_owned();
    let technology = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "create",
        "title": "Mercury",
        "kind": "technology",
        "scope": "Project Orion",
        "summary": "A protocol.",
        "content": "Technology page."
    })));
    assert_eq!(
        technology["page"]["title"],
        "Mercury (technology, Project Orion)"
    );
    let organization = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "get",
        "id": organization_id
    })));
    assert_eq!(
        organization["page"]["title"],
        "Mercury (organization, Project Atlas)"
    );
    assert_eq!(organization["page"]["revision"], 2);
    assert!(
        root.path()
            .join("pages/Mercury (organization, Project Atlas).md")
            .is_file()
    );
    assert!(
        root.path()
            .join("pages/Mercury (technology, Project Orion).md")
            .is_file()
    );
    let disambiguation =
        std::fs::read_to_string(root.path().join("DISAMBIGUATION.md")).expect("disambiguation");
    assert!(disambiguation.contains("## Mercury"));
    assert!(disambiguation.contains("[[Mercury (organization, Project Atlas)]]"));
    assert!(disambiguation.contains("[[Mercury (technology, Project Orion)]]"));
}

#[test]
fn supports_chinese_and_portable_reserved_names_and_rejects_secret_material() {
    let root = tempdir().expect("temporary memory");
    let mut service = MemoryService::open(root.path()).expect("open memory");
    for (title, expected) in [("项目决策", "项目决策.md"), ("CON", "CON note.md")] {
        ok(service.handle(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "operation": "create",
            "title": title,
            "content": "Portable note."
        })));
        assert!(root.path().join("pages").join(expected).is_file());
    }
    let rejected = service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "create",
        "title": "Credentials",
        "content": "api_key = live-secret-value"
    }));
    assert_eq!(rejected["error"]["code"], "invalid_input");
    assert!(!root.path().join("pages/Credentials.md").exists());
    for request in [
        json!({
            "protocolVersion": PROTOCOL_VERSION,
            "operation": "create",
            "title": "Unsafe source",
            "sources": ["https://user:password@example.test/reference"],
            "content": "Ordinary note."
        }),
        json!({
            "protocolVersion": PROTOCOL_VERSION,
            "operation": "create",
            "title": "sk-live-token",
            "content": "Ordinary note."
        }),
        json!({
            "protocolVersion": PROTOCOL_VERSION,
            "operation": "global_save",
            "target": "memory",
            "expectedRevision": 0,
            "content": "password = live-secret-value"
        }),
    ] {
        assert_eq!(service.handle(request)["error"]["code"], "invalid_input");
    }
    assert!(!root.path().join("MEMORY.md").exists());
}

#[test]
fn rejects_conflicting_rename_before_mutation_and_reopens_views_idempotently() {
    let root = tempdir().expect("temporary memory");
    let mut service = MemoryService::open(root.path()).expect("open memory");
    let first = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "create",
        "title": "First",
        "content": "First body."
    })));
    ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "create",
        "title": "Second",
        "aliases": ["Occupied"],
        "content": "Second body."
    })));
    let first_id = first["page"]["id"].as_str().expect("first id").to_owned();
    let before = std::fs::read_to_string(root.path().join("pages/First.md")).expect("first page");
    let conflict = service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "update",
        "id": first_id,
        "expectedRevision": 1,
        "title": "Occupied"
    }));
    assert_eq!(conflict["error"]["code"], "conflict");
    assert_eq!(
        std::fs::read_to_string(root.path().join("pages/First.md")).expect("unchanged first"),
        before
    );
    drop(service);
    let mut service = MemoryService::open(root.path()).expect("first reopen");
    let generation = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "snapshot"
    })))["generation"]
        .as_u64()
        .expect("generation");
    drop(service);
    let mut service = MemoryService::open(root.path()).expect("second reopen");
    let reopened_generation = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "snapshot"
    })))["generation"]
        .as_u64()
        .expect("reopened generation");
    assert_eq!(reopened_generation, generation);
}

#[test]
fn rolls_back_page_links_and_views_when_the_git_commit_cannot_be_written() {
    let root = tempdir().expect("temporary memory");
    let mut service = MemoryService::open(root.path()).expect("open memory");
    let target = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "create",
        "title": "Rollback Target",
        "content": "Target body."
    })));
    let id = target["page"]["id"].as_str().expect("target id").to_owned();
    ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "create",
        "title": "Rollback Source",
        "content": "Links to [[Rollback Target]]."
    })));
    let index_before = std::fs::read_to_string(root.path().join("INDEX.md")).expect("index before");
    let git_index = root.path().join(".git/index");
    let saved_index = root.path().join(".git/index.saved");
    std::fs::rename(&git_index, &saved_index).expect("save git index");
    std::fs::create_dir(&git_index).expect("block git index writes");

    let failed = service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "update",
        "id": id,
        "expectedRevision": 1,
        "title": "Renamed Target"
    }));
    assert_eq!(failed["error"]["code"], "internal");
    assert!(root.path().join("pages/Rollback Target.md").is_file());
    assert!(!root.path().join("pages/Renamed Target.md").exists());
    assert!(
        std::fs::read_to_string(root.path().join("pages/Rollback Source.md"))
            .expect("restored inbound page")
            .contains("[[Rollback Target]]")
    );
    assert_eq!(
        std::fs::read_to_string(root.path().join("INDEX.md")).expect("restored index"),
        index_before
    );

    std::fs::remove_dir(&git_index).expect("remove blocked git index");
    std::fs::rename(&saved_index, &git_index).expect("restore git index");
    let current = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "get",
        "id": id
    })));
    assert_eq!(current["page"]["title"], "Rollback Target");
    assert_eq!(current["page"]["revision"], 1);
}

#[test]
fn migrates_legacy_id_named_pages_without_changing_identity_or_revision() {
    let root = tempdir().expect("temporary memory");
    let mut service = MemoryService::open(root.path()).expect("open memory");
    let created = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "create",
        "title": "Herdr",
        "content": "Legacy page."
    })));
    let id = created["page"]["id"].as_str().expect("page id").to_owned();
    let create_version = created["version"].as_str().expect("version").to_owned();
    drop(service);

    let natural_path = root.path().join("pages/Herdr.md");
    let legacy_path = root.path().join("pages").join(format!("{id}.md"));
    std::fs::rename(&natural_path, &legacy_path).expect("simulate legacy page");
    let legacy_relative = Path::new("pages").join(format!("{id}.md"));
    commit_paths(
        root.path(),
        &[Path::new("pages/Herdr.md"), legacy_relative.as_path()],
        "test: legacy layout",
    );

    let mut service = MemoryService::open(root.path()).expect("reopen and migrate");
    assert!(natural_path.is_file());
    assert!(!legacy_path.exists());
    let page = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "get",
        "id": id
    })));
    assert_eq!(page["page"]["id"], id);
    assert_eq!(page["page"]["revision"], 1);
    let historical = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "get_version",
        "id": id,
        "version": create_version
    })));
    assert_eq!(historical["version"]["page"]["content"], "Legacy page.");
}

#[test]
fn rejects_invalid_and_unbounded_requests_without_mutating() {
    let root = tempdir().expect("temporary memory");
    let mut service = MemoryService::open(root.path()).expect("open memory");

    for request in [
        json!({"protocolVersion": 2, "operation": "list"}),
        json!({"protocolVersion": PROTOCOL_VERSION, "operation": "list", "path": "/tmp"}),
        json!({
            "protocolVersion": PROTOCOL_VERSION,
            "operation": "global_save",
            "target": "memory",
            "expectedRevision": 0,
            "content": "😀".repeat(2_001)
        }),
        json!({
            "protocolVersion": PROTOCOL_VERSION,
            "operation": "create",
            "title": "Oversized",
            "content": "x".repeat(64_001)
        }),
        json!({
            "protocolVersion": PROTOCOL_VERSION,
            "operation": "create",
            "title": "Illegal|Title",
            "content": "body"
        }),
    ] {
        let response = service.handle(request);
        assert_eq!(response["ok"], false);
        assert_eq!(response["error"]["code"], "invalid_input");
    }

    let listed = ok(service.handle(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "operation": "list"
    })));
    assert!(listed["pages"].as_array().expect("pages").is_empty());
}

#[test]
fn recovers_exactly_before_or_after_true_child_crashes() {
    let create_failpoints = [
        "write:INDEX.md",
        "write:pages/Crash Target.md",
        "git_publish",
        "git_published",
        "git_index",
        "search_rebuild",
        "wal_cleanup",
    ];
    for failpoint in create_failpoints {
        run_crash_case(failpoint, "create");
    }
    run_crash_case("write:DISAMBIGUATION.md", "create_disambiguated");
    let delete_failpoints = [
        "write:.swarmx/tombstones/{id}.md",
        "delete:pages/Crash Target.md",
        "write:INDEX.md",
        "git_publish",
        "git_published",
        "git_index",
        "search_rebuild",
        "wal_cleanup",
    ];
    for failpoint in delete_failpoints {
        run_crash_case(failpoint, "delete");
    }
}

#[test]
fn memory_crash_child() {
    let Ok(root) = std::env::var("SWARMX_MEM_CRASH_ROOT") else {
        return;
    };
    let operation = std::env::var("SWARMX_MEM_CRASH_OPERATION").expect("crash operation");
    let mut service = MemoryService::open(Path::new(&root)).expect("child open memory");
    match operation.as_str() {
        "create" => {
            let response = service.handle(json!({
                "protocolVersion": PROTOCOL_VERSION,
                "operation": "create",
                "title": "Crash Target",
                "content": "Crash-safe content."
            }));
            ok(response);
        }
        "delete" => {
            let id = std::env::var("SWARMX_MEM_CRASH_ID").expect("crash page id");
            let response = service.handle(json!({
                "protocolVersion": PROTOCOL_VERSION,
                "operation": "delete",
                "id": id,
                "expectedRevision": 1
            }));
            ok(response);
        }
        "create_disambiguated" => {
            let response = service.handle(json!({
                "protocolVersion": PROTOCOL_VERSION,
                "operation": "create",
                "title": "Crash Target",
                "kind": "person",
                "content": "Second crash-safe content."
            }));
            ok(response);
        }
        operation => panic!("unknown crash operation: {operation}"),
    }
}

fn run_crash_case(failpoint: &str, operation: &str) {
    let root = tempdir().expect("crash temporary memory");
    let page_id = if matches!(operation, "delete" | "create_disambiguated") {
        let mut service = MemoryService::open(root.path()).expect("setup memory");
        let created = ok(service.handle(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "operation": "create",
            "title": "Crash Target",
            "content": "Crash-safe content."
        })));
        let id = created["page"]["id"]
            .as_str()
            .expect("setup page id")
            .to_owned();
        drop(service);
        if operation == "delete" {
            id
        } else {
            String::new()
        }
    } else {
        let service = MemoryService::open(root.path()).expect("setup memory");
        drop(service);
        String::new()
    };
    let failpoint = failpoint.replace("{id}", &page_id);
    let mut command = Command::new(std::env::current_exe().expect("memory test executable"));
    command
        .arg("--exact")
        .arg("memory_crash_child")
        .arg("--nocapture")
        .env("SWARMX_MEM_CRASH_ROOT", root.path())
        .env("SWARMX_MEM_CRASH_OPERATION", operation)
        .env("SWARMX_MEM_FAILPOINT", &failpoint);
    if !page_id.is_empty() {
        command.env("SWARMX_MEM_CRASH_ID", &page_id);
    }
    let status = command.status().expect("run crash child");
    assert!(
        !status.success(),
        "failpoint {failpoint} did not kill child"
    );

    let first_head = {
        let mut service = MemoryService::open(root.path()).expect("recover first restart");
        let pages = ok(service.handle(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "operation": "list"
        })))["pages"]
            .as_array()
            .expect("page list")
            .clone();
        if operation.starts_with("create") {
            assert!(pages.is_empty() || pages.iter().any(|page| page["title"] == "Crash Target"));
        } else {
            assert!(pages.is_empty() || pages.iter().any(|page| page["id"] == page_id));
        }
        git_head(root.path())
    };
    let second_head = {
        let _service = MemoryService::open(root.path()).expect("recover second restart");
        git_head(root.path())
    };
    assert_eq!(first_head, second_head, "recovery created an extra commit");
    let transactions = root.path().join(".runtime/transactions");
    assert_eq!(
        std::fs::read_dir(transactions)
            .expect("transactions directory")
            .filter_map(|entry| entry.ok())
            .count(),
        0,
        "WAL remained after recovery for {failpoint}"
    );
}

fn git_head(root: &Path) -> Option<String> {
    git2::Repository::open(root)
        .expect("open recovered repository")
        .head()
        .ok()
        .and_then(|head| head.target())
        .map(|head| head.to_string())
}

fn ok(response: Value) -> Value {
    assert_eq!(response["protocolVersion"], PROTOCOL_VERSION);
    assert_eq!(response["ok"], true, "response: {response}");
    response["result"].clone()
}

fn commit_paths(root: &Path, paths: &[&Path], message: &str) {
    let repository = git2::Repository::open(root).expect("open git repository");
    let mut index = repository.index().expect("git index");
    for path in paths {
        let absolute = root.join(path);
        if absolute.is_file() {
            index.add_path(path).expect("stage file");
        } else {
            index.remove_path(path).expect("stage deletion");
        }
    }
    index.write().expect("write git index");
    let tree_id = index.write_tree().expect("write tree");
    let tree = repository.find_tree(tree_id).expect("find tree");
    let parent = repository
        .head()
        .expect("git head")
        .peel_to_commit()
        .expect("parent commit");
    let signature = git2::Signature::now("test", "test@localhost").expect("signature");
    repository
        .commit(
            Some("HEAD"),
            &signature,
            &signature,
            message,
            &tree,
            &[&parent],
        )
        .expect("commit");
}
