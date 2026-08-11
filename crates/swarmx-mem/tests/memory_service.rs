use serde_json::{Value, json};
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
    assert!(root.path().join("pages").join(format!("{id}.md")).is_file());

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

fn ok(response: Value) -> Value {
    assert_eq!(response["protocolVersion"], PROTOCOL_VERSION);
    assert_eq!(response["ok"], true, "response: {response}");
    response["result"].clone()
}
