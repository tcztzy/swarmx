use crate::hook::Hook;

#[test]
fn test_hook_default() {
    let hook = Hook::default();
    assert!(hook.on_start.is_none());
    assert!(hook.on_end.is_none());
    assert!(hook.on_handoff.is_none());
    assert!(hook.on_chunk.is_none());
}

#[test]
fn test_hook_serde() {
    let hook = Hook {
        on_start: Some("start".to_string()),
        on_end: Some("end".to_string()),
        ..Default::default()
    };
    let json = serde_json::to_string(&hook).unwrap();
    let deserialized: Hook = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.on_start, Some("start".to_string()));
    assert_eq!(deserialized.on_end, Some("end".to_string()));
}
