use super::*;
use std::cell::Cell;

thread_local! {
    static MOCK_TIME: Cell<u64> = const { Cell::new(1000) };
}

fn mock_clock() -> u64 {
    MOCK_TIME.with(|t| t.get())
}

fn set_time(t: u64) {
    MOCK_TIME.with(|t2| t2.set(t));
}

#[test]
fn empty_cache_needs_fetch() {
    set_time(1000);
    let cache: SWRCache<&str, String> = SWRCache::with_clock(60, mock_clock);
    let read = cache.read(&"key");
    assert_eq!(read.data, SwrData::Empty);
    assert!(read.needs_fetch);
}

#[test]
fn fresh_data_no_fetch() {
    set_time(1000);
    let mut cache = SWRCache::with_clock(60, mock_clock);
    cache.set("key", "value".to_string());

    set_time(1030); // 30s later, within 60s TTL
    let read = cache.read(&"key");
    assert_eq!(read.data, SwrData::Fresh("value".to_string()));
    assert!(!read.needs_fetch);
}

#[test]
fn stale_data_triggers_fetch() {
    set_time(1000);
    let mut cache = SWRCache::with_clock(60, mock_clock);
    cache.set("key", "value".to_string());

    set_time(1061); // 61s later, past TTL
    let read = cache.read(&"key");
    assert_eq!(read.data, SwrData::Stale("value".to_string()));
    assert!(read.needs_fetch);
}

#[test]
fn mark_fetching_prevents_duplicate_fetch() {
    set_time(1000);
    let mut cache = SWRCache::with_clock(60, mock_clock);
    cache.set("key", "value".to_string());

    set_time(1061);
    let read1 = cache.read(&"key");
    assert!(read1.needs_fetch);

    cache.mark_fetching(&"key");

    let read2 = cache.read(&"key");
    assert!(!read2.needs_fetch); // already fetching
    assert_eq!(read2.data, SwrData::Stale("value".to_string()));
}

#[test]
fn resolve_makes_data_fresh() {
    set_time(1000);
    let mut cache = SWRCache::with_clock(60, mock_clock);
    cache.mark_fetching(&"key");

    set_time(1001);
    cache.resolve("key", "fresh_value".to_string());

    let read = cache.read(&"key");
    assert_eq!(read.data, SwrData::Fresh("fresh_value".to_string()));
    assert!(!read.needs_fetch);
}

#[test]
fn resolve_error_preserves_stale_data() {
    set_time(1000);
    let mut cache = SWRCache::with_clock(60, mock_clock);
    cache.set("key", "old_value".to_string());

    set_time(1061);
    cache.mark_fetching(&"key");

    set_time(1062);
    cache.resolve_error(&"key", "network error".to_string());

    let read = cache.read(&"key");
    assert_eq!(
        read.data,
        SwrData::Error {
            error: "network error".to_string(),
            stale: Some("old_value".to_string()),
        }
    );
}

#[test]
fn invalidate_forces_refetch() {
    set_time(1000);
    let mut cache = SWRCache::with_clock(60, mock_clock);
    cache.set("key", "value".to_string());

    let read = cache.read(&"key");
    assert!(!read.needs_fetch);

    cache.invalidate(&"key");
    let read = cache.read(&"key");
    assert!(read.needs_fetch);
    assert_eq!(read.data, SwrData::Stale("value".to_string()));
}

#[test]
fn peek_returns_ref_without_triggering_logic() {
    set_time(1000);
    let mut cache = SWRCache::with_clock(60, mock_clock);
    cache.set("key", 42u32);
    assert_eq!(cache.peek(&"key"), Some(&42));
    assert_eq!(cache.peek(&"missing"), None);
}

#[test]
fn clear_removes_all() {
    set_time(1000);
    let mut cache = SWRCache::with_clock(60, mock_clock);
    cache.set("a", 1);
    cache.set("b", 2);
    cache.clear();
    assert_eq!(cache.peek(&"a"), None);
    assert_eq!(cache.peek(&"b"), None);
}

#[test]
fn error_on_empty_key_no_stale() {
    set_time(1000);
    let mut cache: SWRCache<&str, String> = SWRCache::with_clock(60, mock_clock);
    cache.resolve_error(&"key", "fail".to_string());

    let read = cache.read(&"key");
    assert_eq!(
        read.data,
        SwrData::Error {
            error: "fail".to_string(),
            stale: None,
        }
    );
}
