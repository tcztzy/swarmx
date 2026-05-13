use std::collections::HashMap;
use std::hash::Hash;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FetchStatus {
    Idle,
    Fetching,
}

#[derive(Clone, Debug)]
struct Entry<V> {
    data: Option<V>,
    error: Option<String>,
    fetched_at: u64,
    status: FetchStatus,
}

/// What a consumer sees when reading from the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwrData<V> {
    /// No data yet, first fetch needed.
    Empty,
    /// Fresh data within TTL.
    Fresh(V),
    /// Stale data — still usable, but revalidation should happen.
    Stale(V),
    /// Error occurred; stale data may still be available.
    Error { error: String, stale: Option<V> },
}

/// Result of `read()` — current data state + whether a fetch should be triggered.
#[derive(Debug, Clone)]
pub struct SwrRead<V> {
    pub data: SwrData<V>,
    pub needs_fetch: bool,
}

/// Stale-While-Revalidate cache for Iced MVU architecture.
///
/// Usage in update():
///   let read = cache.read(&key, now);
///   // render from read.data (Fresh/Stale/Empty/Error)
///   if read.needs_fetch {
///       cache.mark_fetching(&key);
///       return Task::perform(fetch_fn(), |result| Message::Fetched(result));
///   }
///
/// When fetch completes:
///   cache.resolve(key, value);  // or cache.resolve_error(key, err);
pub struct SWRCache<K, V> {
    entries: HashMap<K, Entry<V>>,
    ttl_secs: u64,
    clock: fn() -> u64,
}

impl<K: Clone + Eq + Hash, V: Clone> SWRCache<K, V> {
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            entries: HashMap::new(),
            ttl_secs,
            clock: default_clock,
        }
    }

    #[cfg(test)]
    fn with_clock(ttl_secs: u64, clock: fn() -> u64) -> Self {
        Self {
            entries: HashMap::new(),
            ttl_secs,
            clock,
        }
    }

    /// Read current state for a key. Returns data + whether fetch needed.
    pub fn read(&self, key: &K) -> SwrRead<V> {
        let now = (self.clock)();
        match self.entries.get(key) {
            None => SwrRead {
                data: SwrData::Empty,
                needs_fetch: true,
            },
            Some(entry) => {
                let fresh = now.saturating_sub(entry.fetched_at) < self.ttl_secs;
                let already_fetching = entry.status == FetchStatus::Fetching;

                let data = match (&entry.data, &entry.error) {
                    (_, Some(err)) => SwrData::Error {
                        error: err.clone(),
                        stale: entry.data.clone(),
                    },
                    (Some(v), None) if fresh => SwrData::Fresh(v.clone()),
                    (Some(v), None) => SwrData::Stale(v.clone()),
                    (None, None) => SwrData::Empty,
                };

                let needs_fetch = !fresh && !already_fetching;
                SwrRead { data, needs_fetch }
            }
        }
    }

    /// Mark key as currently being fetched. Call before spawning Task.
    pub fn mark_fetching(&mut self, key: &K) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.status = FetchStatus::Fetching;
        } else {
            self.entries.insert(
                key.clone(),
                Entry {
                    data: None,
                    error: None,
                    fetched_at: 0,
                    status: FetchStatus::Fetching,
                },
            );
        }
    }

    /// Complete a successful fetch.
    pub fn resolve(&mut self, key: K, value: V) {
        let now = (self.clock)();
        self.entries.insert(
            key,
            Entry {
                data: Some(value),
                error: None,
                fetched_at: now,
                status: FetchStatus::Idle,
            },
        );
    }

    /// Complete a failed fetch. Preserves stale data if available.
    pub fn resolve_error(&mut self, key: &K, error: String) {
        let now = (self.clock)();
        if let Some(entry) = self.entries.get_mut(key) {
            entry.error = Some(error);
            entry.fetched_at = now;
            entry.status = FetchStatus::Idle;
        } else {
            self.entries.insert(
                key.clone(),
                Entry {
                    data: None,
                    error: Some(error),
                    fetched_at: now,
                    status: FetchStatus::Idle,
                },
            );
        }
    }

    /// Peek at cached data without triggering revalidation logic.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.entries.get(key).and_then(|e| e.data.as_ref())
    }

    /// Directly set data (for local mutations that don't need a fetch).
    pub fn set(&mut self, key: K, value: V) {
        self.resolve(key, value);
    }

    /// Invalidate a key — next read will trigger fetch.
    pub fn invalidate(&mut self, key: &K) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.fetched_at = 0;
            entry.error = None;
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

fn default_clock() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests;
