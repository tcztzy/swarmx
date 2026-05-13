//! Utilities.

use chrono::Utc;
use rand::Rng;

use std::path::PathBuf;

/// OpenAI compatible timestamp.
pub fn now() -> i64 {
    Utc::now().timestamp()
}

/// Current working directory, falling back to /.
pub fn default_cwd() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"))
}

const RANDOM_STRING_CHARS: &[u8] =
    b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

/// Return a securely generated random string.
pub fn get_random_string(length: usize) -> String {
    let mut rng = rand::rng();
    (0..length)
        .map(|_| {
            let idx = rng.random_range(0..RANDOM_STRING_CHARS.len());
            RANDOM_STRING_CHARS[idx] as char
        })
        .collect()
}

#[cfg(test)]
mod tests;
