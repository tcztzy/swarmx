use crate::utils::{get_random_string, now};

#[test]
fn test_now_positive() {
    let t = now();
    assert!(t > 0);
}

#[test]
fn test_random_string_length() {
    let s = get_random_string(10);
    assert_eq!(s.len(), 10);
}

#[test]
fn test_random_string_unique() {
    let a = get_random_string(16);
    let b = get_random_string(16);
    assert_ne!(a, b);
}

#[test]
fn test_random_string_chars() {
    let s = get_random_string(100);
    assert!(s.chars().all(|c| c.is_ascii_alphanumeric()));
}
