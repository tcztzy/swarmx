use super::*;
use crate::theme::ThemePreference;
use crate::tokens::{Density, DesignTokens};

fn tokens() -> DesignTokens {
    DesignTokens::for_density(Density::Comfortable)
}

fn theme() -> iced_shadcn::Theme {
    ThemePreference::Dark.shadcn_theme()
}

#[test]
fn view_idle_state_does_not_panic() {
    let t = tokens();
    let th = theme();
    let _ = view("", false, &[], "", 0, None, &t, &th);
}

#[test]
fn view_loading_state_does_not_panic() {
    let t = tokens();
    let th = theme();
    let _ = view("hello", true, &[], "", 0, None, &t, &th);
}

#[test]
fn view_with_model_selector_does_not_panic() {
    let t = tokens();
    let th = theme();
    let models = vec!["gpt-4o".to_string(), "gpt-4.1".to_string()];
    let _ = view("hi", false, &models, "gpt-4o", 0, None, &t, &th);
}

#[test]
fn view_with_cwd_does_not_panic() {
    let t = tokens();
    let th = theme();
    let _ = view(
        "",
        false,
        &[],
        "",
        0,
        Some("/home/user/projects/swarmx"),
        &t,
        &th,
    );
}

#[test]
fn view_long_cwd_does_not_panic() {
    let t = tokens();
    let th = theme();
    let long_cwd = "/very/long/path/that/exceeds/fifty/characters/to/test/truncation/behavior";
    let _ = view("", false, &[], "", 0, Some(long_cwd), &t, &th);
}

#[test]
fn view_empty_models_does_not_panic() {
    let t = tokens();
    let th = theme();
    let _ = view("test", false, &[], "claude-sonnet-4-5", 1, None, &t, &th);
}
