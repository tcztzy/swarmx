use super::*;

#[test]
fn theme_preference_serde_round_trip() {
    for pref in ThemePreference::all() {
        let s = serde_json::to_string(pref).unwrap();
        let back: ThemePreference = serde_json::from_str(&s).unwrap();
        assert_eq!(pref, &back, "round-trip failed for {:?}", pref);
    }
}

#[test]
fn theme_preference_default_is_dark() {
    assert_eq!(ThemePreference::default(), ThemePreference::Dark);
}

#[test]
fn theme_preference_label_is_stable() {
    assert_eq!(ThemePreference::Dark.label(), "Dark");
    assert_eq!(ThemePreference::TokyoNight.label(), "Tokyo Night");
    assert_eq!(ThemePreference::CatppuccinMocha.label(), "Catppuccin Mocha");
    assert_eq!(ThemePreference::CatppuccinLatte.label(), "Catppuccin Latte");
}

#[test]
fn is_dark_theme_distinguishes_palettes() {
    assert!(is_dark_theme(&ThemePreference::Dark.shadcn_theme()));
    assert!(!is_dark_theme(
        &ThemePreference::CatppuccinLatte.shadcn_theme()
    ));
}

#[test]
fn agent_color_returns_distinct_brand_hue_per_icon() {
    use crate::harness::Harness;

    let theme = ThemePreference::Dark.shadcn_theme();
    let claude = agent_color(Harness::ClaudeCode, &theme);
    let codex = agent_color(Harness::Codex, &theme);
    let opencode = agent_color(Harness::OpenCode, &theme);
    let hermes = agent_color(Harness::Hermes, &theme);
    let openclaw = agent_color(Harness::OpenClaw, &theme);

    let all = [claude, codex, opencode, hermes, openclaw];
    for i in 0..all.len() {
        for j in (i + 1)..all.len() {
            assert_ne!(all[i], all[j], "colours collide at {} {}", i, j);
        }
    }

    assert!((claude.r - 0xD9 as f32 / 255.0).abs() < 0.01);
    assert!((claude.g - 0x77 as f32 / 255.0).abs() < 0.01);
    assert!((claude.b - 0x57 as f32 / 255.0).abs() < 0.01);
}

#[test]
fn agent_color_for_swarmx_uses_theme_primary() {
    use crate::harness::Harness;

    let theme = ThemePreference::Dark.shadcn_theme();
    let c = agent_color(Harness::SwarmX, &theme);
    assert_eq!(c, theme.palette.primary);
}
