use crate::settings::Settings;

#[test]
fn test_settings_default() {
    let settings = Settings::default();
    assert!(!settings.openai_base_url.is_empty());
}

#[test]
fn test_settings_agents_md_empty() {
    let settings = Settings::default();
    let content = settings.get_agents_md_content();
    assert!(content.is_empty());
}
