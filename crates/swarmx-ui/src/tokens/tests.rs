use super::*;

#[test]
fn comfortable_spacing_matches_legacy_literals() {
    let t = DesignTokens::for_density(Density::Comfortable);
    assert_eq!(t.space_1, 4.0);
    assert_eq!(t.space_2, 8.0);
    assert_eq!(t.space_3, 12.0);
    assert_eq!(t.space_4, 16.0);
    assert_eq!(t.space_5, 24.0);
    assert_eq!(t.space_6, 32.0);
    assert_eq!(t.space_7, 48.0);
    assert_eq!(t.space_8, 64.0);
}

#[test]
fn comfortable_text_sizes_match_legacy() {
    let t = DesignTokens::for_density(Density::Comfortable);
    assert_eq!(t.text_xs.size, 11.0);
    assert_eq!(t.text_sm.size, 12.0);
    assert_eq!(t.text_base.size, 14.0);
    assert_eq!(t.text_md.size, 15.0);
    assert_eq!(t.text_lg.size, 18.0);
    assert_eq!(t.text_xl.size, 22.0);
    assert_eq!(t.text_2xl.size, 28.0);
}

#[test]
fn comfortable_layout_caps_match_legacy() {
    let t = DesignTokens::for_density(Density::Comfortable);
    assert_eq!(t.sidebar_min_width, 180.0);
    assert_eq!(t.sidebar_max_width, 480.0);
    assert_eq!(t.conversation_max_width, 760.0);
}

#[test]
fn compact_currently_clones_comfortable_pending_phase10() {
    let comf = DesignTokens::for_density(Density::Comfortable);
    let comp = DesignTokens::for_density(Density::Compact);
    assert_eq!(comf.space_3, comp.space_3);
    assert_eq!(comf.text_base.size, comp.text_base.size);
}
