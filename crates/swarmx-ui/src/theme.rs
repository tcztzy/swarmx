use iced_shadcn::Theme;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThemePreference {
    #[default]
    Dark,
    TokyoNight,
    CatppuccinMocha,
    CatppuccinLatte,
}

impl ThemePreference {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Dark => "Dark",
            Self::TokyoNight => "Tokyo Night",
            Self::CatppuccinMocha => "Catppuccin Mocha",
            Self::CatppuccinLatte => "Catppuccin Latte",
        }
    }

    pub fn iced_theme(&self) -> iced::Theme {
        match self {
            Self::Dark => iced::Theme::Dark,
            Self::TokyoNight => iced::Theme::TokyoNight,
            Self::CatppuccinMocha => iced::Theme::CatppuccinMocha,
            Self::CatppuccinLatte => iced::Theme::CatppuccinLatte,
        }
    }

    pub fn shadcn_theme(&self) -> Theme {
        match self {
            Self::CatppuccinLatte => Theme::light(),
            _ => Theme::dark(),
        }
    }

    pub fn all() -> &'static [ThemePreference] {
        &[
            Self::Dark,
            Self::TokyoNight,
            Self::CatppuccinMocha,
            Self::CatppuccinLatte,
        ]
    }
}

pub fn is_dark_theme(t: &Theme) -> bool {
    t.palette.background.r < 0.5
}

/// Brand accent colour per agent harness. Used for thin strips and tinted backgrounds.
/// Never use this for body text — contrast is not guaranteed against the palette.
pub fn agent_color(harness: crate::harness::Harness, theme: &Theme) -> iced::Color {
    use crate::harness::Harness;

    fn rgb(r: u8, g: u8, b: u8) -> iced::Color {
        iced::Color::from_rgb(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
    }
    match harness {
        Harness::ClaudeCode => rgb(0xD9, 0x77, 0x57),
        Harness::Codex => rgb(0x10, 0xA3, 0x7F),
        Harness::OpenCode => rgb(0x7C, 0x3A, 0xED),
        Harness::Hermes => rgb(0x63, 0x66, 0xF1),
        Harness::OpenClaw => rgb(0x0E, 0xA5, 0xE9),
        Harness::SwarmX => theme.palette.primary,
    }
}

#[cfg(test)]
mod tests;
