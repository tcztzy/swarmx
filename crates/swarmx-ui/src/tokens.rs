use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Density {
    #[default]
    Comfortable,
    Compact,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextStyle {
    pub size: f32,
    pub line_height: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DesignTokens {
    pub space_0: f32,
    pub space_1: f32,
    pub space_2: f32,
    pub space_3: f32,
    pub space_4: f32,
    pub space_5: f32,
    pub space_6: f32,
    pub space_7: f32,
    pub space_8: f32,

    pub text_xs: TextStyle,
    pub text_sm: TextStyle,
    pub text_base: TextStyle,
    pub text_md: TextStyle,
    pub text_lg: TextStyle,
    pub text_xl: TextStyle,
    pub text_2xl: TextStyle,

    pub radius_xs: f32,
    pub radius_sm: f32,
    pub radius_md: f32,
    pub radius_lg: f32,
    pub radius_xl: f32,

    pub sidebar_min_width: f32,
    pub sidebar_max_width: f32,
    pub sidebar_default_width: f32,
    pub conversation_max_width: f32,

    pub motion_fast: Duration,
    pub motion_base: Duration,
    pub motion_slow: Duration,
}

impl DesignTokens {
    pub fn for_density(density: Density) -> Self {
        let comfortable = Self {
            space_0: 0.0,
            space_1: 4.0,
            space_2: 8.0,
            space_3: 12.0,
            space_4: 16.0,
            space_5: 24.0,
            space_6: 32.0,
            space_7: 48.0,
            space_8: 64.0,

            text_xs: TextStyle {
                size: 11.0,
                line_height: 14.0,
            },
            text_sm: TextStyle {
                size: 12.0,
                line_height: 16.0,
            },
            text_base: TextStyle {
                size: 14.0,
                line_height: 22.0,
            },
            text_md: TextStyle {
                size: 15.0,
                line_height: 24.0,
            },
            text_lg: TextStyle {
                size: 18.0,
                line_height: 26.0,
            },
            text_xl: TextStyle {
                size: 22.0,
                line_height: 30.0,
            },
            text_2xl: TextStyle {
                size: 28.0,
                line_height: 36.0,
            },

            radius_xs: 4.0,
            radius_sm: 6.0,
            radius_md: 8.0,
            radius_lg: 12.0,
            radius_xl: 16.0,

            sidebar_min_width: 180.0,
            sidebar_max_width: 480.0,
            sidebar_default_width: 240.0,
            conversation_max_width: 760.0,

            motion_fast: Duration::from_millis(120),
            motion_base: Duration::from_millis(200),
            motion_slow: Duration::from_millis(320),
        };
        match density {
            Density::Comfortable | Density::Compact => comfortable,
        }
    }
}

impl Default for DesignTokens {
    fn default() -> Self {
        Self::for_density(Density::default())
    }
}

#[cfg(test)]
mod tests;
