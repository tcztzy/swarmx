use iced::Element;
use iced::widget::{column, text};
use iced_shadcn::Theme;

use crate::app::Message;

pub fn view<'a>(_t: &'a Theme) -> Element<'a, Message> {
    column![
        text("SwarmX").size(18),
        text(env!("CARGO_PKG_VERSION")).size(12),
    ]
    .into()
}
