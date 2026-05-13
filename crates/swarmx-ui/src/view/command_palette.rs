use iced::widget::{column, container, text};
use iced::{Alignment, Element, Length};
use iced_shadcn::Theme;

use crate::app::Message;

pub fn view<'a>(_t: &'a Theme) -> Element<'a, Message> {
    container(column![text("Command palette — coming in Phase 9")].align_x(Alignment::Center))
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x(Length::Fill)
        .center_y(Length::Fill)
        .into()
}
