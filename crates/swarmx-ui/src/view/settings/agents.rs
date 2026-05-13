use std::sync::OnceLock;

use iced::widget::{Space, column, container, row, scrollable, text};
use iced::{Element, Length};
use iced_shadcn::{
    ButtonProps, ButtonSize, ButtonVariant, CardProps, CardSize, CardVariant, InputProps,
    SelectProps, SeparatorOrientation, SeparatorProps, Theme, button, card, input, select,
    separator,
};

use crate::app::{App, Message};
use crate::harness::Harness;
use crate::instance::{AgentInstance, ModelProvider, ProviderKind};

fn kind_labels() -> &'static [String] {
    static LABELS: OnceLock<Vec<String>> = OnceLock::new();
    LABELS.get_or_init(|| {
        ProviderKind::all()
            .iter()
            .map(|k| k.label().to_string())
            .collect()
    })
}

fn harness_labels() -> &'static [String] {
    static LABELS: OnceLock<Vec<String>> = OnceLock::new();
    LABELS.get_or_init(|| {
        Harness::all()
            .iter()
            .map(|h| h.label().to_string())
            .collect()
    })
}

pub fn view<'a>(app: &'a App, t: &'a Theme) -> Element<'a, Message> {
    let heading = text("Agents").size(16).color(t.palette.foreground);
    let desc = text("Providers and agent instances. Changes save automatically.")
        .size(12)
        .color(t.palette.muted_foreground);

    let provider_cards: Vec<Element<_>> = app
        .providers
        .iter()
        .enumerate()
        .map(|(i, p)| provider_card(i, p, app.pending_delete_provider, t))
        .collect();

    let add_provider = button(
        "Add Provider",
        Some(Message::AddProvider),
        ButtonProps::new()
            .variant(ButtonVariant::Outline)
            .size(ButtonSize::Size2),
        t,
    );

    let instance_cards: Vec<Element<_>> = app
        .instances
        .iter()
        .enumerate()
        .map(|(i, inst)| {
            instance_card(
                i,
                inst,
                &app.providers,
                &app.provider_labels,
                app.pending_delete_instance,
                t,
            )
        })
        .collect();

    let add_instance_msg = if app.providers.is_empty() {
        None
    } else {
        Some(Message::AddInstance)
    };
    let add_instance = button(
        "Add Instance",
        add_instance_msg,
        ButtonProps::new()
            .variant(ButtonVariant::Outline)
            .size(ButtonSize::Size2),
        t,
    );

    let sep = separator(
        SeparatorProps::new().orientation(SeparatorOrientation::Horizontal),
        t,
    );

    container(scrollable(
        column![
            heading,
            Space::new().height(4.0),
            desc,
            Space::new().height(16.0),
            text("Providers").size(13).color(t.palette.foreground),
            Space::new().height(8.0),
            column(provider_cards).spacing(12).width(Length::Fill),
            Space::new().height(8.0),
            add_provider,
            Space::new().height(24.0),
            sep,
            Space::new().height(16.0),
            text("Agent Instances").size(13).color(t.palette.foreground),
            Space::new().height(8.0),
            column(instance_cards).spacing(12).width(Length::Fill),
            Space::new().height(8.0),
            add_instance,
        ]
        .spacing(0),
    ))
    .width(Length::Fill)
    .into()
}

fn provider_card<'a>(
    i: usize,
    p: &'a ModelProvider,
    pending: Option<usize>,
    t: &'a Theme,
) -> Element<'a, Message> {
    let label_row = labeled_row(
        "Label",
        input(
            &p.label,
            "Provider name",
            Some(move |s| Message::UpdateProviderLabel(i, s)),
            InputProps::new(),
            t,
        )
        .width(Length::Fill)
        .into(),
        t,
    );

    let kind_sel = select(
        kind_labels(),
        Some(p.kind.label().to_string()),
        "Kind",
        move |v: String| {
            let k = ProviderKind::all()
                .into_iter()
                .find(|k| k.label() == v)
                .unwrap_or(ProviderKind::Custom);
            Message::UpdateProviderKind(i, k)
        },
        SelectProps::new(),
        t,
    );
    let kind_row = labeled_row("Kind", Element::from(kind_sel), t);

    let base_url_row = labeled_row(
        "Base URL",
        input(
            p.base_url.as_deref().unwrap_or(""),
            "optional",
            Some(move |s| Message::UpdateProviderBaseUrl(i, s)),
            InputProps::new(),
            t,
        )
        .width(Length::Fill)
        .into(),
        t,
    );

    let api_placeholder = if p.api_key_ref.is_some() {
        "●●●●●●  (stored in keychain)"
    } else {
        "Not set"
    };
    let api_row = labeled_row(
        "API Key",
        input(
            "",
            api_placeholder,
            Some(move |s| Message::SetProviderApiKey(i, s)),
            InputProps::new(),
            t,
        )
        .width(Length::Fill)
        .into(),
        t,
    );

    let default_model_row = labeled_row(
        "Default Model",
        input(
            &p.default_model,
            "model name",
            Some(move |s| Message::UpdateProviderDefaultModel(i, s)),
            InputProps::new(),
            t,
        )
        .width(Length::Fill)
        .into(),
        t,
    );

    let del_row = delete_btn(
        i,
        pending,
        Message::DeleteProvider(i),
        Message::ConfirmDeleteProvider(i),
        Message::CancelDeleteProvider,
        t,
    );

    let body = column![
        label_row,
        Space::new().height(8.0),
        kind_row,
        Space::new().height(8.0),
        base_url_row,
        Space::new().height(8.0),
        api_row,
        Space::new().height(8.0),
        default_model_row,
        Space::new().height(12.0),
        del_row,
    ]
    .width(Length::Fill);

    card(
        body,
        CardProps::new()
            .variant(CardVariant::Surface)
            .size(CardSize::Size1),
        t,
    )
    .into()
}

fn instance_card<'a>(
    i: usize,
    inst: &'a AgentInstance,
    providers: &'a [ModelProvider],
    provider_labels: &'a [String],
    pending: Option<usize>,
    t: &'a Theme,
) -> Element<'a, Message> {
    let label_row = labeled_row(
        "Label",
        input(
            &inst.label,
            "Agent name",
            Some(move |s| Message::UpdateInstanceLabel(i, s)),
            InputProps::new(),
            t,
        )
        .width(Length::Fill)
        .into(),
        t,
    );

    let harness_sel = select(
        harness_labels(),
        Some(inst.harness.label().to_string()),
        "Harness",
        move |v: String| {
            let h = Harness::all()
                .into_iter()
                .find(|h| h.label() == v)
                .unwrap_or(Harness::ClaudeCode);
            Message::UpdateInstanceHarness(i, h)
        },
        SelectProps::new(),
        t,
    );
    let harness_row = labeled_row("Harness", Element::from(harness_sel), t);

    let current_prov_label = providers
        .iter()
        .find(|p| p.id == inst.provider_id)
        .map(|p| p.label.clone());
    let prov_sel = select(
        provider_labels,
        current_prov_label,
        "Provider",
        move |v: String| {
            let pid = providers
                .iter()
                .find(|p| p.label == v)
                .map(|p| p.id.clone())
                .unwrap_or_default();
            Message::UpdateInstanceProviderId(i, pid)
        },
        SelectProps::new(),
        t,
    );
    let prov_row = labeled_row("Provider", Element::from(prov_sel), t);

    let available: &[String] = providers
        .iter()
        .find(|p| p.id == inst.provider_id)
        .map(|p| p.available_models.as_slice())
        .unwrap_or(&[]);
    let model_sel = select(
        available,
        Some(inst.model.clone()),
        "Model",
        move |m: String| Message::UpdateInstanceModel(i, m),
        SelectProps::new(),
        t,
    );
    let model_row = labeled_row("Model", Element::from(model_sel), t);

    let instr_row = labeled_row(
        "Instructions",
        input(
            inst.instructions.as_deref().unwrap_or(""),
            "optional",
            Some(move |s| Message::UpdateInstanceInstructions(i, s)),
            InputProps::new(),
            t,
        )
        .width(Length::Fill)
        .into(),
        t,
    );

    let cwd_str = inst
        .default_cwd
        .as_ref()
        .and_then(|p| p.to_str())
        .unwrap_or("");
    let cwd_row = labeled_row(
        "Default CWD",
        input(
            cwd_str,
            "optional",
            Some(move |s| Message::UpdateInstanceDefaultCwd(i, s)),
            InputProps::new(),
            t,
        )
        .width(Length::Fill)
        .into(),
        t,
    );

    let del_row = delete_btn(
        i,
        pending,
        Message::DeleteInstance(i),
        Message::ConfirmDeleteInstance(i),
        Message::CancelDeleteInstance,
        t,
    );

    let body = column![
        label_row,
        Space::new().height(8.0),
        harness_row,
        Space::new().height(8.0),
        prov_row,
        Space::new().height(8.0),
        model_row,
        Space::new().height(8.0),
        instr_row,
        Space::new().height(8.0),
        cwd_row,
        Space::new().height(12.0),
        del_row,
    ]
    .width(Length::Fill);

    card(
        body,
        CardProps::new()
            .variant(CardVariant::Surface)
            .size(CardSize::Size1),
        t,
    )
    .into()
}

fn labeled_row<'a>(
    label: &'a str,
    widget: Element<'a, Message>,
    t: &'a Theme,
) -> Element<'a, Message> {
    row![
        text(label)
            .size(12)
            .color(t.palette.muted_foreground)
            .width(Length::Fixed(120.0)),
        widget,
    ]
    .align_y(iced::Alignment::Center)
    .into()
}

fn delete_btn<'a>(
    idx: usize,
    pending: Option<usize>,
    on_arm: Message,
    on_confirm: Message,
    on_cancel: Message,
    t: &'a Theme,
) -> Element<'a, Message> {
    if pending == Some(idx) {
        row![
            button(
                "Confirm Delete",
                Some(on_confirm),
                ButtonProps::new()
                    .variant(ButtonVariant::Destructive)
                    .size(ButtonSize::Size1),
                t,
            ),
            Space::new().width(8.0),
            button(
                "Cancel",
                Some(on_cancel),
                ButtonProps::new()
                    .variant(ButtonVariant::Ghost)
                    .size(ButtonSize::Size1),
                t,
            ),
        ]
        .align_y(iced::Alignment::Center)
        .into()
    } else {
        button(
            "Delete",
            Some(on_arm),
            ButtonProps::new()
                .variant(ButtonVariant::Outline)
                .size(ButtonSize::Size1),
            t,
        )
        .into()
    }
}

#[cfg(test)]
mod tests;
