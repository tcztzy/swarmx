import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { MessageContent } from "./message-content.js";

describe("MessageContent", () => {
  it("renders conversational inline code as Markdown code", () => {
    const html = renderToStaticMarkup(
      createElement(MessageContent, { kind: "message", content: "Use `HF_HOME` for cache." }),
    );

    expect(html).toContain("<code>HF_HOME</code>");
    expect(html).not.toContain("`HF_HOME`");
  });

  it("keeps tool output as literal text", () => {
    const html = renderToStaticMarkup(
      createElement(MessageContent, {
        kind: "tool_result",
        content: "Use `HF_HOME` for cache.",
      }),
    );

    expect(html).not.toContain("<code>");
    expect(html).toContain("`HF_HOME`");
  });

  it("escapes raw HTML in conversational Markdown", () => {
    const html = renderToStaticMarkup(
      createElement(MessageContent, {
        kind: "message",
        content: '<img src=x onerror="alert(1)"> `HF_HOME`',
      }),
    );

    expect(html).not.toContain("<img");
    expect(html).toContain("&lt;img");
    expect(html).toContain("<code>HF_HOME</code>");
  });

  it("renders remote Markdown images directly", () => {
    const html = renderToStaticMarkup(
      createElement(MessageContent, {
        kind: "message",
        content: "![Diagram](https://example.com/diagram.png)",
      }),
    );

    expect(html).toContain("<img");
    expect(html).toContain('alt="Diagram"');
    expect(html).toContain('src="https://example.com/diagram.png"');
  });

  it("does not render local Markdown images as broken browser paths before loading", () => {
    const html = renderToStaticMarkup(
      createElement(MessageContent, {
        kind: "message",
        content: "![Screenshot](/var/folders/session-shot.png)",
      }),
    );

    expect(html).toContain("run-event__image-placeholder");
    expect(html).toContain("Loading image");
    expect(html).not.toContain('src="/var/folders/session-shot.png"');
  });
});
