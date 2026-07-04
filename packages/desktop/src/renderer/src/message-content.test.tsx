/** @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MessageContent } from "./message-content.js";

vi.mock("./code-highlighter.js", () => ({
  highlightCodeBlock: vi.fn(async (codeText: string, language: string) => {
    if (language !== "ts" && language !== "typescript") return null;
    return {
      lines: codeText.split("\n").map((line) => ({
        tokens: [{ color: "#f97583", content: line, fontStyle: 0 }],
      })),
    };
  }),
}));

describe("MessageContent", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
    vi.restoreAllMocks();
  });

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

  it("blocks remote Markdown images by default", () => {
    const html = renderToStaticMarkup(
      createElement(MessageContent, {
        kind: "message",
        content: "![Diagram](https://example.com/diagram.png)",
      }),
    );

    expect(html).toContain("run-event__image-placeholder");
    expect(html).toContain("Remote image blocked: Diagram");
    expect(html).not.toContain("<img");
    expect(html).not.toContain('src="https://example.com/diagram.png"');
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

  it("renders fenced code blocks with a stable language label", () => {
    const html = renderToStaticMarkup(
      createElement(MessageContent, {
        kind: "message",
        content: "```ts\nconst token = `literal`;\n```",
      }),
    );

    expect(html).toContain("run-event__code-block");
    expect(html).toContain("run-event__code-header");
    expect(html).toContain("run-event__code-language");
    expect(html).toContain("Copy code");
    expect(html).toContain('data-language="ts"');
    expect(html).toContain("const token = `literal`;");
  });

  it("copies the exact fenced code text without labels or markup", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });

    render(
      createElement(MessageContent, {
        kind: "message",
        content: "```ts\nconst token = `literal`;\n```",
      }),
    );

    fireEvent.click(screen.getByRole("button", { name: "Copy code" }));

    await waitFor(() => {
      expect(writeText).toHaveBeenCalledWith("const token = `literal`;\n");
      expect(screen.getByRole("button", { name: "Code copied" })).toBeTruthy();
    });
  });

  it("enhances known-language code blocks with offline highlighting after fallback render", async () => {
    const { container } = render(
      createElement(MessageContent, {
        kind: "message",
        content: "```ts\nconst answer: number = 42;\n```",
      }),
    );

    expect(container.querySelector(".run-event__code-fallback")).toBeTruthy();

    await waitFor(
      () => {
        expect(container.querySelector(".run-event__code-highlight .line")).toBeTruthy();
      },
      { timeout: 10_000 },
    );
    expect(container.textContent).toContain("const answer: number = 42;");
  });

  it("renders inline and display math offline with KaTeX", () => {
    const html = renderToStaticMarkup(
      createElement(MessageContent, {
        kind: "message",
        content: "Energy $E=mc^2$.\n\n$$\n\\int_0^1 x\\,dx\n$$",
      }),
    );

    expect(html).toContain("katex");
    expect(html).toContain("katex-display");
    expect(html).toContain("E=mc^2");
    expect(html).toContain("\\int_0^1 x\\,dx");
  });

  it("supports escaped inline and display math delimiters", () => {
    const html = renderToStaticMarkup(
      createElement(MessageContent, {
        kind: "message",
        content: "Inline \\(a+b\\).\n\n\\[a^2+b^2=c^2\\]",
      }),
    );

    expect(html).toContain("katex");
    expect(html).toContain("katex-display");
    expect(html).toContain("a+b");
    expect(html).toContain("a^2+b^2=c^2");
  });

  it("does not parse obvious currency or shell prompt dollars as math", () => {
    const html = renderToStaticMarkup(
      createElement(MessageContent, {
        kind: "message",
        content: "Price is $5 and shell $ prompt.",
      }),
    );

    expect(html).not.toContain("katex");
    expect(html).toContain("Price is $5 and shell $ prompt.");
  });

  it("keeps invalid math local and preserves surrounding message content", () => {
    const html = renderToStaticMarkup(
      createElement(MessageContent, {
        kind: "message",
        content: "Before $\\notacommand{$ after",
      }),
    );

    expect(html).toContain("Before");
    expect(html).toContain("katex-error");
    expect(html).toContain("\\notacommand{");
    expect(html).toContain("after");
  });

  it("keeps math-looking tool output literal", () => {
    const html = renderToStaticMarkup(
      createElement(MessageContent, {
        kind: "tool_result",
        content: "Energy $E=mc^2$.",
      }),
    );

    expect(html).not.toContain("katex");
    expect(html).toContain("Energy $E=mc^2$.");
  });
});
