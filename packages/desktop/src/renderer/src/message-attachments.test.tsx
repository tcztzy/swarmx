/** @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { DesktopMediaAttachment } from "../../shared/desktop-api.js";
import { MessageAttachments } from "./message-attachments.js";

afterEach(cleanup);

describe("MessageAttachments", () => {
  it("V563 renders typed sizes and routes preview for each attachment kind", () => {
    const attachments = [
      attachment("image", "diagram.png", 1),
      attachment("audio", "sample.wav", 1025),
      attachment("video", "clip.mp4", 11 * 1024 * 1024),
      attachment("pdf", "brief.pdf", 1536 * 1024),
      attachment("text", "notes.txt", 20),
      attachment("file", "archive.zip", 30),
    ];
    const onPreview = vi.fn();
    render(<MessageAttachments attachments={attachments} onPreview={onPreview} />);

    expect(screen.getByText("image · 1 B")).toBeTruthy();
    expect(screen.getByText("audio · 2 KB")).toBeTruthy();
    expect(screen.getByText("video · 11 MB")).toBeTruthy();
    expect(screen.getByText("pdf · 1.5 MB")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Preview brief.pdf" }));
    expect(onPreview).toHaveBeenCalledWith(attachments[3]);
  });

  it("disables attachment actions when no preview handler is available", () => {
    render(<MessageAttachments attachments={[attachment("file", "archive.zip", 30)]} />);

    expect(
      (screen.getByRole("button", { name: "Preview archive.zip" }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });
});

function attachment(
  kind: DesktopMediaAttachment["kind"],
  name: string,
  sizeBytes: number,
): DesktopMediaAttachment {
  return {
    id: name,
    name,
    kind,
    mimeType: "application/octet-stream",
    sizeBytes,
    uri: `file:///managed/${name}`,
    source: "user",
  };
}
