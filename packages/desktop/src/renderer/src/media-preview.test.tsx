/** @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  DesktopMediaAttachment,
  DesktopMediaPreview,
  SwarmxAPI,
} from "../../shared/desktop-api.js";
import { MediaPreviewPanel } from "./media-preview.js";

afterEach(cleanup);

describe("MediaPreviewPanel", () => {
  it.each([
    ["image", "image/png", "diagram.png", "img"],
    ["pdf", "application/pdf", "brief.pdf", "iframe"],
    ["audio", "audio/wav", "sample.wav", "audio"],
    ["video", "video/mp4", "clip.mp4", "video"],
    ["text", "text/markdown", "notes.md", "pre"],
    ["file", "application/zip", "archive.zip", "file"],
  ] as const)("renders the %s preview variant", async (kind, mimeType, name, expectedElement) => {
    const attachment = mediaAttachment(kind, mimeType, name);
    const preview: DesktopMediaPreview = {
      status: "available",
      attachment,
      ...(kind === "text" ? { text: "# Notes" } : {}),
      ...(kind !== "text" && kind !== "file"
        ? { previewUrl: `swarmx-media://asset/${"a".repeat(64)}/${name}` }
        : {}),
    };
    const { container } = render(
      <MediaPreviewPanel
        api={mediaApi(preview)}
        attachment={attachment}
        onClose={() => undefined}
      />,
    );

    await waitFor(() => expect(screen.queryByText("Loading preview")).toBeNull());
    if (expectedElement === "file") {
      expect(screen.getByText("application/zip")).toBeTruthy();
      expect(screen.getByRole("button", { name: "Open in default app" })).toBeTruthy();
    } else if (expectedElement === "iframe") {
      expect(screen.getByTitle("PDF preview: brief.pdf")).toBeTruthy();
    } else {
      expect(container.querySelector(expectedElement)).toBeTruthy();
    }
  });

  it("renders a stable unavailable state", async () => {
    const attachment = mediaAttachment("image", "image/png", "missing.png");
    render(
      <MediaPreviewPanel
        api={mediaApi({
          status: "unavailable",
          attachment,
          error: "Managed media changed after it was imported.",
        })}
        attachment={attachment}
        onClose={() => undefined}
      />,
    );

    expect(await screen.findByText("Preview unavailable")).toBeTruthy();
    expect(screen.getByText(/changed after it was imported/i)).toBeTruthy();
  });

  it("focuses close, restores actions through Main, and closes on Escape", async () => {
    const attachment = mediaAttachment("image", "image/png", "diagram.png");
    const preview: DesktopMediaPreview = {
      status: "available",
      attachment,
      previewUrl: `swarmx-media://asset/${"a".repeat(64)}/diagram.png`,
    };
    const api = mediaApi(preview);
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<MediaPreviewPanel api={api} attachment={attachment} onClose={onClose} />);

    const close = screen.getByRole("button", { name: "Close preview" });
    expect(document.activeElement).toBe(close);
    await user.click(screen.getByRole("button", { name: "Reveal diagram.png" }));
    await user.click(screen.getByRole("button", { name: "Open diagram.png" }));
    expect(api.revealMediaAttachment).toHaveBeenCalledWith(attachment);
    expect(api.openMediaAttachment).toHaveBeenCalledWith(attachment);

    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

function mediaAttachment(
  kind: DesktopMediaAttachment["kind"],
  mimeType: string,
  name: string,
): DesktopMediaAttachment {
  return {
    id: `${kind}:${name}`,
    name,
    kind,
    mimeType,
    sizeBytes: 1234,
    uri: `file:///managed/${name}`,
    source: "user",
  };
}

function mediaApi(preview: DesktopMediaPreview) {
  return {
    previewMediaAttachment: vi.fn(async () => preview),
    openMediaAttachment: vi.fn(async () => ({ opened: true })),
    revealMediaAttachment: vi.fn(async () => ({ revealed: true })),
  } as unknown as SwarmxAPI & {
    previewMediaAttachment: ReturnType<typeof vi.fn>;
    openMediaAttachment: ReturnType<typeof vi.fn>;
    revealMediaAttachment: ReturnType<typeof vi.fn>;
  };
}
