import { ExternalLink, File, FileAudio, FolderOpen, Loader2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type {
  DesktopMediaAttachment,
  DesktopMediaPreview,
  SwarmxAPI,
} from "../../shared/desktop-api.js";
import { attachmentIcon, formatMediaBytes } from "./message-attachments.js";
import { errorMessage } from "./text-utils.js";

export function MediaPreviewPanel({
  api,
  attachment,
  onClose,
}: {
  api: SwarmxAPI;
  attachment: DesktopMediaAttachment;
  onClose: () => void;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const [preview, setPreview] = useState<DesktopMediaPreview | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setPreview(null);
    setActionError(null);
    void api
      .previewMediaAttachment(attachment)
      .then((result) => {
        if (active) setPreview(result);
      })
      .catch((error) => {
        if (!active) return;
        setPreview({
          status: "unavailable",
          attachment,
          error: errorMessage(error),
        });
      });
    return () => {
      active = false;
    };
  }, [api, attachment]);

  useEffect(() => {
    closeRef.current?.focus();
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  async function runMediaAction(action: () => Promise<unknown>): Promise<void> {
    setActionError(null);
    try {
      const result = await action();
      if (
        result &&
        typeof result === "object" &&
        "error" in result &&
        typeof result.error === "string"
      ) {
        setActionError(result.error);
      }
    } catch (error) {
      setActionError(errorMessage(error));
    }
  }

  return (
    <aside className="runtime-right-panel media-preview" aria-label={`Preview ${attachment.name}`}>
      <header className="media-preview__header">
        <span className="media-preview__kind">
          <MediaIcon attachment={attachment} />
        </span>
        <span className="media-preview__title">
          <small>Preview</small>
          <strong title={attachment.name}>{attachment.name}</strong>
        </span>
        <button
          type="button"
          className="media-preview__icon-button"
          onClick={() => void runMediaAction(() => api.revealMediaAttachment(attachment))}
          aria-label={`Reveal ${attachment.name}`}
          title="Reveal in file browser"
        >
          <FolderOpen aria-hidden="true" />
        </button>
        <button
          type="button"
          className="media-preview__icon-button"
          onClick={() => void runMediaAction(() => api.openMediaAttachment(attachment))}
          aria-label={`Open ${attachment.name}`}
          title="Open in default app"
        >
          <ExternalLink aria-hidden="true" />
        </button>
        <button
          ref={closeRef}
          type="button"
          className="media-preview__icon-button"
          onClick={onClose}
          aria-label="Close preview"
          title="Close preview"
        >
          <X aria-hidden="true" />
        </button>
      </header>

      <div className="media-preview__body">
        {!preview ? (
          <div className="media-preview__state">
            <Loader2 className="is-spinning" aria-hidden="true" />
            <span>Loading preview</span>
          </div>
        ) : preview.status === "unavailable" ? (
          <output className="media-preview__state media-preview__state--error">
            <MediaIcon attachment={attachment} />
            <strong>Preview unavailable</strong>
            <span>{preview.error ?? "The local file is no longer available."}</span>
          </output>
        ) : attachment.kind === "image" && preview.previewUrl ? (
          <div className="media-preview__image-stage">
            <img src={preview.previewUrl} alt={attachment.name} />
          </div>
        ) : attachment.kind === "pdf" && preview.previewUrl ? (
          <iframe
            className="media-preview__pdf"
            src={preview.previewUrl}
            title={`PDF preview: ${attachment.name}`}
          />
        ) : attachment.kind === "audio" && preview.previewUrl ? (
          <div className="media-preview__media-stage">
            <FileAudio aria-hidden="true" />
            <strong>{attachment.name}</strong>
            {/* biome-ignore lint/a11y/useMediaCaption: local user media has no host-authored caption track */}
            <audio src={preview.previewUrl} controls preload="metadata" />
          </div>
        ) : attachment.kind === "video" && preview.previewUrl ? (
          <div className="media-preview__video-stage">
            {/* biome-ignore lint/a11y/useMediaCaption: local user media has no host-authored caption track */}
            <video src={preview.previewUrl} controls preload="metadata" />
          </div>
        ) : attachment.kind === "text" ? (
          <pre className="media-preview__text">{preview.text ?? ""}</pre>
        ) : (
          <div className="media-preview__state">
            <File aria-hidden="true" />
            <strong>{attachment.name}</strong>
            <span>{attachment.mimeType}</span>
            <span>{formatMediaBytes(attachment.sizeBytes)}</span>
            <button
              type="button"
              onClick={() => void runMediaAction(() => api.openMediaAttachment(attachment))}
            >
              Open in default app
            </button>
          </div>
        )}
        {actionError && (
          <p className="media-preview__action-error" role="alert">
            {actionError}
          </p>
        )}
      </div>
    </aside>
  );
}

function MediaIcon({ attachment }: { attachment: DesktopMediaAttachment }) {
  const Icon = attachmentIcon(attachment);
  return <Icon aria-hidden="true" />;
}
