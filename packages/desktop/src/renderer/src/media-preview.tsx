import { ExternalLink, File, FileAudio, FolderOpen, Loader2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type {
  DesktopMediaAttachment,
  DesktopMediaPreview,
  SwarmxAPI,
} from "../../shared/desktop-api.js";
import { AttachmentIcon, formatMediaBytes } from "./message-attachments.js";
import { errorMessage } from "./text-utils.js";
import { rightPanelVariants } from "./ui-primitives.js";

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
    <aside
      className={rightPanelVariants({ kind: "media" })}
      aria-label={`Preview ${attachment.name}`}
    >
      <header className="media-preview__header [min-width:0] [height:48px] [flex:0_0_48px] [padding:0_8px_0_13px] [display:flex] [align-items:center] [gap:5px] [border-bottom:1px_solid_var(--border-subtle)] [background:var(--card-solid)]">
        <span className="media-preview__kind [width:28px] [height:28px] [flex:0_0_28px] [display:grid] [place-items:center] [color:var(--muted)] [background:var(--card-hover)] [border:1px_solid_var(--border-subtle)] [border-radius:8px] [&_svg]:[width:15px] [&_svg]:[height:15px]">
          <AttachmentIcon attachment={attachment} />
        </span>
        <span className="media-preview__title [min-width:0] [margin-right:auto] [display:flex] [flex-direction:column] [&_small]:[color:var(--muted-foreground)] [&_small]:[font-size:9px] [&_small]:[font-weight:650] [&_small]:[letter-spacing:0.05em] [&_small]:[line-height:1.2] [&_small]:[text-transform:uppercase] [&_strong]:[overflow:hidden] [&_strong]:[color:var(--foreground)] [&_strong]:[font-size:12px] [&_strong]:[font-weight:600] [&_strong]:[line-height:1.35] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap]">
          <small>Preview</small>
          <strong title={attachment.name}>{attachment.name}</strong>
        </span>
        <button
          type="button"
          className="media-preview__icon-button [width:30px] [height:30px] [flex:0_0_30px] [display:grid] [place-items:center] [color:var(--muted-foreground)] [background:transparent] [border:0] [border-radius:8px] [cursor:pointer] [&_svg]:[width:15px] [&_svg]:[height:15px]"
          onClick={() => void runMediaAction(() => api.revealMediaAttachment(attachment))}
          aria-label={`Reveal ${attachment.name}`}
          title="Reveal in file browser"
        >
          <FolderOpen aria-hidden="true" />
        </button>
        <button
          type="button"
          className="media-preview__icon-button [width:30px] [height:30px] [flex:0_0_30px] [display:grid] [place-items:center] [color:var(--muted-foreground)] [background:transparent] [border:0] [border-radius:8px] [cursor:pointer] [&_svg]:[width:15px] [&_svg]:[height:15px]"
          onClick={() => void runMediaAction(() => api.openMediaAttachment(attachment))}
          aria-label={`Open ${attachment.name}`}
          title="Open in default app"
        >
          <ExternalLink aria-hidden="true" />
        </button>
        <button
          ref={closeRef}
          type="button"
          className="media-preview__icon-button [width:30px] [height:30px] [flex:0_0_30px] [display:grid] [place-items:center] [color:var(--muted-foreground)] [background:transparent] [border:0] [border-radius:8px] [cursor:pointer] [&_svg]:[width:15px] [&_svg]:[height:15px]"
          onClick={onClose}
          aria-label="Close preview"
          title="Close preview"
        >
          <X aria-hidden="true" />
        </button>
      </header>

      <div className="media-preview__body [position:relative] [min-width:0] [min-height:0] [flex:1] [overflow:auto] [background:var(--background)]">
        {!preview ? (
          <div className="media-preview__state [min-height:100%] [padding:32px] [display:flex] [align-items:center] [justify-content:center] [flex-direction:column] [gap:10px] [color:var(--muted-foreground)] [text-align:center] [&_>_svg]:[width:34px] [&_>_svg]:[height:34px] [&_>_svg]:[color:var(--muted)] [&_strong]:[max-width:min(100%,_420px)] [&_strong]:[overflow:hidden] [&_strong]:[color:var(--foreground)] [&_strong]:[font-size:13px] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap] [&_span]:[max-width:460px] [&_span]:[font-size:11px] [&_span]:[line-height:1.5] [&_button]:[margin-top:5px] [&_button]:[padding:7px_11px] [&_button]:[color:var(--foreground)] [&_button]:[background:var(--card)] [&_button]:[border:1px_solid_var(--border)] [&_button]:[border-radius:8px] [&_button]:[cursor:pointer]">
            <Loader2
              className="is-spinning [animation:spin_0.9s_linear_infinite]"
              aria-hidden="true"
            />
            <span>Loading preview</span>
          </div>
        ) : preview.status === "unavailable" ? (
          <output className="media-preview__state media-preview__state--error [min-height:100%] [padding:32px] [display:flex] [align-items:center] [justify-content:center] [flex-direction:column] [gap:10px] [color:var(--muted-foreground)] [text-align:center] [&_>_svg]:[width:34px] [&_>_svg]:[height:34px] [&_strong]:[max-width:min(100%,_420px)] [&_strong]:[overflow:hidden] [&_strong]:[color:var(--foreground)] [&_strong]:[font-size:13px] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap] [&_span]:[max-width:460px] [&_span]:[font-size:11px] [&_span]:[line-height:1.5] [&_button]:[margin-top:5px] [&_button]:[padding:7px_11px] [&_button]:[color:var(--foreground)] [&_button]:[background:var(--card)] [&_button]:[border:1px_solid_var(--border)] [&_button]:[border-radius:8px] [&_button]:[cursor:pointer] [&_>_svg]:[color:var(--danger)]">
            <AttachmentIcon attachment={attachment} />
            <strong>Preview unavailable</strong>
            <span>{preview.error ?? "The local file is no longer available."}</span>
          </output>
        ) : attachment.kind === "image" && preview.previewUrl ? (
          <div className="media-preview__image-stage [min-width:100%] [min-height:100%] [padding:22px] [display:grid] [place-items:center] [background:var(--background)] [&_img]:[display:block] [&_img]:[max-width:100%] [&_img]:[max-height:calc(100vh_-_132px)] [&_img]:[object-fit:contain] [&_img]:[border-radius:4px] [&_img]:[box-shadow:0_12px_34px_rgba(0,_0,_0,_0.2)]">
            <img src={preview.previewUrl} alt={attachment.name} />
          </div>
        ) : attachment.kind === "pdf" && preview.previewUrl ? (
          <iframe
            className="media-preview__pdf [width:100%] [height:100%] [min-height:520px] [display:block] [background:#f2f2f2] [border:0]"
            src={preview.previewUrl}
            title={`PDF preview: ${attachment.name}`}
          />
        ) : attachment.kind === "audio" && preview.previewUrl ? (
          <div className="media-preview__media-stage [min-height:100%] [padding:32px] [display:flex] [align-items:center] [justify-content:center] [flex-direction:column] [gap:10px] [color:var(--muted-foreground)] [text-align:center] [&_>_svg]:[width:34px] [&_>_svg]:[height:34px] [&_>_svg]:[color:var(--muted)] [&_strong]:[max-width:min(100%,_420px)] [&_strong]:[overflow:hidden] [&_strong]:[color:var(--foreground)] [&_strong]:[font-size:13px] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap] [&_audio]:[width:min(100%,_430px)] [&_audio]:[margin-top:7px]">
            <FileAudio aria-hidden="true" />
            <strong>{attachment.name}</strong>
            {/* biome-ignore lint/a11y/useMediaCaption: local user media has no host-authored caption track */}
            <audio src={preview.previewUrl} controls preload="metadata" />
          </div>
        ) : attachment.kind === "video" && preview.previewUrl ? (
          <div className="media-preview__video-stage [min-height:100%] [padding:18px] [display:grid] [place-items:center] [background:#050505] [&_video]:[width:100%] [&_video]:[max-height:calc(100vh_-_118px)]">
            {/* biome-ignore lint/a11y/useMediaCaption: local user media has no host-authored caption track */}
            <video src={preview.previewUrl} controls preload="metadata" />
          </div>
        ) : attachment.kind === "text" ? (
          <pre className="media-preview__text [min-height:100%] [margin:0] [padding:20px_22px] [overflow:auto] [color:var(--foreground)] [background:var(--background)] [font-family:var(--font-mono)] [font-size:12px] [line-height:1.65] [tab-size:2] [white-space:pre-wrap] [word-break:break-word]">
            {preview.text ?? ""}
          </pre>
        ) : (
          <div className="media-preview__state [min-height:100%] [padding:32px] [display:flex] [align-items:center] [justify-content:center] [flex-direction:column] [gap:10px] [color:var(--muted-foreground)] [text-align:center] [&_>_svg]:[width:34px] [&_>_svg]:[height:34px] [&_>_svg]:[color:var(--muted)] [&_strong]:[max-width:min(100%,_420px)] [&_strong]:[overflow:hidden] [&_strong]:[color:var(--foreground)] [&_strong]:[font-size:13px] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap] [&_span]:[max-width:460px] [&_span]:[font-size:11px] [&_span]:[line-height:1.5] [&_button]:[margin-top:5px] [&_button]:[padding:7px_11px] [&_button]:[color:var(--foreground)] [&_button]:[background:var(--card)] [&_button]:[border:1px_solid_var(--border)] [&_button]:[border-radius:8px] [&_button]:[cursor:pointer]">
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
          <p
            className="media-preview__action-error [position:sticky] [right:12px] [bottom:12px] [left:12px] [margin:0_12px_12px] [padding:8px_10px] [background:color-mix(in_srgb,_var(--danger)_10%,_var(--card-solid))] [border:1px_solid_color-mix(in_srgb,_var(--danger)_32%,_transparent)] [border-radius:8px] [font-size:11px] [color:var(--danger)]"
            role="alert"
          >
            {actionError}
          </p>
        )}
      </div>
    </aside>
  );
}
