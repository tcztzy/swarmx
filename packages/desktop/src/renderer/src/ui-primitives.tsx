import { cva, cx, type VariantProps } from "class-variance-authority";
import type React from "react";

const buttonVariants = cva(
  [
    "button inline-flex cursor-pointer items-center justify-center gap-[7px] rounded-app border border-transparent text-[13px] font-[620] leading-none shadow-inset",
    "transition-[transform,background-color,border-color,color,opacity,box-shadow] duration-[160ms] ease-app-out",
    "enabled:hover:-translate-y-px disabled:cursor-not-allowed disabled:opacity-[0.52] disabled:transform-none motion-reduce:transform-none",
    "focus-visible:border-[rgba(149,233,255,0.48)] focus-visible:shadow-[0_0_0_3px_rgba(149,233,255,0.09),var(--shadow-inset)]",
    "[&_[data-icon]]:size-3.5 max-520:h-[34px]",
  ],
  {
    variants: {
      variant: {
        default: [
          "text-primary-foreground",
          "[background:var(--button-default-background)] [border-color:var(--button-default-border)] [box-shadow:var(--button-default-shadow)]",
          "enabled:hover:[background:var(--button-default-background-hover)]",
        ],
        secondary: "border-border bg-card-hover text-foreground",
        ghost: [
          "bg-transparent text-muted",
          "enabled:hover:border-border-subtle enabled:hover:[background:var(--button-ghost-background-hover)] enabled:hover:text-foreground",
        ],
        destructive: "border-[rgba(248,113,113,0.26)] bg-danger-muted text-danger",
      },
      size: {
        sm: "h-[34px] px-[11px]",
        md: "h-9 px-[13px]",
        icon: "size-[34px] p-0 [&_[data-icon]]:size-4",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "md",
    },
  },
);

interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export function Button({
  children,
  className,
  variant,
  size,
  type = "button",
  ...props
}: ButtonProps) {
  return (
    <button type={type} className={buttonVariants({ variant, size, className })} {...props}>
      {children}
    </button>
  );
}

export const badgeVariants = cva(
  [
    "badge inline-flex h-[26px] items-center gap-1.5 whitespace-nowrap rounded-full border px-[9px] text-xs font-[620] shadow-inset",
    "border-border-subtle bg-[rgba(255,255,255,0.045)] text-muted backdrop-blur-[14px] backdrop-saturate-[1.4]",
    "[&_[data-icon]]:size-3.5",
  ],
  {
    variants: {
      tone: {
        neutral: null,
        loading: "[&_svg]:animate-[spin_900ms_linear_infinite]",
        success:
          "border-[rgba(52,211,153,0.22)] bg-success-muted text-success [&_svg]:animate-[spin_900ms_linear_infinite]",
        danger: "border-[rgba(248,113,113,0.26)] bg-danger-muted text-danger",
      },
    },
    defaultVariants: {
      tone: "neutral",
    },
  },
);

interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, tone, ...props }: BadgeProps) {
  return <span className={badgeVariants({ tone, className })} {...props} />;
}

export const rightPanelVariants = cva(
  "runtime-right-panel [min-width:0] [min-height:0] [display:flex] [flex-direction:column] [border-left:1px_solid_var(--border-subtle)] max-860:[position:static] max-860:[width:100%]",
  {
    variants: {
      kind: {
        workspace:
          "workspace-panel [overflow:hidden] [padding:0] [gap:0] [background:var(--card-solid)] [box-shadow:none]",
        media:
          "media-preview [overflow:hidden] [padding:0] [gap:0] [background:var(--background)] [box-shadow:none]",
        doctor:
          "doctor-panel [overflow-y:auto] [padding:14px] [gap:12px] [background:var(--card-solid)] [box-shadow:-14px_0_34px_rgba(0,_0,_0,_0.12)]",
      },
    },
  },
);

export const doctorNoticeVariants = cva(
  "doctor-notice [min-width:0] [padding:9px_10px] [display:grid] [grid-template-columns:15px_minmax(0,_1fr)] [align-items:start] [gap:8px] [border:1px_solid_var(--border-subtle)] [border-radius:8px] [font-size:11px] [line-height:1.45] [&_svg]:[width:15px] [&_svg]:[height:15px]",
  {
    variants: {
      tone: {
        neutral: "[color:var(--foreground)] [background:var(--input)]",
        error:
          "doctor-notice--error [color:var(--danger)] [background:var(--danger-muted)] [border-color:rgba(248,_113,_113,_0.24)]",
        success:
          "doctor-notice--success [color:var(--success)] [background:var(--success-muted)] [border-color:rgba(52,_211,_153,_0.24)]",
      },
    },
    defaultVariants: {
      tone: "neutral",
    },
  },
);

export { cx };
