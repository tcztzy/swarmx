import { z } from "zod";
import type { DesktopEventContract, DesktopInvokeContract } from "./base.js";

export const DESKTOP_BROWSER_LIMITS = Object.freeze({
  idBytes: 256,
  inputBytes: 32 * 1024,
  urlBytes: 128 * 1024,
  titleBytes: 8 * 1024,
  errorBytes: 8 * 1024,
});

const UTF8_ENCODER = new TextEncoder();
const boundedText = (maximumBytes: number) =>
  z.string().refine((value) => UTF8_ENCODER.encode(value).byteLength <= maximumBytes, {
    message: `text exceeds the ${maximumBytes}-byte transport limit`,
  });
const BrowserCreateIdSchema = boundedText(DESKTOP_BROWSER_LIMITS.idBytes);
const BrowserIdSchema = z
  .string()
  .min(1)
  .refine((value) => UTF8_ENCODER.encode(value).byteLength <= DESKTOP_BROWSER_LIMITS.idBytes, {
    message: "Browser id exceeds the transport limit",
  });
const BrowserInputSchema = boundedText(DESKTOP_BROWSER_LIMITS.inputBytes);
const BrowserCoordinateSchema = z.custom<number>((value) => typeof value === "number", {
  message: "Expected a number",
});

export const DesktopBrowserBoundsSchema = z
  .object({
    x: BrowserCoordinateSchema,
    y: BrowserCoordinateSchema,
    width: BrowserCoordinateSchema,
    height: BrowserCoordinateSchema,
  })
  .strict();

export const DesktopBrowserStateSchema = z
  .object({
    id: BrowserIdSchema,
    url: boundedText(DESKTOP_BROWSER_LIMITS.urlBytes),
    title: boundedText(DESKTOP_BROWSER_LIMITS.titleBytes),
    loading: z.boolean(),
    canGoBack: z.boolean(),
    canGoForward: z.boolean(),
    error: z
      .string()
      .min(1)
      .refine(
        (value) => UTF8_ENCODER.encode(value).byteLength <= DESKTOP_BROWSER_LIMITS.errorBytes,
        { message: "Browser error exceeds the transport limit" },
      )
      .optional(),
  })
  .strict();

export const DesktopBrowserCreateInputSchema = z
  .object({
    id: BrowserCreateIdSchema.optional(),
    url: BrowserInputSchema.optional(),
    bounds: DesktopBrowserBoundsSchema.optional(),
    visible: z.boolean().optional(),
  })
  .strict();

const BrowserIdInputSchema = z.object({ id: BrowserIdSchema }).strict();
const BrowserUpdatedSchema = z.object({ updated: z.boolean() }).strict();
const BrowserDestroyedSchema = z.object({ destroyed: z.boolean() }).strict();

export type DesktopBrowserBounds = z.infer<typeof DesktopBrowserBoundsSchema>;
export type DesktopBrowserState = z.infer<typeof DesktopBrowserStateSchema>;
export type DesktopBrowserCreateInput = z.infer<typeof DesktopBrowserCreateInputSchema>;

export const BrowserInvokeContracts = {
  "browser:create": {
    kind: "invoke",
    args: z.tuple([DesktopBrowserCreateInputSchema.optional()]),
    result: DesktopBrowserStateSchema,
    audit: "intent_outcome",
  },
  "browser:navigate": {
    kind: "invoke",
    args: z.tuple([z.object({ id: BrowserIdSchema, url: BrowserInputSchema }).strict()]),
    result: DesktopBrowserStateSchema,
    audit: "intent_outcome",
  },
  "browser:back": {
    kind: "invoke",
    args: z.tuple([BrowserIdInputSchema]),
    result: DesktopBrowserStateSchema,
    audit: "intent_outcome",
  },
  "browser:forward": {
    kind: "invoke",
    args: z.tuple([BrowserIdInputSchema]),
    result: DesktopBrowserStateSchema,
    audit: "intent_outcome",
  },
  "browser:reload": {
    kind: "invoke",
    args: z.tuple([BrowserIdInputSchema]),
    result: DesktopBrowserStateSchema,
    audit: "intent_outcome",
  },
  "browser:setBounds": {
    kind: "invoke",
    args: z.tuple([z.object({ id: BrowserIdSchema, bounds: DesktopBrowserBoundsSchema }).strict()]),
    result: BrowserUpdatedSchema,
    audit: "failure_only",
  },
  "browser:setVisible": {
    kind: "invoke",
    args: z.tuple([z.object({ id: BrowserIdSchema, visible: z.boolean() }).strict()]),
    result: BrowserUpdatedSchema,
    audit: "failure_only",
  },
  "browser:destroy": {
    kind: "invoke",
    args: z.tuple([BrowserIdInputSchema]),
    result: BrowserDestroyedSchema,
    audit: "intent_outcome",
  },
} as const satisfies Record<string, DesktopInvokeContract>;

export const BrowserEventContracts = {
  "browser:state": {
    kind: "event",
    payload: DesktopBrowserStateSchema,
  },
} as const satisfies Record<string, DesktopEventContract>;

export interface DesktopBrowserApi {
  createBrowser(params?: DesktopBrowserCreateInput): Promise<DesktopBrowserState>;
  navigateBrowser(id: string, url: string): Promise<DesktopBrowserState>;
  backBrowser(id: string): Promise<DesktopBrowserState>;
  forwardBrowser(id: string): Promise<DesktopBrowserState>;
  reloadBrowser(id: string): Promise<DesktopBrowserState>;
  setBrowserBounds(id: string, bounds: DesktopBrowserBounds): Promise<{ updated: boolean }>;
  setBrowserVisible(id: string, visible: boolean): Promise<{ updated: boolean }>;
  destroyBrowser(id: string): Promise<{ destroyed: boolean }>;
  onBrowserState(listener: (state: DesktopBrowserState) => void): () => void;
}
