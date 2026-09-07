import { z } from "zod";

export function projectUrl(path: string) {
  const scope = /^\/projects\/[a-zA-Z0-9_-]+(?=\/|$)/u.exec(window.location.pathname)?.[0] ?? "";
  return `${scope}${path}`;
}

export function projectFetch(path: string, init?: RequestInit) {
  return fetch(projectUrl(path), init);
}

export async function api<T>(path: string, schema: z.ZodType<T>, init?: RequestInit): Promise<T> {
  const response = await projectFetch(path, init);
  const value: unknown = await response.json();
  if (!response.ok) {
    const error = z.object({ error: z.string() }).safeParse(value);
    throw new Error(error.success ? error.data.error : `Request failed (${response.status}).`);
  }
  return schema.parse(value);
}

export function jsonRequest(body: unknown, method = "POST", signal?: AbortSignal): RequestInit {
  return {
    method,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    ...(signal ? { signal } : {}),
  };
}

export async function scienceTool<T>(
  name: string,
  action: string,
  request: unknown,
  schema: z.ZodType<T>,
  signal?: AbortSignal,
): Promise<T> {
  const result = await api(
    `/api/v1/tools/${name}`,
    z.object({ data: schema }),
    jsonRequest({ action, request }, "POST", signal),
  );
  window.dispatchEvent(new Event("swarmx:science-changed"));
  return result.data;
}

export function download(name: string, content: string, mime = "application/json") {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
