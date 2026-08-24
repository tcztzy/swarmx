import { readFileSync } from "node:fs";
import type { IncomingMessage, ServerResponse } from "node:http";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import type { Context } from "@deepseek-ai/cordis";
import type {} from "@deepseek-ai/dsh-host-webserver";

export const name = "swarmx-markdown-file-links";
export const inject = ["webServer"];

const RC2_LINK_BRANCH = 'case"link":return oa(n.url,Rt(n.children,{...l,inLink:!0}),i);';
const SWARMX_LINK_BRANCH =
  'case"link":{const u=Rt(n.children,{...l,inLink:!0}),c=l.fileMentions?.resolveLink?.(n.url);return c!==void 0?f.jsx("button",{type:"button",className:dr.fileMention,title:c.title,"aria-label":c.label,onClick:c.open,children:u},i):oa(n.url,u,i)}';

interface FrontendModule {
  readonly pathname: string;
  readonly source: string;
}

/** Add the one file-link seam missing from dsh-web-frontend 0.1.1-rc.2. */
export function transformMarkdownFileLinks(source: string): string {
  const first = source.indexOf(RC2_LINK_BRANCH);
  if (first === -1) {
    throw new Error("SwarmX Markdown link seam is absent from the installed DSH frontend");
  }
  if (source.indexOf(RC2_LINK_BRANCH, first + RC2_LINK_BRANCH.length) !== -1) {
    throw new Error("SwarmX Markdown link seam must occur exactly once");
  }
  return source.replace(RC2_LINK_BRANCH, SWARMX_LINK_BRANCH);
}

function frontendModule(): FrontendModule {
  const require = createRequire(import.meta.url);
  const indexPath = require.resolve("@deepseek-ai/dsh-web-frontend/dist/index.html");
  const html = readFileSync(indexPath, "utf8");
  const matches = [...html.matchAll(/src="(\/assets\/index-[^"]+\.js)"/gu)];
  if (matches.length !== 1 || matches[0]?.[1] === undefined) {
    throw new Error("SwarmX expected exactly one DSH frontend module script");
  }
  const pathname = matches[0][1];
  const sourcePath = join(dirname(indexPath), `.${pathname}`);
  return { pathname, source: transformMarkdownFileLinks(readFileSync(sourcePath, "utf8")) };
}

function serveModule(source: string, request: IncomingMessage, response: ServerResponse): void {
  if (request.method !== "GET" && request.method !== "HEAD") {
    response.statusCode = 405;
    response.setHeader("Allow", "GET, HEAD");
    response.end();
    return;
  }
  response.statusCode = 200;
  response.setHeader("Content-Type", "text/javascript; charset=utf-8");
  response.setHeader("Cache-Control", "no-cache");
  response.setHeader("Content-Length", Buffer.byteLength(source));
  response.end(request.method === "HEAD" ? undefined : source);
}

/** Override only the prebuilt Markdown module asset; the DSH fallback owns everything else. */
export function apply(ctx: Context): () => void {
  const module = frontendModule();
  return ctx.webServer.register({
    kind: "exact",
    path: module.pathname,
    handler: (request, response) => serveModule(module.source, request, response),
  });
}
