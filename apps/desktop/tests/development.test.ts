import { once } from "node:events";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, it, vi } from "vitest";
import { ProductServices } from "../src/host/product-services.js";
import { startHost } from "../src/host/server.js";
import { resolveWorkspace } from "../src/host/workspace-settings.js";

it("serves authenticated Vite transforms and live HMR on the Host origin with a CSP nonce", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-dev-"));
  await writeFile(join(root, "vite.config.ts"), "export default {};");
  await writeFile(
    join(root, "index.html"),
    '<html><head></head><body><script>globalThis.boot = true;</script><script type="module" src="/counter.ts"></script></body></html>',
  );
  const source = (value: number) =>
    `export const count: number = ${value}; if (import.meta.hot) import.meta.hot.accept();`;
  await writeFile(join(root, "counter.ts"), source(1));
  const products = await ProductServices.create({
    productHome: join(root, "product"),
    workspace: await resolveWorkspace(root),
  });
  const host = await startHost({
    products,
    workspace: products.options.workspace,
    rendererRoot: root,
    development: true,
  });
  let socket: WebSocket | undefined;
  try {
    const origin = host.internalUrl;
    expect((await fetch(`${origin}/@vite/client`)).status).toBe(401);
    const launch = await fetch(host.issueLaunchUrl(), { redirect: "manual" });
    const cookie = launch.headers.get("set-cookie")?.split(";")[0] ?? "";
    const headers = { cookie, origin };
    const page = await fetch(`${origin}/projects/${products.options.workspace.id}/`, { headers });
    const csp = page.headers.get("content-security-policy") ?? "";
    const nonce = /'nonce-([^']+)'/u.exec(csp)?.[1];
    expect(nonce).toBeTruthy();
    expect(csp).toContain("connect-src 'self'");
    const html = await page.text();
    expect(html).toContain("/@vite/client");
    expect(html).toContain(`nonce="${nonce}"`);
    expect(
      (
        await fetch(`${origin}/counter.ts`, {
          headers: { cookie, origin: "https://untrusted.example" },
        })
      ).status,
    ).toBe(403);
    const module = await (await fetch(`${origin}/counter.ts`, { headers })).text();
    expect(module).toContain("count = 1");
    expect(module).not.toContain("count: number");
    expect((await fetch(`${origin}/api/v1/settings`, { headers })).status).toBe(200);

    const client = await (await fetch(`${origin}/@vite/client`, { headers })).text();
    const token = /const wsToken = "([^"]+)"/u.exec(client)?.[1];
    expect(token).toBeTruthy();
    socket = new WebSocket(`${origin.replace("http:", "ws:")}/?token=${token}`, "vite-hmr");
    const messages: { type: string; updates?: { path: string }[] }[] = [];
    socket.addEventListener("message", (event) => messages.push(JSON.parse(String(event.data))));
    await once(socket, "open");
    await writeFile(join(root, "counter.ts"), source(2));
    await vi.waitFor(
      () =>
        expect(messages).toContainEqual(
          expect.objectContaining({
            type: "update",
            updates: expect.arrayContaining([expect.objectContaining({ path: "/counter.ts" })]),
          }),
        ),
      { timeout: 5000 },
    );
    expect(await (await fetch(`${origin}/counter.ts`, { headers })).text()).toContain("count = 2");
  } finally {
    socket?.close();
    await host.dispose();
    await rm(root, { recursive: true, force: true });
  }
}, 15_000);
