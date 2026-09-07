// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { projectFetch, projectUrl } from "../src/renderer/api.js";
import { i18n, t } from "../src/renderer/i18n.js";
import { ProjectNav } from "../src/renderer/projects.js";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  window.history.replaceState({}, "", "/");
});

it("keeps API, stream and artifact URLs tied to the project in this window", async () => {
  window.history.replaceState({}, "", "/projects/first/");
  const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(Response.json({}));
  vi.stubGlobal("fetch", fetchMock);
  expect(projectUrl("/api/ag-ui?agent=swarm")).toBe("/projects/first/api/ag-ui?agent=swarm");
  expect(projectUrl("/api/v1/artifacts/figure/content")).toBe(
    "/projects/first/api/v1/artifacts/figure/content",
  );
  await projectFetch("/api/v1/memory", { method: "GET" });
  expect(fetchMock).toHaveBeenCalledWith("/projects/first/api/v1/memory", { method: "GET" });
});

it.each(["zh", "en"])(
  "registers a named project and surfaces opening failures without losing the current project in %s",
  async (language) => {
    await i18n.changeLanguage(language);
    window.history.replaceState({}, "", "/projects/first/");
    const fetchMock = vi.fn<typeof fetch>(async (path) =>
      path === "/projects/first/api/v1/projects"
        ? Response.json({ id: "second", label: "Other research", root: "/other" }, { status: 201 })
        : Response.json({ error: "Project directory is unavailable" }, { status: 409 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    render(
      <ProjectNav
        projects={[{ id: "first", label: "Current research", root: "/current" }]}
        current="first"
        onSettings={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: t("添加项目") }));
    fireEvent.change(screen.getByLabelText(t("项目名称")), { target: { value: "Other research" } });
    fireEvent.change(screen.getByLabelText(t("项目目录")), { target: { value: "/other" } });
    fireEvent.click(screen.getByRole("button", { name: t("添加并打开") }));
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/projects/first/api/v1/projects",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ label: "Other research", root: "/other" }),
        }),
      ),
    );
    expect((await screen.findByRole("alert")).textContent).toContain("unavailable");
    expect(window.location.pathname).toBe("/projects/first/");
    expect(
      screen
        .getByRole("button", { name: t("打开项目 {{name}}", { name: "Current research" }) })
        .getAttribute("aria-current"),
    ).toBe("true");
  },
);
