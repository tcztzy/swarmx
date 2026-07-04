import { describe, expect, it, vi } from "vitest";
import { N8nImportError, Swarm, importN8nWorkflow } from "../src/index.js";

describe("importN8nWorkflow", () => {
  it("converts n8n workflow nodes and connections into SwarmConfig", () => {
    const result = importN8nWorkflow({
      id: "workflow-1",
      name: "Customer Intake",
      sourceN8nVersion: "1.99.0",
      active: false,
      nodes: [
        {
          id: "node-1",
          name: "Manual Trigger",
          type: "n8n-nodes-base.manualTrigger",
          typeVersion: 1,
          position: [0, 0],
          parameters: {},
        },
        {
          id: "node-2",
          name: "HTTP Request",
          type: "n8n-nodes-base.httpRequest",
          typeVersion: 4,
          position: [320, 0],
          parameters: {
            url: "https://example.com/customers",
            authentication: "predefinedCredentialType",
          },
          credentials: {
            httpBasicAuth: {
              id: "cred-1",
              name: "Production Basic Auth",
              data: { password: "secret" },
            },
          },
          notes: "Fetch customer data.",
        },
        {
          id: "node-3",
          name: "OpenAI Chat Model",
          type: "@n8n/n8n-nodes-langchain.lmChatOpenAi",
          typeVersion: 1.2,
          position: [640, 0],
          parameters: {
            model: "gpt-4o-mini",
          },
        },
      ],
      connections: {
        "Manual Trigger": {
          main: [[{ node: "HTTP Request", type: "main", index: 0 }]],
        },
        "HTTP Request": {
          main: [[{ node: "OpenAI Chat Model", type: "main", index: 0 }]],
        },
      },
    });

    expect(result.config).toEqual(
      expect.objectContaining({
        name: "Customer Intake",
        root: "Manual_Trigger",
        edges: [
          { source: "Manual_Trigger", target: "HTTP_Request" },
          { source: "HTTP_Request", target: "OpenAI_Chat_Model" },
        ],
      }),
    );
    expect(result.nodeMap).toEqual({
      "Manual Trigger": "Manual_Trigger",
      "HTTP Request": "HTTP_Request",
      "OpenAI Chat Model": "OpenAI_Chat_Model",
    });
    expect(result.config.nodes.HTTP_Request).toEqual(
      expect.objectContaining({
        kind: "agent",
        agent: expect.objectContaining({
          name: "HTTP_Request",
          parameters: expect.objectContaining({
            harness: expect.objectContaining({
              software: expect.objectContaining({
                name: "n8n-import",
                version: "1.99.0",
              }),
            }),
            n8n: expect.objectContaining({
              id: "node-2",
              name: "HTTP Request",
              type: "n8n-nodes-base.httpRequest",
              typeVersion: 4,
              position: [320, 0],
              notes: "Fetch customer data.",
              parameters: expect.objectContaining({
                url: "https://example.com/customers",
              }),
              credentials: {
                httpBasicAuth: {
                  type: "httpBasicAuth",
                  id: "cred-1",
                  name: "Production Basic Auth",
                },
              },
            }),
          }),
        }),
      }),
    );
    const importedCredentials = result.config.nodes.HTTP_Request.agent?.parameters.n8n as {
      credentials?: Record<string, Record<string, unknown>>;
    };
    expect(importedCredentials.credentials?.httpBasicAuth).not.toHaveProperty("data");
    expect(result.config.nodes.OpenAI_Chat_Model.agent?.model).toBe("gpt-4o-mini");
    expect(() => new Swarm(result.config)).not.toThrow();
  });

  it("returns actionable errors for invalid n8n JSON", () => {
    expect(() => importN8nWorkflow("{")).toThrow(N8nImportError);
    expect(() => importN8nWorkflow({ name: "missing nodes" })).toThrow(/nodes array/);
    expect(() => importN8nWorkflow({ nodes: [{ type: "n8n-nodes-base.noOp" }] })).toThrow(
      /needs a name/,
    );
  });

  it("imports missing targets and cycle-forming edges as warnings", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const result = importN8nWorkflow({
      name: "Looping Workflow",
      nodes: [
        { name: "A", type: "n8n-nodes-base.manualTrigger", parameters: {} },
        { name: "B", type: "n8n-nodes-base.noOp", parameters: {} },
      ],
      connections: {
        A: {
          main: [[{ node: "B", type: "main", index: 0 }]],
        },
        B: {
          main: [
            [
              { node: "A", type: "main", index: 0 },
              { node: "Missing", type: "main" },
            ],
          ],
        },
      },
    });

    expect(result.config.edges).toEqual([
      { source: "A", target: "B" },
      { source: "B", target: "A", condition: "false" },
    ]);
    expect(result.warnings.join("\n")).toMatch(/cycle-forming/);
    expect(result.warnings.join("\n")).toMatch(/missing target node "Missing"/);
    expect(() => new Swarm(result.config)).not.toThrow();
    warn.mockRestore();
  });
});
