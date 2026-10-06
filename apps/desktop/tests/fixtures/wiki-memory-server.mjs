import { appendFileSync } from "node:fs";
import { createInterface } from "node:readline";

const input = createInterface({ input: process.stdin });
input.on("line", (line) => {
  const message = JSON.parse(line);
  if (message.method === "tools/call")
    appendFileSync(process.env.HOST_WIKI_TEST_DISPATCH_LOG, `${JSON.stringify(message.params)}\n`);
  if (message.id === undefined) return;
  if (message.method === "tools/call" && message.params.arguments.query === "wait") return;
  const result =
    message.method === "initialize"
      ? {
          protocolVersion: message.params.protocolVersion,
          capabilities: { tools: {} },
          serverInfo: { name: "synthetic-wiki", version: "0.0.0" },
        }
      : {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                query: "synthetic",
                totalRecords: 1,
                records: [
                  {
                    datasetId: "knowledge",
                    documentId: "knowledge/synthetic.md",
                    documentName: "synthetic.md",
                    score: 0.5,
                    priority: "P2",
                    content: "Synthetic SDK lifecycle excerpt",
                  },
                ],
              }),
            },
          ],
        };
  process.stdout.write(`${JSON.stringify({ jsonrpc: "2.0", id: message.id, result })}\n`);
});
