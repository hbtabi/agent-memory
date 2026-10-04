import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import { ExtractiveSummarizer, OpenAICompatibleSummarizer, summarizerFromEnv } from "../src/summarizer.js";

test("OpenAI-compatible summariser posts a chat completion", async () => {
  let body: any;
  let auth = "";
  const server = createServer((req, res) => {
    let data = "";
    req.on("data", (c) => (data += c));
    req.on("end", () => {
      body = JSON.parse(data);
      auth = String(req.headers.authorization);
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ choices: [{ message: { content: "Amara prefers oat milk." } }] }));
    });
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const { port } = server.address() as AddressInfo;
  try {
    const s = new OpenAICompatibleSummarizer("sk-test", `http://127.0.0.1:${port}/v1`, "tiny");
    assert.equal(await s.summarize(["oat milk please", "no dairy"], ["oat"]), "Amara prefers oat milk.");
    assert.equal(body.model, "tiny");
    assert.equal(auth, "Bearer sk-test");
  } finally {
    server.close();
  }
});

test("falls back to extractive summary when the API is unreachable", async () => {
  const s = new OpenAICompatibleSummarizer("k", "http://127.0.0.1:9/v1", "m", 1000);
  assert.equal(await s.summarize(["Likes tea.", "Likes green tea."], ["tea"]), "Likes tea; Likes green tea.");
});

test("summarizerFromEnv picks the backend from env vars", () => {
  assert.ok(summarizerFromEnv({}) instanceof ExtractiveSummarizer);
  assert.ok(summarizerFromEnv({ OPENAI_API_KEY: "sk" }) instanceof OpenAICompatibleSummarizer);
  assert.ok(summarizerFromEnv({ OPENAI_API_KEY: "sk", AGENT_MEMORY_OFFLINE: "1" }) instanceof ExtractiveSummarizer);
});
