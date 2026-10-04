import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { AgentMemory, DAY, ExtractiveSummarizer, estimateImportance, recency } from "../src/index.js";
import type { Summarizer } from "../src/index.js";

function setup(start = Date.UTC(2026, 0, 1)) {
  const clock = { now: start };
  let i = 0;
  const mem = new AgentMemory({ clock: () => clock.now, idFactory: () => `m${++i}` });
  return { mem, clock, start };
}

test("remember stores records with estimated importance", () => {
  const { mem } = setup();
  const m = mem.remember("I am allergic to walnuts, never suggest them!", { tags: ["Health"] });
  assert.equal(m.kind, "episodic");
  assert.deepEqual(m.tags, ["health"]);
  assert.ok(m.importance >= 0.7);
  assert.equal(mem.remember("ok thanks").importance, 0.05);
  assert.throws(() => mem.remember("   "));
});

test("near-duplicates reinforce instead of duplicating", () => {
  const { mem } = setup();
  const a = mem.remember("Amara prefers oat milk in her coffee");
  const b = mem.remember("Amara prefers oat milk in her coffee.", { tags: ["coffee"] });
  assert.equal(a.id, b.id);
  assert.equal(mem.size, 1);
  assert.equal(b.accessCount, 1);
  assert.deepEqual(b.tags, ["coffee"]);
  // different kind is a separate memory
  mem.remember("Amara prefers oat milk in her coffee", { kind: "semantic" });
  assert.equal(mem.size, 2);
});

test("recall ranks by relevance, recency and importance with a breakdown", () => {
  const { mem, clock, start } = setup();
  mem.remember("Amara likes oat milk flat whites", { importance: 0.6 });
  clock.now = start + 1 * DAY;
  mem.remember("The menu deadline is Friday 17 October", { importance: 0.7 });
  mem.remember("We chatted about rainy weather", { importance: 0.1 });
  const res = mem.recall("oat milk", { reinforce: false });
  assert.equal(res.length, 1);
  const s = res[0]!.score;
  assert.ok(s.relevance > 0 && s.recency < 1 && s.recency > 0.7 && s.importance === 0.6);
  assert.ok(Math.abs(s.total - (0.6 * s.relevance + 0.2 * s.recency + 0.2 * s.importance)) < 1e-9);
  assert.deepEqual(mem.recall("quantum chromodynamics"), []);
});

test("recency breaks ties between equally relevant memories", () => {
  const { mem, clock, start } = setup();
  mem.remember("standup notes: shipped the login page", { importance: 0.5 });
  clock.now = start + 6 * DAY;
  mem.remember("standup notes: shipped the billing page", { importance: 0.5 });
  const [first] = mem.recall("standup notes shipped page");
  assert.match(first!.memory.text, /billing/);
});

test("recall reinforces accessed memories and filters by kind/tags", () => {
  const { mem } = setup();
  const m = mem.remember("Deploy runs on Fridays", { tags: ["ops"] });
  mem.remember("Deploy checklist lives in Notion", { kind: "semantic", tags: ["ops", "docs"] });
  mem.recall("deploy");
  assert.equal(m.accessCount, 1);
  assert.equal(mem.recall("deploy", { kind: "semantic" }).length, 1);
  assert.equal(mem.recall("deploy", { tags: ["docs"] })[0]!.memory.kind, "semantic");
});

test("episodic memories decay faster than semantic ones", () => {
  const { mem, start } = setup();
  const e = mem.remember("had lunch at noon");
  const s = mem.remember("user's name is Sam", { kind: "semantic" });
  const later = start + 3 * DAY;
  assert.ok(Math.abs(recency(e, later) - 0.5) < 1e-9); // one episodic half-life
  assert.ok(recency(s, later) > 0.95);
});

test("prune removes weak memories, respects maxItems and never drops pinned ones", () => {
  const { mem, clock, start } = setup();
  mem.remember("weather small talk", { importance: 0.05 });
  mem.remember("walnut allergy", { importance: 0.1, pinned: true });
  mem.remember("company is Kettle and Crumb", { kind: "semantic", importance: 0.8 });
  mem.remember("asked about invoices", { importance: 0.4 });
  clock.now = start + 40 * DAY;
  const removed = mem.prune({ minStrength: 0.15 });
  assert.deepEqual(removed, ["m1"]);
  mem.prune({ maxItems: 1 });
  assert.deepEqual(mem.all().map((m) => m.id).sort(), ["m2", "m3"]);
});

test("consolidate turns related episodes into a semantic fact", async () => {
  const { mem, clock, start } = setup();
  mem.remember("Amara asked for Instagram caption ideas", { tags: ["marketing"] });
  clock.now = start + DAY;
  mem.remember("Amara wants to post on Instagram three times a week");
  mem.remember("Amara asked about scheduling Instagram posts");
  mem.remember("The printer deadline is Friday");
  const { created, clusters } = await mem.consolidate({ similarity: 0.15 });
  assert.equal(created.length, 1);
  assert.equal(clusters[0]!.length, 3);
  const fact = created[0]!;
  assert.equal(fact.kind, "semantic");
  assert.equal(fact.derivedFrom?.length, 3);
  assert.ok(fact.tags.includes("marketing"));
  assert.ok(mem.get("m1")!.tags.includes("consolidated"));
  // consolidated episodes are collapsed behind their fact in recall
  const res = mem.recall("instagram posts", { k: 5 });
  assert.equal(res[0]!.memory.id, fact.id);
  assert.ok(res.every((r) => r.memory.kind === "semantic" || !fact.derivedFrom!.includes(r.memory.id)));
  // running again does not re-consolidate the same episodes
  assert.equal((await mem.consolidate({ similarity: 0.15 })).created.length, 0);
});

test("consolidate uses a pluggable summariser", async () => {
  const calls: string[][] = [];
  const fake: Summarizer = { name: "fake", summarize: async (texts) => (calls.push(texts), "User is into Instagram growth.") };
  const mem = new AgentMemory({ summarizer: fake });
  mem.remember("Instagram reels did well");
  mem.remember("Instagram posting schedule");
  const { created } = await mem.consolidate({ similarity: 0.1 });
  assert.equal(created[0]!.text, "User is into Instagram growth.");
  assert.equal(calls[0]!.length, 2);
});

test("extractive summariser keeps distinct facts within budget", async () => {
  const out = await new ExtractiveSummarizer(80).summarize(
    ["The menu deadline is Friday.", "Discussed the autumn menu.", "The menu deadline is Friday."],
    ["menu", "deadline"],
  );
  assert.equal(out, "The menu deadline is Friday; Discussed the autumn menu.");
});

test("buildContext includes pinned memories first and respects maxChars", () => {
  const { mem } = setup();
  mem.remember("Allergic to walnuts", { pinned: true });
  mem.remember("Prefers oat milk in coffee");
  mem.remember("Coffee machine is a La Marzocco");
  const ctx = mem.buildContext("coffee", { k: 5 });
  assert.match(ctx, /^Always remember:\n- Allergic to walnuts\nRelevant memories:\n- \[episodic\]/);
  assert.ok(mem.buildContext("coffee", { maxChars: 60 }).length <= 60);
  assert.equal(new AgentMemory().buildContext("anything"), "");
});

test("JSON file backend round-trips and writes atomically", async () => {
  const dir = await mkdtemp(join(tmpdir(), "agent-memory-"));
  const file = join(dir, "nested", "mem.json");
  const mem = await AgentMemory.open(file);
  mem.remember("Sam's birthday is 3 March", { kind: "semantic", tags: ["personal"] });
  await mem.save();
  const raw = JSON.parse(await readFile(file, "utf8"));
  assert.equal(raw.version, 1);
  assert.equal(raw.memories.length, 1);
  const again = await AgentMemory.open(file);
  assert.equal(again.size, 1);
  assert.equal(again.recall("birthday")[0]!.memory.tags[0], "personal");
});

test("estimateImportance heuristics", () => {
  assert.ok(estimateImportance("Our budget is £2k and the deadline is Monday") > estimateImportance("nice weather today"));
  assert.equal(estimateImportance("thanks!"), 0.05);
});
