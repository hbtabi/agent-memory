import assert from "node:assert/strict";
import { test } from "node:test";
import { TfIdfIndex, cosine } from "../src/tfidf.js";
import { stem, tokenize } from "../src/tokenize.js";

test("tokenize lowercases, drops stopwords and stems", () => {
  assert.deepEqual(tokenize("She PREFERS oat milk, and the café's flat whites!"), ["prefer", "oat", "milk", "cafe", "flat", "white"]);
  assert.equal(stem("preferred"), "prefer");
  assert.equal(stem("stopped"), "stop");
  assert.equal(stem("berries"), "berry");
  assert.equal(stem("glass"), "glass");
});

test("search ranks the most similar document first", () => {
  const idx = new TfIdfIndex();
  idx.add("a", "Amara prefers oat milk in her coffee");
  idx.add("b", "The autumn menu deadline is Friday");
  idx.add("c", "Instagram reels of latte art perform well");
  const res = idx.search("what milk goes in the coffee?");
  assert.deepEqual([...res.keys()], ["a"]);
  assert.ok(res.get("a")! > 0.3);
});

test("rare terms weigh more than common ones (idf)", () => {
  const idx = new TfIdfIndex();
  for (let i = 0; i < 5; i++) idx.add(`d${i}`, `coffee note ${i}`);
  idx.add("rare", "coffee walnut allergy");
  assert.ok(idx.idf("walnut") > idx.idf("coffee"));
});

test("add/remove keeps document frequencies consistent", () => {
  const idx = new TfIdfIndex();
  idx.add("x", "alpha beta");
  idx.add("y", "beta gamma");
  idx.remove("x");
  assert.equal(idx.size, 1);
  assert.equal(idx.search("alpha").size, 0);
  idx.add("y", "delta"); // re-adding replaces
  assert.equal(idx.search("gamma").size, 0);
  assert.equal(idx.search("delta").size, 1);
});

test("vectors are L2-normalised so self-similarity is 1", () => {
  const idx = new TfIdfIndex();
  idx.add("a", "one two two three");
  idx.add("b", "four");
  const v = idx.vector("a");
  assert.ok(Math.abs(cosine(v, v) - 1) < 1e-9);
});
