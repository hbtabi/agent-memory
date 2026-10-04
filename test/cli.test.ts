import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { main, parseArgs } from "../src/cli.js";
import { runDemo } from "../src/demo.js";

function capture(): { logs: string[]; restore: () => void } {
  const logs: string[] = [];
  const orig = [console.log, console.error];
  console.log = (...a: unknown[]) => void logs.push(a.join(" "));
  console.error = (...a: unknown[]) => void logs.push(a.join(" "));
  return { logs, restore: () => ([console.log, console.error] = orig) };
}

test("parseArgs handles flags and positionals", () => {
  assert.deepEqual(parseArgs(["recall", "oat", "milk", "-k", "3", "--json"]), {
    positional: ["recall", "oat", "milk"],
    flags: { k: "3", json: true },
  });
});

test("CLI add → recall → stats → forget against a JSON file", async () => {
  const file = join(await mkdtemp(join(tmpdir(), "am-cli-")), "m.json");
  const cap = capture();
  try {
    assert.equal(await main(["add", "Amara", "prefers", "oat", "milk", "--tags", "coffee", "--file", file]), 0);
    assert.equal(await main(["add", "Deadline", "is", "Friday", "--kind", "semantic", "--file", file]), 0);
    assert.equal(await main(["recall", "oat", "milk", "--json", "--file", file]), 0);
    const json = JSON.parse(cap.logs.at(-1)!);
    assert.equal(json.length, 1);
    assert.equal(await main(["stats", "--file", file]), 0);
    assert.match(cap.logs.at(-1)!, /2 memories · 1 episodic · 1 semantic/);
    assert.equal(await main(["forget", json[0].memory.id, "--file", file]), 0);
    assert.equal(await main(["forget", "nope", "--file", file]), 1);
    assert.equal(await main(["add", "x", "--kind", "weird", "--file", file]), 2);
    assert.equal(await main(["bogus", "--file", file]), 1);
  } finally {
    cap.restore();
  }
});

test("demo runs offline end to end", async () => {
  const prev = process.env.OPENAI_API_KEY;
  delete process.env.OPENAI_API_KEY;
  try {
    const mem = await runDemo({ quiet: true });
    const s = mem.stats();
    assert.ok(s.semantic >= 3);
    assert.ok(mem.all().some((m) => m.pinned));
    assert.match(mem.buildContext("coffee recipe for the autumn menu"), /walnut/);
  } finally {
    if (prev !== undefined) process.env.OPENAI_API_KEY = prev;
  }
});
