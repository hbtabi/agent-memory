import { AgentMemory } from "./memory.js";
import { DAY, HOUR, recency } from "./scoring.js";
import { summarizerFromEnv } from "./summarizer.js";
import { bar, c, trunc } from "./ui.js";

/** A fictional two-week conversation between an assistant and a café owner. */
const EPISODES: [number, string, string[]][] = [
  [0, "Amara said she runs Kettle & Crumb, a small café in Hackney.", ["profile"]],
  [0.1, "Amara mentioned she always drinks oat milk flat whites, never dairy.", ["preference", "coffee"]],
  [0.2, "ok thanks", []],
  [1, "Asked for Instagram caption ideas for the new cinnamon buns.", ["marketing", "instagram"]],
  [2, "Amara wants to post on Instagram three times a week but struggles to find time.", ["marketing", "instagram"]],
  [3, "Discussed the new autumn menu: pumpkin soup, sourdough toasties and chai.", ["menu"]],
  [4, "The autumn menu must be printed by Friday 17 October, deadline from the printer.", ["menu", "deadline"]],
  [5, "Amara asked again about scheduling Instagram posts in advance with a free tool.", ["marketing", "instagram"]],
  [6, "She is allergic to walnuts, so no walnut recipes in suggestions!", ["preference", "health"]],
  [7, "Talked about the weather being rainy and footfall being slow.", ["smalltalk"]],
  [8, "Amara prefers oat milk in every coffee recipe we draft for the menu.", ["preference", "coffee"]],
  [9, "Helped write a reply to a 3-star Google review about slow service.", ["reviews"]],
  [10, "She wants more Google reviews; budget for marketing is £150 a month.", ["reviews", "marketing"]],
  [12, "Instagram reels of latte art got 4x more views than photos.", ["marketing", "instagram"]],
];

export async function runDemo(opts: { file?: string; quiet?: boolean } = {}): Promise<AgentMemory> {
  const start = Date.UTC(2026, 9, 1, 9, 0);
  let now = start;
  const mem = new AgentMemory({ clock: () => now, summarizer: summarizerFromEnv(), idFactory: counter() });
  const log = opts.quiet ? () => {} : (s = "") => console.log(s);

  log(c.bold("1. Two weeks of conversations → episodic memories") + c.dim("  (importance estimated offline)"));
  for (const [day, text, tags] of EPISODES) {
    now = start + day * DAY;
    const pinned = tags.includes("health"); // safety-critical facts get pinned
    const m = mem.remember(text, { tags, pinned });
    log(`  ${c.dim(`day ${String(Math.floor(day)).padStart(2)}`)}  ${bar(m.importance, 8)} ${m.importance.toFixed(2)}  ${trunc(text, 76)}${pinned ? c.yellow(" [pinned]") : ""}`);
  }

  now = start + 14 * DAY;
  log("\n" + c.bold("2. Consolidation: cluster related episodes → semantic facts"));
  const { created, clusters } = await mem.consolidate({ similarity: 0.2 });
  created.forEach((f, i) => {
    log(`  ${c.magenta("◆")} ${trunc(f.text, 110)}`);
    log(c.dim(`    from ${clusters[i]!.length} episodes · importance ${f.importance.toFixed(2)}`));
  });

  log("\n" + c.bold("3. Recall on day 14") + c.dim("  score = 0.6·relevance + 0.2·recency + 0.2·importance"));
  for (const q of ["What milk should I use in Amara's coffee?", "When is the menu deadline?", "Ideas to grow her Instagram"]) {
    log(`  ${c.cyan("?")} ${c.bold(q)}`);
    for (const { memory, score } of mem.recall(q, { k: 3 })) {
      const kind = memory.kind === "semantic" ? c.magenta("sem") : c.dim("epi");
      log(
        `    ${score.total.toFixed(2)}  ${kind}  ${c.dim(`rel ${score.relevance.toFixed(2)} rec ${score.recency.toFixed(2)} imp ${score.importance.toFixed(2)}`)}  ${trunc(memory.text, 60)}`,
      );
    }
  }

  log("\n" + c.bold("4. Decay") + c.dim("  episodic half-life 3 days, semantic 60 days"));
  const small = mem.all("episodic").find((m) => m.tags.includes("smalltalk"))!;
  const fact = created[0];
  for (const d of [8, 14, 30]) {
    const t = start + d * DAY;
    const line = `  day ${String(d).padStart(2)}  smalltalk ${bar(recency(small, t, mem.scoring))} ${recency(small, t, mem.scoring).toFixed(2)}`;
    log(fact ? `${line}   semantic fact ${bar(recency(fact, t, mem.scoring))} ${recency(fact, t, mem.scoring).toFixed(2)}` : line);
  }

  now = start + 30 * DAY;
  const before = new Map(mem.all().map((m) => [m.id, m.text]));
  const removed = mem.prune({ minStrength: 0.12 });
  log("\n" + c.bold("5. Prune on day 30") + c.dim(`  strength = ½·importance + ½·recency < 0.12 → removed ${removed.length}:`));
  for (const id of removed) log(c.dim(`    ✗ ${trunc(before.get(id) ?? id, 90)}`));

  now = start + 30 * DAY + 2 * HOUR;
  log("\n" + c.bold("6. Context block for the next LLM call"));
  const ctx = mem.buildContext("Draft a new coffee recipe for the autumn menu", { k: 4 });
  for (const line of ctx.split("\n")) log("  " + c.green(trunc(line, 112)));

  const s = mem.stats();
  log("\n" + c.dim(`${s.total} memories kept (${s.episodic} episodic, ${s.semantic} semantic) · avg strength ${s.avgStrength.toFixed(2)}`));
  if (opts.file) {
    const fileMem = await AgentMemory.open(opts.file);
    for (const r of mem.all()) fileMem.remember(r.text, { kind: r.kind, tags: r.tags, importance: r.importance });
    await fileMem.save();
    log(c.dim(`saved to ${opts.file}`));
  }
  return mem;
}

function counter(): () => string {
  let i = 0;
  return () => `mem_${String(++i).padStart(3, "0")}`;
}
