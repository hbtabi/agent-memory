#!/usr/bin/env node
import { realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { AgentMemory, ago } from "./memory.js";
import { runDemo } from "./demo.js";
import { strength } from "./scoring.js";
import { summarizerFromEnv } from "./summarizer.js";
import type { MemoryKind } from "./types.js";
import { bar, c, trunc } from "./ui.js";

const HELP = `agent-memory: long-term memory for AI agents

Usage: agent-memory <command> [options]

Commands:
  add <text>          store a memory   [--kind episodic|semantic] [--tags a,b] [--importance 0..1]
  recall <query>      ranked retrieval [-k 5] [--kind ...] [--tags a,b] [--json]
  context <query>     prompt-ready context block [-k 5] [--max-chars 1200]
  list                list memories    [--kind ...] [--json]
  forget <id>         delete a memory
  consolidate         turn related episodes into semantic facts [--similarity 0.3] [--min-cluster 2]
  prune               drop weak memories [--max N] [--min-strength 0.2]
  stats               summary statistics
  demo                run the offline demo (fictional café owner, simulated 30 days)

Global: --file <path> (default: ./agent.memory.json)`;

interface Parsed {
  positional: string[];
  flags: Record<string, string | boolean>;
}

export function parseArgs(argv: string[]): Parsed {
  const positional: string[] = [];
  const flags: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a.startsWith("-")) {
      const key = a.replace(/^-+/, "");
      const next = argv[i + 1];
      if (next !== undefined && !next.startsWith("-")) {
        flags[key] = next;
        i++;
      } else flags[key] = true;
    } else positional.push(a);
  }
  return { positional, flags };
}

function kindFlag(v: string | boolean | undefined): MemoryKind | undefined {
  if (v === undefined) return undefined;
  if (v === "episodic" || v === "semantic") return v;
  throw new Error("--kind must be episodic or semantic");
}

function num(v: string | boolean | undefined, fallback: number): number {
  if (v === undefined || v === true) return fallback;
  const n = Number(v);
  if (Number.isNaN(n)) throw new Error(`expected a number, got '${v}'`);
  return n;
}

const tagsFlag = (v: string | boolean | undefined) => (typeof v === "string" ? v.split(",").map((t) => t.trim()).filter(Boolean) : undefined);

export async function main(argv: string[]): Promise<number> {
  const { positional, flags } = parseArgs(argv);
  const [cmd, ...rest] = positional;
  if (!cmd || flags.help || flags.h) {
    console.log(HELP);
    return cmd || flags.help || flags.h ? 0 : 1;
  }
  const file = typeof flags.file === "string" ? flags.file : "agent.memory.json";
  try {
    if (cmd === "demo") {
      await runDemo({ file: typeof flags.file === "string" ? flags.file : undefined });
      return 0;
    }
    const mem = await AgentMemory.open(file, { summarizer: summarizerFromEnv() });
    const text = rest.join(" ");
    switch (cmd) {
      case "add": {
        if (!text) throw new Error("add needs some text");
        const imp = flags.importance !== undefined ? num(flags.importance, 0.5) : undefined;
        const m = mem.remember(text, { kind: kindFlag(flags.kind) ?? "episodic", tags: tagsFlag(flags.tags), importance: imp });
        await mem.save();
        console.log(`${c.green("✓")} ${m.id} [${m.kind}] importance ${m.importance.toFixed(2)}${m.accessCount ? c.dim(" (reinforced existing memory)") : ""}`);
        return 0;
      }
      case "recall": {
        if (!text) throw new Error("recall needs a query");
        const res = mem.recall(text, { k: num(flags.k, 5), kind: kindFlag(flags.kind), tags: tagsFlag(flags.tags) });
        await mem.save();
        if (flags.json) console.log(JSON.stringify(res, null, 2));
        else if (!res.length) console.log(c.dim("no relevant memories"));
        else
          for (const { memory, score } of res)
            console.log(`${bar(score.total)} ${score.total.toFixed(2)}  ${c.dim(memory.id)} ${memory.kind === "semantic" ? c.magenta("sem") : "epi"}  ${trunc(memory.text, 80)}`);
        return 0;
      }
      case "context": {
        const out = mem.buildContext(text, { k: num(flags.k, 5), maxChars: num(flags["max-chars"], 1200) });
        await mem.save();
        console.log(out || c.dim("no relevant memories"));
        return 0;
      }
      case "list": {
        const list = mem.all(kindFlag(flags.kind));
        if (flags.json) console.log(JSON.stringify(list, null, 2));
        else
          for (const m of list)
            console.log(`${c.dim(m.id)} ${m.kind === "semantic" ? c.magenta("sem") : "epi"} ${bar(strength(m, mem.now()), 6)} ${c.dim(ago(mem.now() - m.createdAt).padEnd(9))} ${trunc(m.text, 80)}`);
        return 0;
      }
      case "forget": {
        const ok = mem.forget(rest[0] ?? "");
        await mem.save();
        console.log(ok ? `${c.green("✓")} forgot ${rest[0]}` : c.red(`no memory with id '${rest[0]}'`));
        return ok ? 0 : 1;
      }
      case "consolidate": {
        const { created } = await mem.consolidate({ similarity: num(flags.similarity, 0.3), minClusterSize: num(flags["min-cluster"], 2) });
        await mem.save();
        console.log(created.length ? created.map((f) => `${c.magenta("◆")} ${f.text}`).join("\n") : c.dim("nothing to consolidate"));
        return 0;
      }
      case "prune": {
        const removed = mem.prune({
          maxItems: flags.max !== undefined ? num(flags.max, Infinity) : undefined,
          minStrength: flags["min-strength"] !== undefined ? num(flags["min-strength"], 0) : undefined,
        });
        await mem.save();
        console.log(`removed ${removed.length} memories`);
        return 0;
      }
      case "stats": {
        const s = mem.stats();
        console.log(`${s.total} memories · ${s.episodic} episodic · ${s.semantic} semantic · avg importance ${s.avgImportance.toFixed(2)} · avg strength ${s.avgStrength.toFixed(2)}`);
        return 0;
      }
      default:
        console.error(`unknown command '${cmd}'\n\n${HELP}`);
        return 1;
    }
  } catch (err) {
    console.error(c.red(`error: ${(err as Error).message}`));
    return 2;
  }
}

function invokedDirectly(): boolean {
  try {
    return Boolean(process.argv[1]) && realpathSync(process.argv[1]!) === fileURLToPath(import.meta.url);
  } catch {
    return false;
  }
}
if (invokedDirectly()) main(process.argv.slice(2)).then((code) => process.exit(code));
