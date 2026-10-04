import { randomUUID } from "node:crypto";
import { DEFAULT_SCORING, effectiveImportance, estimateImportance, recency, strength, type ScoringConfig } from "./scoring.js";
import { InMemoryStore, JsonFileStore, type MemoryStore } from "./store.js";
import { ExtractiveSummarizer, type Summarizer } from "./summarizer.js";
import { TfIdfIndex } from "./tfidf.js";
import type { MemoryKind, MemoryRecord, RecallOptions, RecallResult, RememberOptions } from "./types.js";

export interface AgentMemoryOptions {
  store?: MemoryStore;
  clock?: () => number;
  scoring?: Partial<ScoringConfig>;
  /** Cosine similarity above which a new memory reinforces an existing one instead of duplicating it. */
  dedupeThreshold?: number;
  summarizer?: Summarizer;
  idFactory?: () => string;
}

export interface PruneOptions {
  maxItems?: number;
  minStrength?: number;
  kind?: MemoryKind;
}

export interface ConsolidateOptions {
  similarity?: number;
  minClusterSize?: number;
  /** Only consolidate episodes at least this old (ms). */
  minAge?: number;
}

export interface ConsolidationResult {
  created: MemoryRecord[];
  clusters: string[][];
}

const CONSOLIDATED = "consolidated";

export class AgentMemory {
  private records = new Map<string, MemoryRecord>();
  private index = new TfIdfIndex();
  readonly store: MemoryStore;
  readonly scoring: ScoringConfig;
  private clock: () => number;
  private dedupeThreshold: number;
  private summarizer: Summarizer;
  private idFactory: () => string;

  constructor(opts: AgentMemoryOptions = {}) {
    this.store = opts.store ?? new InMemoryStore();
    this.clock = opts.clock ?? Date.now;
    this.scoring = {
      ...DEFAULT_SCORING,
      ...opts.scoring,
      halfLife: { ...DEFAULT_SCORING.halfLife, ...opts.scoring?.halfLife },
      weights: { ...DEFAULT_SCORING.weights, ...opts.scoring?.weights },
    };
    this.dedupeThreshold = opts.dedupeThreshold ?? 0.9;
    this.summarizer = opts.summarizer ?? new ExtractiveSummarizer();
    this.idFactory = opts.idFactory ?? (() => `mem_${randomUUID().slice(0, 8)}`);
  }

  /** Open (or create) a JSON-file-backed memory. */
  static async open(path: string, opts: Omit<AgentMemoryOptions, "store"> = {}): Promise<AgentMemory> {
    const mem = new AgentMemory({ ...opts, store: new JsonFileStore(path) });
    await mem.load();
    return mem;
  }

  async load(): Promise<void> {
    this.records.clear();
    this.index = new TfIdfIndex();
    for (const r of await this.store.load()) this.insert(r);
  }

  async save(): Promise<void> {
    await this.store.save(this.all());
  }

  get size(): number {
    return this.records.size;
  }

  now(): number {
    return this.clock();
  }

  private insert(r: MemoryRecord): void {
    this.records.set(r.id, r);
    this.index.add(r.id, `${r.text} ${r.tags.join(" ")}`);
  }

  /** Store a memory. Near-duplicates of the same kind reinforce the existing memory instead. */
  remember(text: string, opts: RememberOptions = {}): MemoryRecord {
    const clean = text.replace(/\s+/g, " ").trim();
    if (!clean) throw new Error("cannot remember empty text");
    const kind = opts.kind ?? "episodic";
    const now = this.clock();
    const importance = clamp(opts.importance ?? estimateImportance(clean));
    const sameKind = [...this.records.values()].filter((r) => r.kind === kind).map((r) => r.id);
    let bestId: string | undefined;
    let best = 0;
    for (const [id, s] of this.index.search(clean, sameKind)) if (s > best) [bestId, best] = [id, s];
    if (bestId && best >= this.dedupeThreshold) {
      const existing = this.records.get(bestId)!;
      existing.accessCount += 1;
      existing.lastAccessedAt = now;
      existing.importance = Math.max(existing.importance, importance);
      existing.tags = [...new Set([...existing.tags, ...(opts.tags ?? [])])];
      if (opts.pinned) existing.pinned = true;
      return existing;
    }
    const record: MemoryRecord = {
      id: this.idFactory(),
      kind,
      text: clean,
      tags: [...new Set((opts.tags ?? []).map((t) => t.toLowerCase()))],
      importance,
      createdAt: now,
      lastAccessedAt: now,
      accessCount: 0,
      ...(opts.source ? { source: opts.source } : {}),
      ...(opts.pinned ? { pinned: true } : {}),
      ...(opts.metadata ? { metadata: opts.metadata } : {}),
    };
    this.insert(record);
    return record;
  }

  get(id: string): MemoryRecord | undefined {
    return this.records.get(id);
  }

  all(kind?: MemoryKind): MemoryRecord[] {
    const list = [...this.records.values()];
    return (kind ? list.filter((r) => r.kind === kind) : list).sort((a, b) => a.createdAt - b.createdAt);
  }

  forget(id: string): boolean {
    this.index.remove(id);
    return this.records.delete(id);
  }

  /** Rank memories by weighted relevance (TF-IDF cosine), recency (exponential decay) and importance. */
  recall(query: string, opts: RecallOptions = {}): RecallResult[] {
    const { k = 5, kind, tags, minScore = 0, minRelevance = 0.08, collapse = true, reinforce = true } = opts;
    const now = this.clock();
    const w = this.scoring.weights;
    const wsum = w.relevance + w.recency + w.importance || 1;
    const candidates = [...this.records.values()].filter(
      (r) => (!kind || r.kind === kind) && (!tags?.length || tags.every((t) => r.tags.includes(t.toLowerCase()))),
    );
    const rel = this.index.search(query, candidates.map((r) => r.id));
    const results: RecallResult[] = [];
    for (const r of candidates) {
      const relevance = rel.get(r.id) ?? 0;
      if (relevance < minRelevance || relevance === 0) continue; // must actually be about the query
      const rec = recency(r, now, this.scoring);
      const imp = effectiveImportance(r, this.scoring);
      const total = (w.relevance * relevance + w.recency * rec + w.importance * imp) / wsum;
      if (total >= minScore) results.push({ memory: r, score: { relevance, recency: rec, importance: imp, total } });
    }
    results.sort((a, b) => b.score.total - a.score.total || b.memory.createdAt - a.memory.createdAt);
    let ranked = results;
    if (collapse) {
      const shown = new Set<string>();
      for (const { memory } of results) if (memory.kind === "semantic") memory.derivedFrom?.forEach((id) => shown.add(id));
      ranked = results.filter(({ memory }) => !(memory.kind === "episodic" && shown.has(memory.id)));
    }
    const top = ranked.slice(0, k);
    if (reinforce) {
      for (const { memory } of top) {
        memory.accessCount += 1;
        memory.lastAccessedAt = now;
      }
    }
    return top;
  }

  /** Format pinned + top-ranked memories as a compact block for an LLM prompt, within a character budget. */
  buildContext(query: string, opts: RecallOptions & { maxChars?: number } = {}): string {
    const { maxChars = 1200, ...rest } = opts;
    const now = this.clock();
    const pinned = this.all().filter((m) => m.pinned);
    const pinnedIds = new Set(pinned.map((m) => m.id));
    const relevant = this.recall(query, rest).map((r) => r.memory).filter((m) => !pinnedIds.has(m.id));
    const lines: string[] = [];
    let used = 0;
    const push = (line: string): boolean => {
      if (used + line.length + 1 > maxChars) return false;
      lines.push(line);
      used += line.length + 1;
      return true;
    };
    if (pinned.length && push("Always remember:")) for (const m of pinned) if (!push(`- ${m.text}`)) break;
    if (relevant.length && push("Relevant memories:"))
      for (const m of relevant) if (!push(`- [${m.kind}] ${m.text} (${ago(now - m.createdAt)})`)) break;
    return lines.join("\n");
  }

  /** Remove weak memories (low importance and long unused). Returns removed ids. */
  prune(opts: PruneOptions = {}): string[] {
    const now = this.clock();
    const pool = this.all(opts.kind)
      .filter((r) => !r.pinned)
      .map((r) => ({ r, s: strength(r, now, this.scoring) }));
    const removed = new Set<string>();
    if (opts.minStrength !== undefined) for (const { r, s } of pool) if (s < opts.minStrength) removed.add(r.id);
    if (opts.maxItems !== undefined) {
      const survivors = pool.filter(({ r }) => !removed.has(r.id)).sort((a, b) => b.s - a.s);
      for (const { r } of survivors.slice(Math.max(0, opts.maxItems))) removed.add(r.id);
    }
    for (const id of removed) this.forget(id);
    return [...removed];
  }

  /**
   * Cluster related episodic memories (greedy single-link on TF-IDF cosine) and distil each
   * cluster into a semantic memory. Source episodes are tagged and down-weighted, not deleted.
   */
  async consolidate(opts: ConsolidateOptions = {}): Promise<ConsolidationResult> {
    const { similarity = 0.3, minClusterSize = 2, minAge = 0 } = opts;
    const now = this.clock();
    const episodes = this.all("episodic").filter((r) => !r.tags.includes(CONSOLIDATED) && now - r.createdAt >= minAge);
    const vectors = new Map(episodes.map((e) => [e.id, this.index.vector(e.id)]));
    const unassigned = new Set(episodes.map((e) => e.id));
    const clusters: string[][] = [];
    for (const seed of episodes) {
      if (!unassigned.has(seed.id)) continue;
      unassigned.delete(seed.id);
      const cluster = [seed.id];
      for (let i = 0; i < cluster.length; i++) {
        const v = vectors.get(cluster[i]!)!;
        for (const other of [...unassigned]) {
          if (dot(v, vectors.get(other)!) >= similarity) {
            unassigned.delete(other);
            cluster.push(other);
          }
        }
      }
      if (cluster.length >= minClusterSize) clusters.push(cluster);
    }
    const created: MemoryRecord[] = [];
    for (const ids of clusters) {
      const members = ids.map((id) => this.records.get(id)!);
      const themes = this.index.topTerms(ids, 5);
      const text = await this.summarizer.summarize(members.map((m) => m.text), themes);
      const importance = clamp(Math.max(...members.map((m) => m.importance)) + 0.1 * Math.log2(members.length));
      const tags = [...new Set(members.flatMap((m) => m.tags))];
      const fact = this.remember(text, { kind: "semantic", importance, tags, source: "consolidation" });
      fact.derivedFrom = [...new Set([...(fact.derivedFrom ?? []), ...ids])];
      for (const m of members) {
        m.tags = [...new Set([...m.tags, CONSOLIDATED])];
        m.importance = Math.round(m.importance * 0.5 * 100) / 100;
      }
      created.push(fact);
    }
    return { created, clusters };
  }

  stats(): { total: number; episodic: number; semantic: number; avgImportance: number; avgStrength: number } {
    const all = this.all();
    const now = this.clock();
    const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
    return {
      total: all.length,
      episodic: all.filter((r) => r.kind === "episodic").length,
      semantic: all.filter((r) => r.kind === "semantic").length,
      avgImportance: avg(all.map((r) => r.importance)),
      avgStrength: avg(all.map((r) => strength(r, now, this.scoring))),
    };
  }
}

function clamp(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function dot(a: Map<string, number>, b: Map<string, number>): number {
  let s = 0;
  for (const [t, w] of a) s += w * (b.get(t) ?? 0);
  return s;
}

export function ago(ms: number): string {
  const m = Math.round(ms / 60_000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 48) return `${h}h ago`;
  return `${Math.round(h / 24)}d ago`;
}
