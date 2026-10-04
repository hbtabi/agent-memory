import type { MemoryKind, MemoryRecord } from "./types.js";

export const HOUR = 3_600_000;
export const DAY = 24 * HOUR;

export interface ScoringConfig {
  /** Half-life of recency in ms, per memory kind. */
  halfLife: Record<MemoryKind, number>;
  weights: { relevance: number; recency: number; importance: number };
  /** Each recall adds this much to effective importance (log-scaled), capped at 1. */
  accessBoost: number;
}

export const DEFAULT_SCORING: ScoringConfig = {
  halfLife: { episodic: 3 * DAY, semantic: 60 * DAY },
  weights: { relevance: 0.6, recency: 0.2, importance: 0.2 },
  accessBoost: 0.08,
};

/** Exponential decay: 1.0 now, 0.5 after one half-life. Uses the later of creation and last access. */
export function recency(m: MemoryRecord, now: number, cfg: ScoringConfig = DEFAULT_SCORING): number {
  const age = Math.max(0, now - Math.max(m.createdAt, m.lastAccessedAt));
  return Math.pow(0.5, age / cfg.halfLife[m.kind]);
}

export function effectiveImportance(m: MemoryRecord, cfg: ScoringConfig = DEFAULT_SCORING): number {
  return Math.min(1, m.importance + cfg.accessBoost * Math.log1p(m.accessCount));
}

/** Memory "strength" used for pruning: important or recently used memories survive. */
export function strength(m: MemoryRecord, now: number, cfg: ScoringConfig = DEFAULT_SCORING): number {
  return 0.5 * effectiveImportance(m, cfg) + 0.5 * recency(m, now, cfg);
}

const IMPORTANT_CUES: [RegExp, number][] = [
  [/\b(always|never|must|important|remember|don't forget|deadline|allerg\w*|password|budget)\b/i, 0.25],
  [/\b(i|we) (prefer|like|love|hate|dislike|want|need|am|work|live|run|own)\b/i, 0.2],
  [/\b(prefers|likes|loves|hates|dislikes|wants|needs|runs|owns|works at|lives in|is allergic)\b/i, 0.2],
  [/\b(my|our) (name|birthday|wife|husband|partner|son|daughter|company|business|goal)\b/i, 0.2],
  [/\b\d{1,2}(st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/i, 0.1],
  [/[£$€]\s?\d|\b\d+(\.\d+)?\s?(k|m|%)\b/i, 0.1],
  [/!/, 0.05],
];
const TRIVIAL = new Set(["ok", "okay", "thanks", "thank", "you", "cool", "nice", "great", "lol", "hi", "hello", "hey", "yes", "no", "sure", "cheers", "bye"]);

/** Heuristic 0..1 importance when the caller doesn't provide one (no LLM needed). */
export function estimateImportance(text: string): number {
  const words = text.toLowerCase().split(/[^a-z]+/).filter(Boolean);
  if (words.length > 0 && words.every((w) => TRIVIAL.has(w))) return 0.05;
  let score = 0.3;
  for (const [re, w] of IMPORTANT_CUES) if (re.test(text)) score += w;
  return Math.round(Math.min(1, score) * 100) / 100;
}
