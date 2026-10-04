/** Episodic = things that happened ("user asked about X on Tuesday"). Semantic = durable facts ("user is vegetarian"). */
export type MemoryKind = "episodic" | "semantic";

export interface MemoryRecord {
  id: string;
  kind: MemoryKind;
  text: string;
  tags: string[];
  /** 0..1, how much this memory matters regardless of when it happened. */
  importance: number;
  createdAt: number;
  lastAccessedAt: number;
  accessCount: number;
  source?: string;
  /** Pinned memories are never pruned and always included in buildContext(). */
  pinned?: boolean;
  /** For semantic memories produced by consolidation: the episodic ids they summarise. */
  derivedFrom?: string[];
  metadata?: Record<string, unknown>;
}

export interface RememberOptions {
  kind?: MemoryKind;
  tags?: string[];
  importance?: number;
  source?: string;
  pinned?: boolean;
  metadata?: Record<string, unknown>;
}

export interface RecallOptions {
  k?: number;
  kind?: MemoryKind;
  tags?: string[];
  minScore?: number;
  /** Ignore memories whose TF-IDF relevance is below this (default 0.08). */
  minRelevance?: number;
  /** Hide episodes whose consolidated semantic fact is already in the results (default true). */
  collapse?: boolean;
  /** Update access stats (reinforcement). Default true. */
  reinforce?: boolean;
}

export interface ScoreBreakdown {
  relevance: number;
  recency: number;
  importance: number;
  total: number;
}

export interface RecallResult {
  memory: MemoryRecord;
  score: ScoreBreakdown;
}

export interface MemoryFile {
  version: 1;
  memories: MemoryRecord[];
}
