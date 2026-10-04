import { tokenize } from "./tokenize.js";

export type SparseVector = Map<string, number>;

/**
 * Incremental TF-IDF index. Term frequencies are stored per document; IDF is computed
 * on demand from document frequencies, so adding or removing a document is O(terms).
 */
export class TfIdfIndex {
  private tf = new Map<string, Map<string, number>>();
  private df = new Map<string, number>();

  get size(): number {
    return this.tf.size;
  }

  has(id: string): boolean {
    return this.tf.has(id);
  }

  add(id: string, text: string): void {
    if (this.tf.has(id)) this.remove(id);
    const counts = new Map<string, number>();
    for (const t of tokenize(text)) counts.set(t, (counts.get(t) ?? 0) + 1);
    this.tf.set(id, counts);
    for (const term of counts.keys()) this.df.set(term, (this.df.get(term) ?? 0) + 1);
  }

  remove(id: string): void {
    const counts = this.tf.get(id);
    if (!counts) return;
    for (const term of counts.keys()) {
      const n = (this.df.get(term) ?? 1) - 1;
      if (n <= 0) this.df.delete(term);
      else this.df.set(term, n);
    }
    this.tf.delete(id);
  }

  /** Smoothed IDF: ln((1 + N) / (1 + df)) + 1. */
  idf(term: string): number {
    return Math.log((1 + this.size) / (1 + (this.df.get(term) ?? 0))) + 1;
  }

  private weigh(counts: Map<string, number>): SparseVector {
    const v: SparseVector = new Map();
    let norm = 0;
    for (const [term, c] of counts) {
      const w = (1 + Math.log(c)) * this.idf(term); // sublinear tf
      v.set(term, w);
      norm += w * w;
    }
    norm = Math.sqrt(norm) || 1;
    for (const [term, w] of v) v.set(term, w / norm);
    return v;
  }

  vector(id: string): SparseVector {
    const counts = this.tf.get(id);
    return counts ? this.weigh(counts) : new Map();
  }

  queryVector(text: string): SparseVector {
    const counts = new Map<string, number>();
    for (const t of tokenize(text)) counts.set(t, (counts.get(t) ?? 0) + 1);
    return this.weigh(counts);
  }

  /** Cosine similarity between a query and every document (only non-zero scores returned). */
  search(text: string, ids?: Iterable<string>): Map<string, number> {
    const q = this.queryVector(text);
    const out = new Map<string, number>();
    if (q.size === 0) return out;
    for (const id of ids ?? this.tf.keys()) {
      const s = cosine(q, this.vector(id));
      if (s > 0) out.set(id, s);
    }
    return out;
  }

  /** Top terms of a document by TF-IDF weight, handy for labelling clusters. */
  topTerms(ids: string[], n = 5): string[] {
    const agg = new Map<string, number>();
    for (const id of ids) for (const [t, w] of this.vector(id)) agg.set(t, (agg.get(t) ?? 0) + w);
    return [...agg.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, n).map(([t]) => t);
  }
}

export function cosine(a: SparseVector, b: SparseVector): number {
  const [small, large] = a.size <= b.size ? [a, b] : [b, a];
  let dot = 0;
  for (const [t, w] of small) {
    const o = large.get(t);
    if (o !== undefined) dot += w * o;
  }
  return dot; // vectors are L2-normalised
}
