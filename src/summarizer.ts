/** Turns a cluster of related episodic memories into one semantic memory. */
export interface Summarizer {
  readonly name: string;
  summarize(texts: string[], themes: string[]): Promise<string>;
}

/**
 * Offline, deterministic: keeps the distinct facts of a cluster, most representative first
 * (highest overlap with the cluster's top TF-IDF terms), within a character budget.
 */
export class ExtractiveSummarizer implements Summarizer {
  readonly name = "extractive";

  constructor(private maxChars = 320) {}

  async summarize(texts: string[], themes: string[]): Promise<string> {
    const scored = texts.map((t, i) => {
      const words = t.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
      const hits = words.filter((w) => themes.some((th) => w.startsWith(th))).length;
      return { t: t.replace(/\s+/g, " ").trim().replace(/[.!]+$/, ""), score: hits / Math.sqrt(words.length || 1), i };
    });
    scored.sort((a, b) => b.score - a.score || a.i - b.i);
    const parts: string[] = [];
    let used = 0;
    for (const { t } of scored) {
      if (parts.some((p) => p.toLowerCase() === t.toLowerCase())) continue;
      if (parts.length && used + t.length + 3 > this.maxChars) break;
      parts.push(t);
      used += t.length + 3;
    }
    return parts.join("; ") + ".";
  }
}

/** Any OpenAI-compatible chat endpoint. Falls back to extractive summarisation on any error. */
export class OpenAICompatibleSummarizer implements Summarizer {
  readonly name = "openai-compatible";
  private fallback = new ExtractiveSummarizer();

  constructor(
    private apiKey: string,
    private baseUrl = "https://api.openai.com/v1",
    private model = "gpt-4o-mini",
    private timeoutMs = 20_000,
  ) {}

  async summarize(texts: string[], themes: string[]): Promise<string> {
    try {
      const res = await fetch(`${this.baseUrl.replace(/\/$/, "")}/chat/completions`, {
        method: "POST",
        headers: { Authorization: `Bearer ${this.apiKey}`, "Content-Type": "application/json" },
        signal: AbortSignal.timeout(this.timeoutMs),
        body: JSON.stringify({
          model: this.model,
          temperature: 0.2,
          messages: [
            {
              role: "system",
              content:
                "You consolidate an AI agent's episodic memories into one durable fact. Reply with a single sentence " +
                "in third person, no preamble. Only state what the observations support.",
            },
            { role: "user", content: texts.map((t, i) => `${i + 1}. ${t}`).join("\n") },
          ],
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as { choices?: { message?: { content?: string } }[] };
      const text = data.choices?.[0]?.message?.content?.trim();
      if (!text) throw new Error("empty completion");
      return text;
    } catch {
      return this.fallback.summarize(texts, themes);
    }
  }
}

export function summarizerFromEnv(env: NodeJS.ProcessEnv = process.env): Summarizer {
  const key = env.OPENAI_API_KEY?.trim();
  if (!key || env.AGENT_MEMORY_OFFLINE === "1") return new ExtractiveSummarizer();
  return new OpenAICompatibleSummarizer(key, env.OPENAI_BASE_URL ?? undefined, env.AGENT_MEMORY_MODEL ?? undefined);
}
