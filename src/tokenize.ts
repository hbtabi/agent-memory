const STOPWORDS = new Set(
  ("a an and are as at be been but by can could did do does for from had has have he her his i if in into is it its " +
    "just me my of on or our she so than that the their them then there they this to too up us was we were what when " +
    "where which who will with would you your yours about also any some very really").split(" "),
);

/** Tiny suffix-stripping stemmer: good enough to match "prefers/preferred/preferring" → "prefer". */
export function stem(word: string): string {
  if (word.length <= 4) return word;
  for (const [suffix, min] of [["ingly", 4], ["edly", 4], ["ing", 4], ["ies", 3], ["ied", 3], ["ed", 4], ["ly", 4], ["es", 4], ["s", 3]] as const) {
    if (word.endsWith(suffix) && word.length - suffix.length >= min) {
      if (suffix === "es" && !/(s|x|z|ch|sh)es$/.test(word)) continue; // "whites" -> "white", "boxes" -> "box"
      if (suffix === "s" && /(ss|us|is)$/.test(word)) continue; // "glass", "status", "analysis"
      let base = word.slice(0, -suffix.length);
      if (suffix === "ies" || suffix === "ied") base += "y";
      if (/(.)\1$/.test(base) && !/(ll|ss|ff|zz)$/.test(base)) base = base.slice(0, -1); // "stopped" -> "stop"
      return base;
    }
  }
  return word;
}

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 1 && !STOPWORDS.has(t))
    .map(stem);
}
