const on = (): boolean =>
  !["", "0", undefined].includes(process.env.FORCE_COLOR) || (Boolean(process.stdout.isTTY) && !process.env.NO_COLOR);
const wrap = (code: string) => (s: string | number) => (on() ? `\x1b[${code}m${s}\x1b[0m` : String(s));

export const c = {
  bold: wrap("1"),
  dim: wrap("90"),
  green: wrap("32"),
  yellow: wrap("33"),
  cyan: wrap("36"),
  magenta: wrap("35"),
  red: wrap("31"),
};

export function bar(value: number, width = 10): string {
  const n = Math.round(Math.max(0, Math.min(1, value)) * width);
  const color = value >= 0.66 ? c.green : value >= 0.33 ? c.yellow : c.red;
  return color("█".repeat(n)) + c.dim("░".repeat(width - n));
}

export function trunc(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}
