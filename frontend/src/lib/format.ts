/** Number and text formatting. Every percentage on a page also carries its count (plan §1). */

export const DASH = '—';

export const int = (n: number | null | undefined) => (n == null ? DASH : Math.round(n).toLocaleString('en-US'));

export const pct = (x: number | null | undefined, digits = 1) => (x == null ? DASH : `${(x * 100).toFixed(digits)}%`);

/** "24 / 40 (60.0%)": a count with its denominator, then the share. */
export const countOf = (n: number | null | undefined, d: number | null | undefined) =>
  n == null || !d ? DASH : `${n} / ${d} (${pct(n / d)})`;

export const num = (x: number | null | undefined, digits = 3) => (x == null || Number.isNaN(x) ? DASH : x.toFixed(digits));

export const sci = (x: number | null | undefined) =>
  x == null ? DASH : Math.abs(x) >= 0.01 || x === 0 ? x.toPrecision(3) : x.toExponential(1);

export const pValue = (p: number | null | undefined) => (p == null ? DASH : p < 0.001 ? '< 0.001' : p.toFixed(3));

export const shortSha = (s: string | null | undefined, n = 12) => (s ? `${s.slice(0, n)}…` : DASH);

export function duration(s: number | null | undefined): string {
  if (s == null) return DASH;
  if (s < 90) return `${Math.round(s)} s`;
  if (s < 5400) return `${Math.round(s / 60)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

export const usd = (x: number | null | undefined) => (x == null ? DASH : `$${x.toFixed(2)}`);

export const when = (iso: string | null | undefined) => (iso ? iso.replace('T', ' ').replace(/(\.\d+)?(\+00:00|Z)$/, ' UTC') : DASH);

/** "[mms_issue]a|b[PERSONA:Easy]" → its parts, for a readable task cell. */
export function taskParts(id: string): { symptom: string; issues: string[]; persona: string } | null {
  const m = /^\[(\w+)\]([\w|]*)\[PERSONA:(\w+)\]$/.exec(id);
  if (!m) return null;
  return { symptom: m[1], issues: m[2].split('|').filter(Boolean), persona: m[3] };
}

export const SYMPTOM: Record<string, string> = {
  mms_issue: 'MMS not sending',
  mobile_data_issue: 'mobile data slow or off',
  service_issue: 'no service',
};

/** Max of a list, ignoring nulls; 0 when empty. */
export const maxOf = (xs: (number | null | undefined)[]) => xs.reduce<number>((m, x) => (x != null && x > m ? x : m), 0);

/** A tidy upper bound for an axis: 1, 2 or 5 × 10^k at or above x. */
export function niceMax(x: number): number {
  if (x <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(x));
  for (const m of [1, 2, 5, 10]) if (m * p >= x) return m * p;
  return 10 * p;
}
