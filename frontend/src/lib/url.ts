/**
 * The address grammar, after tau2-loop's: one id per thing, the path names the subject, the query
 * string holds the lens (lens, view, vs, rule, task, status, q, source, symptom, offset), and a
 * detail opens inside its list. Telecom task ids carry `[`, `]` and `|`, so every id is a single
 * escaped path segment. Every link in the viewer is built here; routes.test.tsx drives them.
 */
import type { Lens, RowStatus } from './api';

type Query = Record<string, string | number | null | undefined>;

export function withQuery(path: string, q: Query = {}): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(q)) if (v != null && v !== '') sp.set(k, String(v));
  const s = sp.toString();
  return s ? `${path}?${s}` : path;
}

const seg = encodeURIComponent;

export const overviewPath = () => '/';
export const dataPath = () => '/data';
export const dataVersionPath = (v: string, view?: 'coverage' | null) => withQuery(`/data/${seg(v)}`, { view });

export type RowsLens = {
  status?: RowStatus | 'all' | null;
  rule?: string | null;
  task?: string | null;
  symptom?: string | null;
  source?: string | null;
  q?: string | null;
  offset?: number | null;
};
/** A rows lens with its defaults dropped (status all, offset 0), so one view has one address. */
const lensQuery = (q: RowsLens): Query => ({ ...q, status: q.status === 'all' ? null : q.status, offset: q.offset || null });
export const rowsPath = (v: string, q: RowsLens = {}) => withQuery(`/data/${seg(v)}/rows`, lensQuery(q));
/** A row opens inside its list, so it carries the list's lens; `lens` is the row's own (chat by default). */
export const rowPath = (v: string, simId: string, lens?: Lens | null, keep: RowsLens = {}) =>
  withQuery(`/data/${seg(v)}/rows/${seg(simId)}`, { lens: lens === 'chat' ? null : lens, ...lensQuery(keep) });
export const reviewPath = (v: string) => `/data/${seg(v)}/review`;

export const trainPath = () => '/train';
export const trainRunPath = (run: string, vs?: string | null) => withQuery(`/train/${seg(run)}`, { vs });
export const evalPath = () => '/eval';
export const evalRunPath = (run: string, vs?: string | null) => withQuery(`/eval/${seg(run)}`, { vs });
export const evalConversationPath = (run: string, task: string, trial: number) =>
  `/eval/${seg(run)}/${seg(task)}/t${trial}`;
export const methodPath = () => '/method';

export const LENSES: Lens[] = ['chat', 'tokens', 'raw'];
export const isLens = (v: string | null): v is Lens => v === 'chat' || v === 'tokens' || v === 'raw';
