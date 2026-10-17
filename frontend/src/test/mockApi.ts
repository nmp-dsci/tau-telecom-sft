/**
 * A fake `serving/app.py`: answers the viewer's fetches from api.json, the real API's answers on
 * the offline fixtures (tests/fixtures/make_api_fixtures.py). A URL is matched exactly, then with
 * its default query values dropped, then by path alone; anything else is a 404.
 */
import { vi } from 'vitest';
import fixture from './api.json';

export const ids = fixture.ids;
const answers = fixture.answers as Record<string, unknown>;

export type Reply = unknown | { status: number; body: unknown };
export type Overrides = Record<string, Reply | ((url: URL, body: unknown) => Reply)>;

const DEFAULTS: Record<string, string> = { status: 'all', offset: '0', limit: '50' };

const isStatus = (r: unknown): r is { status: number; body: unknown } =>
  typeof r === 'object' && r !== null && 'status' in r && 'body' in r && Object.keys(r).length === 2;

function answer(url: URL): unknown {
  const path = decodeURIComponent(url.pathname);
  const exact = `${path}${url.search}`;
  if (exact in answers) return answers[exact];
  const sp = new URLSearchParams();
  url.searchParams.forEach((v, k) => {
    if (DEFAULTS[k] !== v) sp.set(k, v);
  });
  const trimmed = `${path}${sp.toString() ? `?${sp}` : ''}`;
  if (trimmed in answers) return answers[trimmed];
  const loose = Object.keys(answers).find((k) => k.split('?')[0] === path);
  return loose ? answers[loose] : undefined;
}

/** Stub global fetch. Overrides are keyed `"GET /api/train"` (path only, no query string). */
export function mockApi(overrides: Overrides = {}) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://viewer.test');
    const method = (init?.method ?? 'GET').toUpperCase();
    const key = `${method} ${decodeURIComponent(url.pathname)}`;
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    let reply: Reply;
    if (key in overrides) {
      const o = overrides[key];
      reply = typeof o === 'function' ? (o as (u: URL, b: unknown) => Reply)(url, body) : o;
    } else if (method === 'GET') {
      const d = answer(url);
      reply = d === undefined ? { status: 404, body: { detail: `Not found: ${url.pathname}` } } : d;
    } else {
      reply = { status: 405, body: { detail: 'not mocked' } };
    }
    const [status, payload] = isStatus(reply) ? [reply.status, reply.body] : [200, reply];
    return { ok: status >= 200 && status < 300, status, json: async () => structuredClone(payload) } as unknown as Response;
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

export function fetched(fetchMock: ReturnType<typeof mockApi>): string[] {
  return fetchMock.mock.calls.map(([u]) => String(u));
}
