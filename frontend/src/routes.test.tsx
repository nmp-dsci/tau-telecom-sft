/**
 * Every route renders against the real API's answers on the offline fixtures (src/test/api.json):
 * the populated state, the empty state before a milestone has run, the `?vs=` and `?lens=`
 * lenses, the review's one write, and errors. Then the URL grammar: every address the viewer
 * builds lands on the page it names, telecom's `[ ] |` task ids included.
 */
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { RouterProvider, createMemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  dataPath,
  dataVersionPath,
  evalConversationPath,
  evalPath,
  evalRunPath,
  methodPath,
  reviewPath,
  rowPath,
  rowsPath,
  trainPath,
  trainRunPath,
} from './lib/url';
import { routes } from './routes';
import { type Overrides, fetched, ids, mockApi } from './test/mockApi';

afterEach(() => {
  vi.unstubAllGlobals();
});

function renderAt(url: string, overrides: Overrides = {}) {
  const fetchMock = mockApi(overrides);
  const router = createMemoryRouter(routes, { initialEntries: [url] });
  render(<RouterProvider router={router} />);
  return { router, fetchMock };
}

const h1 = (name: RegExp | string) => screen.findByRole('heading', { level: 1, name });
const EMPTY_RUNS: Overrides = {
  'GET /api/train': { runs: [] },
  'GET /api/eval': { runs: [] },
};

describe('shell', () => {
  it('shows the brand, the five tabs and the health pill', async () => {
    renderAt('/');
    expect(await screen.findByText(/^api ok · /)).toBeTruthy();
    const nav = screen.getByRole('navigation', { name: 'Pages' });
    expect(within(nav).getAllByRole('link').map((a) => a.textContent)).toEqual(['Overview', 'Data', 'Train', 'Eval', 'Method']);
    expect(screen.getByText(/mlflow ○ down/)).toBeTruthy();
  });

  it('says when the API is down, in words', async () => {
    renderAt('/', { 'GET /healthz': { status: 503, body: { detail: 'down' } } });
    expect(await screen.findByText('✕ api down')).toBeTruthy();
  });

  it('an unknown address says so', async () => {
    renderAt('/nowhere');
    expect(await h1('No page here')).toBeTruthy();
  });
});

describe('/ overview', () => {
  it('leads with the question, the results table and the chain', async () => {
    renderAt('/');
    expect(await h1(/beat its own base on tau2 telecom/)).toBeTruthy();
    const table = screen.getByRole('table');
    for (const r of ['R0', 'R1', 'R2', 'R3', 'R4', 'R5']) expect(within(table).getByText(r)).toBeTruthy();
    const chain = screen.getByRole('list', { name: /chain for fixture-smoke/ });
    expect(within(chain).getByRole('link', { name: 'fixture-smoke' }).getAttribute('href')).toBe(trainRunPath(ids.train));
    expect(within(chain).getByRole('link', { name: 'fixture tuned 0.6B' }).getAttribute('href')).toBe(evalRunPath(ids.tuned, ids.base));
    expect(screen.getByRole('heading', { name: /of 9 ticked/ })).toBeTruthy();
  });

  it('before any run, the chain says which milestone fills it', async () => {
    renderAt('/', { 'GET /api/overview': { ...(await import('./test/api.json')).answers['/api/overview'], chains: [], smoke: [] } });
    expect(await screen.findByText(/M2's make train-smoke/)).toBeTruthy();
  });
});

describe('/data', () => {
  it('lists the versions with their counts and review state', async () => {
    renderAt(dataPath());
    expect(await h1('Dataset versions')).toBeTruthy();
    const link = screen.getByRole('link', { name: ids.version });
    expect(link.getAttribute('href')).toBe(dataVersionPath(ids.version));
    expect(screen.getByText(/0 \/ \d+ read/)).toBeTruthy();
  });

  it('a version shows the funnel, every rule, and links each drop count to its rows', async () => {
    renderAt(dataVersionPath(ids.version));
    expect(await h1(/Dataset smoke: 8 clean conversations from 17/)).toBeTruthy();
    expect(screen.getByRole('list', { name: 'F1 to F10, in order' }).querySelectorAll('li')).toHaveLength(10);
    const f4 = screen.getAllByRole('link', { name: '1' }).find((a) => a.getAttribute('href')?.includes('rule=F4'));
    expect(f4?.getAttribute('href')).toBe(rowsPath(ids.version, { status: 'dropped', rule: 'F4' }));
    expect(screen.getByText(/counted here and never listed/)).toBeTruthy();
  });

  it('the coverage lens counts test tasks without naming them', async () => {
    const { fetchMock } = renderAt(dataVersionPath(ids.version, 'coverage'));
    expect(await screen.findByText(/root issues appear in the test tasks/)).toBeTruthy();
    expect(fetched(fetchMock)).toContain(`/api/data/${ids.version}/coverage`);
    expect(screen.getByText('test tasks are counted, never named', { exact: false })).toBeTruthy();
  });
});

describe('/data/:v/rows', () => {
  it('lists kept and dropped rows and filters by the query', async () => {
    const { fetchMock } = renderAt(rowsPath(ids.version, { status: 'dropped', rule: 'F4' }));
    expect(await screen.findByText(/dropped rows dropped by F4/)).toBeTruthy();
    expect(fetched(fetchMock).some((u) => u.includes('status=dropped') && u.includes('rule=F4'))).toBe(true);
    expect(screen.getAllByText(/F4: called/).length).toBeGreaterThan(0);
  });

  it('opens a row inside the list in the chat lens, with three lens links', async () => {
    renderAt(rowPath(ids.version, ids.kept));
    expect(await screen.findByRole('heading', { level: 2, name: /Row / })).toBeTruthy();
    const lenses = screen.getByRole('navigation', { name: 'Lens' });
    expect(within(lenses).getAllByRole('link').map((a) => a.getAttribute('href'))).toEqual([
      rowPath(ids.version, ids.kept),
      rowPath(ids.version, ids.kept, 'tokens'),
      rowPath(ids.version, ids.kept, 'raw'),
      rowsPath(ids.version),
    ]);
    expect(await screen.findAllByText('system prompt')).toBeTruthy();
    expect(screen.getAllByText('trained').length).toBeGreaterThan(0);
  });

  it('the raw lens greys what the agent never saw; a dropped row marks the turn that broke its rule', async () => {
    renderAt(rowPath(ids.version, ids.kept, 'raw'));
    expect((await screen.findAllByText("○ not in the agent's view")).length).toBeGreaterThan(0);
    vi.unstubAllGlobals();
    renderAt(rowPath(ids.version, ids.f4, 'raw'));
    expect((await screen.findAllByText('✕ broke the rule')).length).toBe(1);
  });

  it('the tokens lens marks trained text and counts it', async () => {
    renderAt(rowPath(ids.version, ids.kept, 'tokens'));
    expect(await screen.findByText(/tokens trained \(/)).toBeTruthy();
    expect(document.querySelectorAll('.tk-trained').length).toBeGreaterThan(0);
  });
});

describe('/data/:v/review', () => {
  it('shows the sample, refuses a sign-off before every verdict, and saves one verdict', async () => {
    const { fetchMock } = renderAt(reviewPath(ids.version), {
      'POST /api/data/smoke/review': (_u: URL, body: unknown) => ({ saved: { ...(body as object), at: 'now' } }),
    });
    expect(await h1(/Read what the model will learn/)).toBeTruthy();
    expect(screen.getByText('○ not signed off')).toBeTruthy();
    expect((screen.getByRole('button', { name: /Sign off on smoke/ }) as HTMLButtonElement).disabled).toBe(true);
    const group = screen.getAllByRole('group', { name: /Verdict on / })[0];
    fireEvent.click(within(group).getByRole('button', { name: '✓ good' }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([, init]) => (init as RequestInit | undefined)?.method === 'POST')).toBe(true));
    const post = fetchMock.mock.calls.find(([, init]) => (init as RequestInit | undefined)?.method === 'POST');
    expect(JSON.parse(String((post?.[1] as RequestInit).body))).toMatchObject({ kind: 'verdict', verdict: 'good' });
  });
});

describe('/train', () => {
  it('lists every job in a sortable table', async () => {
    renderAt(trainPath());
    expect(await h1('Every fine-tuning job')).toBeTruthy();
    expect(screen.getByRole('link', { name: 'fixture-smoke' }).getAttribute('href')).toBe(trainRunPath(ids.train));
    fireEvent.click(screen.getByRole('button', { name: /final loss/ }));
    expect(screen.getByRole('button', { name: /final loss ↓/ }).getAttribute('aria-pressed')).toBe('true');
  });

  it('before M2, says what fills it', async () => {
    renderAt(trainPath(), EMPTY_RUNS);
    expect(await screen.findByText(/M2's make train-smoke/)).toBeTruthy();
  });

  it('a job shows its curves, its config and the mask check', async () => {
    renderAt(trainRunPath(ids.train));
    expect(await h1(/fixture-smoke: Qwen\/Qwen3-0.6B on smoke/)).toBeTruthy();
    for (const t of ['Training loss', 'Learning rate', 'Gradient norm', 'Tokens per second', 'Memory (GB)'])
      expect(screen.getByText(t)).toBeTruthy();
    expect(screen.getAllByText(/loss mask is the agent's turns|every agent turn trained/).length).toBeGreaterThan(0);
    expect(screen.getByRole('link', { name: 'fixture tuned 0.6B' }).getAttribute('href')).toBe(evalRunPath(ids.tuned));
  });

  it('?vs= overlays another run and diffs the configs', async () => {
    renderAt(trainRunPath(ids.train, ids.train));
    expect(await screen.findByText(/What changed between the two configs: 0 setting/)).toBeTruthy();
  });
});

describe('/eval', () => {
  it('lists every scored model with counts', async () => {
    renderAt(evalPath());
    expect(await h1('Every scored model')).toBeTruthy();
    expect(screen.getByText('2 / 3 (66.7%)')).toBeTruthy();
    expect(screen.getByText('1 / 3 (33.3%)')).toBeTruthy();
  });

  it('a tuned run defaults to its base: fixed, broke and p', async () => {
    renderAt(evalRunPath(ids.tuned));
    expect(await h1(/fixture tuned 0.6B/)).toBeTruthy();
    expect(screen.getByRole('heading', { level: 2, name: /Fixed 1\s*, broke 0\s*, same 2/ })).toBeTruthy();
    expect(screen.getByText('✓ served prompt = trained bytes')).toBeTruthy();
    const cells = screen.getAllByRole('link', { name: /trial 1: (pass|fail)/ });
    expect(cells[0].getAttribute('href')).toBe(evalConversationPath(ids.tuned, ids.task, 1));
  });

  it('opens one scored conversation', async () => {
    renderAt(evalConversationPath(ids.tuned, ids.task, 1));
    expect((await screen.findAllByText(/reward 1/)).length).toBeGreaterThan(0);
  });
});

describe('/method', () => {
  it('reads the recipe and the loss from the latest job, and shows one row as trained', async () => {
    renderAt(methodPath());
    expect(await h1(/every agent turn in it is a target/)).toBeTruthy();
    expect(screen.getAllByText('assistant_only_loss').length).toBeGreaterThan(0);
    expect(screen.getByText('Training loss')).toBeTruthy();
    expect(document.querySelectorAll('.tk-trained').length).toBe(2);
  });
});

describe('the URL grammar', () => {
  it('escapes telecom task ids into one path segment and round-trips them', () => {
    const task = '[mms_issue]a|b[PERSONA:Easy]';
    const path = evalConversationPath('run1', task, 2);
    expect(path).toBe('/eval/run1/%5Bmms_issue%5Da%7Cb%5BPERSONA%3AEasy%5D/t2');
    expect(rowsPath('v1', { task })).toBe('/data/v1/rows?task=%5Bmms_issue%5Da%7Cb%5BPERSONA%3AEasy%5D');
  });

  it('drops default lens values, so one thing has one address', () => {
    expect(rowsPath('v1', { status: 'all', offset: 0 })).toBe('/data/v1/rows');
    expect(rowPath('v1', 'abc', 'chat')).toBe('/data/v1/rows/abc');
    expect(rowPath('v1', 'abc', 'raw', { status: 'dropped', rule: 'F4' })).toBe('/data/v1/rows/abc?lens=raw&status=dropped&rule=F4');
    expect(dataVersionPath('v1', 'coverage')).toBe('/data/v1?view=coverage');
  });
});
