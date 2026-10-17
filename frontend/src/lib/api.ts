/**
 * The API contract, typed: src/tau_telecom_sft/serving/app.py, reading the run folders that
 * docs/run_folders.md fixes. The viewer reads nothing else, and never MLflow (it links to it).
 */
import { useEffect, useState } from 'react';

// ── shared ────────────────────────────────────────────────────────────────
export type Health = { ok: true; code_sha: string; mlflow: 'up' | 'down' };
export type Milestone = { id: string; title: string; done: boolean; detail: string };
export type Tokens = { min: number; median: number; p90: number; p99?: number; max: number; prompt_and_tools?: number };

export type DatasetSummary = {
  version: string;
  built: string;
  code_sha: string;
  n_loaded: number;
  n_kept: number;
  n_dropped: number;
  training_tasks_covered: number;
  training_tasks_total: number;
  train_sha256: string;
  sources: string[];
  tokens: Tokens;
  max_tokens: number;
  token_counter: string;
  rows_here: boolean;
  review: { reviewed: number; total: number; signed_off: ReviewRecord | null };
};

export type FilterRow = { rule: string; keep_if: string; in: number; dropped: number };
export type SourceRow = { id: string; label: string; file: string; bytes: number; sha256: string; conversations: number };
export type Stats = {
  dataset: string;
  built: string;
  code_sha: string;
  split_version: number;
  split_sha256: string;
  system_prompt_sha256: string;
  tools_sha256: string;
  tau2_pin: string;
  sources: SourceRow[];
  n_loaded: number;
  n_kept: number;
  n_dropped: number;
  filters: FilterRow[];
  drop_reasons: Record<string, number>;
  kept_by_task: Record<string, number>;
  kept_by_symptom: Record<string, number>;
  kept_by_root_issue: Record<string, number>;
  kept_by_source: Record<string, number>;
  training_tasks_covered: number;
  training_tasks_total: number;
  token_counter: string;
  max_tokens: number;
  tokens: Tokens;
  tokens_hist?: { from: number; n: number }[];
  reached_f10?: number;
  over_limit?: Record<string, number>;
  agent_turns?: { median: number; max: number };
  source_checks?: Record<string, unknown>[];
  train_sha256: string;
};

export type DataVersion = { summary: DatasetSummary; stats: Stats; card: string; withheld: number };

export type Coverage = {
  version: string;
  thin_below: number;
  issues: { issue: string; kept_rows: number; train_tasks: number; test_tasks: number; thin: boolean }[];
  symptoms: { symptom: string; kept_rows: number; test_tasks: number }[];
  tasks_covered: number;
  tasks_total: number;
  note: string;
};

export type RowStatus = 'kept' | 'dropped';
export type RowItem = {
  sim_id: string;
  status: RowStatus;
  task_id: string;
  symptom: string | null;
  source: string;
  trial: number | null;
  agent_turns: number | null;
  rule: string | null;
  reason: string | null;
};
export type RowList = {
  version: string;
  rows_here: boolean;
  total: number;
  counts: { kept: number; dropped: number; withheld: number };
  n_kept: number;
  rules: string[];
  sources: string[];
  symptoms: string[];
  items: RowItem[];
};

export type Lens = 'chat' | 'tokens' | 'raw';
export type ChatMessage = {
  role: 'system' | 'assistant' | 'user' | 'tool';
  content: string;
  tool_calls: { id: string | null; name: string; arguments: string }[];
  tool_call_id: string | null;
};
export type RawMessage = {
  i: number;
  role: 'assistant' | 'user' | 'tool' | 'system';
  content: string | null;
  tool_calls: { id: string; name: string; arguments: Record<string, unknown> }[];
  requestor: string | null;
  error: boolean;
  tool_call_id: string | null;
  in_agent_view: boolean;
  offending: boolean;
};
export type Segment = { text: string; trained: boolean };
export type TokenView = { tokenizer: string; template: string; n_tokens: number; n_trained: number; segments: Segment[] };
export type RowDetail = {
  version: string;
  sim_id: string;
  status: RowStatus;
  meta: { task_id: string; source: string; trial: number | null; agent_turns: number | null; reward?: number | null; symptom?: string };
  drop: { rule: string; reason: string } | null;
  in_review_sample: boolean;
  lens: Lens;
  raw_here: boolean;
  missing?: string;
  messages?: ChatMessage[] | RawMessage[] | null;
  n_tools?: number;
  termination?: string;
  reward?: number | null;
  tokens?: TokenView;
};

export type Verdict = 'good' | 'bad' | 'note';
export type ReviewRecord = { kind: 'verdict' | 'sign-off'; sim_id: string | null; verdict: Verdict | null; note: string; at: string };
export type ReviewItem = {
  sim_id: string;
  status: RowStatus;
  task_id: string;
  source: string;
  trial: number | null;
  rule: string | null;
  reason: string | null;
  latest: ReviewRecord | null;
};
export type ReviewStatus = {
  version: string;
  seed: number;
  drawn: string;
  items: ReviewItem[];
  reviewed: number;
  total: number;
  signed_off: ReviewRecord | null;
  verdicts: Verdict[];
  log: ReviewRecord[];
};

// ── runs ──────────────────────────────────────────────────────────────────
export type Purpose = 'smoke' | 'result';
export type Adapter = { path: string; sha256: string; bytes: number } | null;

export type EvalSummary = {
  run_id: string;
  label: string;
  purpose: Purpose;
  results_row: string | null;
  created: string;
  model: { name?: string; base_model?: string; train_run?: string | null; adapter_sha256?: string | null };
  split: string;
  trials: number;
  tau2_loop_run: string | null;
  n_tasks: number | null;
  passed: number | null;
  pass_hat: Record<string, number | null>;
  prompt_bytes_ok: boolean | null;
  tool_calls_ok: boolean | null;
  vs: { run_id?: string; fixed?: number; broke?: number; same?: number; p?: number | null } | null;
  mlflow_url: string | null;
};

export type TrainSummary = {
  run_id: string;
  label: string;
  purpose: Purpose;
  status: 'running' | 'finished' | 'failed';
  created: string;
  base_model: string;
  dataset: string;
  train_sha256: string;
  n_rows: number;
  device: string;
  instance: string;
  duration_s: number | null;
  cost_usd: number | null;
  steps: number | null;
  final_loss: number | null;
  peak_mem_gb: number | null;
  settings: Record<string, string | number | boolean | null>;
  adapter: Adapter;
  mask_ok: boolean | null;
  scored: EvalSummary[];
  mlflow_url: string | null;
};

export type Metric = {
  step: number;
  epoch?: number;
  loss?: number;
  grad_norm?: number;
  learning_rate?: number;
  tokens?: number;
  tokens_per_s?: number;
  mem_gb?: number;
  elapsed_s?: number;
};

export type MaskCheck = { ok: boolean; row_sim_id?: string; assistant_turns?: number; trained_turns?: number; problems: string[] };
export type Mask = {
  row_sim_id: string;
  task_id?: string;
  n_tokens: number;
  n_trained: number;
  chat_template?: string;
  segments: Segment[];
  check: MaskCheck;
};

export type TrainRunDoc = Record<string, unknown> & {
  run_id: string;
  config?: Record<string, unknown>;
  config_file?: string;
  code_sha?: string;
  image_tag?: string | null;
  tokenizer?: string;
  rows_selected?: string;
  finished?: string;
  tokens_trained?: number;
  epochs?: number;
  mask_check?: MaskCheck;
  mlflow_run_id?: string;
};
export type TrainRun = { run: TrainRunDoc; summary: TrainSummary; config_flat: Record<string, unknown>; metrics: Metric[]; mask: Mask | null };

export type LoopResult = {
  task_id: string;
  trial: number;
  reward: number;
  correct: boolean;
  termination_reason?: string;
  n_agent_turns?: number;
  n_tool_calls?: number;
  n_tool_errors?: number;
  duration_ms?: number;
  error?: string | null;
};
export type CompareTask = { task_id: string; a: number[]; b: number[]; change: 'fixed' | 'broke' | 'same' };
export type Compare = { vs: EvalSummary; n_tasks: number; fixed: number; broke: number; same: number; p: number | null; tasks: CompareTask[] };
export type Served = {
  requests: number;
  system_prompt_sha256: string[];
  tools_sha256: string[];
  with_tool_calls: number;
  text_tool_calls: number;
  latency_s_median: number | null;
  prompt_tokens_max: number | null;
};
export type EvalRunDoc = Record<string, unknown> & {
  run_id: string;
  serving?: { server?: string; precision?: string; endpoint?: string; sampling?: Record<string, number>; max_tokens?: number };
  prompt_bytes?: Record<string, unknown> & { ok?: boolean };
  tool_calls?: Record<string, unknown> & { ok?: boolean };
  tau2_loop_version?: string;
};
export type EvalRun = {
  run: EvalRunDoc;
  summary: EvalSummary;
  grid: { task_id: string; trials: number[] }[];
  trials: number;
  pass_hat: Record<string, number | null>;
  results: LoopResult[];
  served: Served;
  loop_run: Record<string, unknown> | null;
  compare: Compare | null;
};
export type EvalConversation = {
  run_id: string;
  task_id: string;
  trial: number;
  reward: number | null;
  termination: string;
  result: LoopResult | null;
  messages: RawMessage[];
};

export type Chain = {
  dataset: string;
  train_run: string;
  train_label: string;
  purpose: Purpose;
  adapter_sha256: string | null;
  evals: EvalSummary[];
  bases: EvalSummary[];
};
export type ResultRow = { row: string; model: string; milestone: string; run: EvalSummary | null };
export type Overview = {
  project: 'tau-telecom-sft';
  code_sha: string;
  milestones: Milestone[];
  datasets: DatasetSummary[];
  train_runs: TrainSummary[];
  eval_runs: EvalSummary[];
  results: ResultRow[];
  smoke: EvalSummary[];
  chains: Chain[];
};

export type Method = {
  train_run: TrainRunDoc | null;
  metrics: Metric[];
  mask: Mask | null;
  dataset: { version: string; n_kept: number; tokens: Tokens; max_tokens: number; token_counter: string } | null;
};

// ── endpoints ─────────────────────────────────────────────────────────────
type Params = Record<string, string | number | null | undefined>;

/** `?a=1&b=2`, skipping empty values; '' when nothing is set. */
export function qs(params: Params): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v != null && v !== '') sp.set(k, String(v));
  const s = sp.toString();
  return s ? `?${s}` : '';
}

const seg = encodeURIComponent;

export type RowQuery = {
  status?: 'all' | RowStatus;
  rule?: string | null;
  task?: string | null;
  symptom?: string | null;
  source?: string | null;
  q?: string | null;
  offset?: number;
  limit?: number;
};

export const api = {
  health: () => '/healthz',
  overview: () => '/api/overview',
  data: () => '/api/data',
  dataVersion: (v: string) => `/api/data/${seg(v)}`,
  coverage: (v: string) => `/api/data/${seg(v)}/coverage`,
  rows: (v: string, p: RowQuery) => `/api/data/${seg(v)}/rows${qs(p)}`,
  row: (v: string, simId: string, lens: Lens) => `/api/data/${seg(v)}/rows/${seg(simId)}${qs({ lens })}`,
  review: (v: string) => `/api/data/${seg(v)}/review`,
  trainRuns: () => '/api/train',
  trainRun: (run: string) => `/api/train/${seg(run)}`,
  evalRuns: () => '/api/eval',
  evalRun: (run: string, vs?: string | null) => `/api/eval/${seg(run)}${qs({ vs })}`,
  evalConversation: (run: string, task: string, trial: number) => `/api/eval/${seg(run)}/${seg(task)}/t${trial}`,
  method: () => '/api/method',
};

// ── fetching ──────────────────────────────────────────────────────────────
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

/** The app may run under a sub-path (SHOWCASE_BASE); API paths are rooted there too. */
function abs(url: string): string {
  const base = (import.meta.env.BASE_URL ?? '/').replace(/\/$/, '');
  return url.startsWith('/') ? `${base}${url}` : url;
}

async function read<T>(r: Response, url: string): Promise<T> {
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new ApiError((j as { detail?: string }).detail ?? `${r.status} ${url}`, r.status);
  return j as T;
}

export async function get<T>(url: string): Promise<T> {
  return read<T>(await fetch(abs(url)), url);
}

export async function post<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(abs(url), {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  return read<T>(r, url);
}

/** GET a URL whenever it changes; `url: null` fetches nothing. Stale data stays until the new arrives. */
export function useGet<T>(url: string | null): { data: T | null; error: string | null; status: number | null; reload: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<number | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!url) {
      setData(null);
      return;
    }
    let live = true;
    setError(null);
    get<T>(url)
      .then((d) => {
        if (live) {
          setData(d);
          setStatus(200);
        }
      })
      .catch((e: unknown) => {
        if (!live) return;
        setData(null);
        setError(e instanceof Error ? e.message : String(e));
        setStatus(e instanceof ApiError ? e.status : null);
      });
    return () => {
      live = false;
    };
  }, [url, tick]);
  return { data, error, status, reload: () => setTick((t) => t + 1) };
}
