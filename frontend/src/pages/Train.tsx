import { useState } from 'react';
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { type Metric, type TrainRun, type TrainSummary, api, useGet } from '../lib/api';
import { DASH, duration, int, num, sci, shortSha, usd, when } from '../lib/format';
import { KV, Loading, MlflowLink, NotYet, OkBadge, PurposeBadge, type Series, LineChart, TokensView } from '../lib/ui';
import { dataVersionPath, evalRunPath, rowPath, trainRunPath } from '../lib/url';

type SortKey = 'created' | 'label' | 'learning_rate' | 'num_train_epochs' | 'lora_r' | 'max_length' | 'final_loss' | 'pass';

const sortValue = (r: TrainSummary, k: SortKey): number | string => {
  if (k === 'created' || k === 'label') return String(r[k] ?? '');
  if (k === 'final_loss') return r.final_loss ?? Number.POSITIVE_INFINITY;
  if (k === 'pass') return bestPass(r) ?? -1;
  const v = r.settings[k];
  return typeof v === 'number' ? v : Number(v ?? Number.NaN);
};

function bestPass(r: TrainSummary): number | null {
  const ps = r.scored.map((e) => e.pass_hat['3'] ?? e.pass_hat['1'] ?? null).filter((x): x is number => x != null);
  return ps.length ? Math.max(...ps) : null;
}

/** `/train`: one sortable table of every fine-tuning job, the settings that move a score first. */
export function TrainList() {
  const { data, error } = useGet<{ runs: TrainSummary[] }>(api.trainRuns());
  const [sort, setSort] = useState<{ k: SortKey; desc: boolean }>({ k: 'created', desc: true });
  if (!data) return <Loading error={error} what="the training runs" />;
  const rows = [...data.runs].sort((a, b) => {
    const x = sortValue(a, sort.k);
    const y = sortValue(b, sort.k);
    const c = typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y));
    return sort.desc ? -c : c;
  });
  const head = (k: SortKey, label: string, numeric = false) => (
    <th className={numeric ? 'num' : undefined} aria-sort={sort.k === k ? (sort.desc ? 'descending' : 'ascending') : 'none'}>
      <button type="button" className="sort" aria-pressed={sort.k === k} onClick={() => setSort({ k, desc: sort.k === k ? !sort.desc : true })}>
        {label}
        {sort.k === k ? (sort.desc ? ' ↓' : ' ↑') : ''}
      </button>
    </th>
  );
  return (
    <>
      <p className="eyebrow">Train</p>
      <h1>Every fine-tuning job</h1>
      <p className="lede">
        LoRA SFT in TRL, the loss on the agent's turns only. Each job's folder in <code>runs/</code> is the record: its config, the dataset
        hash it trained on, where it ran, its loss per step and the loss-mask check. MLflow is the link.
      </p>
      {rows.length === 0 ? (
        <NotYet what="training run" by="M2's make train-smoke" />
      ) : (
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                {head('label', 'run')}
                {head('created', 'started')}
                <th>data</th>
                {head('learning_rate', 'lr', true)}
                {head('num_train_epochs', 'epochs', true)}
                {head('lora_r', 'rank', true)}
                {head('max_length', 'max length', true)}
                <th className="num">steps</th>
                {head('final_loss', 'final loss', true)}
                <th>where</th>
                {head('pass', 'pass^3 (pass^1) it scored', true)}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.run_id}>
                  <td>
                    <Link to={trainRunPath(r.run_id)}>{r.label}</Link> <PurposeBadge purpose={r.purpose} />{' '}
                    {r.status !== 'finished' && <span className={`badge ${r.status === 'failed' ? 'bad' : 'warn'}`}>{r.status}</span>}
                  </td>
                  <td className="tag">{when(r.created)}</td>
                  <td>
                    <Link to={dataVersionPath(r.dataset)}>{r.dataset}</Link> <span className="tag">{int(r.n_rows)} rows</span>
                  </td>
                  <td className="num">{sci(r.settings.learning_rate as number | undefined)}</td>
                  <td className="num">{r.settings.num_train_epochs ?? DASH}</td>
                  <td className="num">{r.settings.lora_r ?? DASH}</td>
                  <td className="num">{int(r.settings.max_length as number | undefined)}</td>
                  <td className="num">{r.steps ?? DASH}</td>
                  <td className="num">{num(r.final_loss)}</td>
                  <td className="tag">
                    {r.device} · {r.instance}
                  </td>
                  <td className="num">
                    {r.scored.length === 0 ? DASH : r.scored.map((e) => (
                      <Link key={e.run_id} to={evalRunPath(e.run_id)}>
                        {e.pass_hat['3'] != null ? num(e.pass_hat['3'], 2) : `(${num(e.pass_hat['1'], 2)})`}
                      </Link>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

function series(metrics: Metric[], key: keyof Metric, name: string, alt = false): Series {
  return {
    name,
    alt,
    points: metrics.filter((m) => typeof m[key] === 'number').map((m) => [m.step, m[key] as number]),
  };
}

function epochMarks(metrics: Metric[]): { x: number; label: string }[] {
  const out: { x: number; label: string }[] = [];
  let last = 0;
  for (const m of metrics) {
    const e = Math.floor(m.epoch ?? 0);
    if (e > last && m.epoch != null) {
      out.push({ x: m.step, label: `epoch ${e}` });
      last = e;
    }
  }
  return out;
}

/** `/train/:run[?vs=<run>]`: one job, its curves, its mask check; `vs` overlays another and diffs the configs. */
export function TrainRunPage() {
  const { run = '' } = useParams();
  const [sp] = useSearchParams();
  const vs = sp.get('vs');
  const navigate = useNavigate();
  const { data, error } = useGet<TrainRun>(api.trainRun(run));
  const other = useGet<TrainRun>(vs ? api.trainRun(vs) : null);
  const all = useGet<{ runs: TrainSummary[] }>(api.trainRuns());
  if (!data) return <Loading error={error} what={`training run ${run}`} />;
  const s = data.summary;
  const r = data.run;
  const m = data.metrics;
  const o = other.data;
  const loss: Series[] = [series(m, 'loss', s.label)];
  if (o) loss.push(series(o.metrics, 'loss', o.summary.label, true));
  const marks = epochMarks(m);
  const diff = o
    ? Object.keys({ ...data.config_flat, ...o.config_flat })
        .filter((k) => JSON.stringify(data.config_flat[k]) !== JSON.stringify(o.config_flat[k]))
        .sort()
    : [];
  return (
    <>
      <p className="crumb">
        <Link to="/train">train</Link> / {run}
      </p>
      <h1>
        {s.label}: <em>{s.base_model}</em> on {s.dataset}
      </h1>
      <div className="row badges">
        <PurposeBadge purpose={s.purpose} />
        <span className={`badge ${s.status === 'finished' ? 'ok' : s.status === 'failed' ? 'bad' : 'warn'}`}>{s.status}</span>
        <OkBadge ok={s.mask_ok} yes="loss mask is the agent's turns" no="✕ loss mask check failed" />
        <MlflowLink url={s.mlflow_url} />
        <label className="choice">
          <span className="tag">compare with</span>
          <select value={vs ?? ''} onChange={(e) => navigate(trainRunPath(run, e.target.value || null))}>
            <option value="">nothing</option>
            {(all.data?.runs ?? [])
              .filter((x) => x.run_id !== run)
              .map((x) => (
                <option key={x.run_id} value={x.run_id}>
                  {x.label} · {x.run_id}
                </option>
              ))}
          </select>
        </label>
      </div>
      {s.purpose === 'smoke' && (
        <div className="callout">
          <p>
            A smoke job proves the pipeline (M2): the same TRL code as the GPU job, a few steps on the Mac. Its loss says the plumbing works,
            not that the model learned telecom.
          </p>
        </div>
      )}
      <div className="kpis">
        <div className="stat">
          <div className="n">{s.steps ?? DASH}</div>
          <div className="l">optimiser steps</div>
        </div>
        <div className="stat">
          <div className="n">{num(s.final_loss)}</div>
          <div className="l">final loss</div>
        </div>
        <div className="stat">
          <div className="n">{duration(s.duration_s)}</div>
          <div className="l">wall clock</div>
        </div>
        <div className="stat">
          <div className="n">{s.peak_mem_gb != null ? `${num(s.peak_mem_gb, 1)} GB` : DASH}</div>
          <div className="l">peak memory</div>
        </div>
        <div className="stat">
          <div className="n">{usd(s.cost_usd)}</div>
          <div className="l">cost</div>
        </div>
      </div>

      <section aria-labelledby="curves-h">
        <span className="eyebrow">metrics.jsonl, one line per step</span>
        <h2 id="curves-h">Did training behave?</h2>
        {m.length === 0 ? (
          <NotYet what="logged step" by="the job's first logging step" />
        ) : (
          <div className="charts">
            <LineChart title="Training loss" series={loss} marks={marks} caption={o ? `Dashed: ${o.summary.label}.` : 'Per optimiser step, epoch ends marked.'} />
            <LineChart title="Learning rate" series={[series(m, 'learning_rate', 'learning rate')]} digits={6} />
            <LineChart title="Gradient norm" series={[series(m, 'grad_norm', 'grad norm')]} />
            <LineChart title="Tokens per second" series={[series(m, 'tokens_per_s', 'tokens/s')]} digits={0} />
            <LineChart title="Memory (GB)" series={[series(m, 'mem_gb', 'memory')]} digits={1} />
          </div>
        )}
      </section>

      {o && (
        <section aria-labelledby="diff-h">
          <span className="eyebrow">?vs={o.summary.run_id}</span>
          <h2 id="diff-h">What changed between the two configs: {diff.length} setting(s)</h2>
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>setting</th>
                  <th>{s.label}</th>
                  <th>{o.summary.label}</th>
                </tr>
              </thead>
              <tbody>
                {diff.map((k) => (
                  <tr key={k}>
                    <td className="mono">{k}</td>
                    <td className="mono">{JSON.stringify(data.config_flat[k] ?? null)}</td>
                    <td className="mono">{JSON.stringify(o.config_flat[k] ?? null)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section aria-labelledby="job-h">
        <span className="eyebrow">run.json</span>
        <h2 id="job-h">The job</h2>
        <div className="cols">
          <KV
            rows={[
              ['run id', <code key="r">{s.run_id}</code>],
              ['started', when(s.created)],
              ['finished', when(r.finished as string | undefined)],
              ['code', <code key="c">{String(r.code_sha ?? DASH)}</code>],
              ['image', r.image_tag ? <code key="i">{String(r.image_tag)}</code> : 'none (the Mac)'],
              ['config file', <code key="f">{String(r.config_file ?? DASH)}</code>],
              ['base model', s.base_model],
              ['tokenizer', String(r.tokenizer ?? DASH)],
              ['where', `${s.device} · ${s.instance}`],
            ]}
          />
          <KV
            rows={[
              ['dataset', <Link key="d" to={dataVersionPath(s.dataset)}>{s.dataset}</Link>],
              ['train.jsonl sha256', <code key="h">{shortSha(s.train_sha256, 16)}</code>],
              ['rows trained', `${int(s.n_rows)}${r.rows_selected ? ` (${String(r.rows_selected)})` : ''}`],
              ['tokens trained', int(r.tokens_trained as number | undefined)],
              ['adapter', s.adapter ? <code key="a">{shortSha(s.adapter.sha256, 16)}</code> : DASH],
              ['adapter size', s.adapter ? `${int(s.adapter.bytes)} bytes` : DASH],
              ['MLflow run', String(r.mlflow_run_id ?? DASH)],
            ]}
          />
        </div>
        <details className="facts">
          <summary>Every setting ({Object.keys(data.config_flat).length})</summary>
          <div className="tablewrap">
            <table>
              <tbody>
                {Object.entries(data.config_flat).map(([k, val]) => (
                  <tr key={k}>
                    <td className="mono">{k}</td>
                    <td className="mono">{JSON.stringify(val)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </section>

      <section aria-labelledby="mask-h">
        <span className="eyebrow">mask.json: the loss-mask check on one row</span>
        <h2 id="mask-h">Are the trained tokens exactly the agent's turns?</h2>
        {!data.mask ? (
          <NotYet what="mask check" by="the job's start" />
        ) : (
          <>
            <div className="row badges">
              <OkBadge ok={data.mask.check.ok} yes="every agent turn trained, nothing else" no="✕ the mask is wrong" />
              <span className="tag">
                {data.mask.check.trained_turns ?? DASH} of {data.mask.check.assistant_turns ?? DASH} agent turns trained · row{' '}
                <Link to={rowPath(s.dataset, data.mask.row_sim_id, 'tokens')}>{data.mask.row_sim_id.slice(0, 8)}</Link>
              </span>
            </div>
            {data.mask.check.problems.length > 0 && (
              <ul className="tight">
                {data.mask.check.problems.map((p) => (
                  <li key={p}>{p}</li>
                ))}
              </ul>
            )}
            <TokensView segments={data.mask.segments} nTokens={data.mask.n_tokens} nTrained={data.mask.n_trained} />
          </>
        )}
      </section>

      <section aria-labelledby="scored-h">
        <span className="eyebrow">What its adapter scored</span>
        <h2 id="scored-h">Scored runs</h2>
        {s.scored.length === 0 ? (
          <NotYet what="scored run of this adapter" by="tau2-loop, through serve-mac" />
        ) : (
          <ul className="tight">
            {s.scored.map((e) => (
              <li key={e.run_id}>
                <Link to={evalRunPath(e.run_id)}>{e.label}</Link> · pass^1 {num(e.pass_hat['1'], 2)} · {e.passed ?? DASH} of{' '}
                {(e.n_tasks ?? 0) * (e.trials || 1)} conversations
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}
