import { Link } from 'react-router-dom';
import { type Chain, type EvalSummary, type Overview as OverviewDoc, api, useGet } from '../lib/api';
import { DASH, countOf, int, pct, shortSha } from '../lib/format';
import { Loading, NotYet, PurposeBadge } from '../lib/ui';
import { dataVersionPath, evalRunPath, reviewPath, trainRunPath } from '../lib/url';

/** `/`: where the project stands, the results table (R0 to R5, with counts), and the chain. */
export function Overview() {
  const { data, error } = useGet<OverviewDoc>(api.overview());
  if (!data) return <Loading error={error} what="the overview" />;
  const ds = data.datasets[data.datasets.length - 1];
  return (
    <>
      <p className="eyebrow">tau-telecom-sft · overview</p>
      <h1>
        Does a small open model, fine-tuned on <em>clean</em> teacher conversations, beat its own base on tau2 telecom?
      </h1>
      <p className="lede">
        One chain, every link inspectable: <b>dataset version → training run → adapter → scored model</b>. The student is
        scored in tau2-loop on split v3's 40 test tasks with the same Haiku 5.5 customer your Claude v2 agent faced. Each
        test task is worth 2.5 points, so every share below carries its count.
      </p>

      <div className="verdict headline" role="group" aria-label="Where it stands">
        <div className="stat accent">
          <div className="n">{ds ? int(ds.n_kept) : DASH}</div>
          <div className="l">
            clean training rows in {ds ? ds.version : 'no dataset yet'}
            {ds && <span className="tag">of {int(ds.n_loaded)} board conversations · {ds.training_tasks_covered} / {ds.training_tasks_total} training tasks</span>}
          </div>
        </div>
        <div className="stat">
          <div className="n">{data.train_runs.length}</div>
          <div className="l">
            training run{data.train_runs.length === 1 ? '' : 's'}
            <span className="tag">{data.train_runs.filter((t) => t.purpose === 'result').length} for results · the rest smoke</span>
          </div>
        </div>
        <div className="stat">
          <div className="n">{data.eval_runs.length}</div>
          <div className="l">
            scored run{data.eval_runs.length === 1 ? '' : 's'}
            <span className="tag">{data.results.filter((r) => r.run).length} of 6 results rows filled</span>
          </div>
        </div>
        <div className={`stat ${ds?.review.signed_off ? 'accent' : 'amber'}`}>
          <div className="n">
            {ds ? `${ds.review.reviewed} / ${ds.review.total}` : DASH}
          </div>
          <div className="l">
            sample rows read{' '}
            {ds && (
              <span className="tag">
                {ds.review.signed_off ? `signed off ${ds.review.signed_off.at}` : <Link to={reviewPath(ds.version)}>not signed off: read them →</Link>}
              </span>
            )}
          </div>
        </div>
      </div>

      <section aria-labelledby="results-h">
        <span className="eyebrow">The results table (plan §8)</span>
        <h2 id="results-h">40 test tasks × 4 trials per row, paired by task against its own untrained base</h2>
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th>row</th>
                <th>model</th>
                <th className="num">pass^1</th>
                <th className="num">pass^3</th>
                <th className="num">vs base</th>
                <th>filled by</th>
              </tr>
            </thead>
            <tbody>
              {data.results.map((r) => (
                <tr key={r.row} className={r.run ? '' : 'dim'}>
                  <td className="k">{r.row}</td>
                  <td>{r.run ? <Link to={evalRunPath(r.run.run_id)}>{r.model}</Link> : r.model}</td>
                  <td className="num">{r.run ? passCell(r.run, '1') : DASH}</td>
                  <td className="num">{r.run ? passCell(r.run, '3') : DASH}</td>
                  <td className="num">{r.run?.vs ? `+${r.run.vs.fixed ?? 0} / −${r.run.vs.broke ?? 0}` : DASH}</td>
                  <td className="tag">{r.milestone}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="sub">
          Only a run marked <code>purpose: result</code> fills a row. The Mac proof of concept (M2) is listed below as smoke: its
          scores are not results.
        </p>
      </section>

      <section aria-labelledby="chain-h">
        <span className="eyebrow">The chain</span>
        <h2 id="chain-h">From a dataset version to the scores its adapter earned</h2>
        {data.chains.length === 0 ? (
          <NotYet what="training run" by="M2's make train-smoke" />
        ) : (
          data.chains.map((c) => <ChainRow key={c.train_run} chain={c} />)
        )}
        {data.smoke.length > 0 && (
          <>
            <h3>Smoke runs</h3>
            <ul className="tight">
              {data.smoke.map((e) => (
                <li key={e.run_id}>
                  <Link to={evalRunPath(e.run_id)}>{e.label}</Link> · {e.model.name ?? e.model.base_model} ·{' '}
                  {countOf(e.passed, (e.n_tasks ?? 0) * (e.trials || 1))} conversations passed
                </li>
              ))}
            </ul>
          </>
        )}
      </section>

      <section aria-labelledby="ms-h">
        <span className="eyebrow">Milestones (ai_specs/s00_project_plan.md)</span>
        <h2 id="ms-h">
          {data.milestones.filter((m) => m.done).length} of {data.milestones.length} ticked
        </h2>
        <ol className="milestones">
          {data.milestones.map((m) => (
            <li key={m.id}>
              <span className={`badge ${m.done ? 'ok' : ''}`}>{m.done ? `✓ ${m.id}` : `○ ${m.id}`}</span>
              <span>
                <b>{m.title}</b> <span className="detail">{m.detail}</span>
              </span>
            </li>
          ))}
        </ol>
      </section>
    </>
  );
}

function passCell(e: EvalSummary, k: string) {
  const v = e.pass_hat[k];
  if (v == null) return DASH;
  return `${pct(v)} (${Math.round(v * (e.n_tasks ?? 0) * 10) / 10} / ${e.n_tasks})`;
}

function ChainRow({ chain }: { chain: Chain }) {
  return (
    <ol className="chain" aria-label={`chain for ${chain.train_label}`}>
      <li>
        <span className="eyebrow">dataset</span>
        <b>
          <Link to={dataVersionPath(chain.dataset)}>{chain.dataset}</Link>
        </b>
      </li>
      <li className="key">
        <span className="eyebrow">training run</span>
        <b>
          <Link to={trainRunPath(chain.train_run)}>{chain.train_label}</Link>
        </b>
        <PurposeBadge purpose={chain.purpose} />
      </li>
      <li>
        <span className="eyebrow">adapter</span>
        <b className="mono">{shortSha(chain.adapter_sha256)}</b>
      </li>
      <li className="key">
        <span className="eyebrow">scored</span>
        {chain.evals.length === 0 && chain.bases.length === 0 ? (
          <span className="tag">not scored yet</span>
        ) : (
          <ul className="tight">
            {chain.bases.map((e) => (
              <li key={e.run_id}>
                base: <Link to={evalRunPath(e.run_id)}>{e.label}</Link> · {countOf(e.passed, (e.n_tasks ?? 0) * (e.trials || 1))}
              </li>
            ))}
            {chain.evals.map((e) => (
              <li key={e.run_id}>
                tuned: <Link to={evalRunPath(e.run_id, chain.bases[0]?.run_id)}>{e.label}</Link> ·{' '}
                {countOf(e.passed, (e.n_tasks ?? 0) * (e.trials || 1))}
              </li>
            ))}
          </ul>
        )}
      </li>
    </ol>
  );
}
