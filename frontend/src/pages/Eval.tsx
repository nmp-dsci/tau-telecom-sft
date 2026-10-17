import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { type EvalConversation, type EvalRun, type EvalSummary, api, useGet } from '../lib/api';
import { DASH, countOf, duration, num, pValue, pct, shortSha, when } from '../lib/format';
import { Grid, KV, Loading, MlflowLink, NotYet, OkBadge, PassBadge, PurposeBadge, RawView, TaskCell } from '../lib/ui';
import { evalConversationPath, evalRunPath, trainRunPath } from '../lib/url';

const conversations = (e: EvalSummary) => (e.n_tasks ?? 0) * (e.trials || 1);

/** `/eval`: every scored model, with pass^k and its counts. */
export function EvalList() {
  const { data, error } = useGet<{ runs: EvalSummary[] }>(api.evalRuns());
  if (!data) return <Loading error={error} what="the scored runs" />;
  return (
    <>
      <p className="eyebrow">Eval</p>
      <h1>Every scored model</h1>
      <p className="lede">
        tau2-loop plays the conversations on this Mac (its orchestrator, the phone, the grader and the Haiku 5.5 customer); only the student's
        turns come from the served model. Each run here is a copy of its tau2-loop run folder, read and never written back.
      </p>
      {data.runs.length === 0 ? (
        <NotYet what="scored run" by="M2's smoke scoring in tau2-loop" />
      ) : (
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th>run</th>
                <th>model</th>
                <th>split</th>
                <th className="num">passed</th>
                <th className="num">pass^1</th>
                <th>served bytes</th>
                <th>tool calls</th>
                <th>started</th>
              </tr>
            </thead>
            <tbody>
              {data.runs.map((e) => (
                <tr key={e.run_id}>
                  <td>
                    <Link to={evalRunPath(e.run_id)}>{e.label}</Link> <PurposeBadge purpose={e.purpose} />
                  </td>
                  <td>
                    {e.model.name ?? e.model.base_model}{' '}
                    {e.model.train_run ? (
                      <Link className="tag" to={trainRunPath(e.model.train_run)}>
                        tuned · {e.model.train_run}
                      </Link>
                    ) : (
                      <span className="tag">untrained base</span>
                    )}
                  </td>
                  <td className="tag">{e.split}</td>
                  <td className="num">{countOf(e.passed, conversations(e))}</td>
                  <td className="num">{pct(e.pass_hat['1'])}</td>
                  <td>
                    <OkBadge ok={e.prompt_bytes_ok} yes="trained bytes" no="✕ differ" />
                  </td>
                  <td>
                    <OkBadge ok={e.tool_calls_ok} yes="structured" no="✕ as text" />
                  </td>
                  <td className="tag">{when(e.created)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

/** `/eval/:run[?vs=<run>]`: pass^k, the task × trial grid, the served-bytes checks, and the paired gate against a base. */
export function EvalRunPage() {
  const { run = '' } = useParams();
  const [sp] = useSearchParams();
  const vs = sp.get('vs');
  const navigate = useNavigate();
  const { data, error } = useGet<EvalRun>(api.evalRun(run, vs));
  const all = useGet<{ runs: EvalSummary[] }>(api.evalRuns());
  if (!data) return <Loading error={error} what={`scored run ${run}`} />;
  const s = data.summary;
  const r = data.run;
  const sampling = r.serving?.sampling ?? {};
  const cmp = data.compare;
  const ks = Object.keys(data.pass_hat).sort();
  return (
    <>
      <p className="crumb">
        <Link to="/eval">eval</Link> / {run}
      </p>
      <h1>
        {s.label}: <em>{countOf(s.passed, conversations(s))}</em> conversations passed
      </h1>
      <div className="row badges">
        <PurposeBadge purpose={s.purpose} />
        <OkBadge ok={s.prompt_bytes_ok} yes="served prompt = trained bytes" no="✕ served prompt differs from training" />
        <OkBadge ok={s.tool_calls_ok} yes="tool calls came back structured" no="✕ tool calls came back as text" />
        <MlflowLink url={s.mlflow_url} />
        <label className="choice">
          <span className="tag">compare with</span>
          <select value={vs ?? ''} onChange={(e) => navigate(evalRunPath(run, e.target.value || null))}>
            <option value="">{(r.vs as { run_id?: string } | undefined)?.run_id ? 'its base (default)' : 'nothing'}</option>
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
            Smoke (M2): training tasks only, one trial, a 0.6B model served on the Mac. It proves a tuned adapter can be served and scored
            end to end; its score is not a result.
          </p>
        </div>
      )}
      <div className="kpis">
        {ks.length === 0 ? (
          <div className="stat empty-stat">
            <div className="n">{DASH}</div>
            <div className="l">no trials scored</div>
          </div>
        ) : (
          ks.map((k) => (
            <div key={k} className={`stat ${k === '3' ? 'accent' : ''}`}>
              <div className="n">{pct(data.pass_hat[k])}</div>
              <div className="l">
                pass^{k}
                <span className="tag">over {data.grid.length} task(s)</span>
              </div>
            </div>
          ))
        )}
        <div className="stat">
          <div className="n">{data.served.requests}</div>
          <div className="l">model calls served</div>
        </div>
      </div>

      {cmp && (
        <section aria-labelledby="cmp-h">
          <span className="eyebrow">Paired by task against {cmp.vs.label}</span>
          <h2 id="cmp-h">
            Fixed <span className="change-fixed">{cmp.fixed}</span>, broke <span className="change-broke">{cmp.broke}</span>, same {cmp.same} · one-sided
            sign test p = {pValue(cmp.p)}
          </h2>
          <p className="sub">
            A task's score is its pass fraction over trials; fixed means higher than the base, broke means lower. This is tau2-loop's gate. On{' '}
            {cmp.n_tasks} shared task(s).
          </p>
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>task</th>
                  <th className="num">{s.label}</th>
                  <th className="num">{cmp.vs.label}</th>
                  <th>change</th>
                </tr>
              </thead>
              <tbody>
                {cmp.tasks.map((t) => (
                  <tr key={t.task_id}>
                    <td>
                      <TaskCell id={t.task_id} />
                    </td>
                    <td className="num">
                      {t.a.reduce((x, y) => x + y, 0)} / {t.a.length}
                    </td>
                    <td className="num">
                      {t.b.reduce((x, y) => x + y, 0)} / {t.b.length}
                    </td>
                    <td className={`change-${t.change}`}>{t.change === 'fixed' ? '▲ fixed' : t.change === 'broke' ? '▼ broke' : '= same'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section aria-labelledby="grid-h">
        <span className="eyebrow">Every conversation</span>
        <h2 id="grid-h">
          {data.grid.length} task(s) × {data.trials} trial(s)
        </h2>
        {data.grid.length === 0 ? (
          <NotYet what="scored conversation" by="the tau2-loop run" />
        ) : (
          <Grid rows={data.grid} trials={data.trials} to={(t, n) => evalConversationPath(run, t, n)} />
        )}
        {data.results.length > 0 && (
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>task</th>
                  <th className="num">trial</th>
                  <th>result</th>
                  <th>ended by</th>
                  <th className="num">agent turns</th>
                  <th className="num">tool calls</th>
                  <th className="num">duration</th>
                </tr>
              </thead>
              <tbody>
                {data.results.map((x) => (
                  <tr key={`${x.task_id}-${x.trial}`}>
                    <td>
                      <TaskCell id={x.task_id} to={evalConversationPath(run, x.task_id, x.trial)} />
                    </td>
                    <td className="num">t{x.trial}</td>
                    <td>
                      <PassBadge pass={x.correct} /> {x.error && <span className="badge bad">{x.error}</span>}
                    </td>
                    <td className="tag">{x.termination_reason ?? DASH}</td>
                    <td className="num">{x.n_agent_turns ?? DASH}</td>
                    <td className="num">{x.n_tool_calls ?? DASH}</td>
                    <td className="num">{x.duration_ms != null ? duration(x.duration_ms / 1000) : DASH}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section aria-labelledby="serve-h">
        <span className="eyebrow">How it was served (plan §8's rules)</span>
        <h2 id="serve-h">The same bytes it was trained on, and tool calls as tool calls</h2>
        <div className="cols">
          <KV
            rows={[
              ['model', s.model.name ?? s.model.base_model ?? DASH],
              ['base', s.model.base_model ?? DASH],
              ['trained by', s.model.train_run ? <Link key="t" to={trainRunPath(s.model.train_run)}>{s.model.train_run}</Link> : 'untrained'],
              ['adapter', shortSha(s.model.adapter_sha256, 16)],
              ['server', `${r.serving?.server ?? DASH} · ${r.serving?.precision ?? DASH}`],
              ['endpoint', r.serving?.endpoint ?? DASH],
              ['sampling', Object.keys(sampling).length ? Object.entries(sampling).map(([k, v]) => `${k} ${v}`).join(' · ') : DASH],
              ['tau2-loop run', s.tau2_loop_run ?? DASH],
              ['tau2-loop version', String(r.tau2_loop_version ?? DASH)],
            ]}
          />
          <KV
            rows={[
              ['requests recorded', String(data.served.requests)],
              ['system prompt sha256 served', data.served.system_prompt_sha256.map((x) => shortSha(x, 16)).join(', ') || DASH],
              ['tools sha256 served', data.served.tools_sha256.map((x) => shortSha(x, 16)).join(', ') || DASH],
              ['expected (the dataset card)', shortSha((r.prompt_bytes as Record<string, string> | undefined)?.expected_system_prompt_sha256, 16)],
              ['replies with tool calls', String(data.served.with_tool_calls)],
              ['tool calls written as text', String(data.served.text_tool_calls)],
              ['median latency', data.served.latency_s_median != null ? `${num(data.served.latency_s_median, 2)} s` : DASH],
              ['longest prompt', data.served.prompt_tokens_max != null ? `${data.served.prompt_tokens_max} tokens` : DASH],
            ]}
          />
        </div>
      </section>
    </>
  );
}

/** `/eval/:run/:task/:trial`: one scored conversation, every message, the customer's phone calls greyed. */
export function EvalConversationPage() {
  const { run = '', task = '', trial = 't1' } = useParams();
  const n = Number(trial.replace(/^t/, '')) || 1;
  const { data, error } = useGet<EvalConversation>(api.evalConversation(run, task, n));
  return (
    <>
      <p className="crumb">
        <Link to="/eval">eval</Link> / <Link to={evalRunPath(run)}>{run}</Link> / t{n}
      </p>
      <h1 className="qt">
        <TaskCell id={task} />
      </h1>
      {!data ? (
        <Loading error={error} what="the conversation" />
      ) : (
        <>
          <div className="row badges">
            <PassBadge pass={data.reward == null ? null : data.reward >= 1} />
            <span className="tag">
              reward {data.reward ?? DASH} · ended by {data.termination} · {data.messages.length} messages
            </span>
            {data.result?.error && <span className="badge bad">{data.result.error}</span>}
          </div>
          <RawView messages={data.messages} />
        </>
      )}
    </>
  );
}
