import { Link, useParams, useSearchParams } from 'react-router-dom';
import { type Coverage, type DataVersion as DataVersionDoc, type DatasetSummary, api, useGet } from '../lib/api';
import { DASH, SYMPTOM, int, pct, shortSha, when } from '../lib/format';
import { Bars, Funnel, Histogram, KV, Loading, NotYet } from '../lib/ui';
import { dataVersionPath, reviewPath, rowsPath } from '../lib/url';

/** `/data`: every dataset version, newest last, each with its funnel's ends and its sign-off. */
export function Data() {
  const { data, error } = useGet<{ versions: DatasetSummary[] }>(api.data());
  if (!data) return <Loading error={error} what="the datasets" />;
  return (
    <>
      <p className="eyebrow">Data</p>
      <h1>Dataset versions</h1>
      <p className="lede">
        A version is append-only: a new rule means a new version. Each one is the board's conversations through the agent view and F1 to
        F10, written as one chat-and-tools row per conversation.
      </p>
      {data.versions.length === 0 ? (
        <NotYet what="dataset version" by="M1's make build" />
      ) : (
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th>version</th>
                <th className="num">kept</th>
                <th className="num">of</th>
                <th className="num">training tasks</th>
                <th className="num">tokens, median / max</th>
                <th>sources</th>
                <th>review</th>
                <th>train.jsonl</th>
              </tr>
            </thead>
            <tbody>
              {data.versions.map((v) => (
                <tr key={v.version}>
                  <td>
                    <Link to={dataVersionPath(v.version)}>{v.version}</Link>
                  </td>
                  <td className="num">{int(v.n_kept)}</td>
                  <td className="num">{int(v.n_loaded)}</td>
                  <td className="num">
                    {v.training_tasks_covered} / {v.training_tasks_total}
                  </td>
                  <td className="num">
                    {int(v.tokens?.median)} / {int(v.tokens?.max)}
                  </td>
                  <td className="tag">{v.sources.join(' · ')}</td>
                  <td>
                    <Link to={reviewPath(v.version)}>
                      {v.review.signed_off ? '✓ signed off' : `${v.review.reviewed} / ${v.review.total} read`}
                    </Link>
                  </td>
                  <td className="mono">{shortSha(v.train_sha256)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

/** `/data/:v[?view=coverage]`: one version's funnel, sources, lengths, and what it covers. */
export function DataVersion() {
  const { v = '' } = useParams();
  const [sp] = useSearchParams();
  const view = sp.get('view') === 'coverage' ? 'coverage' : 'funnel';
  const { data, error } = useGet<DataVersionDoc>(api.dataVersion(v));
  if (!data) return <Loading error={error} what={`dataset ${v}`} />;
  const st = data.stats;
  const s = data.summary;
  return (
    <>
      <p className="crumb">
        <Link to="/data">data</Link> / {v}
      </p>
      <h1>
        Dataset {v}: <em>{int(st.n_kept)}</em> clean conversations from {int(st.n_loaded)}
      </h1>
      <p className="lede">
        Built {when(st.built)} at code <code>{st.code_sha}</code>. {st.training_tasks_covered} of the {st.training_tasks_total} training tasks
        have at least one row. Counted with the {st.token_counter}.
      </p>
      <div className="row badges">
        <Link className="btn btn-sm" to={rowsPath(v)}>
          Browse the rows
        </Link>
        <Link className="btn btn-sm" to={reviewPath(v)}>
          {s.review.signed_off ? '✓ Signed off' : `Review the sample (${s.review.reviewed} / ${s.review.total})`}
        </Link>
        {!s.rows_here && <span className="badge warn">train.jsonl is not on this machine: kept rows cannot open</span>}
      </div>
      <nav className="lenses" aria-label="View">
        <Link to={dataVersionPath(v)} aria-current={view === 'funnel'}>
          funnel
        </Link>
        <Link to={dataVersionPath(v, 'coverage')} aria-current={view === 'coverage'}>
          coverage
        </Link>
      </nav>
      {view === 'coverage' ? <CoverageView v={v} /> : <FunnelView doc={data} />}
    </>
  );
}

function FunnelView({ doc }: { doc: DataVersionDoc }) {
  const st = doc.stats;
  const v = st.dataset;
  const reasons = Object.entries(st.drop_reasons).sort((a, b) => b[1] - a[1]);
  return (
    <>
      <section aria-labelledby="funnel-h">
        <span className="eyebrow">F1 → F10, in order (plan §5)</span>
        <h2 id="funnel-h">Every drop is charged to the first rule it failed</h2>
        <Funnel rows={st.filters} total={st.n_loaded} />
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th>rule</th>
                <th>keep a conversation only if</th>
                <th className="num">in</th>
                <th className="num">dropped</th>
                <th className="num">kept</th>
              </tr>
            </thead>
            <tbody>
              {st.filters.map((f) => (
                <tr key={f.rule}>
                  <td className="k">{f.rule}</td>
                  <td>{f.keep_if}</td>
                  <td className="num">{f.in}</td>
                  <td className="num">
                    {f.dropped ? (
                      <Link to={rowsPath(v, { status: 'dropped', rule: f.rule })}>{f.dropped}</Link>
                    ) : (
                      0
                    )}
                  </td>
                  <td className="num">{f.in - f.dropped}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="sub">
          {doc.withheld} of the drops are test tasks or their twins: they are counted here and never listed (AGENTS.md rule 1).
        </p>
        <details className="facts">
          <summary>Why they were dropped ({reasons.length} reasons)</summary>
          <div className="tablewrap">
            <table>
              <tbody>
                {reasons.map(([k, n]) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td className="num">{n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </section>

      <section aria-labelledby="len-h">
        <span className="eyebrow">Lengths, in the student's tokens</span>
        <h2 id="len-h">
          Median {int(st.tokens.median)}, max {int(st.tokens.max)} against F10's limit of {int(st.max_tokens)}
        </h2>
        <div className="charts">
          <Histogram
            title="Kept rows by length (1,024-token bins)"
            bins={st.tokens_hist ?? []}
            binWidth={1024}
            limit={st.max_tokens}
            caption={
              <>
                The system prompt and the 13 tools alone take <b>{int(st.tokens.prompt_and_tools)}</b> tokens. p90 {int(st.tokens.p90)}, p99{' '}
                {int(st.tokens.p99)}. The dashed line is F10's limit, when it falls inside the axis.
              </>
            }
          />
          {st.over_limit && (
            <div className="tablewrap">
              <table>
                <caption className="tag">What a limit would cut, of the {st.reached_f10} rows that reached F10</caption>
                <thead>
                  <tr>
                    <th className="num">limit</th>
                    <th className="num">rows over it</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(st.over_limit).map(([k, n]) => (
                    <tr key={k}>
                      <td className="num">{int(Number(k))}</td>
                      <td className="num">{n}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      <section aria-labelledby="kept-h">
        <span className="eyebrow">What was kept</span>
        <h2 id="kept-h">By source and by symptom</h2>
        <div className="cols">
          <Bars
            rows={Object.entries(st.kept_by_source).map(([k, n]) => ({
              key: k,
              n,
              label: <Link to={rowsPath(v, { status: 'kept', source: k.split(' ')[0] })}>{k}</Link>,
            }))}
            caption="Rows per teacher"
          />
          <Bars
            rows={Object.entries(st.kept_by_symptom).map(([k, n]) => ({
              key: k,
              n,
              label: <Link to={rowsPath(v, { status: 'kept', symptom: k })}>{SYMPTOM[k] ?? k}</Link>,
            }))}
            caption="Rows per symptom"
          />
        </div>
      </section>

      <section aria-labelledby="src-h">
        <span className="eyebrow">Sources and the bytes this version promises</span>
        <h2 id="src-h">Where the rows came from</h2>
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th>id</th>
                <th>source</th>
                <th>file</th>
                <th className="num">bytes</th>
                <th>sha256</th>
                <th className="num">conversations</th>
              </tr>
            </thead>
            <tbody>
              {st.sources.map((s) => (
                <tr key={s.id}>
                  <td className="k">{s.id}</td>
                  <td>{s.label}</td>
                  <td className="mono">{s.file}</td>
                  <td className="num">{int(s.bytes)}</td>
                  <td className="mono">{shortSha(s.sha256, 16)}</td>
                  <td className="num">{s.conversations}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <KV
          rows={[
            ['train.jsonl sha256', <code key="t">{st.train_sha256}</code>],
            ['split', `v${st.split_version}, sha256 ${shortSha(st.split_sha256, 16)}`],
            ['tau2 pin', <code key="p">{st.tau2_pin.slice(0, 12)}</code>],
            ['system prompt sha256', <code key="s">{st.system_prompt_sha256}</code>],
            ['tools sha256', <code key="o">{st.tools_sha256}</code>],
          ]}
        />
        {st.source_checks && st.source_checks.length > 0 && <SourceChecks checks={st.source_checks} />}
      </section>

      <details className="facts">
        <summary>The dataset card, as committed (card.md)</summary>
        <pre>{doc.card}</pre>
      </details>
    </>
  );
}

const CHECKS: [string, string][] = [
  ['agent_llm', 'agent model, as tau2 recorded it'],
  ['agent_llm_args', 'agent sampling'],
  ['customer_llm', 'customer model'],
  ['phone_tool_results', 'customer phone-tool results (removed from every row)'],
  ['agent_tool_results', 'agent tool results'],
  ['assistant_turns_with_reasoning', 'assistant turns with reasoning in raw_data (dropped)'],
  ['assistant_turns_with_text_and_calls', 'assistant turns with text and calls (kept as is)'],
  ['system_prompt_stored', 'system prompt stored in the file'],
  ['policy_matches_pin', "the file's policy is the pin's"],
  ['licence', 'licence'],
];

function SourceChecks({ checks }: { checks: Record<string, unknown>[] }) {
  const show = (x: unknown) => (x == null ? DASH : typeof x === 'object' ? JSON.stringify(x) : String(x));
  return (
    <>
      <h3>The five checks after downloading (plan §4)</h3>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>check</th>
              {checks.map((c) => (
                <th key={String(c.source)}>
                  {String(c.source)} {String(c.model ?? '')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {CHECKS.map(([k, label]) => (
              <tr key={k}>
                <td>{label}</td>
                {checks.map((c) => (
                  <td key={String(c.source)} className="mono">
                    {show(c[k])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function CoverageView({ v }: { v: string }) {
  const { data, error } = useGet<Coverage>(api.coverage(v));
  if (!data) return <Loading error={error} what="coverage" />;
  const thin = data.issues.filter((i) => i.thin);
  const needed = data.issues.filter((i) => i.test_tasks > 0);
  return (
    <section aria-labelledby="cov-h">
      <span className="eyebrow">Coverage: what M7's trigger reads</span>
      <h2 id="cov-h">
        {needed.length} root issues appear in the test tasks; {thin.length} of them have fewer than {data.thin_below} kept rows
      </h2>
      <p className="sub">
        Every test task recombines root issues the training tasks teach (plan §1). This lens counts kept rows per root issue against how many
        test tasks need it. Test tasks are counted, never named.
      </p>
      <Bars
        rows={data.issues.map((i) => ({
          key: i.issue,
          n: i.kept_rows,
          flag: i.thin,
          title: `${i.kept_rows} kept rows · ${i.train_tasks} training tasks · ${i.test_tasks} test tasks need it`,
          label: (
            <span>
              <span className="mono">{i.issue}</span> <span className="tag">· {i.test_tasks} test</span>
            </span>
          ),
        }))}
        caption={`Kept rows per root issue, most-needed by the test first. ⚑ marks an issue the test needs with under ${data.thin_below} rows.`}
      />
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>root issue</th>
              <th className="num">kept rows</th>
              <th className="num">training tasks with it</th>
              <th className="num">test tasks that need it</th>
              <th>thin?</th>
            </tr>
          </thead>
          <tbody>
            {data.issues.map((i) => (
              <tr key={i.issue}>
                <td className="mono">{i.issue}</td>
                <td className="num">{i.kept_rows}</td>
                <td className="num">{i.train_tasks}</td>
                <td className="num">{i.test_tasks}</td>
                <td>{i.thin ? <span className="badge warn">⚑ thin</span> : i.test_tasks ? <span className="badge ok">✓ covered</span> : DASH}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h3>By symptom</h3>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>symptom</th>
              <th className="num">kept rows</th>
              <th className="num">test tasks</th>
              <th className="num">rows per test task</th>
            </tr>
          </thead>
          <tbody>
            {data.symptoms.map((s) => (
              <tr key={s.symptom}>
                <td>{SYMPTOM[s.symptom] ?? s.symptom}</td>
                <td className="num">{s.kept_rows}</td>
                <td className="num">{s.test_tasks}</td>
                <td className="num">{s.test_tasks ? (s.kept_rows / s.test_tasks).toFixed(1) : DASH}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="tag">
        {data.tasks_covered} / {data.tasks_total} training tasks covered ({pct(data.tasks_covered / Math.max(data.tasks_total, 1))}).
      </p>
    </section>
  );
}
