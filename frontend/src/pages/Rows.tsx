import { type FormEvent, useEffect, useState } from 'react';
import { Link, Outlet, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { type ChatMessage, type RawMessage, type RowDetail as RowDetailDoc, type RowList, type RowStatus, api, useGet } from '../lib/api';
import { DASH, SYMPTOM, int } from '../lib/format';
import { ChatView, Choice, Empty, Loading, RawView, StatusBadge, TaskCell, TokensView } from '../lib/ui';
import { LENSES, type RowsLens, dataVersionPath, isLens, reviewPath, rowPath, rowsPath } from '../lib/url';

const PAGE = 50;

function lensOf(sp: URLSearchParams): RowsLens {
  const status = sp.get('status');
  return {
    status: status === 'kept' || status === 'dropped' ? (status as RowStatus) : 'all',
    rule: sp.get('rule'),
    task: sp.get('task'),
    symptom: sp.get('symptom'),
    source: sp.get('source'),
    q: sp.get('q'),
    offset: Number(sp.get('offset') ?? 0) || 0,
  };
}

/** `/data/:v/rows`: kept and dropped conversations, filtered by the query; a row opens above the list. */
export function Rows() {
  const { v = '', id } = useParams();
  const [sp] = useSearchParams();
  const lens = lensOf(sp);
  const navigate = useNavigate();
  const { data, error } = useGet<RowList>(
    api.rows(v, { status: lens.status ?? 'all', rule: lens.rule, task: lens.task, symptom: lens.symptom, source: lens.source, q: lens.q, offset: lens.offset ?? 0, limit: PAGE }),
  );
  const [q, setQ] = useState(lens.q ?? '');
  useEffect(() => setQ(lens.q ?? ''), [lens.q]);
  const go = (next: RowsLens) => navigate(rowsPath(v, { ...lens, offset: 0, ...next }));
  const submit = (e: FormEvent) => {
    e.preventDefault();
    go({ q: q || null });
  };
  return (
    <>
      <p className="crumb">
        <Link to="/data">data</Link> / <Link to={dataVersionPath(v)}>{v}</Link> / rows
      </p>
      <h1 className={id ? 'qt' : undefined}>The rows of {v}</h1>
      {id && <Outlet />}
      {!data ? (
        <Loading error={error} what="the rows" />
      ) : (
        <section aria-labelledby="list-h" className={id ? '' : 'first'}>
          <h2 id="list-h" className="qt">
            {int(data.total)} {lens.status === 'all' ? 'rows' : `${lens.status} rows`}
            {lens.rule ? ` dropped by ${lens.rule}` : ''}
          </h2>
          <p className="sub">
            {int(data.counts.kept)} kept and {int(data.counts.dropped)} dropped are listed; {int(data.counts.withheld)} dropped test-task
            conversations are counted, never listed. {!data.rows_here && 'train.jsonl is not on this machine, so kept rows list nothing.'}
          </p>
          <div className="toolbar">
            <Choice
              label="status"
              options={[
                ['all', 'all'],
                ['kept', 'kept'],
                ['dropped', 'dropped'],
              ]}
              value={lens.status ?? 'all'}
              onChange={(s) => go({ status: s, rule: s === 'kept' ? null : lens.rule })}
            />
            <label className="choice">
              <span className="tag">rule</span>
              <select value={lens.rule ?? ''} onChange={(e) => go({ rule: e.target.value || null, status: e.target.value ? 'dropped' : lens.status })}>
                <option value="">any</option>
                {data.rules.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
            </label>
            <label className="choice">
              <span className="tag">symptom</span>
              <select value={lens.symptom ?? ''} onChange={(e) => go({ symptom: e.target.value || null })}>
                <option value="">any</option>
                {data.symptoms.map((s) => (
                  <option key={s} value={s}>
                    {SYMPTOM[s] ?? s}
                  </option>
                ))}
              </select>
            </label>
            <label className="choice">
              <span className="tag">source</span>
              <select value={lens.source ?? ''} onChange={(e) => go({ source: e.target.value || null })}>
                <option value="">any</option>
                {data.sources.map((s) => (
                  <option key={s} value={s.split(' ')[0]}>
                    {s}
                  </option>
                ))}
              </select>
            </label>
            <form className="searchbox" role="search" onSubmit={submit}>
              <input type="search" aria-label="Search task, row id or reason" placeholder="task, row id or reason…" value={q} onChange={(e) => setQ(e.target.value)} />
            </form>
            {lens.task && (
              <span className="badge info">
                task filter on{' '}
                <button type="button" className="linkbtn" onClick={() => go({ task: null })}>
                  clear
                </button>
              </span>
            )}
          </div>
          {data.items.length === 0 ? (
            <Empty>
              <p>No rows match.</p>
            </Empty>
          ) : (
            <div className="tablewrap">
              <table>
                <thead>
                  <tr>
                    <th>row</th>
                    <th>task</th>
                    <th>source</th>
                    <th className="num">trial</th>
                    <th className="num">agent turns</th>
                    <th>why dropped</th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((r) => (
                    <tr key={r.sim_id} className={r.sim_id === id ? 'open' : ''}>
                      <td>
                        <StatusBadge status={r.status} />{' '}
                        <Link className="mono" to={rowPath(v, r.sim_id, null, lens)}>
                          {r.sim_id.slice(0, 8)}
                        </Link>
                      </td>
                      <td>
                        <TaskCell id={r.task_id} to={rowsPath(v, { ...lens, task: r.task_id, offset: 0 })} />
                      </td>
                      <td className="tag">{r.source}</td>
                      <td className="num">{r.trial ?? DASH}</td>
                      <td className="num">{r.agent_turns ?? DASH}</td>
                      <td>{r.rule ? `${r.rule}: ${r.reason}` : DASH}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="pager">
            <button type="button" className="ghost" disabled={(lens.offset ?? 0) === 0} onClick={() => go({ offset: Math.max(0, (lens.offset ?? 0) - PAGE) })}>
              ← previous
            </button>
            <span className="tag">
              {data.total ? (lens.offset ?? 0) + 1 : 0}–{Math.min((lens.offset ?? 0) + PAGE, data.total)} of {data.total}
            </span>
            <button type="button" className="ghost" disabled={(lens.offset ?? 0) + PAGE >= data.total} onClick={() => go({ offset: (lens.offset ?? 0) + PAGE })}>
              next →
            </button>
          </div>
        </section>
      )}
    </>
  );
}

/** `/data/:v/rows/:id?lens=chat|tokens|raw`: one conversation, three ways. */
export function RowDetail() {
  const { v = '', id = '' } = useParams();
  const [sp] = useSearchParams();
  const raw = sp.get('lens');
  const lens = isLens(raw) ? raw : 'chat';
  const keep = lensOf(sp);
  const { data, error } = useGet<RowDetailDoc>(api.row(v, id, lens));
  return (
    <section aria-labelledby="row-h" className="detail-pane">
      <h2 id="row-h" className="qt">
        Row <span className="mono">{id.slice(0, 8)}</span>
      </h2>
      <nav className="lenses" aria-label="Lens">
        {LENSES.map((l) => (
          <Link key={l} to={rowPath(v, id, l, keep)} aria-current={l === lens}>
            {l === 'chat' ? 'chat: as the agent saw it' : l === 'tokens' ? 'tokens: what TRL trains on' : 'raw: the tau2 conversation'}
          </Link>
        ))}
        <Link to={rowsPath(v, keep)}>✕ close</Link>
      </nav>
      {!data ? (
        <Loading error={error} what="the row" />
      ) : (
        <>
          <div className="row badges">
            <StatusBadge status={data.status} />
            {data.drop && (
              <span className="badge bad">
                {data.drop.rule}: {data.drop.reason}
              </span>
            )}
            {data.in_review_sample && (
              <Link className="badge info" to={reviewPath(v)}>
                in the review sample →
              </Link>
            )}
            <span className="tag">
              {data.meta.source} · trial {data.meta.trial ?? DASH} · {data.meta.agent_turns ?? DASH} agent turns
            </span>
          </div>
          <p>
            <TaskCell id={data.meta.task_id} />
          </p>
          {data.status === 'dropped' && lens !== 'raw' && (
            <div className="callout warn">
              <p>
                This conversation was dropped, so none of it is trained on. The {lens} lens shows what its row would have been; the raw lens
                marks the turn that broke {data.drop?.rule ?? 'its rule'} when the rule is about one turn (F4, F5, F7).
              </p>
            </div>
          )}
          {data.missing && (
            <div className="callout warn">
              <p>{data.missing}</p>
            </div>
          )}
          {lens === 'chat' && data.messages && (
            <>
              <p className="sub">
                The agent's view: tau2's stock system prompt, the customer's words, the agent's turns and its own tool results. The{' '}
                {data.n_tools ?? 13} tool schemas ride beside the messages, in <code>tools</code>.
              </p>
              <ChatView messages={data.messages as ChatMessage[]} />
            </>
          )}
          {lens === 'raw' && data.messages && (
            <>
              <p className="sub">
                Every message tau2 wrote: {data.messages.length} in all, {(data.messages as RawMessage[]).filter((m) => !m.in_agent_view).length} of
                them greyed because the agent never saw them (the customer's phone calls and their results). Ended by{' '}
                <code>{data.termination}</code>, reward {data.reward ?? DASH}.
              </p>
              <RawView messages={data.messages as RawMessage[]} />
            </>
          )}
          {lens === 'tokens' && data.tokens && (
            <>
              <p className="sub">
                The row rendered by {data.tokens.template} for <code>{data.tokens.tokenizer}</code>, cut where the assistant mask changes.
                Highlighted text is what the loss is on; everything else is context.
              </p>
              <TokensView segments={data.tokens.segments} nTokens={data.tokens.n_tokens} nTrained={data.tokens.n_trained} />
            </>
          )}
        </>
      )}
    </section>
  );
}
