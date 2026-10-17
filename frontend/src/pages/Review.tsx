import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { type ReviewItem, type ReviewStatus, type Verdict, api, post, useGet } from '../lib/api';
import { DASH, when } from '../lib/format';
import { ErrorNote, Loading, StatusBadge, TaskCell } from '../lib/ui';
import { dataVersionPath, rowPath } from '../lib/url';

const VERDICT_TEXT: Record<Verdict, string> = { good: '✓ good', bad: '✕ bad', note: '✎ note' };

/**
 * `/data/:v/review`: M1's sign-off. Ten kept and ten dropped rows drawn with a fixed seed; each
 * gets good, bad or a note, then the person signs off once. Saved to data/sft/<v>/review.jsonl,
 * append-only (D10). Nothing here is pre-filled: the sign-off is the person's.
 */
export function Review() {
  const { v = '' } = useParams();
  const { data, error, reload } = useGet<ReviewStatus>(api.review(v));
  const [saveError, setSaveError] = useState<string | null>(null);
  const [signNote, setSignNote] = useState('');
  if (!data) return <Loading error={error} what="the review sample" />;
  const save = async (body: Record<string, unknown>) => {
    setSaveError(null);
    try {
      await post(api.review(v), body);
      reload();
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e));
    }
  };
  const ready = data.reviewed === data.total && data.total > 0;
  return (
    <>
      <p className="crumb">
        <Link to="/data">data</Link> / <Link to={dataVersionPath(v)}>{v}</Link> / review
      </p>
      <h1>
        Read what the model will learn: <em>{data.reviewed}</em> of {data.total} rows reviewed
      </h1>
      <p className="lede">
        M1 is done when a person has read 10 kept and 10 dropped conversations and signed off. The sample was drawn on {when(data.drawn)} with
        seed {data.seed}; dropped rows were drawn one rule at a time. Open each row, read it in the chat and raw lenses, then mark it.
      </p>
      {saveError && <ErrorNote error={saveError} what="the verdict" />}
      {data.signed_off ? (
        <div className="callout" role="status">
          <p>
            <span className="badge ok">✓ signed off</span> {when(data.signed_off.at)}
            {data.signed_off.note ? ` · “${data.signed_off.note}”` : ''}
          </p>
        </div>
      ) : (
        <div className="callout warn" role="status">
          <p>
            <span className="badge warn">○ not signed off</span> {data.total - data.reviewed} row(s) still need a verdict.
          </p>
        </div>
      )}
      {(['kept', 'dropped'] as const).map((status) => (
        <section key={status} aria-labelledby={`rv-${status}`}>
          <h2 id={`rv-${status}`} className="qt">
            {status === 'kept' ? 'Ten kept rows: is each one worth copying?' : 'Ten dropped rows: was each one right to drop?'}
          </h2>
          <ol className="review">
            {data.items
              .filter((i) => i.status === status)
              .map((item) => (
                <ReviewRow key={item.sim_id} v={v} item={item} verdicts={data.verdicts} onSave={save} />
              ))}
          </ol>
        </section>
      ))}
      <section className="signoff" aria-labelledby="so-h">
        <h2 id="so-h" className="qt">
          Sign off
        </h2>
        <p className="sub">
          The sign-off is one line in review.jsonl with the time. It is accepted only once every sampled row has a verdict.
        </p>
        <textarea aria-label="Sign-off note" placeholder="optional note" value={signNote} onChange={(e) => setSignNote(e.target.value)} />
        <div className="row">
          <button type="button" disabled={!ready || Boolean(data.signed_off)} onClick={() => save({ kind: 'sign-off', note: signNote })}>
            Sign off on {v}
          </button>
          {!ready && <span className="tag">needs {data.total - data.reviewed} more verdict(s)</span>}
        </div>
      </section>
      {data.log.length > 0 && (
        <details className="facts">
          <summary>review.jsonl ({data.log.length} lines)</summary>
          <pre>{data.log.map((r) => JSON.stringify(r)).join('\n')}</pre>
        </details>
      )}
    </>
  );
}

function ReviewRow({
  v,
  item,
  verdicts,
  onSave,
}: {
  v: string;
  item: ReviewItem;
  verdicts: Verdict[];
  onSave: (b: Record<string, unknown>) => Promise<void>;
}) {
  const [note, setNote] = useState(item.latest?.note ?? '');
  const [busy, setBusy] = useState(false);
  const mark = async (verdict: Verdict) => {
    setBusy(true);
    await onSave({ kind: 'verdict', sim_id: item.sim_id, verdict, note });
    setBusy(false);
  };
  return (
    <li className={item.latest ? 'done' : ''}>
      <div className="row">
        <StatusBadge status={item.status} />
        <Link className="mono" to={rowPath(v, item.sim_id)}>
          {item.sim_id.slice(0, 8)} · chat
        </Link>
        <Link className="tag" to={rowPath(v, item.sim_id, 'raw')}>
          raw
        </Link>
        <Link className="tag" to={rowPath(v, item.sim_id, 'tokens')}>
          tokens
        </Link>
        <span className="tag">
          {item.source} · trial {item.trial ?? DASH}
        </span>
        {item.rule && (
          <span className="badge warn">
            {item.rule}: {item.reason}
          </span>
        )}
      </div>
      <TaskCell id={item.task_id} />
      <textarea aria-label={`Note on ${item.sim_id}`} placeholder="what you noticed (optional)" value={note} onChange={(e) => setNote(e.target.value)} />
      <div className="row" role="group" aria-label={`Verdict on ${item.sim_id}`}>
        {verdicts.map((vd) => (
          <button
            key={vd}
            type="button"
            className={item.latest?.verdict === vd ? '' : 'ghost'}
            aria-pressed={item.latest?.verdict === vd}
            disabled={busy}
            onClick={() => mark(vd)}
          >
            {VERDICT_TEXT[vd]}
          </button>
        ))}
        {item.latest && <span className="tag">saved {when(item.latest.at)}</span>}
      </div>
    </li>
  );
}
