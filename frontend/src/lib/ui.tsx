/**
 * The pieces every page shares. Colour comes only from house.css tokens (classes in styles.css);
 * every status also carries a word and a glyph, never colour alone. Charts are inline SVG for
 * the marks and HTML for every label, so text never stretches with the plot.
 */
import { type ReactNode, useState } from 'react';
import { Link } from 'react-router-dom';
import type { ChatMessage, Purpose, RawMessage, RowStatus, Segment } from './api';
import { DASH, SYMPTOM, int, niceMax, num, taskParts } from './format';

// ── states ────────────────────────────────────────────────────────────────
export function ErrorNote({ error, what }: { error: string; what?: string }) {
  return (
    <div className="callout red" role="alert">
      <span className="badge bad">✕ error</span> {what ? `Could not load ${what}: ` : ''}
      <span className="mono">{error}</span>
    </div>
  );
}

export function Loading({ error, what }: { error: string | null; what?: string }) {
  if (error) return <ErrorNote error={error} what={what} />;
  return (
    <p className="busy" role="status" aria-live="polite">
      <span className="busy-dot" aria-hidden="true" /> Loading{what ? ` ${what}` : ''}…
    </p>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

/** The normal state before a milestone has run: said plainly, with what fills it. */
export function NotYet({ what, by }: { what: string; by: string }) {
  return (
    <Empty>
      <p>
        <span className="badge">○ not yet</span> No {what} yet: <b>{by}</b> fills this.
      </p>
    </Empty>
  );
}

// ── badges ────────────────────────────────────────────────────────────────
export function StatusBadge({ status }: { status: RowStatus }) {
  return status === 'kept' ? <span className="badge ok">✓ kept</span> : <span className="badge warn">✕ dropped</span>;
}

export function PurposeBadge({ purpose }: { purpose: Purpose }) {
  return purpose === 'result' ? (
    <span className="badge rec">result</span>
  ) : (
    <span className="badge info" title="M2's proof of concept: its numbers are not results">
      smoke · not a result
    </span>
  );
}

export function PassBadge({ pass }: { pass: boolean | null | undefined }) {
  if (pass == null) return <span className="badge">{DASH}</span>;
  return pass ? <span className="badge ok">✓ pass</span> : <span className="badge bad">✕ fail</span>;
}

export function OkBadge({ ok, yes, no }: { ok: boolean | null | undefined; yes: string; no: string }) {
  if (ok == null) return <span className="badge">○ {no.replace(/^✕ /, '')}: not checked</span>;
  return ok ? <span className="badge ok">✓ {yes}</span> : <span className="badge bad">✕ {no}</span>;
}

export function MlflowLink({ url }: { url: string | null | undefined }) {
  if (!url)
    return (
      <span className="tag" title="The central MLflow is down, or this run was not logged">
        MLflow: no link
      </span>
    );
  return (
    <a className="tag" href={url} target="_blank" rel="noreferrer">
      open in MLflow ↗
    </a>
  );
}

/** A telecom task id, readable: symptom, its root issues, and the persona. */
export function TaskCell({ id, to }: { id: string; to?: string }) {
  const p = taskParts(id);
  const body = p ? (
    <>
      <span className="task-symptom">{SYMPTOM[p.symptom] ?? p.symptom}</span>
      <span className="task-issues">{p.issues.join(' · ')}</span>
      <span className="tag">persona {p.persona}</span>
    </>
  ) : (
    <span className="mono">{id}</span>
  );
  return (
    <span className="taskcell" title={id}>
      {to ? <Link to={to}>{body}</Link> : body}
    </span>
  );
}

export function Choice<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: [T, string][];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="choice" role="group" aria-label={label}>
      <span className="tag">{label}</span>
      {options.map(([v, text]) => (
        <button key={v} type="button" className="chipbtn" aria-pressed={v === value} onClick={() => onChange(v)}>
          {text}
        </button>
      ))}
    </div>
  );
}

export function KV({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="kv">
      {rows.map(([k, v]) => (
        <div key={k} className="kvrow">
          <dt>{k}</dt>
          <dd>{v ?? DASH}</dd>
        </div>
      ))}
    </dl>
  );
}

// ── charts ────────────────────────────────────────────────────────────────
export type Series = { name: string; points: [number, number][]; alt?: boolean };

/**
 * Lines over steps. `marks` are vertical lines (epoch ends). With two series, the second is
 * drawn dashed in ink (the `?vs=` overlay), and the legend names both.
 */
export function LineChart({
  title,
  series,
  marks = [],
  caption,
  digits = 3,
  xLabel = 'steps',
}: {
  title: string;
  series: Series[];
  marks?: { x: number; label: string }[];
  caption?: ReactNode;
  digits?: number;
  xLabel?: string;
}) {
  const W = 600;
  const H = 160;
  const all = series.flatMap((s) => s.points);
  if (!all.length)
    return (
      <figure className="fig linechart">
        <div className="fig-title">{title}</div>
        <p className="tag">no points logged</p>
      </figure>
    );
  const xMax = Math.max(...all.map((p) => p[0]), 1);
  const yMax = niceMax(Math.max(...all.map((p) => p[1])));
  const x = (v: number) => (v / xMax) * W;
  const y = (v: number) => H - Math.min(Math.max(v / yMax, 0), 1) * H;
  const path = (pts: [number, number][]) =>
    pts.map(([a, b], i) => `${i ? 'L' : 'M'}${x(a).toFixed(1)},${y(b).toFixed(1)}`).join(' ');
  const ticks: [number, string][] = [0, 0.5, 1].map((f) => [f * yMax, num(f * yMax, f === 0 ? 0 : digits)]);
  return (
    <figure className="fig linechart">
      <div className="fig-title">{title}</div>
      {series.length > 1 && (
        <div className="legend">
          {series.map((s) => (
            <span key={s.name} className={`lg ${s.alt ? 'alt' : ''}`}>
              <span className="sw" aria-hidden="true" /> {s.name}
            </span>
          ))}
        </div>
      )}
      <div className="plot">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label={title}>
          {marks.map((m) => (
            <line key={m.x} className="vmark" x1={x(m.x)} x2={x(m.x)} y1={0} y2={H} vectorEffect="non-scaling-stroke" />
          ))}
          {series.map((s) => (
            <g key={s.name}>
              <path className={`line ${s.alt ? 'alt' : ''}`} d={path(s.points)} vectorEffect="non-scaling-stroke" />
              {s.points.length < 40 &&
                s.points.map(([a, b]) => (
                  <circle key={a} className={`dot ${s.alt ? 'alt' : ''}`} cx={x(a)} cy={y(b)} r={2.5} vectorEffect="non-scaling-stroke">
                    <title>
                      {s.name} · step {a}: {num(b, digits)}
                    </title>
                  </circle>
                ))}
            </g>
          ))}
        </svg>
        {ticks.map(([v, label]) => (
          <span key={label} className="ytick tag" style={{ top: `${((y(v) / H) * 100).toFixed(2)}%` }}>
            {label}
          </span>
        ))}
        {marks
          .filter((m) => m.label)
          .map((m) => (
            <span key={m.x} className={`mlabel tag ${m.x > xMax * 0.9 ? 'end' : ''}`} style={{ left: `${((m.x / xMax) * 100).toFixed(2)}%` }}>
              {m.label}
            </span>
          ))}
      </div>
      <div className="xaxis tag">
        <span style={{ left: '0%' }}>0</span>
        <span style={{ left: '100%' }}>
          {int(xMax)} {xLabel}
        </span>
      </div>
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  );
}

/** Bars from a histogram (`from` = bin start), with an optional limit line (F10's). */
export function Histogram({
  title,
  bins,
  binWidth,
  limit,
  caption,
  unit = 'tokens',
}: {
  title: string;
  bins: { from: number; n: number }[];
  binWidth: number;
  limit?: number;
  caption?: ReactNode;
  unit?: string;
}) {
  const W = 600;
  const H = 110;
  if (!bins.length) return <NotYet what="length histogram" by="a dataset build" />;
  const lo = bins[0].from;
  const hi = Math.max(bins[bins.length - 1].from + binWidth, limit && limit < lo + binWidth * 40 ? limit + binWidth : 0);
  const top = Math.max(...bins.map((b) => b.n), 1);
  const x = (v: number) => ((v - lo) / (hi - lo)) * W;
  return (
    <figure className="fig hist">
      <div className="fig-title">{title}</div>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label={title}>
        {bins.map((b) => {
          const h = b.n ? Math.max((b.n / top) * (H - 6), 1) : 0;
          return (
            <rect key={b.from} className="hbar" x={x(b.from) + 1} y={H - h} width={Math.max(x(b.from + binWidth) - x(b.from) - 2, 1)} height={h}>
              <title>
                {int(b.from)}–{int(b.from + binWidth)} {unit}: {b.n}
              </title>
            </rect>
          );
        })}
        {limit != null && limit <= hi && (
          <line className="hmark limit" x1={x(limit)} x2={x(limit)} y1={0} y2={H} vectorEffect="non-scaling-stroke" />
        )}
      </svg>
      <div className="haxis tag">
        <span>{int(lo)}</span>
        <span>
          {int(hi)} {unit}
        </span>
      </div>
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  );
}

/** The F1 → F10 funnel: each rule's input, what it dropped, and what went on. */
export function Funnel({ rows, total }: { rows: { rule: string; keep_if: string; in: number; dropped: number }[]; total: number }) {
  return (
    <ol className="funnel" aria-label="F1 to F10, in order">
      {rows.map((r) => {
        const kept = r.in - r.dropped;
        return (
          <li key={r.rule}>
            <span className="fr mono">{r.rule}</span>
            <span className="fbar" aria-hidden="true">
              <span className="fk" style={{ width: `${(kept / Math.max(total, 1)) * 100}%` }} />
              <span className="fd" style={{ width: `${(r.dropped / Math.max(total, 1)) * 100}%` }} />
            </span>
            <span className="fn mono">
              {r.dropped ? `−${r.dropped}` : '0'} → {kept}
            </span>
            <span className="fw">{r.keep_if}</span>
          </li>
        );
      })}
    </ol>
  );
}

/** A row of horizontal bars, label: value. */
export function Bars({ rows, caption }: { rows: { label: ReactNode; n: number; key: string; flag?: boolean; title?: string }[]; caption?: ReactNode }) {
  const top = Math.max(...rows.map((r) => r.n), 1);
  return (
    <figure className="fig">
      <ul className="bars">
        {rows.map((r) => (
          <li key={r.key} className={r.flag ? 'flag' : ''} title={r.title}>
            <span className="bl">{r.label}</span>
            <span className="bb" aria-hidden="true">
              <span style={{ width: `${(r.n / top) * 100}%` }} />
            </span>
            <span className="bn mono">{r.n}</span>
          </li>
        ))}
      </ul>
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  );
}

// ── conversations ─────────────────────────────────────────────────────────
const ROLE_LABEL: Record<string, string> = {
  system: 'system prompt',
  assistant: 'agent',
  user: 'customer',
  tool: 'tool result',
};

function Fold({ text, limit = 600 }: { text: string; limit?: number }) {
  const [open, setOpen] = useState(false);
  if (text.length <= limit) return <div className="msgtext">{text}</div>;
  return (
    <div className="msgtext">
      {open ? text : `${text.slice(0, limit)}…`}{' '}
      <button type="button" className="linkbtn" onClick={() => setOpen(!open)}>
        {open ? 'show less' : `show all ${int(text.length)} characters`}
      </button>
    </div>
  );
}

/** The chat lens: the conversation as the agent saw it (system, customer, agent, its tool results). */
export function ChatView({ messages }: { messages: ChatMessage[] }) {
  return (
    <ol className="convo">
      {messages.map((m, i) => (
        <li key={i} className={`msg ${m.role}`}>
          <div className="who">
            <span className="tag">{ROLE_LABEL[m.role] ?? m.role}</span>
            {m.role === 'assistant' && <span className="badge ok">trained</span>}
            {m.tool_call_id && <span className="tag">↳ {m.tool_call_id}</span>}
          </div>
          {m.content && <Fold text={m.content} limit={m.role === 'system' ? 400 : 1500} />}
          {m.tool_calls.map((tc, j) => (
            <pre key={j} className="call">
              <code>
                {tc.name}({tc.arguments})
              </code>
            </pre>
          ))}
        </li>
      ))}
    </ol>
  );
}

/** The raw lens: every message tau2 wrote, the ones the agent never saw greyed, a rule's breaks marked. */
export function RawView({ messages }: { messages: RawMessage[] }) {
  return (
    <ol className="convo raw">
      {messages.map((m) => (
        <li
          key={m.i}
          className={`msg ${m.role} ${m.in_agent_view ? '' : 'hidden-from-agent'} ${m.offending ? 'offending' : ''}`}
          id={`m${m.i}`}
        >
          <div className="who">
            <span className="tag mono">#{m.i}</span>
            <span className="tag">
              {m.role === 'tool' ? `tool result (for the ${m.requestor === 'user' ? 'customer' : 'agent'})` : (ROLE_LABEL[m.role] ?? m.role)}
              {m.role === 'user' && m.tool_calls.length ? ': phone tool call' : ''}
            </span>
            {!m.in_agent_view && <span className="badge">○ not in the agent's view</span>}
            {m.error && <span className="badge bad">✕ error</span>}
            {m.offending && <span className="badge bad">✕ broke the rule</span>}
          </div>
          {m.content && <Fold text={m.content} />}
          {m.tool_calls.map((tc, j) => (
            <pre key={j} className="call">
              <code>
                {tc.name}({JSON.stringify(tc.arguments)})
              </code>
            </pre>
          ))}
        </li>
      ))}
    </ol>
  );
}

/** The tokens lens: the exact text TRL trains on, the trained spans marked. */
export function TokensView({ segments, nTokens, nTrained }: { segments: Segment[]; nTokens: number; nTrained: number }) {
  const [showSystem, setShowSystem] = useState(false);
  const first = segments[0];
  const rest = first && !first.trained && first.text.length > 2000 && !showSystem ? segments.slice(1) : segments;
  return (
    <figure className="fig tokens">
      <div className="legend">
        <span className="lg trained">
          <span className="sw" aria-hidden="true" /> trained: the loss is on these tokens
        </span>
        <span className="lg">
          <span className="sw ctx" aria-hidden="true" /> context: read, never trained
        </span>
        <span className="tag">
          {int(nTrained)} of {int(nTokens)} tokens trained ({((nTrained / Math.max(nTokens, 1)) * 100).toFixed(1)}%)
        </span>
      </div>
      {rest !== segments && (
        <p>
          <button type="button" className="linkbtn" onClick={() => setShowSystem(true)}>
            show the system turn ({int(first.text.length)} characters: instructions, policy and the 13 tools)
          </button>
        </p>
      )}
      <pre className="tokentext">
        {rest.map((s, i) => (
          <span key={i} className={s.trained ? 'tk-trained' : 'tk-ctx'}>
            {s.text}
          </span>
        ))}
      </pre>
    </figure>
  );
}

/** The task × trial grid: one cell per conversation, pass or fail in a glyph as well as a colour. */
export function Grid({
  rows,
  trials,
  to,
}: {
  rows: { task_id: string; trials: number[] }[];
  trials: number;
  to: (task: string, trial: number) => string;
}) {
  return (
    <div className="tablewrap">
      <table className="grid-tt">
        <thead>
          <tr>
            <th>task</th>
            {Array.from({ length: trials }, (_, i) => (
              <th key={i} className="num">
                t{i + 1}
              </th>
            ))}
            <th className="num">passed</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.task_id}>
              <td>
                <TaskCell id={r.task_id} />
              </td>
              {Array.from({ length: trials }, (_, i) => {
                const v = r.trials[i];
                return (
                  <td key={i} className={`num cell ${v === 1 ? 'pass' : v === 0 ? 'fail' : ''}`}>
                    {v == null ? (
                      DASH
                    ) : (
                      <Link to={to(r.task_id, i + 1)} aria-label={`${r.task_id} trial ${i + 1}: ${v ? 'pass' : 'fail'}`}>
                        {v ? '✓' : '✕'}
                      </Link>
                    )}
                  </td>
                );
              })}
              <td className="num">
                {r.trials.reduce((a, b) => a + b, 0)} / {r.trials.length}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
