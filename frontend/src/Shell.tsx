import { NavLink, Outlet } from 'react-router-dom';
import { type Health, api, useGet } from './lib/api';

/** The tabs, in the order of the chain: where we are → the data → training → scoring → how. */
const NAV: [string, string][] = [
  ['/', 'Overview'],
  ['/data', 'Data'],
  ['/train', 'Train'],
  ['/eval', 'Eval'],
  ['/method', 'Method'],
];

export function Shell() {
  const { data: health, error } = useGet<Health>(api.health());
  return (
    <>
      <header className="appbar">
        <div className="in">
          <NavLink to="/" className="brand">
            tau-telecom-sft
          </NavLink>
          <nav aria-label="Pages">
            {NAV.map(([to, label]) => (
              <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => (isActive ? 'on' : '')}>
                {label}
              </NavLink>
            ))}
          </nav>
          <HealthPill health={health} error={error} />
        </div>
      </header>
      <main>
        <Outlet />
      </main>
      <footer className="appfoot">
        <div className="in">
          tau-telecom-sft · LoRA SFT of an open student on clean tau2 telecom conversations, scored in tau2-loop on split v3 ·
          on this Mac only (127.0.0.1) · build <code>{health?.code_sha ?? '…'}</code>
        </div>
      </footer>
    </>
  );
}

function HealthPill({ health, error }: { health: Health | null; error: string | null }) {
  if (error)
    return (
      <span className="pill" title={error}>
        <span className="badge bad">✕ api down</span>
      </span>
    );
  if (!health)
    return (
      <span className="pill">
        <span className="badge">api …</span>
      </span>
    );
  return (
    <span className="pill">
      <span className="badge ok" title="FastAPI on 127.0.0.1:8085">
        api ok · {health.code_sha}
      </span>
      <span className={`badge ${health.mlflow === 'up' ? 'ok' : 'warn'}`} title="The central MLflow: the viewer links to it, it never reads from it">
        mlflow {health.mlflow === 'up' ? '● up' : '○ down'}
      </span>
    </span>
  );
}
