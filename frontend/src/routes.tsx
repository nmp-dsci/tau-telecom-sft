import { Link, type RouteObject, useRouteError } from 'react-router-dom';
import { Shell } from './Shell';
import { Data, DataVersion } from './pages/Data';
import { EvalConversationPage, EvalList, EvalRunPage } from './pages/Eval';
import { Method } from './pages/Method';
import { Overview } from './pages/Overview';
import { Review } from './pages/Review';
import { RowDetail, Rows } from './pages/Rows';
import { TrainList, TrainRunPage } from './pages/Train';

/**
 * Every viewer address, as data. tau2-loop's grammar: the path names the subject (a dataset
 * version, a row, a training run, a scored run, one conversation), the query string holds the
 * lens (lens, view, vs, rule, task, status, q), and a row opens inside its list
 * (`/data/:v/rows/:id` renders into the rows page).
 */
export const routes: RouteObject[] = [
  {
    id: 'shell',
    element: <Shell />,
    errorElement: <Crash />,
    children: [
      { id: 'overview', path: '/', element: <Overview /> },
      { id: 'data', path: '/data', element: <Data /> },
      { id: 'data-version', path: '/data/:v', element: <DataVersion /> },
      {
        id: 'rows',
        path: '/data/:v/rows',
        element: <Rows />,
        children: [{ id: 'row', path: ':id', element: <RowDetail /> }],
      },
      { id: 'review', path: '/data/:v/review', element: <Review /> },
      { id: 'train', path: '/train', element: <TrainList /> },
      { id: 'train-run', path: '/train/:run', element: <TrainRunPage /> },
      { id: 'eval', path: '/eval', element: <EvalList /> },
      { id: 'eval-run', path: '/eval/:run', element: <EvalRunPage /> },
      { id: 'eval-conversation', path: '/eval/:run/:task/:trial', element: <EvalConversationPage /> },
      { id: 'method', path: '/method', element: <Method /> },
      { id: 'missing', path: '*', element: <NotFound /> },
    ],
  },
];

function NotFound() {
  return (
    <div className="empty">
      <h1>No page here</h1>
      <p>
        Try the <Link to="/">Overview</Link> or the <Link to="/data">Data</Link>.
      </p>
    </div>
  );
}

function Crash() {
  const err = useRouteError() as Error | undefined;
  return (
    <main>
      <div className="callout red" role="alert">
        <span className="badge bad">✕ error</span> The viewer hit an error: <span className="mono">{err?.message ?? String(err)}</span>
      </div>
      <p>
        <a href="/">Back to the overview</a>
      </p>
    </main>
  );
}
