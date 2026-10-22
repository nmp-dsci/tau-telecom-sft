import React from 'react';
import ReactDOM from 'react-dom/client';
import { RouterProvider, createBrowserRouter } from 'react-router-dom';
// The house UI, vendored into ../.lavish: tokens and components first, then the project's one
// number (--hue, provisional until D8). house.css @imports fonts/plex.css with relative URLs, so
// Vite rebases them and bundles the IBM Plex woff2 files. Nothing in src/ declares a colour of
// its own (scripts/design_lint.mjs enforces it).
import '../../.lavish/house.css';
import '../../.lavish/project.css';
import './styles.css';
import { routes } from './routes';

// SHOWCASE_BASE (nmp-central-ai s06) sets Vite's base; the router follows it.
const basename = import.meta.env.BASE_URL.replace(/\/$/, '') || '/';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={createBrowserRouter(routes, { basename })} />
  </React.StrictMode>,
);
