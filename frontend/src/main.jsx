import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import './styles.css';

const rootElement = document.getElementById('root');
window.addEventListener('error', event => {
  if (!rootElement || rootElement.childElementCount) return;
  rootElement.innerHTML = '<main style="max-width:680px;margin:10vh auto;padding:32px;font:16px system-ui;color:#20352f"><h1>VeriRAG could not load</h1><p>The page loaded, but the app script failed. Open DevTools Console for the error, then refresh.</p></main>';
});

class AppErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { failed: false }; }
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(error) { console.error('VeriRAG render error:', error); }
  render() {
    if (this.state.failed) return <main className="startup-error"><h1>VeriRAG hit a display error</h1><p>Refresh the page. If it happens again, open DevTools Console and share the first red error.</p></main>;
    return this.props.children;
  }
}

document.documentElement.dataset.appMounted = 'true';
createRoot(rootElement).render(<React.StrictMode><AppErrorBoundary><App /></AppErrorBoundary></React.StrictMode>);

