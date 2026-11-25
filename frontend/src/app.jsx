import React, { useEffect, useMemo, useState } from 'react';
import './mobile.css';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api';
const HISTORY_KEY = 'verirag:conversations:v1';
const THEME_KEY = 'verirag:theme';

function loadConversations() {
  const value = readStorage(HISTORY_KEY, []);
  return Array.isArray(value) ? value.filter(item => item && typeof item.id === 'string' && Array.isArray(item.messages)) : [];
}

function readStorage(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch { return fallback; }
}

function Icon({ name, size = 18 }) {
  const common = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true };
  const paths = {
    plus: <><path d="M12 5v14M5 12h14" /></>,
    upload: <><path d="M12 16V4m0 0L7 9m5-5 5 5" /><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" /></>,
    file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6M8 13h8M8 17h8" /></>,
    chat: <><path d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8A8.5 8.5 0 0 1 8.7 3.9a8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8z" /></>,
    trash: <><path d="M3 6h18M8 6V4h8v2m3 0-1 14H6L5 6m4 4v6m6-6v6" /></>,
    sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42" /></>,
    moon: <path d="M20.9 13A9 9 0 0 1 11 3.1 9 9 0 1 0 20.9 13z" />,
    send: <><path d="m22 2-7 20-4-9-9-4 20-7Z" /><path d="M22 2 11 13" /></>,
    close: <><path d="m18 6-12 12M6 6l12 12" /></>,
    menu: <><path d="M4 6h16M4 12h16M4 18h16" /></>,
  };
  return <svg {...common}>{paths[name]}</svg>;
}

function App() {
  const [documents, setDocuments] = useState([]);
  const [conversations, setConversations] = useState(loadConversations);
  const [activeId, setActiveId] = useState(() => loadConversations()[0]?.id || null);
  const [question, setQuestion] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [verifier, setVerifier] = useState('heuristic');
  const [mode, setMode] = useState('live');
  const [pipeline, setPipeline] = useState('verirag');
  const [llmConfigured, setLlmConfigured] = useState(null);
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem(THEME_KEY) === 'dark' ? 'dark' : 'light'; }
    catch { return 'light'; }
  });
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const activeConversation = conversations.find(item => item.id === activeId) || null;
  const messages = activeConversation?.messages || [];
  const sortedConversations = useMemo(() => [...conversations].sort((a, b) => b.updatedAt - a.updatedAt), [conversations]);

  useEffect(() => { try { localStorage.setItem(HISTORY_KEY, JSON.stringify(conversations)); } catch { setError('Chat history could not be saved in this browser.'); } }, [conversations]);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem(THEME_KEY, theme); } catch { /* Theme remains active for this page. */ }
  }, [theme]);

  async function loadDocuments() {
    try {
      const response = await fetch(`${API}/documents`);
      if (!response.ok) throw new Error('Could not load documents.');
      setDocuments(await response.json());
      const health = await fetch(`${API}/health`);
      if (health.ok) {
        const status = await health.json();
        setLlmConfigured(status.llm_configured);
        if (!status.llm_configured) setMode('demo');
      }
      setError('');
    } catch (err) {
      setError(`${err.message} Start the VeriRAG backend, then refresh this page.`);
    }
  }

  useEffect(() => { loadDocuments(); }, []);

  function newChat() {
    setActiveId(null);
    setQuestion('');
    setError('');
  }

  async function upload(event) {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    setBusy(true); setError('');
    try {
      const form = new FormData(); form.append('file', file);
      const response = await fetch(`${API}/documents`, { method: 'POST', body: form });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || 'Upload failed.');
      await loadDocuments();
      setError(body.duplicate ? 'This exact PDF is already indexed.' : 'PDF uploaded and indexed.');
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  async function ask(event) {
    event.preventDefault();
    const prompt = question.trim();
    if (!prompt || busy) return;
    setBusy(true); setError('');
    try {
      const response = await fetch(`${API}/ask`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: prompt, verifier, mode, pipeline }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || 'The question could not be answered.');
      const now = Date.now();
      const id = activeId || `chat-${now}`;
      const existing = conversations.find(item => item.id === id);
      const chat = {
        id,
        title: existing?.title || prompt.slice(0, 48),
        createdAt: existing?.createdAt || now,
        updatedAt: now,
        messages: [...(existing?.messages || []), { question: prompt, result: body }],
      };
      setConversations(items => [chat, ...items.filter(item => item.id !== id)]);
      setActiveId(id);
      setQuestion('');
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  function deleteChat(event, id) {
    event.stopPropagation();
    setConversations(items => items.filter(item => item.id !== id));
    if (activeId === id) newChat();
  }

  return <main className="app-shell" data-theme={theme}>
    <aside className={`sidebar glass-panel ${mobileMenuOpen ? 'mobile-open' : ''}`}>
      <div className="brand-row"><a className="brand" href="#top"><span className="brand-mark">V</span><span>VeriRAG</span></a><span className="brand-caption">Research workspace</span><button className="icon-button mobile-close" onClick={() => setMobileMenuOpen(false)} aria-label="Close menu"><Icon name="close" size={16} /></button></div>
      <button className="new-chat-button" onClick={newChat}><Icon name="plus" size={17} /> New conversation</button>

      <section className="side-section document-section">
        <div className="side-heading"><span>Documents</span><span className="count-pill">{documents.length}</span></div>
        <label className={`upload-card ${busy ? 'is-disabled' : ''}`}>
          <input type="file" accept="application/pdf,.pdf" onChange={upload} disabled={busy} />
          <span className="upload-symbol"><Icon name="upload" /></span>
          <span><strong>{busy ? 'Processing…' : 'Upload a PDF'}</strong><small>PDF · up to 20 MB</small></span>
        </label>
        <div className="document-list">
          {documents.length === 0 ? <p className="side-empty">Your indexed PDFs will appear here.</p> : documents.map(doc => <div className="document-item" key={doc.document_id}>
            <Icon name="file" size={16} /><span className="document-name"><strong title={doc.filename}>{doc.filename}</strong><small>{doc.page_count} pages · {doc.chunk_count} passages</small></span>
          </div>)}
        </div>
      </section>

      <section className="side-section history-section">
        <div className="side-heading"><span>Recent chats</span><span className="count-pill">{conversations.length}</span></div>
        {sortedConversations.length === 0 ? <p className="side-empty">Your conversations will be saved here.</p> : <nav className="history-list" aria-label="Chat history">
          {sortedConversations.map(chat => <button key={chat.id} className={`history-item ${activeId === chat.id ? 'active' : ''}`} onClick={() => { setActiveId(chat.id); setError(''); }}>
            <Icon name="chat" size={15} /><span>{chat.title}</span><span className="delete-chat" role="button" tabIndex="0" aria-label={`Delete ${chat.title}`} onClick={event => deleteChat(event, chat.id)} onKeyDown={event => event.key === 'Enter' && deleteChat(event, chat.id)}><Icon name="trash" size={14} /></span>
          </button>)}
        </nav>}
      </section>

      <div className="sidebar-foot"><span className="status-dot" /> Local research prototype</div>
    </aside>

    <section className="main-area" id="top">
      <header className="topbar glass-panel">
        <div><p className="topbar-kicker">DOCUMENT INTELLIGENCE</p><span className="topbar-title">Ask your library</span></div>
        <div className="topbar-actions"><button className="icon-button mobile-menu-button" onClick={() => setMobileMenuOpen(true)} aria-label="Open documents and chat history"><Icon name="menu" size={17} /></button><span className={`connection-chip ${llmConfigured === false ? 'demo-chip' : ''}`}><span />{llmConfigured === false ? 'Demo ready' : 'Workspace'}</span>
          <button className="icon-button theme-button" onClick={() => setTheme(value => value === 'light' ? 'dark' : 'light')} aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`} title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}><Icon name={theme === 'light' ? 'moon' : 'sun'} /></button>
        </div>
      </header>

      {error && <div className={`notice ${error.includes('already') || error.includes('indexed.') ? 'success' : ''}`} role="status">{error}<button aria-label="Dismiss message" onClick={() => setError('')}><Icon name="close" size={15} /></button></div>}

      <div className="conversation-scroll">
        {messages.length === 0 ? <section className="welcome-state">
          <span className="welcome-orb"><Icon name="chat" size={25} /></span>
          <p className="eyebrow">Evidence first, claim by claim</p>
          <h1>What would you like<br />to understand?</h1>
          <p className="welcome-copy">Ask questions about your PDFs and inspect the sources behind every answer.</p>
          <div className="suggestion-grid">
            {['What does the retention section say?', 'How long are files kept and are they encrypted at rest?', 'Which cloud region stores these documents?'].map(suggestion => <button key={suggestion} onClick={() => setQuestion(suggestion)}>{suggestion}<span>↗</span></button>)}
          </div>
        </section> : <div className="message-list">
          {messages.map((message, index) => <article className="turn" key={`${activeId}-${index}`}>
            <div className="user-message"><span className="avatar user-avatar">A</span><div><small>You</small><p>{message.question}</p></div></div>
            <AnswerResult result={message.result} />
          </article>)}
        </div>}
      </div>

      <div className="composer-wrap">
        <form className="composer glass-panel" onSubmit={ask}>
          <label className="sr-only" htmlFor="question">Ask a question about your documents</label>
          <textarea id="question" value={question} onChange={event => setQuestion(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); ask(event); } }} placeholder={documents.length ? 'Ask anything about your documents…' : 'Upload a PDF, then ask a question…'} rows="2" maxLength="4000" />
          <div className="composer-bottom">
            <div className="composer-settings">
              <select aria-label="Pipeline" value={pipeline} onChange={event => setPipeline(event.target.value)}><option value="verirag">VeriRAG verification</option><option value="baseline">RAG baseline</option></select>
              <select aria-label="Answer mode" value={mode} onChange={event => setMode(event.target.value)}><option value="live">Live LLM</option><option value="demo">Deterministic demo</option></select>
              <select aria-label="Verification method" value={verifier} onChange={event => setVerifier(event.target.value)} disabled={mode === 'demo' || pipeline === 'baseline'}><option value="heuristic">Similarity heuristic</option><option value="llm">LLM verifier</option></select>
            </div>
            <button className="send-button" type="submit" disabled={busy || !question.trim()}>{busy ? <span className="spinner" /> : <Icon name="send" size={17} />}<span>{busy ? 'Thinking' : 'Ask'}</span></button>
          </div>
          <p className="composer-note">{mode === 'demo' ? 'Deterministic demo · scripted responses, no LLM call' : llmConfigured === false ? 'Live answers need an API key · demo mode is ready' : 'Answers are grounded in your indexed documents'}</p>
        </form>
        <p className="privacy-note">PDFs stay on this machine. Live LLM requests send retrieved passages to your configured provider.</p>
      </div>
    </section>
  </main>;
}

function AnswerResult({ result }) {
  return <div className="assistant-message">
    <div className="assistant-heading"><span className="avatar assistant-avatar">V</span><div><small>VeriRAG</small><span>{result.generation_mode === 'deterministic_demo' ? 'DETERMINISTIC DEMO · NOT LLM OUTPUT' : result.pipeline === 'baseline' ? 'CONVENTIONAL RAG BASELINE' : 'CLAIM VERIFICATION'}</span></div><time>{Math.round(result.latency_ms)} ms</time></div>
    <div className="answer-body">{result.answer}</div>
    <div className="answer-citations"><strong>Sources</strong>{result.citations.length ? result.citations.map(citation => <span className="citation-chip" key={citation.chunk_id}><Icon name="file" size={13} />{citation.filename} · p.{citation.page_number}</span>) : <span className="muted-inline">No source citation selected</span>}</div>
    {result.claims?.length > 0 && <section className="claim-results"><div className="claim-section-title"><strong>Claim checks</strong><span>{result.verifier_method.replaceAll('_', ' ')}</span></div>
      {result.claims.map((claim, index) => <details className="claim-card" key={`${index}-${claim.claim}`}>
        <summary><span className={`status-badge status-${claim.label.toLowerCase()}`}>{claim.label.replaceAll('_', ' ')}</span><span className="claim-summary">{claim.claim}</span><span className="expand-mark">⌄</span></summary>
        <p className="claim-explanation">{claim.explanation}</p>
        <EvidenceGroup title="Verifier-selected evidence" items={claim.evidence} selected />
        <EvidenceGroup title="Claim-specific retrieved passages" items={claim.retrieved_evidence} />
      </details>)}
    </section>}
    <details className="answer-evidence"><summary>Question-level passages ({result.retrieved_evidence.length})</summary>{result.retrieved_evidence.map(item => <EvidenceItem key={item.chunk_id} item={item} />)}</details>
  </div>;
}

function EvidenceGroup({ title, items = [], selected = false }) {
  return <div className="evidence-group"><strong>{title}</strong>{items.length ? items.map(item => <EvidenceItem key={item.chunk_id} item={item} selected={selected} />) : <p className="evidence-empty">No {selected ? 'evidence selected' : 'passages retrieved'}.</p>}</div>;
}

function EvidenceItem({ item, selected = false }) {
  return <details className="evidence-item"><summary><span>{item.filename} · page {item.page_number}</span>{item.similarity != null && <small>{item.similarity.toFixed(3)}</small>}</summary><p>{item.text}</p></details>;
}

export default App;
