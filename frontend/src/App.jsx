import { useEffect, useState } from 'react';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api';

function App() {
  const [documents, setDocuments] = useState([]);
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [verifier, setVerifier] = useState('heuristic');

  async function loadDocuments() {
    try {
      const response = await fetch(`${API}/documents`);
      if (!response.ok) throw new Error('Could not load the document list.');
      setDocuments(await response.json());
    } catch (err) { setError(`${err.message} Is the backend running?`); }
  }

  useEffect(() => { loadDocuments(); }, []);

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
      setError(body.duplicate ? 'This exact PDF is already indexed.' : 'PDF added and indexed.');
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  async function ask(event) {
    event.preventDefault();
    if (!question.trim()) return;
    setBusy(true); setError(''); setResult(null);
    try {
      const response = await fetch(`${API}/ask`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, verifier }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || 'The question could not be answered.');
      setResult(body);
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  return <main className="shell">
    <header className="topbar">
      <a className="brand" href="#top"><span className="brand-mark">V</span> VeriRAG</a>
      <span className="prototype-tag">Research prototype</span>
    </header>
    <section className="hero" id="top">
      <p className="eyebrow">Evidence first, claim by claim</p>
      <h1>Ask your documents.<br /><em>Inspect every answer.</em></h1>
      <p className="intro">A local document question-answering prototype that shows its sources and assesses the factual claims it makes.</p>
    </section>

    <section className="workspace">
      <aside className="sidebar panel">
        <div className="section-heading"><div><p className="eyebrow">Your library</p><h2>Documents</h2></div><span className="count">{documents.length}</span></div>
        <label className={`upload ${busy ? 'disabled' : ''}`}>
          <input type="file" accept="application/pdf,.pdf" onChange={upload} disabled={busy} />
          <span className="upload-icon">＋</span><span><strong>Add a PDF</strong><small>Up to 20 MB · stored locally</small></span>
        </label>
        <div className="doc-list">
          {documents.length === 0 ? <p className="empty">Your uploaded PDFs will appear here.</p> : documents.map(doc =>
            <article className="doc-card" key={doc.document_id}>
              <span className="pdf-icon">PDF</span><div className="doc-info"><strong title={doc.filename}>{doc.filename}</strong><small>{doc.page_count} pages · {doc.chunk_count} passages</small></div>
            </article>)}
        </div>
        <div className="privacy-note"><span>●</span><p>PDFs and the search index stay on this machine. Questions sent to an LLM provider include retrieved passages.</p></div>
      </aside>

      <section className="main-panel">
        <form className="ask-card panel" onSubmit={ask}>
          <p className="eyebrow">Ask a question</p>
          <label className="sr-only" htmlFor="question">Question about your documents</label>
          <textarea id="question" value={question} onChange={e => setQuestion(e.target.value)} placeholder="What do the uploaded documents say about…?" rows="3" maxLength="4000" />
          <div className="ask-controls">
            <label className="select-label">Verification
              <select value={verifier} onChange={e => setVerifier(e.target.value)}><option value="heuristic">Similarity heuristic</option><option value="llm">Evidence-conditioned LLM</option></select>
            </label>
            <button className="ask-button" disabled={busy || !question.trim()}>{busy ? <><span className="spinner" /> Working</> : <>Ask VeriRAG <span>↗</span></>}</button>
          </div>
          <p className="microcopy">Generated answers require LLM_API_KEY; uploads and retrieval run locally.</p>
        </form>

        {error && <div className={`notice ${error.includes('already') || error.includes('indexed') ? 'success' : ''}`} role="status">{error}</div>}

        {result ? <div className="results">
          <section className="answer-card panel">
            <div className="section-heading"><div><p className="eyebrow">Generated response</p><h2>Your answer</h2></div><span className="latency">{result.latency_ms} ms</span></div>
            <p className="answer-text">{result.answer}</p>
            <div className="citation-list">
              <h3>Sources cited</h3>
              {result.citations.length ? result.citations.map(c => <div className="citation" key={c.chunk_id}><span>↗</span><strong>{c.filename}</strong><small>Page {c.page_number}</small></div>) : <p className="empty">The answer did not cite a retrieved passage.</p>}
            </div>
          </section>

          <section className="panel claims-panel">
            <div className="section-heading"><div><p className="eyebrow">Evidence assessment</p><h2>Claim by claim</h2></div><span className="method-tag">{result.verifier_method.replaceAll('_', ' ')}</span></div>
            {result.claims.length ? <div className="claim-list">{result.claims.map((claim, index) => <article className="claim" key={`${index}-${claim.claim}`}>
              <div className="claim-title"><span className={`status status-${claim.label.toLowerCase()}`}>{claim.label.replaceAll('_', ' ')}</span>{claim.raw_similarity != null && <span className="score">similarity {claim.raw_similarity.toFixed(2)}</span>}</div>
              <p>{claim.claim}</p><small className="explanation">{claim.explanation}</small>
              {claim.evidence.map(c => <details className="evidence" key={c.chunk_id}><summary>{c.filename} · page {c.page_number}</summary><blockquote>{c.text}</blockquote></details>)}
            </article>)}</div> : <p className="empty">No factual claims were extracted.</p>}
            <p className="disclaimer">Similarity is a topical signal, not proof of factual support. Verification results depend on the documents and verifier quality.</p>
          </section>

          <details className="panel retrieved-panel"><summary><span><p className="eyebrow">Debug view</p><strong>Retrieved passages ({result.retrieved_evidence.length})</strong></span><span className="chevron">⌄</span></summary>
            {result.retrieved_evidence.map(item => <article className="retrieved" key={item.chunk_id}><div><strong>{item.filename} · page {item.page_number}</strong><small>cosine similarity {item.similarity.toFixed(3)}</small></div><p>{item.text}</p></article>)}
          </details>
        </div> : <section className="empty-state panel"><span className="empty-mark">V.</span><h2>Answers with receipts.</h2><p>Upload a PDF, ask a question, then inspect the passages and claim assessments behind the response.</p><div className="steps"><span><b>01</b> Add a PDF</span><span><b>02</b> Ask a question</span><span><b>03</b> Review evidence</span></div></section>}
      </section>
    </section>
    <footer><span>VeriRAG · research prototype</span><span>Verification can inform review; it cannot guarantee truth.</span></footer>
  </main>;
}

export default App;

