// frontend/src/App.jsx
import React, { useState } from 'react'
import UploadDropzone from './components/UploadDropzone'
import ChatPane from './components/ChatPane'
import SourcePanel from './components/SourcePanel'
import StatusBar from './components/StatusBar'
import Logo from './components/Logo'

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, message: '' }
  }
  static getDerivedStateFromError(err) {
    return { hasError: true, message: err?.message || 'Unknown error' }
  }
  componentDidCatch(err, info) {
    console.error('ErrorBoundary caught:', err, info)
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center h-full gap-4 px-8 text-center">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          <div>
            <p className="text-sm font-semibold text-ink-primary mb-1">Something went wrong</p>
            <p className="text-xs text-ink-tertiary max-w-xs leading-relaxed">{this.state.message}</p>
          </div>
          <button
            onClick={() => this.setState({ hasError: false, message: '' })}
            className="text-xs px-3 py-1.5 rounded-lg border border-border text-ink-secondary
                       hover:bg-surface-elevated hover:text-ink-primary transition-colors cursor-pointer"
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

function LockIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
      <path d="M7 11V7a5 5 0 0110 0v4" />
    </svg>
  )
}

function FileTextIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <polyline points="10 9 9 9 8 9" />
    </svg>
  )
}

function LogOutIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" />
      <polyline points="16 17 21 12 16 7" />
      <line x1="21" y1="12" x2="9" y2="12" />
    </svg>
  )
}

export default function App() {
  const [sessionId, setSessionId]     = useState(null)
  const [ingestStats, setIngestStats] = useState(null)
  const [sourceChunks, setSourceChunks] = useState([])
  const [lastLatency, setLastLatency] = useState(null)
  const [docName, setDocName]         = useState('')

  function handleIngestComplete(data) {
    setSessionId(data.session_id)
    setIngestStats(data)
    setSourceChunks([])
    setDocName(data.doc_filename || 'Document')
  }

  function handleCitations(chunks) {
    setSourceChunks(chunks)
  }

  function handleDone(data) {
    setLastLatency(data.latency_ms)
  }

  function handleEndSession() {
    setSessionId(null)
    setIngestStats(null)
    setSourceChunks([])
    setLastLatency(null)
    setDocName('')
  }

  if (!sessionId) {
    return (
      <div className="min-h-screen bg-surface-base flex flex-col items-center justify-center p-8">
        {/* Background gradient */}
        <div
          className="pointer-events-none fixed inset-0"
          style={{
            background: 'radial-gradient(ellipse 80% 60% at 50% -10%, rgba(59,130,246,0.07) 0%, transparent 70%)',
          }}
        />

        <div className="relative z-10 flex flex-col items-center w-full max-w-lg">
          {/* Wordmark */}
          <div className="flex flex-col items-center gap-1 mb-8">
            <Logo />
            <p className="text-xs text-ink-tertiary tracking-widest uppercase" style={{ letterSpacing: '0.12em' }}>
              Board Intelligence
            </p>
          </div>

          {/* Upload card */}
          <div className="w-full bg-surface-card border border-border rounded-2xl p-8 shadow-card_lg">
            <p className="text-ink-secondary text-sm text-center mb-6">
              Upload a confidential board meeting PDF. All processing
              happens entirely on local hardware with zero network egress.
            </p>
            <UploadDropzone
              apiBase={API_BASE}
              onComplete={handleIngestComplete}
            />
          </div>

          {/* Trust badge */}
          <div className="mt-6 flex items-center gap-1.5 text-xs text-ink-tertiary">
            <LockIcon />
            <span>Air-gapped processing - no data leaves this machine</span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen bg-surface-base flex flex-col overflow-hidden">
      {/* Header */}
      <header className="shrink-0 border-b border-border-subtle bg-surface-panel flex items-center justify-between px-5 py-3">
        <div className="flex items-center gap-4">
          <Logo />
          {docName && (
            <>
              <span className="text-border text-xs select-none">/</span>
              <div className="flex items-center gap-1.5 text-ink-tertiary text-xs">
                <FileTextIcon />
                <span className="max-w-48 truncate">{docName}</span>
              </div>
            </>
          )}
        </div>

        <button
          onClick={handleEndSession}
          className="flex items-center gap-1.5 text-xs text-ink-tertiary hover:text-status-error
                     transition-colors duration-150 cursor-pointer px-2 py-1 rounded-lg
                     hover:bg-status-error/10"
          aria-label="End session"
        >
          <LogOutIcon />
          <span>End Session</span>
        </button>
      </header>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Chat area */}
        <div className="flex-1 flex flex-col min-w-0 bg-surface-base">
          <ErrorBoundary>
            <ChatPane
              apiBase={API_BASE}
              sessionId={sessionId}
              onCitations={handleCitations}
              onDone={handleDone}
            />
          </ErrorBoundary>
        </div>

        {/* Source panel */}
        <div className="w-80 shrink-0 border-l border-border-subtle bg-surface-panel flex flex-col overflow-hidden">
          <SourcePanel chunks={sourceChunks} />
        </div>
      </div>

      {/* Status bar */}
      <StatusBar
        ingestStats={ingestStats}
        lastLatency={lastLatency}
        docName={docName}
      />
    </div>
  )
}
