// frontend/src/components/UploadDropzone.jsx
import { useState, useRef } from 'react'

function UploadIcon() {
  return (
    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  )
}

function SpinnerIcon() {
  return (
    <svg className="animate-spin_slow" width="28" height="28" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="#334155" strokeWidth="2.5" />
      <path d="M12 2a10 10 0 0110 10" stroke="#3B82F6" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  )
}

function CheckIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#22C55E" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polyline points="9 12 11 14 15 10" />
    </svg>
  )
}

function AlertIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#EF4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  )
}

export default function UploadDropzone({ apiBase, onComplete }) {
  const [dragging, setDragging]   = useState(false)
  const [uploading, setUploading] = useState(false)
  const [done, setDone]           = useState(false)
  const [progress, setProgress]   = useState('')
  const [error, setError]         = useState(null)
  const inputRef = useRef(null)

  async function uploadFile(file) {
    if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
      setError('Only PDF files are accepted.')
      return
    }
    setError(null)
    setDone(false)
    setUploading(true)
    setProgress('Uploading and parsing PDF...')

    const form = new FormData()
    form.append('file', file)

    try {
      const resp = await fetch(`${apiBase}/ingest`, {
        method: 'POST',
        body: form,
      })
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        const detail = body.detail
        const msg = typeof detail === 'string'
          ? detail
          : (detail?.error || detail?.message || `Server error (HTTP ${resp.status})`)
        throw new Error(msg)
      }
      const data = await resp.json()
      data.doc_filename = file.name
      setProgress(`Indexed ${data.page_count} pages, ${data.chunk_count} chunks`)
      setDone(true)
      setTimeout(() => onComplete(data), 600)
    } catch (err) {
      setError(err.message)
      setProgress('')
    } finally {
      setUploading(false)
    }
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) uploadFile(file)
  }

  function onInputChange(e) {
    const file = e.target.files[0]
    if (file) uploadFile(file)
    e.target.value = ''
  }

  const isActive = dragging && !uploading

  return (
    <div>
      <div
        role="button"
        tabIndex={uploading ? -1 : 0}
        aria-label="Upload PDF - click or drag and drop"
        onDragOver={(e) => { e.preventDefault(); if (!uploading) setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !uploading && inputRef.current.click()}
        onKeyDown={(e) => { if ((e.key === 'Enter' || e.key === ' ') && !uploading) inputRef.current.click() }}
        className={[
          'relative w-full rounded-xl border-2 border-dashed p-10',
          'flex flex-col items-center justify-center gap-4',
          'transition-all duration-200 select-none',
          uploading || done
            ? 'pointer-events-none'
            : 'cursor-pointer',
          isActive
            ? 'border-accent bg-accent/8 shadow-glow_sm'
            : 'border-border hover:border-border-bright hover:bg-surface-elevated/40',
        ].join(' ')}
        style={isActive ? { background: 'rgba(59,130,246,0.06)' } : {}}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={onInputChange}
          aria-hidden="true"
        />

        {/* Icon area */}
        <div className={[
          'w-14 h-14 rounded-2xl flex items-center justify-center transition-all duration-200',
          isActive
            ? 'bg-accent text-white shadow-glow_accent'
            : 'bg-surface-elevated text-ink-tertiary',
        ].join(' ')}>
          {uploading && <SpinnerIcon />}
          {done && <CheckIcon />}
          {!uploading && !done && <UploadIcon />}
        </div>

        {/* Text */}
        <div className="text-center">
          {uploading && (
            <>
              <p className="text-sm font-medium text-ink-primary">{progress}</p>
              <p className="text-xs text-ink-tertiary mt-1">
                Building vector index and BM25 index...
              </p>
            </>
          )}
          {done && (
            <>
              <p className="text-sm font-medium text-status-online">{progress}</p>
              <p className="text-xs text-ink-tertiary mt-1">
                Launching chat interface...
              </p>
            </>
          )}
          {!uploading && !done && (
            <>
              <p className="text-sm font-semibold text-ink-primary">
                {isActive ? 'Drop to upload' : 'Drop your board PDF here'}
              </p>
              <p className="text-xs text-ink-tertiary mt-1">
                or click to browse files
              </p>
            </>
          )}
        </div>

        {/* File type hint */}
        {!uploading && !done && (
          <span className="text-xs text-ink-disabled px-2 py-0.5 rounded border border-border-subtle">
            PDF only
          </span>
        )}
      </div>

      {/* Error state */}
      {error && (
        <div className="mt-3 flex items-start gap-2 px-3 py-2.5 rounded-lg bg-status-error/10 border border-status-error/20">
          <span className="shrink-0 mt-0.5"><AlertIcon /></span>
          <p className="text-sm text-status-error leading-snug">{error}</p>
        </div>
      )}
    </div>
  )
}
