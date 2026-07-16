// frontend/src/components/ChatPane.jsx
import { useState, useRef, useEffect, useCallback } from 'react'

// ---- Icons ----

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  )
}

function BotIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="11" width="18" height="10" rx="2" />
      <circle cx="12" cy="5" r="2" />
      <path d="M12 7v4" />
      <line x1="8" y1="16" x2="8" y2="16" />
      <line x1="16" y1="16" x2="16" y2="16" />
    </svg>
  )
}

function CopyIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
    </svg>
  )
}

function CheckSmIcon({ color }) {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke={color || '#22C55E'} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  )
}

function WarnIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  )
}

function AlertTriangleIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0 mt-0.5">
      <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  )
}

function SparkleIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17l-6.2 4.3 2.4-7.4L2 9.4h7.6z" />
    </svg>
  )
}

function QuoteIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 21c3 0 7-1 7-8V5c0-1.25-.756-2.017-2-2H4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V20c0 1 0 1 1 1z" />
      <path d="M15 21c3 0 7-1 7-8V5c0-1.25-.757-2.017-2-2h-4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2h.75c0 2.25.25 4-2.75 4v3c0 1 0 1 1 1z" />
    </svg>
  )
}

// ---- Citation badge ----

function CitationBadge({ page }) {
  return (
    <span
      className="inline-flex items-center px-1.5 py-0.5 rounded border border-citation/25 bg-citation/10 text-citation font-medium mx-0.5 cursor-default"
      title={`Page ${page}`}
      style={{ fontSize: '10px', lineHeight: 1 }}
    >
      p.{page}
    </span>
  )
}

// ---- Inline renderer ----

function UnverifiedBadge({ keyVal }) {
  return (
    <span
      key={keyVal}
      className="inline-flex items-center px-1.5 py-0.5 rounded border border-status-warning/30 bg-status-warning/10 text-status-warning font-medium mx-0.5 cursor-default"
      title="Citation could not be verified against retrieved evidence"
      style={{ fontSize: '10px', lineHeight: 1 }}
    >
      unverified
    </span>
  )
}

function renderInline(text, keyPrefix) {
  const regex = /(\*\*[^*\n]+\*\*|\*[^*\n]+\*|\[Page \d+\]|\[unverifiable citation\])/g
  const parts = []
  let last = 0
  let idx = 0
  let m
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) {
      parts.push(<span key={`${keyPrefix}-t${idx++}`}>{text.slice(last, m.index)}</span>)
    }
    if (m[0].startsWith('**')) {
      parts.push(
        <strong key={`${keyPrefix}-b${idx++}`} className="font-semibold text-ink-primary">
          {m[0].slice(2, -2)}
        </strong>
      )
    } else if (m[0].startsWith('*')) {
      parts.push(
        <em key={`${keyPrefix}-i${idx++}`} className="italic text-ink-secondary">
          {m[0].slice(1, -1)}
        </em>
      )
    } else if (m[0] === '[unverifiable citation]') {
      parts.push(<UnverifiedBadge key={`${keyPrefix}-u${idx++}`} />)
    } else {
      const pageNum = m[0].match(/\[Page (\d+)\]/)[1]
      parts.push(<CitationBadge key={`${keyPrefix}-c${idx++}`} page={pageNum} />)
    }
    last = m.index + m[0].length
  }
  if (last < text.length) {
    parts.push(<span key={`${keyPrefix}-t${idx}`}>{text.slice(last)}</span>)
  }
  return parts
}

// ---- Text cleaning utilities ----

function stripThink(text) {
  // Strip complete think blocks
  let out = text.replace(/<think>[\s\S]*?<\/think>/g, '')
  // Strip any unclosed think block still streaming (everything from last <think> onwards)
  const lastOpen  = out.lastIndexOf('<think>')
  const lastClose = out.lastIndexOf('</think>')
  if (lastOpen !== -1 && lastOpen > lastClose) {
    out = out.slice(0, lastOpen)
  }
  return out.trim()
}

function splitInlineHeadings(text) {
  // Model sometimes puts "**Heading:** - bullet" on one line -- split it
  return text.replace(/(\*\*[^*\n]+\*\*:?)\s+([-*] |\d+\. )/g, '$1\n$2')
}

// ---- Block markdown renderer ----

function MessageBody({ text, flaggedVerbs }) {
  const isThinkingOnly = text.includes('<think>') && !text.includes('</think>')
  const clean = splitInlineHeadings(stripThink(text))

  if (!clean && isThinkingOnly) {
    return (
      <span className="flex items-center gap-2 text-xs text-ink-tertiary">
        <span className="flex gap-0.5">
          {[0, 200, 400].map(d => (
            <span key={d} className="w-1 h-1 rounded-full bg-ink-disabled animate-pulse_dot" style={{ animationDelay: `${d}ms` }} />
          ))}
        </span>
        <span className="italic">Reasoning...</span>
      </span>
    )
  }

  const lines = clean.split('\n')
  const elements = []

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    if (/^#{1,3} /.test(line)) {
      const hashes = line.match(/^(#+) /)[1].length
      const content = line.replace(/^#+ /, '')
      if (hashes === 1) {
        elements.push(
          <p key={i} className="font-semibold text-ink-primary mt-3 mb-0.5 first:mt-0">
            {renderInline(content, `h${i}`)}
          </p>
        )
      } else if (hashes === 2) {
        elements.push(
          <p key={i} className="font-medium text-ink-tertiary text-xs uppercase tracking-wider mt-3 mb-1 first:mt-0">
            {renderInline(content, `h${i}`)}
          </p>
        )
      } else {
        elements.push(
          <p key={i} className="font-medium text-ink-primary mt-2 mb-0.5 first:mt-0">
            {renderInline(content, `h${i}`)}
          </p>
        )
      }
      continue
    }

    // Bold heading pattern: **Text:** alone on a line
    if (/^\*\*[^*]+\*\*:?\s*$/.test(line.trim())) {
      elements.push(
        <p key={i} className="font-semibold text-ink-primary mt-3 mb-0.5 first:mt-0">
          {renderInline(line.trim(), `bh${i}`)}
        </p>
      )
      continue
    }

    if (/^[-*] /.test(line)) {
      const bulletText = line.slice(2).toLowerCase()
      const isWarned = flaggedVerbs && [...flaggedVerbs].some(v => {
        const re = new RegExp(`\\b${v.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`, 'i')
        return re.test(bulletText)
      })
      elements.push(
        <div
          key={i}
          className={[
            'flex gap-2 ml-1 my-0.5 rounded-sm',
            isWarned ? 'border-l-2 border-status-warning/60 pl-1.5 -ml-0.5 bg-status-warning/5' : '',
          ].join(' ')}
          title={isWarned ? 'This bullet contains a flagged verb -- see warnings below' : undefined}
        >
          <span className="text-ink-disabled select-none shrink-0 mt-px" style={{ fontSize: '10px', lineHeight: '20px' }}>
            -
          </span>
          <span className="text-ink-secondary leading-relaxed">
            {renderInline(line.slice(2), `li${i}`)}
          </span>
        </div>
      )
      continue
    }

    if (/^\d+\. /.test(line)) {
      const match = line.match(/^(\d+)\. (.*)$/) || ['', '1', line]
      elements.push(
        <div key={i} className="flex gap-2 ml-1 my-0.5">
          <span className="text-ink-disabled font-mono text-xs select-none shrink-0 w-4 text-right" style={{ lineHeight: '22px' }}>
            {match[1]}.
          </span>
          <span className="text-ink-secondary leading-relaxed">
            {renderInline(match[2] || '', `ol${i}`)}
          </span>
        </div>
      )
      continue
    }

    if (line.trim() === '') {
      elements.push(<div key={i} className="h-1" />)
      continue
    }

    elements.push(
      <p key={i} className="text-ink-secondary leading-relaxed my-0.5">
        {renderInline(line, `p${i}`)}
      </p>
    )
  }

  return (
    <span className="block space-y-0">
      {isThinkingOnly && (
        <span className="flex items-center gap-1.5 text-xs text-ink-tertiary mb-2">
          <span className="flex gap-0.5">
            {[0, 200, 400].map(d => (
              <span key={d} className="w-1 h-1 rounded-full bg-ink-disabled animate-pulse_dot" style={{ animationDelay: `${d}ms` }} />
            ))}
          </span>
          <span className="italic">Reasoning...</span>
        </span>
      )}
      {elements}
    </span>
  )
}

// ---- Copy button ----

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)
  function handleCopy() {
    const clean = text.replace(/<think>[\s\S]*?<\/think>/g, '').trim()
    navigator.clipboard.writeText(clean).catch(() => {})
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }
  return (
    <button
      onClick={handleCopy}
      className="opacity-0 group-hover:opacity-100 transition-opacity duration-150
                 p-1 rounded text-ink-disabled hover:text-ink-secondary cursor-pointer"
      aria-label="Copy response"
      title="Copy"
    >
      {copied ? <CheckSmIcon /> : <CopyIcon />}
    </button>
  )
}

// ---- Per-message evaluation footer ----

function EvalFooter({ msg }) {
  const clean = (msg.text || '').replace(/<think>[\s\S]*?<\/think>/g, '')
  const citCount = (clean.match(/\[Page \d+\]/g) || []).length
  const hasMeta = citCount > 0 || msg.eval_warnings !== undefined || msg.latency_ms !== undefined
  if (!hasMeta) return null

  const verified  = msg.eval_warnings !== undefined && msg.eval_warnings.length === 0
  const warnCount = msg.eval_warnings ? msg.eval_warnings.length : 0

  return (
    <div className="flex items-center gap-3 mt-2 pt-2 border-t border-border-subtle text-xs text-ink-disabled flex-wrap">

      {citCount > 0 && (
        <span className="flex items-center gap-1">
          <QuoteIcon />
          <span className="text-citation">{citCount} citation{citCount !== 1 ? 's' : ''}</span>
        </span>
      )}

      {msg.eval_warnings !== undefined && (
        verified ? (
          <span className="flex items-center gap-1 text-status-online">
            <CheckSmIcon color="#22C55E" />
            Verified
          </span>
        ) : (
          <span className="flex items-center gap-1 text-status-warning">
            <WarnIcon />
            {warnCount} warning{warnCount !== 1 ? 's' : ''}
          </span>
        )
      )}

      {msg.token_count !== undefined && (
        <span>{msg.token_count} tokens</span>
      )}

      {msg.latency_ms !== undefined && (
        <span className="ml-auto">
          {msg.latency_ms >= 1000
            ? `${(msg.latency_ms / 1000).toFixed(1)}s`
            : `${msg.latency_ms}ms`}
        </span>
      )}
    </div>
  )
}

// ---- Warnings detail (expandable) ----

function WarningsDetail({ warnings }) {
  const [open, setOpen] = useState(false)
  if (!warnings || warnings.length === 0) return null
  return (
    <div className="mt-1.5">
      <button
        onClick={() => setOpen(v => !v)}
        className="text-xs text-status-warning/80 hover:text-status-warning transition-colors cursor-pointer"
      >
        {open ? 'Hide warnings' : 'Show warnings'}
      </button>
      {open && (
        <ul className="mt-1.5 space-y-1">
          {warnings.map((w, i) => {
            const msg = typeof w === 'string' ? w : w.message
            return (
              <li key={i} className="text-xs text-ink-tertiary pl-2 border-l border-status-warning/30">
                {msg}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

// ---- Example starter questions ----

const EXAMPLE_QUESTIONS = [
  'What are the key decisions in this document?',
  'Summarize the financial highlights.',
  'What action items were assigned and to whom?',
  'What risks or concerns were discussed?',
]

// ---- Main ChatPane ----

export default function ChatPane({ apiBase, sessionId, onCitations, onDone }) {
  const [messages, setMessages]   = useState([])
  const [input, setInput]         = useState('')
  const [streaming, setStreaming] = useState(false)
  const bottomRef    = useRef(null)
  const textareaRef  = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendQuestion = useCallback(async (questionText) => {
    const question = (questionText || input).trim()
    if (!question || streaming) return
    setInput('')

    const userMsg      = { role: 'user',      text: question }
    const assistantMsg = { role: 'assistant', text: '', streaming: true }
    setMessages(prev => [...prev, userMsg, assistantMsg])
    setStreaming(true)

    try {
      const resp = await fetch(`${apiBase}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Session-ID': sessionId },
        body: JSON.stringify({ question, use_hyde: true }),
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)

      const reader  = resp.body.getReader()
      const decoder = new TextDecoder()
      let buffer    = ''
      let eventType = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.slice(6).trim()
          } else if (line.startsWith('data:')) {
            const raw = line.slice(5).trim()
            if (!raw) continue
            try {
              const obj = JSON.parse(raw)

              if (eventType === 'chunk' && obj.text) {
                setMessages(prev => {
                  const msgs = [...prev]
                  const last = msgs[msgs.length - 1]
                  if (last?.role === 'assistant') {
                    msgs[msgs.length - 1] = { ...last, text: last.text + obj.text }
                  }
                  return msgs
                })
              }

              else if (eventType === 'citations' && obj.chunks) {
                onCitations(obj.chunks)
              }

              else if (eventType === 'corrected' && obj.text !== undefined) {
                // Replace streamed tokens with the citation-validated version.
                // Any fabricated [Page N] tags are already [unverifiable citation].
                setMessages(prev => {
                  const msgs = [...prev]
                  const last = msgs[msgs.length - 1]
                  if (last?.role === 'assistant') {
                    msgs[msgs.length - 1] = { ...last, text: obj.text }
                  }
                  return msgs
                })
              }

              else if (eventType === 'done') {
                // Attach latency + tokens to the message for inline eval display
                setMessages(prev => {
                  const msgs = [...prev]
                  const last = msgs[msgs.length - 1]
                  if (last?.role === 'assistant') {
                    msgs[msgs.length - 1] = {
                      ...last,
                      latency_ms:  obj.latency_ms,
                      token_count: obj.total_tokens,
                    }
                  }
                  return msgs
                })
                onDone(obj)
              }

              else if (eventType === 'verification') {
                // Attach verification result to the message
                setMessages(prev => {
                  const msgs = [...prev]
                  const last = msgs[msgs.length - 1]
                  if (last?.role === 'assistant') {
                    msgs[msgs.length - 1] = {
                      ...last,
                      eval_warnings: obj.warnings || [],
                    }
                  }
                  return msgs
                })
              }

              else if (eventType === 'error') {
                const errMsg = obj.error || 'An error occurred during generation.'
                setMessages(prev => {
                  const msgs = [...prev]
                  const last = msgs[msgs.length - 1]
                  if (last?.role === 'assistant') {
                    msgs[msgs.length - 1] = {
                      ...last,
                      text: errMsg,
                      isError: true,
                      streaming: false,
                    }
                  }
                  return msgs
                })
              }

            } catch (_) {}
            eventType = null
          }
        }
      }
    } catch (err) {
      setMessages(prev => {
        const msgs = [...prev]
        const last = msgs[msgs.length - 1]
        if (last?.role === 'assistant') {
          msgs[msgs.length - 1] = { ...last, text: `Error: ${err.message}`, streaming: false }
        }
        return msgs
      })
    } finally {
      setMessages(prev => {
        const msgs = [...prev]
        const last = msgs[msgs.length - 1]
        if (last?.role === 'assistant') {
          msgs[msgs.length - 1] = { ...last, streaming: false }
        }
        return msgs
      })
      setStreaming(false)
    }
  }, [input, streaming, apiBase, sessionId, onCitations, onDone])

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendQuestion()
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-5 py-6 space-y-5">

        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-8 pb-10">
            <div className="flex flex-col items-center gap-3 text-center">
              <div className="w-12 h-12 rounded-2xl bg-surface-elevated flex items-center justify-center text-ink-tertiary">
                <SparkleIcon />
              </div>
              <p className="text-ink-secondary text-sm font-medium">
                Ask anything about the board document
              </p>
              <p className="text-ink-tertiary text-xs max-w-xs">
                Answers are grounded in the uploaded document with page citations.
              </p>
            </div>
            <div className="w-full max-w-md space-y-2">
              {EXAMPLE_QUESTIONS.map((q, i) => (
                <button
                  key={i}
                  onClick={() => !streaming && sendQuestion(q)}
                  className="w-full text-left text-xs text-ink-tertiary px-3 py-2.5 rounded-lg
                             border border-border-subtle bg-surface-card
                             hover:border-border hover:text-ink-secondary hover:bg-surface-elevated
                             transition-all duration-150 cursor-pointer"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={['flex animate-fade_in', msg.role === 'user' ? 'justify-end' : 'justify-start'].join(' ')}
          >
            {msg.role === 'assistant' && (
              <div className="w-7 h-7 rounded-lg bg-surface-elevated border border-border-subtle
                              flex items-center justify-center shrink-0 mr-2.5 mt-1 text-ink-tertiary">
                <BotIcon />
              </div>
            )}

            <div
              className={[
                'group relative max-w-xl rounded-2xl px-4 py-3 text-sm',
                msg.role === 'user'
                  ? 'bg-accent text-white rounded-tr-sm'
                  : 'bg-surface-card border border-border-subtle text-ink-secondary rounded-tl-sm shadow-card',
              ].join(' ')}
            >
              {msg.role === 'user' ? (
                <p className="leading-relaxed whitespace-pre-wrap">{msg.text}</p>
              ) : msg.isError ? (
                <div className="flex items-start gap-2.5 text-status-warning">
                  <AlertTriangleIcon />
                  <div>
                    <p className="font-semibold text-sm mb-0.5">Generation failed</p>
                    <p className="text-xs opacity-80 leading-relaxed">{msg.text}</p>
                    <p className="text-xs opacity-50 mt-2">
                      Try rephrasing your question. If the error persists, reload the page.
                    </p>
                  </div>
                </div>
              ) : (
                <>
                  <MessageBody
                    text={msg.text}
                    flaggedVerbs={new Set(
                      (msg.eval_warnings || [])
                        .map(w => (typeof w === 'string' ? w : w.message).match(/"([^"]+)"/)?.[1])
                        .filter(Boolean)
                    )}
                  />

                  {msg.streaming && (
                    <span className="inline-flex items-center gap-0.5 ml-1 align-middle">
                      {[0, 150, 300].map(d => (
                        <span key={d} className="w-1 h-3 bg-ink-disabled rounded-full animate-pulse_dot" style={{ animationDelay: `${d}ms` }} />
                      ))}
                    </span>
                  )}

                  {/* Inline evaluation footer */}
                  {!msg.streaming && <EvalFooter msg={msg} />}

                  {/* Warnings detail */}
                  {!msg.streaming && <WarningsDetail warnings={msg.eval_warnings} />}

                  {/* Copy action */}
                  {!msg.streaming && msg.text && (
                    <div className="flex justify-end -mb-0.5">
                      <CopyButton text={msg.text} />
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="shrink-0 border-t border-border-subtle bg-surface-panel px-4 py-3">
        <div className="flex gap-2.5 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask a question about the board document..."
            rows={1}
            disabled={streaming}
            aria-label="Question input"
            className="flex-1 resize-none rounded-xl border border-border bg-surface-elevated
                       px-4 py-2.5 text-sm text-ink-primary placeholder:text-ink-disabled
                       focus:outline-none focus:border-accent focus:shadow-glow_sm
                       disabled:opacity-40 disabled:cursor-not-allowed
                       transition-all duration-150 leading-relaxed"
            style={{ minHeight: '42px', maxHeight: '140px' }}
            onInput={e => {
              e.target.style.height = 'auto'
              e.target.style.height = Math.min(e.target.scrollHeight, 140) + 'px'
            }}
          />
          <button
            onClick={() => sendQuestion()}
            disabled={streaming || !input.trim()}
            aria-label="Send message"
            className="shrink-0 w-10 h-10 rounded-xl bg-accent hover:bg-accent-hover
                       text-white flex items-center justify-center
                       disabled:opacity-30 disabled:cursor-not-allowed
                       transition-all duration-150 cursor-pointer shadow-glow_sm
                       hover:shadow-glow_accent active:scale-95"
          >
            <SendIcon />
          </button>
        </div>
        <p className="text-xs text-ink-disabled mt-2 px-1">
          Enter to send, Shift+Enter for newline
        </p>
      </div>
    </div>
  )
}
