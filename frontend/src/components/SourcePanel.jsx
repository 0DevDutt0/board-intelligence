// frontend/src/components/SourcePanel.jsx
import { useState } from 'react'

function LayersIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="12 2 2 7 12 12 22 7 12 2" />
      <polyline points="2 17 12 22 22 17" />
      <polyline points="2 12 12 17 22 12" />
    </svg>
  )
}

function ChevronIcon({ open }) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ transform: open ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 150ms ease' }}
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  )
}

function TableIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
      <line x1="3" y1="9" x2="21" y2="9" />
      <line x1="3" y1="15" x2="21" y2="15" />
      <line x1="9" y1="3" x2="9" y2="21" />
    </svg>
  )
}

function FileIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  )
}

// Score bar (reranker relevance) with human-readable quality label
function ScoreBar({ score }) {
  const pct   = Math.max(0, Math.min(1, score)) * 100
  const color = pct >= 50 ? '#22C55E' : pct >= 25 ? '#F59E0B' : '#64748B'
  const label = pct >= 50 ? 'Strong match' : pct >= 25 ? 'Good match' : 'Weak match'

  return (
    <div className="flex items-center gap-1.5">
      <div className="flex-1 h-1 bg-surface-elevated rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-xs font-medium" style={{ color, fontSize: '10px' }}>
        {label}
      </span>
    </div>
  )
}

// Individual chunk card
function ChunkCard({ chunk, index }) {
  const [expanded, setExpanded] = useState(false)
  const isTable = chunk.chunk_type === 'table'

  return (
    <div className="rounded-xl border border-border-subtle bg-surface-card overflow-hidden">
      {/* Card header */}
      <button
        onClick={() => setExpanded(v => !v)}
        className="w-full flex items-start gap-2.5 px-3 py-2.5 hover:bg-surface-elevated/50
                   transition-colors duration-100 cursor-pointer text-left"
        aria-expanded={expanded}
      >
        {/* Rank badge */}
        <span className="shrink-0 w-5 h-5 rounded-md bg-surface-elevated flex items-center justify-center
                          text-ink-disabled font-mono text-xs border border-border-subtle">
          {index + 1}
        </span>

        <div className="flex-1 min-w-0">
          {/* Page + type */}
          <div className="flex items-center gap-1.5 flex-wrap mb-1">
            <span className="text-xs font-semibold text-ink-primary">
              Page {chunk.page_number}
            </span>
            {chunk.section_heading && (
              <span className="text-xs text-ink-tertiary truncate max-w-32">
                {chunk.section_heading}
              </span>
            )}
            <span
              className={[
                'ml-auto flex items-center gap-1 px-1.5 py-0.5 rounded-md text-xs font-medium border',
                isTable
                  ? 'text-citation bg-citation/10 border-citation/20'
                  : 'text-ink-disabled bg-surface-elevated border-border-subtle',
              ].join(' ')}
            >
              {isTable ? <TableIcon /> : <FileIcon />}
              {chunk.chunk_type}
            </span>
          </div>

          {/* Reranker score */}
          {chunk.reranker_score !== undefined && (
            <ScoreBar score={chunk.reranker_score} />
          )}
        </div>

        <span className="shrink-0 text-ink-disabled mt-0.5">
          <ChevronIcon open={expanded} />
        </span>
      </button>

      {/* Chunk text */}
      {expanded && (
        <div className="px-3 pb-3 border-t border-border-subtle">
          <p className="text-xs text-ink-tertiary leading-relaxed pt-2.5 whitespace-pre-wrap">
            {chunk.text}
          </p>
        </div>
      )}

      {/* Collapsed preview */}
      {!expanded && chunk.text && (
        <div className="px-3 pb-2.5 pt-0">
          <p className="text-xs text-ink-disabled leading-relaxed line-clamp-3">
            {chunk.text}
          </p>
        </div>
      )}
    </div>
  )
}

// Main SourcePanel
export default function SourcePanel({ chunks }) {
  return (
    <div className="flex flex-col h-full">
      {/* Panel header */}
      <div className="shrink-0 flex items-center gap-2 px-4 py-3 border-b border-border-subtle">
        <span className="text-ink-tertiary"><LayersIcon /></span>
        <h2 className="text-xs font-semibold text-ink-secondary uppercase tracking-widest">
          Sources
        </h2>
        {chunks && chunks.length > 0 && (
          <span className="ml-auto text-xs font-medium text-ink-disabled bg-surface-elevated
                           px-1.5 py-0.5 rounded-md border border-border-subtle">
            {chunks.length}
          </span>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {(!chunks || chunks.length === 0) ? (
          <div className="flex flex-col items-center justify-center h-full gap-3 px-6 pb-12">
            <div className="w-10 h-10 rounded-xl bg-surface-elevated border border-border-subtle
                            flex items-center justify-center text-ink-disabled">
              <LayersIcon />
            </div>
            <p className="text-xs text-ink-tertiary text-center leading-relaxed">
              Retrieved passages will appear here after you ask a question.
            </p>
          </div>
        ) : (
          <div className="p-3 space-y-2">
            {chunks.map((chunk, i) => (
              <ChunkCard key={i} chunk={chunk} index={i} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
