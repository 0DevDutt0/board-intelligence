// frontend/src/components/StatusBar.jsx

function DotIcon({ color }) {
  return (
    <span
      className="inline-block w-1.5 h-1.5 rounded-full"
      style={{ backgroundColor: color }}
    />
  )
}

function Divider() {
  return <span className="text-border-bright select-none">|</span>
}

export default function StatusBar({ ingestStats, lastLatency, docName }) {
  if (!ingestStats) return null

  const latencyColor =
    lastLatency == null   ? null :
    lastLatency < 4000    ? '#22C55E' :
    lastLatency < 12000   ? '#F59E0B' :
                            '#EF4444'

  return (
    <div
      className="shrink-0 flex items-center gap-3 px-4 py-1.5 text-xs
                 bg-surface-panel border-t border-border-subtle text-ink-disabled
                 overflow-x-auto whitespace-nowrap"
      role="status"
      aria-live="polite"
    >
      {/* Session active */}
      <span className="flex items-center gap-1.5">
        <DotIcon color="#22C55E" />
        <span className="text-ink-tertiary">Session active</span>
      </span>

      {docName && (
        <>
          <Divider />
          <span className="text-ink-disabled max-w-48 truncate">{docName}</span>
        </>
      )}

      {ingestStats.page_count !== undefined && (
        <>
          <Divider />
          <span>{ingestStats.page_count} pages</span>
        </>
      )}

      {ingestStats.chunk_count !== undefined && (
        <>
          <Divider />
          <span>{ingestStats.chunk_count} chunks</span>
        </>
      )}

      {ingestStats.ingest_time_seconds !== undefined && (
        <>
          <Divider />
          <span>Ingested in {ingestStats.ingest_time_seconds}s</span>
        </>
      )}

      {lastLatency != null && (
        <span className="ml-auto flex items-center gap-1.5">
          <DotIcon color={latencyColor} />
          <span>Last response: {lastLatency}ms</span>
        </span>
      )}

      {/* Offline badge */}
      <span className="ml-auto flex items-center gap-1 px-1.5 py-0.5 rounded border border-border-subtle text-ink-disabled">
        <span style={{ fontSize: '10px' }}>AIR-GAPPED</span>
      </span>
    </div>
  )
}
