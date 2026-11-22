import { useEffect, useRef } from 'react'

const STYLE = {
  ok: { icon: '[✓]', cls: 'text-green-400' },
  progress: { icon: '[⟳]', cls: 'text-yellow-300' },
  error: { icon: '[✗]', cls: 'text-red-400' },
  info: { icon: '[i]', cls: 'text-sky-300' },
}

const fmtTime = (ts) =>
  new Date(ts * 1000).toLocaleTimeString('fr-FR', { hour12: false }) +
  '.' + String(Math.floor((ts % 1) * 1000)).padStart(3, '0')

export default function LogTerminal({ logs, running, onClear }) {
  const endRef = useRef(null)
  useEffect(() => { endRef.current?.scrollIntoView({ block: 'end' }) }, [logs])

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-xl border border-zinc-800 bg-black shadow-2xl">
      <div className="flex items-center gap-2 border-b border-zinc-800 bg-zinc-900 px-4 py-2">
        <span className="h-3 w-3 rounded-full bg-red-500/80" />
        <span className="h-3 w-3 rounded-full bg-yellow-500/80" />
        <span className="h-3 w-3 rounded-full bg-green-500/80" />
        <span className="ml-3 font-mono text-xs text-zinc-400">phishguard@analyzer: ~/logs</span>
        {running && <span className="ml-2 font-mono text-xs text-yellow-300">● live</span>}
        <button
          onClick={onClear}
          disabled={running || logs.length === 0}
          className="ml-auto font-mono text-xs text-zinc-500 hover:text-zinc-200 disabled:opacity-30"
        >
          clear
        </button>
      </div>

      <div className="terminal-scroll min-h-[18rem] flex-1 overflow-y-auto p-4 font-mono text-[13px] leading-relaxed">
        {logs.length === 0 && !running && (
          <p className="text-zinc-600">$ en attente d'une analyse...</p>
        )}
        {logs.map((l, i) => {
          const s = STYLE[l.level] || STYLE.info
          return (
            <div key={i} className="flex gap-3 whitespace-pre-wrap break-all">
              <span className="shrink-0 text-zinc-600">{fmtTime(l.ts)}</span>
              <span className={`shrink-0 ${s.cls}`}>{s.icon}</span>
              <span className={l.level === 'info' ? 'text-zinc-300' : s.cls}>{l.message}</span>
            </div>
          )
        })}
        {running && <span className="cursor-blink text-green-400">█</span>}
        <div ref={endRef} />
      </div>
    </div>
  )
}
