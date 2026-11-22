import { useCallback, useEffect, useRef, useState } from 'react'
import { clearHistory, deleteHistoryItem, getHealth, getHistory, runAnalysis } from './api'
import AnalyzeForm from './components/AnalyzeForm'
import History from './components/History'
import LogTerminal from './components/LogTerminal'
import ResultCard from './components/ResultCard'

const now = () => Date.now() / 1000

export default function App() {
  const [health, setHealth] = useState(null)
  const [logs, setLogs] = useState([])
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const cancelRef = useRef(null)

  const refreshHistory = useCallback(() => getHistory().then(setHistory).catch(() => {}), [])

  // Statut du backend (re-vérifie tant que les modèles ne sont pas prêts)
  useEffect(() => {
    let timer
    const poll = () =>
      getHealth()
        .then((h) => { setHealth(h); if (!h.ready && !h.error) timer = setTimeout(poll, 2000) })
        .catch(() => { setHealth({ ready: false, error: 'Backend injoignable (port 8000)' }); timer = setTimeout(poll, 3000) })
    poll()
    refreshHistory()
    return () => { clearTimeout(timer); cancelRef.current?.() }
  }, [refreshHistory])

  const analyze = async (input) => {
    setRunning(true)
    setResult(null)
    setLogs([{ level: 'progress', message: 'Envoi des fichiers au serveur...', ts: now() }])
    try {
      cancelRef.current = await runAnalysis(input, {
        onLog: (l) => setLogs((prev) => [...prev, l]),
        onResult: (r) => { setResult(r); refreshHistory() },
        onFailed: () => {},
        onDone: () => setRunning(false),
      })
    } catch (err) {
      setLogs((prev) => [...prev, { level: 'error', message: err.message, ts: now() }])
      setRunning(false)
    }
  }

  const ready = health?.ready

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <header className="mb-8 flex flex-wrap items-center gap-4">
        <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-sky-600 text-xl">🛡</div>
        <div>
          <h1 className="text-2xl font-bold text-slate-50">PhishGuard</h1>
          <p className="text-sm text-slate-400">Détection de phishing multimodale · URL + HTML + Screenshot</p>
        </div>
        <div className="ml-auto flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs">
          <span className={`h-2 w-2 rounded-full ${ready ? 'bg-emerald-400' : health?.error ? 'bg-red-500' : 'bg-yellow-400 animate-pulse'}`} />
          <span className="text-slate-300">
            {ready
              ? `Modèles prêts · image sur ${health.device?.toUpperCase()}${health.image_model ? '' : ' (indisponible)'}`
              : health?.error || 'Chargement des modèles…'}
          </span>
        </div>
      </header>

      <main className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        <div className="space-y-6 lg:col-span-2">
          <AnalyzeForm onSubmit={analyze} running={running} disabled={!ready} />
          <ResultCard result={result} />
        </div>
        <div className="lg:col-span-3">
          <LogTerminal logs={logs} running={running} onClear={() => setLogs([])} />
        </div>
      </main>

      <div className="mt-6">
        <History
          items={history}
          selectedId={result?.id}
          onSelect={(h) => !running && setResult(h)}
          onDelete={(id) => deleteHistoryItem(id).then(refreshHistory)}
          onClear={() => clearHistory().then(() => { setHistory([]) })}
        />
      </div>

      {health?.meta?.metrics && (
        <footer className="mt-6 text-center text-xs text-slate-600">
          Fusion LogReg · F1 val (CV 5 plis) = {health.meta.metrics.fusion_val_cv5?.F1} · modèles du {health.meta.created}
        </footer>
      )}
    </div>
  )
}
