import { useEffect, useState } from 'react'

function FileDrop({ label, hint, accept, file, onChange, preview }) {
  const [drag, setDrag] = useState(false)
  return (
    <label
      onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); e.dataTransfer.files[0] && onChange(e.dataTransfer.files[0]) }}
      className={`flex min-h-[7.5rem] cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-3 text-center transition
        ${drag ? 'border-sky-400 bg-sky-500/10' : file ? 'border-slate-600 bg-slate-800/40' : 'border-slate-700 hover:border-slate-500'}`}
    >
      <input type="file" accept={accept} className="hidden" onChange={(e) => onChange(e.target.files[0] || null)} />
      {preview ? (
        <img src={preview} alt="aperçu" className="max-h-20 rounded border border-slate-700 object-contain" />
      ) : (
        <span className="text-2xl text-slate-500">{file ? '📄' : '⬆'}</span>
      )}
      <span className="text-sm font-medium text-slate-300">{file ? file.name : label}</span>
      <span className="text-xs text-slate-500">
        {file ? `${(file.size / 1024).toFixed(1)} Ko` : hint}
      </span>
    </label>
  )
}

export default function AnalyzeForm({ onSubmit, running, disabled }) {
  const [url, setUrl] = useState('')
  const [html, setHtml] = useState(null)
  const [shot, setShot] = useState(null)
  const [preview, setPreview] = useState(null)

  useEffect(() => {
    if (!shot) return setPreview(null)
    const u = URL.createObjectURL(shot)
    setPreview(u)
    return () => URL.revokeObjectURL(u)
  }, [shot])

  const canSubmit = !running && !disabled && (url.trim() || html || shot)

  return (
    <form
      onSubmit={(e) => { e.preventDefault(); canSubmit && onSubmit({ url: url.trim(), html, screenshot: shot }) }}
      className="space-y-4 rounded-xl border border-slate-800 bg-slate-900/60 p-5"
    >
      <div>
        <label htmlFor="url" className="mb-1.5 block text-sm font-medium text-slate-300">URL du site</label>
        <input
          id="url"
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://exemple.com/login"
          className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 font-mono text-sm text-slate-100 placeholder:text-slate-600 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500"
        />
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <FileDrop label="Fichier HTML" hint=".html / .htm" accept=".html,.htm,text/html" file={html} onChange={setHtml} />
        <FileDrop label="Screenshot" hint=".png / .jpg" accept="image/*" file={shot} onChange={setShot} preview={preview} />
      </div>

      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={!canSubmit}
          className="flex-1 rounded-lg bg-sky-600 px-4 py-2.5 font-semibold text-white transition hover:bg-sky-500 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
        >
          {running ? 'Analyse en cours…' : 'Analyser'}
        </button>
        <button
          type="button"
          disabled={running}
          onClick={() => { setUrl(''); setHtml(null); setShot(null) }}
          className="rounded-lg border border-slate-700 px-4 py-2.5 text-sm text-slate-400 hover:border-slate-500 hover:text-slate-200 disabled:opacity-40"
        >
          Réinitialiser
        </button>
      </div>
    </form>
  )
}
