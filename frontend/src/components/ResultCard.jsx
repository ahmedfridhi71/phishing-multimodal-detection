const MODALITIES = [
  { key: 'p_url', label: 'URL', model: 'TF-IDF char + LogReg' },
  { key: 'p_html_texte', label: 'HTML texte', model: 'TF-IDF mots + LinearSVC' },
  { key: 'p_html_structure', label: 'HTML structure', model: 'GradientBoosting' },
  { key: 'p_image', label: 'Screenshot', model: 'ResNet50' },
]

const pct = (v) => `${(v * 100).toFixed(1)} %`

function ScoreBar({ label, model, value }) {
  const missing = value === null || value === undefined
  const color = missing ? 'bg-slate-700' : value >= 0.5 ? 'bg-red-500' : 'bg-emerald-500'
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-sm">
        <span className="text-slate-300">
          {label} <span className="text-xs text-slate-500">· {model}</span>
        </span>
        <span className="font-mono text-slate-200">{missing ? '—' : pct(value)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
        <div className={`h-full ${color} transition-all duration-700`} style={{ width: missing ? '0%' : pct(value) }} />
      </div>
    </div>
  )
}

export default function ResultCard({ result }) {
  if (!result) {
    return (
      <div className="flex h-full min-h-[14rem] items-center justify-center rounded-xl border border-dashed border-slate-700 p-6 text-center text-slate-500">
        Le résultat de l'analyse s'affichera ici.
      </div>
    )
  }

  const phishing = result.label === 'phishing'
  const html = result.details?.html

  return (
    <div className={`rounded-xl border p-6 ${phishing ? 'border-red-500/40 bg-red-950/20' : 'border-emerald-500/40 bg-emerald-950/20'}`}>
      <div className="flex flex-wrap items-center gap-4">
        <span className={`rounded-full px-4 py-1.5 text-sm font-bold uppercase tracking-wider ${phishing ? 'bg-red-500 text-white' : 'bg-emerald-500 text-emerald-950'}`}>
          {phishing ? '⚠ Phishing' : '✓ Légitime'}
        </span>
        <div className="ml-auto text-right">
          <div className="text-xs uppercase tracking-wide text-slate-400">Confiance</div>
          <div className={`font-mono text-3xl font-bold ${phishing ? 'text-red-400' : 'text-emerald-400'}`}>
            {pct(result.confidence)}
          </div>
        </div>
      </div>

      <p className="mt-3 break-all font-mono text-xs text-slate-400" title={result.url}>{result.url || '(pas d\'URL)'}</p>

      <div className="mt-4 h-3 overflow-hidden rounded-full bg-slate-800" title={`p(phishing) = ${result.probability}`}>
        <div className={`h-full ${phishing ? 'bg-red-500' : 'bg-emerald-500'} transition-all duration-700`}
             style={{ width: pct(result.probability) }} />
      </div>
      <div className="mt-1 flex justify-between text-xs text-slate-500">
        <span>Légitime</span>
        <span>p(phishing) = {result.probability.toFixed(3)}</span>
        <span>Phishing</span>
      </div>

      <div className="mt-6 space-y-3">
        {MODALITIES.map((m) => (
          <ScoreBar key={m.key} label={m.label} model={m.model} value={result.scores?.[m.key]} />
        ))}
      </div>

      <div className="mt-5 flex flex-wrap gap-x-6 gap-y-1 border-t border-slate-800 pt-4 text-xs text-slate-400">
        <span>Fusion : <span className="text-slate-200">{result.fusion_method}</span></span>
        <span>Durée : <span className="font-mono text-slate-200">{result.elapsed.toFixed(2)} s</span></span>
        {html && (
          <span>
            HTML : {html.n_forms} form · {html.n_password} mdp · {html.n_inputs} inputs · {html.n_iframes} iframes
          </span>
        )}
      </div>
    </div>
  )
}
