export default function History({ items, selectedId, onSelect, onDelete, onClear }) {
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60">
      <div className="flex items-center border-b border-slate-800 px-5 py-3">
        <h2 className="font-semibold text-slate-100">Historique</h2>
        <span className="ml-2 rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-400">{items.length}</span>
        {items.length > 0 && (
          <button onClick={onClear} className="ml-auto text-xs text-slate-500 hover:text-red-400">
            Tout effacer
          </button>
        )}
      </div>

      {items.length === 0 ? (
        <p className="px-5 py-6 text-sm text-slate-500">Aucune analyse pour l'instant.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-5 py-2 font-medium">Date</th>
                <th className="px-5 py-2 font-medium">URL</th>
                <th className="px-5 py-2 font-medium">Verdict</th>
                <th className="px-5 py-2 text-right font-medium">Confiance</th>
                <th className="px-5 py-2" />
              </tr>
            </thead>
            <tbody>
              {items.map((h) => {
                const phishing = h.label === 'phishing'
                return (
                  <tr
                    key={h.id}
                    onClick={() => onSelect(h)}
                    className={`cursor-pointer border-t border-slate-800/70 hover:bg-slate-800/50 ${selectedId === h.id ? 'bg-slate-800/70' : ''}`}
                  >
                    <td className="whitespace-nowrap px-5 py-2.5 font-mono text-xs text-slate-400">
                      {h.timestamp.replace('T', ' ')}
                    </td>
                    <td className="max-w-[28rem] truncate px-5 py-2.5 font-mono text-xs text-slate-300" title={h.url}>
                      {h.url || <span className="text-slate-600">—</span>}
                    </td>
                    <td className="px-5 py-2.5">
                      <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${phishing ? 'bg-red-500/20 text-red-300' : 'bg-emerald-500/20 text-emerald-300'}`}>
                        {phishing ? 'Phishing' : 'Légitime'}
                      </span>
                    </td>
                    <td className="px-5 py-2.5 text-right font-mono text-xs text-slate-200">
                      {(h.confidence * 100).toFixed(1)} %
                    </td>
                    <td className="px-5 py-2.5 text-right">
                      <button
                        onClick={(e) => { e.stopPropagation(); onDelete(h.id) }}
                        className="text-slate-600 hover:text-red-400"
                        aria-label="Supprimer"
                      >
                        ✕
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
