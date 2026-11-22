const API = '/api'

async function json(res) {
  if (!res.ok) {
    let msg = `HTTP ${res.status}`
    try {
      const body = await res.json()
      msg = body.detail || msg
    } catch { /* corps non JSON */ }
    throw new Error(msg)
  }
  return res.json()
}

export const getHealth = () => fetch(`${API}/health`).then(json)
export const getHistory = () => fetch(`${API}/history`).then(json)
export const clearHistory = () => fetch(`${API}/history`, { method: 'DELETE' }).then(json)
export const deleteHistoryItem = (id) => fetch(`${API}/history/${id}`, { method: 'DELETE' }).then(json)

/**
 * Lance une analyse : upload multipart puis écoute du flux SSE.
 * handlers : { onLog, onResult, onFailed, onDone }
 * Retourne une fonction d'annulation.
 */
export async function runAnalysis({ url, html, screenshot }, handlers) {
  const form = new FormData()
  form.append('url', url)
  if (html) form.append('html', html)
  if (screenshot) form.append('screenshot', screenshot)

  const { job_id } = await fetch(`${API}/analyze`, { method: 'POST', body: form }).then(json)

  const es = new EventSource(`${API}/analyze/${job_id}/stream`)
  let finished = false
  const close = () => { finished = true; es.close() }

  es.addEventListener('log', (e) => handlers.onLog?.(JSON.parse(e.data)))
  es.addEventListener('result', (e) => handlers.onResult?.(JSON.parse(e.data)))
  es.addEventListener('failed', (e) => handlers.onFailed?.(JSON.parse(e.data)))
  es.addEventListener('done', () => { close(); handlers.onDone?.() })
  es.onerror = () => {
    if (finished) return
    close()
    handlers.onLog?.({ level: 'error', message: 'Connexion SSE interrompue', ts: Date.now() / 1000 })
    handlers.onDone?.()
  }
  return close
}
