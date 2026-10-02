/**
 * Query 模块 API — 对接 app/query_process/api/query_service.py (默认 :8001)
 * 开发时由 vite proxy 转发；生产构建后由 query_service 同源托管。
 */

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

// 统一请求超时：避免长时间白屏无反馈
const REQUEST_TIMEOUT_MS = 30_000

async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(url, { ...options, signal: controller.signal })
  } catch (err) {
    if (err?.name === 'AbortError') {
      throw new Error(`请求超时（${timeoutMs / 1000}s）：${url}`)
    }
    throw err
  } finally {
    clearTimeout(timer)
  }
}

export async function postQuery(query, sessionId = null, isStream = true) {
  const res = await fetchWithTimeout(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      session_id: sessionId,
      is_stream: isStream,
    }),
  })
  if (!res.ok) {
    const err = await res.text()
    throw new Error(err || `请求失败 (${res.status})`)
  }
  return res.json()
}

/**
 * 流式查询：POST /query + SSE /stream/{session_id}
 * @param {object} handlers - { onProgress, onDelta, onFinal, onError }
 */
export function streamQuery(query, sessionId, handlers = {}) {
  return new Promise(async (resolve, reject) => {
    let settled = false
    let es = null

    const finish = (result, isError = false) => {
      if (settled) return
      settled = true
      es?.close()
      isError ? reject(result) : resolve(result)
    }

    try {
      const data = await postQuery(query, sessionId, true)
      const sid = data.session_id
      if (!sid) throw new Error('后端未返回 session_id')

      es = new EventSource(`${API_BASE}/stream/${sid}`)
      let answer = ''

      es.addEventListener('ready', () => {
        handlers.onReady?.(sid)
      })

      es.addEventListener('progress', (e) => {
        try {
          const payload = JSON.parse(e.data)
          handlers.onProgress?.(payload)
        } catch { /* ignore */ }
      })

      es.addEventListener('delta', (e) => {
        try {
          const { delta } = JSON.parse(e.data)
          if (delta) {
            answer += delta
            handlers.onDelta?.(delta, answer)
          }
        } catch { /* ignore */ }
      })

      es.addEventListener('final', (e) => {
        try {
          const payload = JSON.parse(e.data)
          finish({
            answer: payload.answer || answer,
            sources: payload.sources ?? [],
            sessionId: sid,
            imageUrls: payload.image_urls ?? [],
          })
        } catch (err) {
          finish(err, true)
        }
      })

      es.addEventListener('error', (e) => {
        try {
          const payload = JSON.parse(e.data)
          finish(new Error(payload.error || '处理失败'), true)
        } catch {
          if (es.readyState === EventSource.CLOSED) {
            if (answer) {
              finish({ answer, sources: [], sessionId: sid })
            } else {
              finish(new Error('SSE 连接中断'), true)
            }
          }
        }
      })

      es.onerror = () => {
        if (!settled && es.readyState === EventSource.CLOSED) {
          if (answer) {
            finish({ answer, sources: [], sessionId: sid })
          } else {
            finish(new Error('无法连接流式服务'), true)
          }
        }
      }
    } catch (err) {
      finish(err, true)
    }
  })
}

export async function fetchHistory(sessionId, limit = 50) {
  const res = await fetchWithTimeout(`${API_BASE}/history/${sessionId}?limit=${limit}`)
  if (!res.ok) throw new Error('获取历史失败')
  return res.json()
}
