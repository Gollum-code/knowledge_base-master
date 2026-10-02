/**
 * Import 模块 API — 对接 app/import_process/api/file_import_service.py (默认 :8000)
 */

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

// 上传超时：大文件走服务端中转，超时时间放宽
const UPLOAD_TIMEOUT_MS = 120_000
const STATUS_TIMEOUT_MS = 15_000

async function fetchWithTimeout(url, options = {}, timeoutMs = STATUS_TIMEOUT_MS) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(url, { ...options, signal: controller.signal })
  } catch (err) {
    if (err?.name === 'AbortError') {
      throw new Error(`请求超时（${timeoutMs / 1000}s）`)
    }
    throw err
  } finally {
    clearTimeout(timer)
  }
}

export async function uploadFiles(files) {
  const formData = new FormData()
  for (const file of files) {
    formData.append('files', file)
  }
  const res = await fetchWithTimeout(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  }, UPLOAD_TIMEOUT_MS)
  if (!res.ok) {
    const err = await res.text()
    throw new Error(err || `上传失败 (${res.status})`)
  }
  return res.json()
}

/**
 * 图片直传：先取预签名 URL，再由浏览器直接 PUT 到对象存储，减少服务端带宽中转。
 * MinIO 不可用或调用失败时抛错，调用方应回退到 uploadFiles。
 */
export async function presignUpload(filename) {
  const res = await fetchWithTimeout(
    `${API_BASE}/presign-upload?filename=${encodeURIComponent(filename)}`,
    { method: 'POST' },
  )
  if (!res.ok) throw new Error(`获取直传地址失败 (${res.status})`)
  return res.json()
}

export async function fetchTaskStatus(taskId) {
  const res = await fetchWithTimeout(`${API_BASE}/status/${taskId}`)
  if (!res.ok) {
    const err = await res.text()
    throw new Error(err || `状态查询失败 (${res.status})`)
  }
  return res.json()
}

function normalizeStatus(data) {
  return {
    status: data.status || 'processing',
    done_list: data.done_list ?? [],
    running_list: data.running_list ?? [],
  }
}

/**
 * SSE 流式订阅任务进度（实时推送，替代轮询）
 * @param {string} taskId
 * @param {{ onUpdate, onComplete, onError }} handlers
 * @returns {() => void} stop
 */
export function streamTaskStatus(taskId, { onUpdate, onComplete, onError }) {
  let es = null
  let stopped = false

  const stop = () => {
    stopped = true
    es?.close()
  }

  const applyUpdate = (raw) => {
    const data = normalizeStatus(raw)
    onUpdate?.(data)
    return data
  }

  const finish = (data) => {
    stop()
    onComplete?.(data)
  }

  // 拉取一次当前状态，避免 SSE 连接前丢失的上传阶段事件
  fetchTaskStatus(taskId)
    .then(applyUpdate)
    .catch(() => {})

  es = new EventSource(`${API_BASE}/stream/${taskId}`)

  es.addEventListener('ready', () => {
    fetchTaskStatus(taskId)
      .then(applyUpdate)
      .catch(() => {})
  })

  es.addEventListener('progress', (e) => {
    try {
      applyUpdate(JSON.parse(e.data))
    } catch { /* ignore */ }
  })

  es.addEventListener('final', (e) => {
    try {
      finish(applyUpdate(JSON.parse(e.data)))
    } catch (err) {
      stop()
      onError?.(err)
    }
  })

  es.addEventListener('error', (e) => {
    try {
      const payload = JSON.parse(e.data)
      stop()
      onError?.(new Error(payload.error || '处理失败'))
    } catch { /* SSE 连接级 error，见 onerror */ }
  })

  es.onerror = () => {
    if (stopped || es.readyState !== EventSource.CLOSED) return
    fetchTaskStatus(taskId)
      .then((raw) => {
        const data = applyUpdate(raw)
        if (data.status === 'completed' || data.status === 'failed') {
          finish(data)
        } else {
          onError?.(new Error('SSE 连接中断'))
        }
      })
      .catch((err) => onError?.(err))
  }

  return stop
}

/** @deprecated 保留兼容，推荐使用 streamTaskStatus */
export function pollTaskStatus(taskId, { interval = 1000, onUpdate, onComplete, onError }) {
  let timer = null
  let stopped = false

  const stop = () => {
    stopped = true
    if (timer) clearInterval(timer)
  }

  const tick = async () => {
    if (stopped) return
    try {
      const data = await fetchTaskStatus(taskId)
      onUpdate?.(normalizeStatus(data))
      if (data.status === 'completed' || data.status === 'failed') {
        stop()
        onComplete?.(normalizeStatus(data))
      }
    } catch (err) {
      stop()
      onError?.(err)
    }
  }

  tick()
  timer = setInterval(tick, interval)
  return stop
}
