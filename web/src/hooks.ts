import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError, errorMessage, post, type Input } from './api'

export function useLoad<T>(load: (signal: AbortSignal) => Promise<T>, revision = 0, interval = 0) {
  const [data, setData] = useState<T>()
  const [error, setError] = useState<string>()
  const [retry, setRetry] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    let next: number
    async function run() {
      try {
        const value = await load(controller.signal)
        if (controller.signal.aborted) return
        setData(value)
        setError(undefined)
      } catch (error) {
        if (!controller.signal.aborted) setError(errorMessage(error))
      }
      if (interval && !controller.signal.aborted) next = window.setTimeout(run, interval)
    }
    void run()
    return () => {
      controller.abort()
      window.clearTimeout(next)
    }
  }, [load, revision, retry, interval])

  return { data, error, reload: () => setRetry((value) => value + 1) }
}

export function useJson<T>(path: string, revision = 0, interval = 0) {
  const load = useCallback((signal: AbortSignal) => api<T>(path, { signal }), [path])
  return useLoad(load, revision, interval)
}

type Pending = { input: Input; key: string }

export function useSubmission(scope: string, onAccepted: (id: string) => void) {
  const storageKey = 'agentserve:pending:' + scope
  const [pending, setPending] = useState<Pending | null>(() => {
    const previous = sessionStorage.getItem(storageKey)
    return previous ? JSON.parse(previous) as Pending : null
  })
  const [error, setError] = useState<string>()
  const [busy, setBusy] = useState(false)
  const inFlight = useRef(false)

  function remember(value: Pending | null) {
    setPending(value)
    if (value) sessionStorage.setItem(storageKey, JSON.stringify(value))
    else sessionStorage.removeItem(storageKey)
  }

  async function submit(input?: Input) {
    if (inFlight.current || (input && pending)) return false
    const attempt = input ? { input, key: crypto.randomUUID() } : pending
    if (!attempt) return false
    inFlight.current = true
    setBusy(true)
    setError(undefined)
    remember(attempt)
    try {
      const result = await post<{ task_id: string }>('/tasks', attempt.input, attempt.key)
      remember(null)
      onAccepted(result.task_id)
      return true
    } catch (error) {
      if (error instanceof ApiError && error.status >= 400 && error.status < 500) remember(null)
      setError(errorMessage(error))
      return false
    } finally {
      inFlight.current = false
      setBusy(false)
    }
  }

  return { submit, pending, error, busy }
}
