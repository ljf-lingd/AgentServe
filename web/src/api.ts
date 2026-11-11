export const databaseId = 'business_orders_v1'
export type Role = 'user' | 'admin'
export type Mode = 'sql' | 'knowledge' | 'metric' | 'analysis'
export type ViewMode = Mode | 'sql_explain'
export type Status = 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED'
export type DateWindow = { start: string; end: string; city?: string }
export type Input = {
  database_id: string
  session_id?: string
  mode?: Mode
  question?: string
  answer_mode?: 'explain'
  as_of?: string
  clause_id?: string
  metric?: DateWindow & { name: string; group_by?: 'city' | 'month' }
  analysis?: DateWindow
  save_to_history?: boolean
}
export type TaskError = { code: string; detail: string }
export type Task = {
  task_id: string
  turn_no: number | null
  status: Status
  input: Input
  error: TaskError | null
  in_history?: boolean
  report: null | {
    answer?: string
    answer_source?: string
    query_result?: { columns: string[]; rows: unknown[][]; truncated: boolean } | null
    citations: { path: string; anchor: string; version: string; quote: string }[]
    metric?: { unit: string }
    analysis?: { units: { revenue_cents: string; order_count: string } }
  }
}
export type Session = { session_id: string; title: string | null; created_at: number; active?: boolean }
export type RecentTask = { task_id: string; title: string | null; mode: ViewMode; status: Status }
export type User = { username: string; role: Role }
export type Access = { mode: 'demo' | 'auth'; user: User | null }

export const modeLabel: Record<ViewMode, string> = {
  sql: '多轮查数', sql_explain: '查数并解释', knowledge: '业务规则', metric: '指标查询', analysis: '经营简报',
}
export const statusLabel: Record<Status, string> = {
  QUEUED: '排队中', RUNNING: '执行中', COMPLETED: '已完成', FAILED: '失败', CANCELLED: '已取消',
}
export const activeTask = (status: Status) => status === 'QUEUED' || status === 'RUNNING'

export function monthIndex(value: string) {
  const [year, month] = value.split('-').map(Number)
  return year * 12 + month - 1
}

export function shiftMonth(value: string, amount: number) {
  const index = monthIndex(value) + amount
  return `${String(Math.floor(index / 12)).padStart(4, '0')}-${String(index % 12 + 1).padStart(2, '0')}`
}

export class ApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message)
  }
}

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options)
  if (!response.ok) {
    if (response.status === 401 && path !== '/auth/login') {
      window.dispatchEvent(new Event('agentserve-session-expired'))
    }
    const body = await response.json().catch(() => null)
    const message = body?.error?.detail ?? ({
      INVALID_CREDENTIALS: '账号或密码错误。',
      LOGIN_RATE_LIMITED: '登录尝试过多，请稍后再试。',
      ADMIN_REQUIRED: '该页面需要管理员权限。',
    } as Record<string, string>)[body?.error?.code]
    throw new ApiError(response.status, message ?? ('请求失败（HTTP ' + response.status + '）'))
  }
  return response.status === 204 ? undefined as T : response.json()
}

export function post<T>(path: string, body?: unknown, key?: string) {
  const headers = new Headers()
  if (body !== undefined) headers.set('Content-Type', 'application/json')
  if (key) headers.set('Idempotency-Key', key)
  return api<T>(path, { method: 'POST', body: JSON.stringify(body), headers })
}

export function errorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : '无法连接服务，请检查连接后重试。'
}
