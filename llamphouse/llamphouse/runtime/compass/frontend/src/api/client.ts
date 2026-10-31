/* ─── The 2.0 API, as Compass reads it ────────────────────────────────────────
 *  /v2 (threads, runs, files, subscriptions) and /v2/compass (what Compass adds:
 *  overview, debug views, agents, triggers, dashboards). Same origin; an API
 *  key, when the deployment asks for one, is kept in this browser only.
 * ───────────────────────────────────────────────────────────────────────────── */

import { reactive } from 'vue'

const KEY = 'llamphouse.compass.key'

export const session = reactive({
  key: (() => { try { return localStorage.getItem(KEY) || new URLSearchParams(location.search).get('key') || '' } catch { return '' } })(),
  needsKey: false,
})

export function setKey(key: string) {
  session.key = key
  session.needsKey = false
  try { key ? localStorage.setItem(KEY, key) : localStorage.removeItem(KEY) } catch { /* private mode */ }
}

function headers(extra?: HeadersInit): HeadersInit {
  return { 'Content-Type': 'application/json', ...(session.key ? { Authorization: `Bearer ${session.key}` } : {}), ...(extra as any) }
}

export class ApiError extends Error {
  constructor(public status: number, public body: string) {
    super(ApiError.describe(status, body))
  }
  static describe(status: number, body: string): string {
    try {
      const parsed = JSON.parse(body)
      const detail = parsed.detail ?? parsed.error?.message ?? parsed.error ?? body
      return `${status}: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`
    } catch { return `${status}: ${body}` }
  }
}

export async function api<T = any>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: headers(init?.headers) })
  if (res.status === 401) session.needsKey = true
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ''))
  const type = res.headers.get('content-type') || ''
  return (type.includes('json') ? res.json() : res.text()) as Promise<T>
}

const post = (path: string, body?: any) => api(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) })
const put = (path: string, body: any) => api(path, { method: 'PUT', body: JSON.stringify(body) })
const del = (path: string) => api(path, { method: 'DELETE' })

/** Server-sent events over fetch (so the API key header goes along): calls `each` per event until aborted. */
export async function follow(path: string, signal: AbortSignal, each: (data: any, event: string) => void): Promise<void> {
  const res = await fetch(path, { headers: headers(), signal })
  if (res.status === 401) session.needsKey = true
  if (!res.ok || !res.body) throw new ApiError(res.status, await res.text().catch(() => ''))
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) return
    buffer += decoder.decode(value, { stream: true })
    let i
    while ((i = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, i)
      buffer = buffer.slice(i + 2)
      let event = 'message'
      const data: string[] = []
      for (const line of block.split('\n')) {
        if (line.startsWith('event: ')) event = line.slice(7)
        else if (line.startsWith('data: ')) data.push(line.slice(6))
      }
      if (data.length) {
        try { each(JSON.parse(data.join('\n')), event) } catch { each(data.join('\n'), event) }
      }
    }
  }
}

/* ─── Types ──────────────────────────────────────────────────────────────── */

export interface Part { kind: string; [k: string]: any }
export interface Sender { kind: string; id: string; name?: string | null; run?: string | null }
export interface Message { id: string; sender: Sender; parts: Part[]; metadata?: Record<string, any>; references?: string[] }
export interface ThreadEvent {
  position: number; kind: string; thread: string; run: string | null; agent: string | null; to: string | null
  at: string | null; message: Message | null; data: any
}
export interface Run {
  id: string; thread: string; agent: string; status: string; parent: string | null; input: Message; output: any
  error: string | null; pending: any; activations: number; children: string[]; created: string[]; unread: Message[]
}
export interface Thread {
  id: string; agent: string; created_at: string | null; forked_from: { thread: string; position: number } | null
  created_by: string | null; closed: boolean; last_position: number; metadata: Record<string, any>
  config: Record<string, any> | null; active_run: Run | null; runs: string[]; created: string[]
}
export interface ThreadRow { id: string; agent: string; created_at: string | null; created_by: string | null; forked_from: string | null; closed: boolean; metadata: Record<string, any>; active_run: { id: string; status: string } | null }
export interface RunRow { id: string; thread_id: string; agent: string; status: string; created_at: string | null; ended_at: string | null }
export interface Why { level: 'info' | 'wait' | 'attention' | 'done' | 'error'; text: string; hint?: string }
export interface StoryItem { position: number; at: string | null; kind: string; text: string; thread?: string; run?: string }
export interface TreeNode { id: string; agent: string; status: string; kind: string; current?: boolean; thread?: string; closed?: boolean; children: TreeNode[] }
export interface Debug {
  id: string; agent: string; thread: string; status: string; why: Why[]; story: StoryItem[]
  steps: { key: string; name: string; n: number | null; value: any }[]
  waits_on: { key: string; filters: any; cursors: Record<string, number>; waiting: boolean }[]
  tree: TreeNode; created_at: string | null; ended_at: string | null; activations: number; generation: number
  lease: { owner: string | null; until: string | null }; ring_at: string | null; wake_at: string | null
  root: string | null; depth: number; parent: string | null; children: string[]; created: string[]
  caller: any; request: any; pending: any; output: any; error: string | null; run_state: any
  session: { id: string; version: number; revisions: number; revising: boolean; messages: number; provider: string } | null
}
export interface SessionView { id: string; agent: string; thread: string; version: number; revisions: number; revising: boolean; forked_from: string | null; messages: (Message & { own: boolean })[]; log: any[] }
export interface Overview {
  since: string; hours: number; threads: number; active: number; runs: number; by_status: Record<string, number>
  by_agent: Record<string, { statuses: Record<string, number>; runs: number; p50_ms: number | null; p95_ms: number | null }>
  timeline: { at: string; status: string; count: number }[]; attention: RunRow[]; failures: RunRow[]
}
export interface AgentView {
  name: string; display_name: string; type: string; description: string; version: string | null; requires: string[]
  class: string; permissions: { agents: string[] | '*'; files: string }; session_provider: string
  stats: { statuses: Record<string, number>; runs: number; p50_ms: number | null; p95_ms: number | null } | null
  schedules: Schedule[]; webhooks: Webhook[]
  config?: { effective: any; changed: Record<string, any>; version: number; updated_by: string | null; updated_at: string | null; schema: any; file: any }
  input_schema?: any; output_schema?: any; source?: string | null
}
export interface Schedule { id: string; agent: string; definition: string; next_at: string | null; data: any }
export interface Webhook { agent: string; name: string; path: string; auth: string; label: string | null; message: any; thread_metadata: any; idempotency: any; busy: string; curl: string }
export interface Info {
  version: string; engine: string; workers: number; role: string; components: Record<string, string | null>
  auth: string[]; access: string; adapters: { kind: string; prefix: string; agents: any }[]; agents: number
  you: { subject: string; tenant: string | null; admin: boolean }; sql: boolean
}
export interface Subscription { id: string; url: string; filter: any; format: string; headers: string[]; owner: string | null; meta: any; created_at: string | null }
export interface Delivery { id: string; seq: number; subscription_id: string; payload: any; attempts: number; next_at: string; delivered_at: string | null; dead: boolean; last_error: string | null; created_at: string }
export interface Span {
  TraceId: string; SpanId: string; ParentSpanId: string; SpanName: string; ServiceName?: string; Duration: number
  StatusCode: string; SpanAttributes: Record<string, any>; ResourceAttributes?: Record<string, string>; Timestamp: string; children?: Span[]
}
export interface TraceRow { TraceId: string; SpanName: string; run_id: string; thread_id: string; agent: string; duration_ms: number; StatusCode: string; Timestamp: string; span_count: number }
export interface ChartDef { id: string; title: string; sql: string; chart_type: 'table' | 'bar' | 'line' | 'bignum' | 'pie'; x_column: string | null; y_columns: string[]; created_at?: number; updated_at?: number }
export interface DashboardChart { chart_id: string; col_span?: 1 | 2 | 3 | 4; height_px?: number }
export type Chart = ChartDef
export interface Dashboard { id: string; title: string; description: string; charts: DashboardChart[]; created_at: number; updated_at: number }
export interface QueryResult { columns: string[]; rows: any[][]; duration_ms: number; error?: string; truncated?: boolean }
export interface AgentInfo { name: string; type: string; description: string; input_schema: any; config_schema: any; output_schema: any }

/* ─── Calls ──────────────────────────────────────────────────────────────── */

const C = '/v2/compass'
const qs = (o: Record<string, any>) => {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(o)) if (v !== undefined && v !== null && v !== '') p.set(k, String(v))
  const s = p.toString()
  return s ? `?${s}` : ''
}

export const v2 = {
  info: () => api<Info>(`${C}/info`),
  overview: (hours = 24) => api<Overview>(`${C}/overview${qs({ hours })}`),
  lookup: (id: string) => api<{ kind: string; id: string; thread: string; runs?: string[] }>(`${C}/lookup/${encodeURIComponent(id)}`),

  agents: () => api<AgentView[]>(`${C}/agents`),
  agent: (name: string) => api<AgentView>(`${C}/agents/${name}`),
  agentInfo: () => api<AgentInfo[]>('/v2/agents'),
  setConfig: (name: string, values: any, if_version?: number) => put(`/v2/agents/${name}/config`, { values, if_version }),

  threads: (o: { agent?: string; top?: boolean; limit?: number; metadata?: Record<string, string> } = {}) => {
    const meta = Object.fromEntries(Object.entries(o.metadata || {}).map(([k, v]) => [`metadata.${k}`, v]))
    return api<ThreadRow[]>(`/v2/threads${qs({ agent: o.agent, top: o.top, limit: o.limit, ...meta })}`)
  },
  thread: (id: string) => api<Thread>(`/v2/threads/${id}`),
  threadTree: (id: string) => api<{ root: string; tree: TreeNode }>(`${C}/threads/${id}/tree`),
  events: (id: string, after = 0) => api<ThreadEvent[]>(`/v2/threads/${id}/events${qs({ after })}`),
  state: (id: string, at?: number) => api<Record<string, { value: any; version: number }>>(`/v2/threads/${id}/state${qs({ at })}`),
  threadRuns: (id: string) => api<Run[]>(`/v2/threads/${id}/runs`),
  newThread: (body: { agent: string; metadata?: any; state?: any; config?: any }) => post('/v2/threads', body) as Promise<Thread>,
  append: (id: string, body: any) => post(`/v2/threads/${id}/messages`, body),
  start: (id: string, body: any) => post(`/v2/threads/${id}/runs`, body) as Promise<Run>,
  closeThread: (id: string) => post(`/v2/threads/${id}/close`),

  runs: (o: { agent?: string; status?: string; limit?: number; before?: string } = {}) => api<RunRow[]>(`${C}/runs${qs(o)}`),
  run: (id: string) => api<Run>(`/v2/runs/${id}`),
  debug: (id: string) => api<Debug>(`${C}/runs/${id}/debug`),
  session: (id: string) => api<SessionView>(`/v2/runs/${id}/session`),
  runEvents: (id: string) => api<any[]>(`/v2/runs/${id}/events`),
  send: (id: string, body: any) => post(`/v2/runs/${id}/messages`, body),
  answer: (id: string, body: any) => post(`/v2/runs/${id}/answer`, body),
  signal: (id: string, key: string, value: any) => post(`/v2/runs/${id}/signal`, { key, value }),
  cancel: (id: string) => post(`/v2/runs/${id}/cancel`),

  traces: (agent?: string, limit = 50) => api<{ traces: TraceRow[]; tracing: boolean; error?: string }>(`${C}/traces${qs({ agent, limit })}`),
  trace: (runId: string) => api<{ run: string; tracing: boolean; spans: Span[] }>(`/v2/runs/${runId}/trace`),

  schedules: () => api<Schedule[]>(`${C}/schedules`),
  fire: (id: string) => post(`${C}/schedules/${encodeURIComponent(id)}/fire`),
  webhooks: () => api<Webhook[]>(`${C}/webhooks`),

  subscriptions: () => api<Subscription[]>('/v2/subscriptions'),
  subscribe: (body: any) => post('/v2/subscriptions', body) as Promise<Subscription>,
  unsubscribe: (id: string) => del(`/v2/subscriptions/${id}`),
  deliveries: (id: string) => api<Delivery[]>(`/v2/subscriptions/${id}/deliveries`),
  retry: (id: string, delivery: string) => post(`/v2/subscriptions/${id}/deliveries/${delivery}/retry`),
}

/* the dashboards' calls, by the names the dashboard views use */
export const compass = {
  listDashboards: () => api<Dashboard[]>(`${C}/dashboards`),
  getDashboard: (id: string) => api<Dashboard>(`${C}/dashboards/${id}`),
  createDashboard: (title: string, description = '') => post(`${C}/dashboards`, { title, description, charts: [] }) as Promise<Dashboard>,
  updateDashboard: (id: string, data: Partial<Pick<Dashboard, 'title' | 'description' | 'charts'>>) => put(`${C}/dashboards/${id}`, data) as Promise<Dashboard>,
  deleteDashboard: (id: string) => del(`${C}/dashboards/${id}`),
  listCharts: () => api<ChartDef[]>(`${C}/charts`),
  getChart: (id: string) => api<ChartDef>(`${C}/charts/${id}`),
  createChart: (data: Partial<ChartDef>) => post(`${C}/charts`, data) as Promise<ChartDef>,
  updateChart: (id: string, data: Partial<ChartDef>) => put(`${C}/charts/${id}`, data) as Promise<ChartDef>,
  deleteChart: (id: string) => del(`${C}/charts/${id}`),
  runQuery: async (sql: string): Promise<QueryResult> => {
    try { return await post(`${C}/query`, { sql }) as QueryResult } catch (e: any) {
      return { columns: [], rows: [], duration_ms: 0, error: e instanceof ApiError ? e.message : String(e) }
    }
  },
  getRunTrace: async (runId: string) => (await v2.trace(runId)).spans,
}

/* ─── Helpers ────────────────────────────────────────────────────────────── */

export function statusBadge(status: string): string {
  switch (status) {
    case 'completed': return 'badge--success'
    case 'running': case 'queued': case 'waiting': return 'badge--info'
    case 'failed': case 'rejected': case 'cancelled': return 'badge--error'
    case 'input_required': case 'auth_required': return 'badge--warning'
    default: return 'badge--neutral'
  }
}

export function formatTs(value: number | string | null | undefined): string {
  if (!value) return '—'
  const d = typeof value === 'number' ? new Date(value * 1000) : new Date(value)
  const pad = (n: number, w = 2) => String(n).padStart(w, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

export function ago(value: string | null | undefined): string {
  if (!value) return '—'
  const s = (Date.now() - new Date(value).getTime()) / 1000
  if (s < 0) return `in ${until(value)}`
  if (s < 60) return `${Math.round(s)}s ago`
  if (s < 3600) return `${Math.round(s / 60)}m ago`
  if (s < 86400) return `${Math.round(s / 3600)}h ago`
  return `${Math.round(s / 86400)}d ago`
}

export function until(value: string | null | undefined): string {
  if (!value) return '—'
  const s = (new Date(value).getTime() - Date.now()) / 1000
  if (s <= 0) return 'now'
  if (s < 60) return `${Math.round(s)}s`
  if (s < 3600) return `${Math.round(s / 60)}m`
  if (s < 86400) return `${Math.round(s / 3600)}h`
  return `${Math.round(s / 86400)}d`
}

export function shortId(id: string | null | undefined): string {
  if (!id) return '—'
  return id.length <= 14 ? id : id.slice(0, 8) + '…' + id.slice(-4)
}

export function durationMs(start: number | string | null, end: number | string | null): string {
  if (!start || !end) return '—'
  const ms = typeof start === 'number' && typeof end === 'number' ? (end - start) * 1000
    : new Date(end as string).getTime() - new Date(start as string).getTime()
  return ms < 1000 ? `${Math.round(ms)}ms` : ms < 60000 ? `${(ms / 1000).toFixed(1)}s` : `${(ms / 60000).toFixed(1)}m`
}

export function textOf(m: Message | null | undefined): string {
  if (!m) return ''
  return (m.parts || []).map(p => p.kind === 'text' ? p.text : p.kind === 'data' ? JSON.stringify(p.data) : `[${p.kind}${p.name ? ' ' + p.name : ''}]`).join(' ')
}

export function fileUrl(uri: string): string {
  const id = uri.split(/[\/:]/).pop()
  return `/v2/files/${encodeURIComponent(id || '')}${session.key ? `?key=${encodeURIComponent(session.key)}` : ''}`
}

export function copy(text: string) {
  navigator.clipboard?.writeText(text).catch(() => {})
}
