/* ─── Compass API Client ─────────────────────────────────────
 *  All calls are relative to the Compass prefix so the same
 *  code works embedded (/compass) and standalone (/).
 * ─────────────────────────────────────────────────────────── */

const BASE = import.meta.env.BASE_URL.replace(/\/+$/, '')

async function api<T = any>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}/api${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers as any) },
    ...init,
  })
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new ApiError(res.status, body)
  }
  return res.json()
}

export class ApiError extends Error {
  constructor(public status: number, public body: string) {
    super(`API ${status}: ${body}`)
  }
}

/* ─── Types ──────────────────────────────────────────────── */

export interface Overview {
  assistants: number
  threads: number
  runs: number
  messages: number
}

/** Wrapper returned by most list endpoints */
interface ListResponse<T> {
  data: T[]
  total: number
}

export interface AgentSkill {
  id: string
  name: string
  description: string
}

export interface Assistant {
  id: string
  name: string
  description: string | null
  model: string
  skills: AgentSkill[]
  created_at: number
  metadata: Record<string, any>
}

export interface Thread {
  id: string
  created_at: number
  metadata: Record<string, any>
  agent_id: string | null
  agent_name: string | null
}

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: MessageContent[]
  created_at: number
  run_id: string | null
  assistant_id: string | null
  agent_name: string | null
  metadata: Record<string, any>
}

export interface MessageContent {
  type: string
  text?: string | { value: string; annotations?: any[] }
  image_file?: { file_id: string }
}

export interface Run {
  id: string
  thread_id: string
  assistant_id: string
  status: string
  model: string
  instructions: string | null
  tools: any[]
  created_at: number
  started_at: number | null
  completed_at: number | null
  failed_at: number | null
  last_error: any | null
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null
  metadata: Record<string, any>
}

export interface RunStep {
  id: string
  run_id: string
  type: 'message_creation' | 'tool_calls'
  status: string
  step_details: any
  created_at: number
  completed_at: number | null
  last_error: any | null
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null
}

export interface Span {
  TraceId: string
  SpanId: string
  ParentSpanId: string
  SpanName: string
  ServiceName: string
  Duration: number
  StatusCode: string
  SpanAttributes: Record<string, string>
  ResourceAttributes: Record<string, string>
  Timestamp: string
  children?: Span[]
}

export interface ConfigParamBase {
  key: string
  label: string
  description: string
  type: string
  default: any
}

export interface NumberParam extends ConfigParamBase {
  type: 'number'
  default: number
  min: number
  max: number
  step: number
}

export interface BooleanParam extends ConfigParamBase {
  type: 'boolean'
  default: boolean
}

export interface SelectParam extends ConfigParamBase {
  type: 'select'
  default: string
  options: string[]
}

export interface PromptParam extends ConfigParamBase {
  type: 'prompt'
  default: string
  variables: string[]
}

export interface StringParam extends ConfigParamBase {
  type: 'string'
  default: string
  max_length: number
}

export type ConfigParam = NumberParam | BooleanParam | SelectParam | PromptParam | StringParam

export interface FlowNode {
  id: string
  agent_id: string
  agent_name: string
  status: string
  dispatch_type: string | null
  duration_ms: number | null
  is_root: boolean
  thread_id: string | null
  created_at: number | null
}

export interface FlowEdge {
  source: string
  target: string
  type: string    // "call_agent" | "handover"
  sequence: number
}

export interface FlowData {
  nodes: FlowNode[]
  edges: FlowEdge[]
  has_flow: boolean
}

/* ─── API Functions ──────────────────────────────────────── */

export const compass = {
  /* Overview */
  overview: () => api<Overview>('/overview'),

  /* Assistants */
  listAssistants: async () => {
    const res = await api<ListResponse<Assistant>>('/assistants')
    return res.data ?? []
  },

  getAssistantConfig: (assistantId: string) =>
    api<{ params: ConfigParam[]; defaults: Record<string, any> }>(`/assistants/${assistantId}/config`),

  setAssistantConfig: (assistantId: string, values: Record<string, any>) =>
    api(`/assistants/${assistantId}/config`, {
      method: 'POST',
      body: JSON.stringify({ values }),
    }),

  /* Threads */
  listThreads: async (limit = 50) => {
    const res = await api<ListResponse<Thread>>(`/threads?limit=${limit}`)
    return res.data ?? []
  },

  /* Messages */
  listMessages: async (threadId: string) => {
    const res = await api<ListResponse<Message>>(`/threads/${threadId}/messages`)
    return res.data ?? []
  },

  /* Runs */
  listRuns: async (threadId: string) => {
    const res = await api<ListResponse<Run>>(`/threads/${threadId}/runs`)
    return res.data ?? []
  },

  getRunConfig: (threadId: string, runId: string) =>
    api(`/threads/${threadId}/runs/${runId}/config`),

  /* Run Steps */
  listRunSteps: async (threadId: string, runId: string) => {
    const res = await api<ListResponse<RunStep>>(`/threads/${threadId}/runs/${runId}/steps`)
    return res.data ?? []
  },

  /* Compare */
  compareRuns: (runIds: string[]) =>
    api(`/compare?run_ids=${runIds.join(',')}`),

  /* Traces */
  listTraces: async (limit = 50) => {
    const res = await api<{ traces: Span[]; total: number }>(`/traces?limit=${limit}`)
    return res.traces ?? []
  },

  getRunTrace: async (runId: string) => {
    const res = await api<{ traces: Span[]; total: number }>(`/traces/${runId}`)
    return res.traces ?? []
  },

  /* Flow */
  getRunFlow: (runId: string) =>
    api<FlowData>(`/runs/${runId}/flow`),
}

/* ─── Helpers ────────────────────────────────────────────── */

export function statusBadge(status: string): string {
  switch (status) {
    case 'completed': return 'badge--success'
    case 'in_progress':
    case 'queued': return 'badge--info'
    case 'failed':
    case 'cancelled':
    case 'expired': return 'badge--error'
    case 'requires_action': return 'badge--warning'
    default: return 'badge--neutral'
  }
}

export function formatTs(epoch: number): string {
  if (!epoch) return '—'
  return new Date(epoch * 1000).toLocaleString()
}

export function shortId(id: string): string {
  if (!id) return '—'
  if (id.length <= 12) return id
  return id.slice(0, 6) + '…' + id.slice(-4)
}

export function durationMs(start: number | null, end: number | null): string {
  if (!start || !end) return '—'
  const ms = (end - start) * 1000
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}
