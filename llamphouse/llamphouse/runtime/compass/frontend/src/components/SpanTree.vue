<script setup lang="ts">
import { ref, computed, onBeforeUnmount } from 'vue'
import type { Span } from '../api/client'
import ContentBlock from './ContentBlock.vue'

const props = defineProps<{
  spans: Span[]
}>()

/* Flatten spans into a tree and then to a renderable list with depth */
interface FlatNode {
  span: Span
  depth: number
  hasChildren: boolean
  id: string
}

const collapsed = ref(new Set<string>())
const selectedSpanId = ref<string | null>(null)
const rootEl = ref<HTMLElement | null>(null)
const inspectorWidth = ref(440)

let resizing = false
let startX = 0
let startWidth = 440

function maxInspectorWidth(): number {
  const total = rootEl.value?.clientWidth ?? 1200
  return Math.max(360, total - 520)
}

function beginResize(e: MouseEvent) {
  resizing = true
  startX = e.clientX
  startWidth = inspectorWidth.value
  window.addEventListener('mousemove', onResizeMove)
  window.addEventListener('mouseup', endResize)
  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
}

function onResizeMove(e: MouseEvent) {
  if (!resizing) return
  const delta = startX - e.clientX
  const next = startWidth + delta
  inspectorWidth.value = Math.min(maxInspectorWidth(), Math.max(340, next))
}

function endResize() {
  resizing = false
  window.removeEventListener('mousemove', onResizeMove)
  window.removeEventListener('mouseup', endResize)
  document.body.style.cursor = ''
  document.body.style.userSelect = ''
}

onBeforeUnmount(() => {
  endResize()
})

function toggle(id: string) {
  const next = new Set(collapsed.value)
  if (next.has(id)) {
    next.delete(id)
  } else {
    next.add(id)
  }
  collapsed.value = next
}

function selectSpan(id: string) {
  selectedSpanId.value = selectedSpanId.value === id ? null : id
}

const flatList = computed<FlatNode[]>(() => {
  // Build parent→children map
  const childrenMap = new Map<string, Span[]>()
  const rootSpans: Span[] = []

  for (const s of props.spans) {
    if (!s.ParentSpanId || !props.spans.some((p) => p.SpanId === s.ParentSpanId)) {
      rootSpans.push(s)
    } else {
      const siblings = childrenMap.get(s.ParentSpanId) || []
      siblings.push(s)
      childrenMap.set(s.ParentSpanId, siblings)
    }
  }

  // DFS flatten
  const result: FlatNode[] = []

  function walk(span: Span, depth: number) {
    const children = childrenMap.get(span.SpanId) || []
    result.push({
      span,
      depth,
      hasChildren: children.length > 0,
      id: span.SpanId,
    })

    if (!collapsed.value.has(span.SpanId)) {
      for (const child of children) {
        walk(child, depth + 1)
      }
    }
  }

  for (const root of rootSpans) {
    walk(root, 0)
  }

  return result
})

function durationLabel(ns: number): string {
  const ms = ns / 1_000_000
  if (ms < 1) return `${Math.round(ns / 1000)}µs`
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function statusColor(code: string): string {
  switch (code) {
    case 'STATUS_CODE_OK': return 'var(--success)'
    case 'STATUS_CODE_ERROR': return 'var(--error)'
    default: return 'var(--text-muted)'
  }
}

/**
 * Return an effective OTel-style status code for a span, promoting
 * `gen_ai.task.status` (e.g. "success" / "error" / "failed") to
 * STATUS_CODE_OK / STATUS_CODE_ERROR when the underlying span status is
 * unset. Some instrumentations (Traceloop, LangChain) leave the OTel
 * status unset and record success via a task attribute instead.
 */
function effectiveStatus(span: any): string {
  const code = span?.StatusCode || ''
  if (code === 'STATUS_CODE_OK' || code === 'STATUS_CODE_ERROR') return code
  const task = String(getAttr(span, 'gen_ai.task.status') || '').toLowerCase()
  if (!task) return code
  if (task === 'success' || task === 'ok' || task === 'succeeded' || task === 'completed') {
    return 'STATUS_CODE_OK'
  }
  if (task === 'error' || task === 'failed' || task === 'failure') {
    return 'STATUS_CODE_ERROR'
  }
  return code
}

function spanIcon(name: string): string {
  if (name.includes('.call_agent')) return '→'
  if (name.includes('.handover')) return '⇒'
  if (name.includes('.openai') || name.includes('chat ')) return 'AI'
  if (name.includes('.streaming') || name.includes('.stream')) return 'ST'
  if (name.includes('.data_store')) return 'DB'
  if (name.includes('.queue')) return 'Q'
  if (name.includes('.worker')) return 'W'
  if (name.includes('.run.create') || name.includes('.run.')) return 'R'
  if (name.includes('.thread')) return 'TH'
  if (name.includes('.message')) return 'M'
  if (name.includes('.client')) return 'CL'
  if (name.includes('.server')) return 'SV'
  return '·'
}

/** SVG icon markup per span type — small, monochrome, currentColor. */
function spanTypeSvg(name: string): string {
  const t = spanType(name)
  const svgs: Record<string, string> = {
    llm: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.9 4.1L18 9l-4.1 1.9L12 15l-1.9-4.1L6 9l4.1-1.9L12 3z"/><path d="M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9L19 15z"/></svg>',
    tool: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a4 4 0 015.7 5.7l-3.5 3.5-7.1 7.1a1 1 0 01-1.4 0l-2.8-2.8a1 1 0 010-1.4l7.1-7.1 2-2z"/><path d="M9 11l4 4"/></svg>',
    dispatch: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 3v6a3 3 0 003 3h6a3 3 0 013 3v6"/><path d="M3 6l3-3 3 3"/><path d="M15 21l3-3-3-3"/></svg>',
    internal: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 00.3 1.9l.1.1a2 2 0 11-2.8 2.8l-.1-.1a1.7 1.7 0 00-1.9-.3 1.7 1.7 0 00-1 1.5V21a2 2 0 11-4 0v-.1a1.7 1.7 0 00-1-1.5 1.7 1.7 0 00-1.9.3l-.1.1a2 2 0 11-2.8-2.8l.1-.1a1.7 1.7 0 00.3-1.9 1.7 1.7 0 00-1.5-1H3a2 2 0 110-4h.1a1.7 1.7 0 001.5-1 1.7 1.7 0 00-.3-1.9l-.1-.1a2 2 0 112.8-2.8l.1.1a1.7 1.7 0 001.9.3H9a1.7 1.7 0 001-1.5V3a2 2 0 114 0v.1a1.7 1.7 0 001 1.5 1.7 1.7 0 001.9-.3l.1-.1a2 2 0 112.8 2.8l-.1.1a1.7 1.7 0 00-.3 1.9V9a1.7 1.7 0 001.5 1H21a2 2 0 110 4h-.1a1.7 1.7 0 00-1.5 1z"/></svg>',
    infra: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v6c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 11v6c0 1.66 4 3 9 3s9-1.34 9-3v-6"/></svg>',
    app: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"/><path d="M3.3 7L12 12l8.7-5"/><path d="M12 22V12"/></svg>',
  }
  return svgs[t] || svgs.app
}

function spanCategory(name: string): string {
  if (name.includes('.call_agent')) return 'Call Agent'
  if (name.includes('.handover')) return 'Handover'
  if (name.includes('.openai') || name.includes('chat ')) return 'LLM'
  if (name.includes('.streaming') || name.includes('.stream')) return 'Stream'
  if (name.includes('.data_store')) return 'Store'
  if (name.includes('.queue')) return 'Queue'
  if (name.includes('.worker')) return 'Worker'
  if (name.includes('.run')) return 'Run'
  if (name.includes('.thread')) return 'Thread'
  if (name.includes('.message')) return 'Message'
  if (name.includes('.client')) return 'Client'
  return ''
}

function spanType(name: string): 'llm' | 'tool' | 'dispatch' | 'internal' | 'infra' | 'app' {
  const lower = (name || '').toLowerCase()
  if (lower.includes('openai') || lower.includes('langchain') || lower.includes('chat ') || lower.includes('gen_ai')) {
    return 'llm'
  }
  if (lower.includes('tool') || lower.includes('function_call')) return 'tool'
  if (lower.includes('call_agent') || lower.includes('handover') || lower.includes('dispatch')) return 'dispatch'
  if (lower.startsWith('llamphouse.')) return 'internal'
  if (lower.includes('queue') || lower.includes('worker') || lower.includes('data_store')) return 'infra'
  return 'app'
}

function spanTypeLabel(name: string): string {
  const t = spanType(name)
  if (t === 'llm') return 'LLM'
  if (t === 'tool') return 'Tool'
  if (t === 'dispatch') return 'Dispatch'
  if (t === 'internal') return 'Internal'
  if (t === 'infra') return 'Infra'
  return 'App'
}

function getAttr(span: Span, key: string): string {
  const raw = span.SpanAttributes?.[key]
  if (raw == null) return ''
  if (typeof raw === 'string') return raw
  if (typeof raw === 'number' || typeof raw === 'boolean') return String(raw)
  try { return JSON.stringify(raw) } catch { return String(raw) }
}

function getAttrRaw(span: Span, key: string): unknown {
  return span.SpanAttributes?.[key]
}

function firstAttr(span: Span, keys: string[]): string {
  for (const key of keys) {
    const v = getAttr(span, key)
    if (v !== '') return v
  }
  return ''
}

function truncate(text: unknown, max = 200): string {
  const s = typeof text === 'string' ? text : String(text ?? '')
  if (!s || s.length <= max) return s
  return s.slice(0, max) + '…'
}

const selectedSpan = computed<Span | null>(() => {
  if (!selectedSpanId.value) return null
  return props.spans.find(s => s.SpanId === selectedSpanId.value) ?? null
})

function prettyJson(value: string): string {
  try {
    return JSON.stringify(JSON.parse(value), null, 2)
  } catch {
    return value
  }
}

function parseJsonObject(value: string): Record<string, any> | null {
  if (!value) return null
  try {
    const parsed = JSON.parse(value)
    return parsed && typeof parsed === 'object' ? parsed : null
  } catch {
    return null
  }
}

function parseDelimitedList(value: unknown): string[] {
  if (value == null || value === '') return []
  if (Array.isArray(value)) return value.map((v) => String(v).trim()).filter(Boolean)
  if (typeof value !== 'string') {
    try {
      const parsed = JSON.parse(String(value))
      if (Array.isArray(parsed)) return parsed.map((v) => String(v).trim()).filter(Boolean)
    } catch { /* ignore */ }
    return [String(value)]
  }
  // Try JSON-encoded array first (e.g. "[\"a\",\"b\"]")
  const trimmed = value.trim()
  if (trimmed.startsWith('[')) {
    try {
      const parsed = JSON.parse(trimmed)
      if (Array.isArray(parsed)) return parsed.map((v) => String(v).trim()).filter(Boolean)
    } catch { /* fall through */ }
  }
  return trimmed.split(',').map((v) => v.trim()).filter(Boolean)
}

function prettyMaybeJson(value: string): string {
  if (!value) return ''
  try {
    return JSON.stringify(JSON.parse(value), null, 2)
  } catch {
    return value
  }
}

function findNumeric(obj: Record<string, any>, keys: string[]): number | null {
  for (const key of keys) {
    const parts = key.split('.')
    let curr: any = obj
    for (const part of parts) {
      if (curr == null || typeof curr !== 'object' || !(part in curr)) {
        curr = undefined
        break
      }
      curr = curr[part]
    }
    const n = Number(curr)
    if (Number.isFinite(n) && n >= 0) return n
  }
  return null
}

function toNum(value: unknown): number | null {
  if (value === '' || value == null) return null
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

function extractTokens(span: Span): { total: number | null; prompt: number | null; completion: number | null } | null {
  // Preferred: OTel GenAI semconv attributes emitted by traceloop / langchain instrumentor
  const genPrompt = toNum(getAttr(span, 'gen_ai.usage.input_tokens'))
  const genCompletion = toNum(getAttr(span, 'gen_ai.usage.output_tokens'))
  const genTotal = toNum(getAttr(span, 'gen_ai.usage.total_tokens'))
  if (genPrompt !== null || genCompletion !== null || genTotal !== null) {
    const total = genTotal ?? ((genPrompt ?? 0) + (genCompletion ?? 0) || null)
    return { total, prompt: genPrompt, completion: genCompletion }
  }

  // Legacy: OpenInference / LangSmith style attrs
  const legacyPrompt = toNum(getAttr(span, 'llm.token_count.prompt'))
  const legacyCompletion = toNum(getAttr(span, 'llm.token_count.completion'))
  const legacyTotal = toNum(getAttr(span, 'llm.token_count.total'))
  if (legacyPrompt !== null || legacyCompletion !== null || legacyTotal !== null) {
    return { total: legacyTotal, prompt: legacyPrompt, completion: legacyCompletion }
  }

  // Fallback: dig into input/output JSON blobs
  const inputObj = parseJsonObject(getAttr(span, 'input.value'))
  const outputObj = parseJsonObject(getAttr(span, 'output.value'))
  const total = outputObj ? findNumeric(outputObj, ['usage.total_tokens', 'total_tokens']) : null
  const prompt = outputObj ? findNumeric(outputObj, ['usage.prompt_tokens', 'prompt_tokens']) : null
  const completion = outputObj ? findNumeric(outputObj, ['usage.completion_tokens', 'completion_tokens']) : null
  if (total !== null || prompt !== null || completion !== null) return { total, prompt, completion }
  if (inputObj) {
    const fallbackPrompt = findNumeric(inputObj, ['usage.prompt_tokens', 'prompt_tokens'])
    const fallbackCompletion = findNumeric(inputObj, ['usage.completion_tokens', 'completion_tokens'])
    const fallbackTotal = findNumeric(inputObj, ['usage.total_tokens', 'total_tokens'])
    if (fallbackPrompt !== null || fallbackCompletion !== null || fallbackTotal !== null) {
      return { total: fallbackTotal, prompt: fallbackPrompt, completion: fallbackCompletion }
    }
  }
  return null
}

/** Normalize a gen_ai message (or system_instruction) into a display object. */
interface ChatMessage {
  role: string
  content: string
  toolCalls?: Array<{ name: string; args: string; id?: string }>
  toolCallId?: string
  name?: string
}

function partsToText(parts: any): string {
  if (parts == null) return ''
  if (typeof parts === 'string') return parts
  if (!Array.isArray(parts)) return stringifyMaybe(parts)
  const out: string[] = []
  for (const p of parts) {
    if (p == null) continue
    if (typeof p === 'string') { out.push(p); continue }
    // gen_ai part shapes: { type: 'text', content: '...' } | { type: 'tool_call', ... } | { type: 'tool_call_response', ... }
    if (p.type === 'text' && typeof p.content === 'string') { out.push(p.content); continue }
    if (typeof p.text === 'string') { out.push(p.text); continue }
    if (typeof p.content === 'string') { out.push(p.content); continue }
    out.push(stringifyMaybe(p))
  }
  return out.join('\n')
}

function normalizeMessage(raw: any): ChatMessage | null {
  if (raw == null) return null
  if (typeof raw === 'string') return { role: 'user', content: raw }
  const role = raw.role || raw.type || 'message'
  let content = ''
  const toolCalls: Array<{ name: string; args: string; id?: string }> = []

  if (Array.isArray(raw.parts)) {
    content = partsToText(raw.parts)
    for (const p of raw.parts) {
      if (p && p.type === 'tool_call') {
        toolCalls.push({
          name: p.name || p.function?.name || 'tool',
          args: stringifyMaybe(p.arguments ?? p.function?.arguments ?? p.args ?? ''),
          id: p.id || p.tool_call_id,
        })
      }
    }
  } else if (typeof raw.content === 'string') {
    content = raw.content
  } else if (Array.isArray(raw.content)) {
    content = partsToText(raw.content)
  } else if (raw.content != null) {
    content = stringifyMaybe(raw.content)
  }

  if (Array.isArray(raw.tool_calls)) {
    for (const tc of raw.tool_calls) {
      toolCalls.push({
        name: tc.function?.name || tc.name || 'tool',
        args: stringifyMaybe(tc.function?.arguments ?? tc.arguments ?? tc.args ?? ''),
        id: tc.id,
      })
    }
  }

  return {
    role,
    content,
    toolCalls: toolCalls.length ? toolCalls : undefined,
    toolCallId: raw.tool_call_id,
    name: raw.name,
  }
}

function parseMessagesAttr(value: string): ChatMessage[] {
  if (!value) return []
  let parsed: any
  try { parsed = JSON.parse(value) } catch { return [] }
  const arr = Array.isArray(parsed) ? parsed : [parsed]
  return arr.map(normalizeMessage).filter((m): m is ChatMessage => m !== null)
}

function extractChatMessages(span: Span): { system: ChatMessage[]; input: ChatMessage[]; output: ChatMessage[] } {
  const system = parseMessagesAttr(getAttr(span, 'gen_ai.system_instructions'))
    .map((m) => ({ ...m, role: 'system' }))
  const input = parseMessagesAttr(getAttr(span, 'gen_ai.input.messages'))
  const output = parseMessagesAttr(getAttr(span, 'gen_ai.output.messages'))
  return { system, input, output }
}

function tokenTotal(span: Span): string {
  const t = extractTokens(span)
  if (!t || t.total == null) return '—'
  return String(t.total)
}

function stringifyMaybe(value: unknown): string {
  if (value == null) return ''
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function extractMessages(span: Span): { input: string | null; output: string | null } {
  const inputObj = parseJsonObject(getAttr(span, 'input.value'))
  const outputObj = parseJsonObject(getAttr(span, 'output.value'))

  const inputMessage = inputObj?.message ?? inputObj?.messages ?? inputObj?.text ?? null
  const outputMessage = outputObj?.message ?? outputObj?.messages ?? outputObj?.text ?? null

  return {
    input: inputMessage != null ? stringifyMaybe(inputMessage) : null,
    output: outputMessage != null ? stringifyMaybe(outputMessage) : null,
  }
}

function extractToolCalls(span: Span): Array<{ name: string; id: string; args: string; output: string }> {
  const result: Array<{ name: string; id: string; args: string; output: string }> = []
  const inputObj = parseJsonObject(getAttr(span, 'input.value'))
  const outputObj = parseJsonObject(getAttr(span, 'output.value'))
  const toolCalls = (inputObj?.tool_calls || inputObj?.tools || outputObj?.tool_calls || []) as any[]

  for (const raw of toolCalls) {
    const call = raw?.root ?? raw
    const fn = call?.function ?? {}
    const name = fn?.name || call?.name || call?.type || 'tool'
    const id = call?.id || call?.tool_call_id || ''
    const args = stringifyMaybe(fn?.arguments ?? call?.arguments ?? '')
    const output = stringifyMaybe(call?.output ?? outputObj?.tool_output ?? '')
    result.push({ name, id, args, output })
  }

  if (result.length === 0 && outputObj?.tool_call_id) {
    result.push({
      name: outputObj?.function_name || 'tool',
      id: String(outputObj.tool_call_id),
      args: stringifyMaybe(inputObj?.arguments ?? inputObj?.tool_arguments ?? ''),
      output: stringifyMaybe(outputObj?.output ?? ''),
    })
  }

  return result
}

const selectedTokens = computed(() => (selectedSpan.value ? extractTokens(selectedSpan.value) : null))
const selectedMessages = computed(() => (selectedSpan.value ? extractMessages(selectedSpan.value) : { input: null, output: null }))
const selectedToolCalls = computed(() => (selectedSpan.value ? extractToolCalls(selectedSpan.value) : []))
const selectedChatMessages = computed(() => (selectedSpan.value ? extractChatMessages(selectedSpan.value) : { system: [], input: [], output: [] }))
const hasChatMessages = computed(() => {
  const c = selectedChatMessages.value
  return c.system.length + c.input.length + c.output.length > 0
})

const selectedLevel = computed<string>(() => {
  if (!selectedSpan.value) return ''
  // Explicit level attribute (Langfuse-style)
  const explicit = firstAttr(selectedSpan.value, [
    'langfuse.observation.level',
    'llamphouse.level',
    'level',
  ])
  if (explicit) return explicit.toUpperCase()
  // Derive from status (including gen_ai.task.status fallback)
  const status = effectiveStatus(selectedSpan.value)
  if (status.includes('ERROR')) return 'ERROR'
  if (status.includes('OK')) return 'DEFAULT'
  return ''
})

const selectedWorkflowNodes = computed(() => {
  if (!selectedSpan.value) return []
  return parseDelimitedList(getAttrRaw(selectedSpan.value, 'gen_ai.workflow.nodes'))
})

const selectedWorkflowEdges = computed(() => {
  if (!selectedSpan.value) return []
  return parseDelimitedList(getAttrRaw(selectedSpan.value, 'gen_ai.workflow.edges'))
})

const selectedTraceLoopInfo = computed(() => {
  if (!selectedSpan.value) return null
  return {
    workflowName: firstAttr(selectedSpan.value, ['traceloop.workflow.name', 'gen_ai.agent.name']),
    entityName: getAttr(selectedSpan.value, 'traceloop.entity.name'),
    entityPath: getAttr(selectedSpan.value, 'traceloop.entity.path'),
    spanKind: getAttr(selectedSpan.value, 'traceloop.span.kind'),
    integration: getAttr(selectedSpan.value, 'traceloop.association.properties.ls_integration'),
    taskStatus: getAttr(selectedSpan.value, 'gen_ai.task.status'),
  }
})

const selectedTaskInput = computed(() => {
  if (!selectedSpan.value) return ''
  return firstAttr(selectedSpan.value, [
    'traceloop.entity.input',
    'gen_ai.task.input',
    'input.value',
  ])
})

const selectedTaskOutput = computed(() => {
  if (!selectedSpan.value) return ''
  return firstAttr(selectedSpan.value, [
    'traceloop.entity.output',
    'gen_ai.task.output',
    'output.value',
  ])
})

const selectedAllAttrs = computed<Array<{ key: string; value: string }>>(() => {
  if (!selectedSpan.value?.SpanAttributes) return []
  return Object.entries(selectedSpan.value.SpanAttributes).map(([key, value]) => ({
    key,
    value: typeof value === 'string' ? value : stringifyMaybe(value),
  }))
})

const selectedSimpleAttrs = computed<Array<{ key: string; value: string }>>(() => {
  const noisy = new Set([
    'input.value',
    'output.value',
    'traceloop.entity.input',
    'traceloop.entity.output',
    'gen_ai.task.input',
    'gen_ai.task.output',
    'gen_ai.workflow.nodes',
    'gen_ai.workflow.edges',
    'gen_ai.input.messages',
    'gen_ai.output.messages',
    'gen_ai.system_instructions',
    'gen_ai.usage.input_tokens',
    'gen_ai.usage.output_tokens',
    'gen_ai.usage.total_tokens',
    'gen_ai.request.model',
    'gen_ai.response.model',
    'gen_ai.provider.name',
    'gen_ai.operation.name',
    'gen_ai.agent.name',
    'service.name',
    'service.version',
    'deployment.environment',
    'langfuse.observation.level',
    'llamphouse.level',
    'level',
  ])
  return selectedAllAttrs.value.filter((row) => !noisy.has(row.key))
})
</script>

<template>
  <div ref="rootEl" class="span-tree">
    <template v-if="flatList.length === 0">
      <div class="empty-state">
        <div class="empty-state__icon"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></div>
        <div class="empty-state__title">No spans found</div>
      </div>
    </template>

    <template v-else>
      <div class="span-tree__layout" :style="{ gridTemplateColumns: `minmax(0, 1fr) 8px ${inspectorWidth}px` }">
        <section class="span-tree__table">
          <div class="span-tree__header">
            <span>Span</span>
            <span>Tokens</span>
            <span>Duration</span>
            <span>Status</span>
          </div>

          <div class="span-tree__rows">
            <div
              v-for="node in flatList"
              :key="node.id"
              class="span-row"
              :class="[
                { 'span-row--selected': selectedSpanId === node.id },
                `span-row--type-${spanType(node.span.SpanName)}`,
              ]"
              :style="{ paddingLeft: (node.depth * 20 + 12) + 'px' }"
              @click="selectSpan(node.id)"
            >
              <div class="span-row__name">
                <span
                  v-for="d in node.depth"
                  :key="'g-' + d"
                  class="span-row__guide"
                ></span>
                <button
                  v-if="node.hasChildren"
                  class="span-row__toggle"
                  :class="{ 'span-row__toggle--open': !collapsed.has(node.id) }"
                  @click.stop="toggle(node.id)"
                  :title="collapsed.has(node.id) ? 'Expand' : 'Collapse'"
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 6 15 12 9 18"/></svg>
                </button>
                <span v-else class="span-row__leaf-dot"></span>
                <span
                  class="span-row__icon"
                  :class="`span-row__icon--${spanType(node.span.SpanName)}`"
                  :title="spanTypeLabel(node.span.SpanName)"
                  v-html="spanTypeSvg(node.span.SpanName)"
                ></span>
                <span class="span-row__label" :title="node.span.SpanName">{{ node.span.SpanName }}</span>
                <span v-if="spanCategory(node.span.SpanName)" class="span-row__tag">{{ spanCategory(node.span.SpanName) }}</span>
                <span v-if="getAttr(node.span, 'assistant.name') || getAttr(node.span, 'assistant.id')" class="span-row__agent-badge">
                  <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0116 0"/></svg>
                  {{ getAttr(node.span, 'assistant.name') || getAttr(node.span, 'assistant.id') }}
                </span>
                <span v-if="getAttr(node.span, 'dispatch.target_agent')" class="span-row__agent">
                  <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
                  {{ getAttr(node.span, 'dispatch.target_agent') }}
                </span>
                <span v-if="getAttr(node.span, 'gen_ai.request.model')" class="span-row__model">{{ getAttr(node.span, 'gen_ai.request.model') }}</span>
              </div>
              <div class="span-row__tokens mono">{{ tokenTotal(node.span) }}</div>
              <div class="span-row__duration mono">{{ durationLabel(node.span.Duration) }}</div>
              <div class="span-row__status">
                <span class="span-status-dot" :style="{ background: statusColor(effectiveStatus(node.span)) }"></span>
              </div>
            </div>
          </div>
        </section>

        <div class="span-tree__resize-handle" @mousedown.prevent="beginResize"></div>

        <aside class="span-inspector">
          <div v-if="selectedSpan" class="span-detail">
            <div class="span-detail__header">
              <span class="span-detail__type-badge" :class="`span-row__type--${spanType(selectedSpan.SpanName)}`">{{ spanTypeLabel(selectedSpan.SpanName) }}</span>
              <span class="span-detail__title" :title="selectedSpan.SpanName">{{ selectedSpan.SpanName }}</span>
              <button class="span-detail__close" @click="selectedSpanId = null" title="Close">✕</button>
            </div>

            <!-- Summary chip row (Langfuse-style) -->
            <div class="span-detail__chips">
              <span class="span-chip">
                <span class="span-chip__dot" :style="{ background: statusColor(effectiveStatus(selectedSpan)) }"></span>
                {{ (effectiveStatus(selectedSpan) || 'UNSET').replace('STATUS_CODE_', '') }}
              </span>
              <span class="span-chip">
                <span class="span-chip__label">Duration</span>
                <span class="span-chip__value mono">{{ durationLabel(selectedSpan.Duration) }}</span>
              </span>
              <span v-if="selectedTokens && selectedTokens.total !== null" class="span-chip">
                <span class="span-chip__label">Tokens</span>
                <span class="span-chip__value mono">{{ selectedTokens.total }}</span>
                <span v-if="selectedTokens.prompt !== null || selectedTokens.completion !== null" class="span-chip__sub mono">
                  ({{ selectedTokens.prompt ?? '—' }}p / {{ selectedTokens.completion ?? '—' }}c)
                </span>
              </span>
              <span v-if="getAttr(selectedSpan, 'gen_ai.request.model')" class="span-chip">
                <span class="span-chip__label">Model</span>
                <span class="span-chip__value mono">{{ getAttr(selectedSpan, 'gen_ai.request.model') }}</span>
              </span>
              <span v-if="getAttr(selectedSpan, 'gen_ai.agent.name') || getAttr(selectedSpan, 'assistant.name')" class="span-chip">
                <span class="span-chip__label">Agent</span>
                <span class="span-chip__value mono">{{ getAttr(selectedSpan, 'gen_ai.agent.name') || getAttr(selectedSpan, 'assistant.name') }}</span>
              </span>
              <span v-if="getAttr(selectedSpan, 'gen_ai.operation.name')" class="span-chip">
                <span class="span-chip__label">Op</span>
                <span class="span-chip__value mono">{{ getAttr(selectedSpan, 'gen_ai.operation.name') }}</span>
              </span>
              <span v-if="getAttr(selectedSpan, 'gen_ai.provider.name')" class="span-chip">
                <span class="span-chip__label">Provider</span>
                <span class="span-chip__value mono">{{ getAttr(selectedSpan, 'gen_ai.provider.name') }}</span>
              </span>
              <span class="span-chip" :class="`span-chip--type-${spanType(selectedSpan.SpanName)}`">
                <span class="span-chip__label">Type</span>
                <span class="span-chip__value">{{ spanTypeLabel(selectedSpan.SpanName) }}</span>
              </span>
              <span v-if="selectedLevel" class="span-chip" :class="`span-chip--level-${selectedLevel.toLowerCase()}`">
                <span class="span-chip__label">Level</span>
                <span class="span-chip__value">{{ selectedLevel }}</span>
              </span>
              <span v-if="getAttr(selectedSpan, 'deployment.environment')" class="span-chip">
                <span class="span-chip__label">Env</span>
                <span class="span-chip__value mono">{{ getAttr(selectedSpan, 'deployment.environment') }}</span>
              </span>
              <span v-if="getAttr(selectedSpan, 'service.version')" class="span-chip">
                <span class="span-chip__label">Version</span>
                <span class="span-chip__value mono">{{ getAttr(selectedSpan, 'service.version') }}</span>
              </span>
              <span v-if="getAttr(selectedSpan, 'service.name')" class="span-chip">
                <span class="span-chip__label">Service</span>
                <span class="span-chip__value mono">{{ getAttr(selectedSpan, 'service.name') }}</span>
              </span>
              <span class="span-chip span-chip--muted">
                <span class="span-chip__label">Span</span>
                <span class="span-chip__value mono">{{ selectedSpan.SpanId.slice(0, 12) }}…</span>
              </span>
            </div>

            <!-- Workflow topology (only when present) -->
            <section v-if="selectedWorkflowNodes.length || selectedWorkflowEdges.length" class="span-detail__section span-detail__card">
              <div class="span-detail__section-title">
                <span>Workflow Topology</span>
                <span class="span-detail__section-count">{{ selectedWorkflowNodes.length }} nodes · {{ selectedWorkflowEdges.length }} edges</span>
              </div>
              <div v-if="selectedWorkflowNodes.length" class="workflow-tags">
                <span v-for="node in selectedWorkflowNodes" :key="'n-' + node" class="workflow-tag">{{ node }}</span>
              </div>
              <div v-if="selectedWorkflowEdges.length" class="workflow-edges">
                <div class="workflow-col__title workflow-col__title--sub">Edges</div>
                <div v-for="edge in selectedWorkflowEdges" :key="'e-' + edge" class="workflow-edge">
                  <template v-for="(part, i) in String(edge).split('->').map(s => s.trim())" :key="edge + '-' + i">
                    <span v-if="i > 0" class="workflow-edge__arrow">→</span>
                    <span class="workflow-edge__node">{{ part }}</span>
                  </template>
                </div>
              </div>
            </section>

            <!-- Chat messages (gen_ai.input.messages / gen_ai.output.messages / gen_ai.system_instructions) -->
            <section v-if="hasChatMessages" class="span-detail__section span-detail__card">
              <div class="span-detail__section-title">
                <span>Messages</span>
                <span class="span-detail__section-count">{{ selectedChatMessages.system.length + selectedChatMessages.input.length + selectedChatMessages.output.length }}</span>
              </div>
              <div class="chat-messages">
                <div
                  v-for="(msg, i) in [...selectedChatMessages.system, ...selectedChatMessages.input, ...selectedChatMessages.output]"
                  :key="'msg-' + i"
                  class="chat-msg"
                  :class="`chat-msg--${msg.role}`"
                >
                  <div class="chat-msg__header">
                    <span class="chat-msg__role">{{ msg.role }}</span>
                    <span v-if="msg.name" class="chat-msg__name">· {{ msg.name }}</span>
                    <span v-if="msg.toolCallId" class="chat-msg__tool-id mono">↳ {{ msg.toolCallId }}</span>
                  </div>
                  <ContentBlock v-if="msg.content" :content="msg.content" :max-height="320" compact />
                  <div v-if="msg.toolCalls && msg.toolCalls.length" class="chat-msg__tools">
                    <div v-for="(tc, j) in msg.toolCalls" :key="'tc-' + i + '-' + j" class="chat-tool">
                      <div class="chat-tool__header">
                        <span class="chat-tool__badge">TOOL CALL</span>
                        <span class="chat-tool__name mono">{{ tc.name }}</span>
                        <span v-if="tc.id" class="chat-tool__id mono">{{ tc.id }}</span>
                      </div>
                      <ContentBlock v-if="tc.args" :content="tc.args" :max-height="220" compact default-mode="json" />
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <!-- Task Input (only if no chat messages) -->
            <section v-if="!hasChatMessages && selectedTaskInput" class="span-detail__section span-detail__card">
              <div class="span-detail__section-title">Task Input</div>
              <ContentBlock :content="selectedTaskInput" :max-height="360" />
            </section>

            <!-- Task Output (only if no chat messages) -->
            <section v-if="!hasChatMessages && selectedTaskOutput" class="span-detail__section span-detail__card">
              <div class="span-detail__section-title">Task Output</div>
              <ContentBlock :content="selectedTaskOutput" :max-height="360" default-mode="markdown" />
            </section>

            <!-- Legacy input/output (only if different from task input/output) -->
            <section v-if="getAttr(selectedSpan, 'input.value') && getAttr(selectedSpan, 'input.value') !== selectedTaskInput" class="span-detail__section span-detail__card">
              <div class="span-detail__section-title">Input</div>
              <ContentBlock :content="getAttr(selectedSpan, 'input.value')" :max-height="300" default-mode="json" />
            </section>

            <section v-if="getAttr(selectedSpan, 'output.value') && getAttr(selectedSpan, 'output.value') !== selectedTaskOutput" class="span-detail__section span-detail__card">
              <div class="span-detail__section-title">Output</div>
              <ContentBlock :content="getAttr(selectedSpan, 'output.value')" :max-height="300" default-mode="json" />
            </section>

            <!-- Legacy messages (dict-shaped input/output) -->
            <section v-if="selectedMessages.input" class="span-detail__section span-detail__card">
              <div class="span-detail__section-title">Messages (Input)</div>
              <ContentBlock :content="selectedMessages.input" :max-height="300" />
            </section>

            <section v-if="selectedMessages.output" class="span-detail__section span-detail__card">
              <div class="span-detail__section-title">Messages (Output)</div>
              <ContentBlock :content="selectedMessages.output" :max-height="300" />
            </section>

            <!-- Tool calls -->
            <section v-if="selectedToolCalls.length > 0" class="span-detail__section span-detail__card">
              <div class="span-detail__section-title">Function / Tool Calls</div>
              <div class="tool-calls">
                <div v-for="(call, idx) in selectedToolCalls" :key="idx" class="tool-call-card">
                  <div class="tool-call-card__header">
                    <span class="tool-call-card__name">{{ call.name }}</span>
                    <span v-if="call.id" class="tool-call-card__id mono">{{ call.id }}</span>
                  </div>
                  <div v-if="call.args" class="tool-call-card__section">
                    <div class="tool-call-card__label">Arguments</div>
                    <ContentBlock :content="call.args" :max-height="260" compact default-mode="json" />
                  </div>
                  <div v-if="call.output" class="tool-call-card__section">
                    <div class="tool-call-card__label">Output</div>
                    <ContentBlock :content="call.output" :max-height="260" compact />
                  </div>
                </div>
              </div>
            </section>

            <!-- Attributes (structured, collapsed by default when we have rich content) -->
            <details class="span-detail__section span-detail__card" :open="selectedSimpleAttrs.length <= 6">
              <summary class="span-detail__section-title span-detail__section-title--summary">
                <span>Attributes</span>
                <span class="span-detail__section-count">{{ selectedSimpleAttrs.length }}</span>
              </summary>
              <div class="span-detail__attrs">
                <div v-for="row in selectedSimpleAttrs" :key="row.key" class="span-detail__attr-row">
                  <span class="span-detail__attr-key">{{ row.key }}</span>
                  <span class="span-detail__attr-val mono">{{ truncate(row.value, 300) }}</span>
                </div>
                <div v-if="selectedSimpleAttrs.length === 0" style="color: var(--text-muted); font-size: 0.8rem;">
                  No additional attributes
                </div>
              </div>
            </details>

            <details class="span-detail__section span-detail__card">
              <summary class="span-detail__section-title span-detail__section-title--summary">
                <span>Raw Attributes JSON</span>
                <span class="span-detail__section-count">{{ selectedAllAttrs.length }}</span>
              </summary>
              <ContentBlock :content="JSON.stringify(selectedSpan.SpanAttributes || {}, null, 2)" :max-height="400" default-mode="json" />
            </details>
          </div>

          <div v-else class="span-inspector__empty">
            <div class="span-inspector__empty-title">Span Inspector</div>
            <div class="span-inspector__empty-text">Select a span on the left to inspect its input/output and attributes.</div>
          </div>
        </aside>
      </div>
    </template>
  </div>
</template>

<style scoped>
.span-tree {
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.span-tree__layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 8px 440px;
  min-height: 520px;
}

.span-tree__table {
  min-width: 0;
}

.span-tree__resize-handle {
  width: 8px;
  cursor: col-resize;
  border-left: 1px solid var(--border);
  border-right: 1px solid var(--border);
  background: linear-gradient(to right, transparent 0, #e2e8f0 50%, transparent 100%);
}

.span-tree__header {
  display: grid;
  grid-template-columns: 1fr 90px 100px 60px;
  padding: 8px 12px;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
}

.span-tree__rows {
  max-height: 640px;
  overflow-y: auto;
}

.span-row {
  display: grid;
  grid-template-columns: 1fr 90px 100px 60px;
  align-items: center;
  padding: 6px 0;
  padding-right: 12px;
  border-bottom: 1px solid var(--border);
  font-size: 0.825rem;
  cursor: pointer;
  transition: background var(--transition);
  position: relative;
}

.span-row::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 3px;
  background: transparent;
  transition: background var(--transition);
}

.span-row--type-llm::before { background: rgba(249, 115, 22, 0.6); }
.span-row--type-tool::before { background: rgba(234, 179, 8, 0.6); }
.span-row--type-dispatch::before { background: rgba(124, 58, 237, 0.6); }
.span-row--type-internal::before { background: rgba(20, 184, 166, 0.55); }
.span-row--type-infra::before { background: rgba(148, 163, 184, 0.55); }
.span-row--type-app::before { background: rgba(59, 130, 246, 0.55); }

.span-row:hover {
  background: var(--bg-hover);
}

.span-row--selected {
  background: var(--accent-dim);
}

.span-row--selected::before {
  background: var(--accent);
  width: 3px;
}

.span-row__name {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}

.span-row__guide {
  display: inline-block;
  width: 1px;
  height: 24px;
  background: var(--border);
  margin-right: 15px;
  flex-shrink: 0;
  opacity: 0.6;
}

.span-row__toggle {
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  padding: 0;
  width: 16px;
  height: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  border-radius: 3px;
  transition: transform 0.15s ease, color var(--transition), background var(--transition);
}

.span-row__toggle:hover {
  color: var(--text-primary);
  background: var(--bg-hover);
}

.span-row__toggle--open {
  transform: rotate(90deg);
}

.span-row__leaf-dot {
  width: 16px;
  height: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  position: relative;
}

.span-row__leaf-dot::after {
  content: '';
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: var(--border);
}

.span-row__icon {
  flex-shrink: 0;
  width: 22px;
  height: 22px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-hover);
  color: var(--text-secondary);
  border-radius: 5px;
  border: 1px solid transparent;
}

.span-row__icon--llm {
  background: rgba(249, 115, 22, 0.12);
  color: #c2410c;
  border-color: rgba(249, 115, 22, 0.25);
}

.span-row__icon--tool {
  background: rgba(234, 179, 8, 0.14);
  color: #a16207;
  border-color: rgba(234, 179, 8, 0.3);
}

.span-row__icon--dispatch {
  background: rgba(124, 58, 237, 0.12);
  color: #6d28d9;
  border-color: rgba(124, 58, 237, 0.25);
}

.span-row__icon--internal {
  background: rgba(20, 184, 166, 0.12);
  color: #0f766e;
  border-color: rgba(20, 184, 166, 0.25);
}

.span-row__icon--infra {
  background: rgba(148, 163, 184, 0.18);
  color: #475569;
  border-color: rgba(148, 163, 184, 0.35);
}

.span-row__icon--app {
  background: rgba(59, 130, 246, 0.1);
  color: #1e40af;
  border-color: rgba(59, 130, 246, 0.22);
}

.span-row__label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.span-row__tag {
  font-size: 0.65rem;
  color: var(--accent);
  background: var(--accent-dim);
  padding: 1px 6px;
  border-radius: 4px;
  flex-shrink: 0;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  font-weight: 600;
}

.span-row__type {
  font-size: 0.64rem;
  padding: 1px 6px;
  border-radius: 4px;
  flex-shrink: 0;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  font-weight: 700;
}

.span-row__type--llm {
  color: #7c2d12;
  background: rgba(249, 115, 22, 0.12);
}

.span-row__type--tool {
  color: #854d0e;
  background: rgba(234, 179, 8, 0.14);
}

.span-row__type--dispatch {
  color: #6d28d9;
  background: rgba(124, 58, 237, 0.12);
}

.span-row__type--internal {
  color: #0f766e;
  background: rgba(20, 184, 166, 0.12);
}

.span-row__type--infra {
  color: #334155;
  background: rgba(148, 163, 184, 0.2);
}

.span-row__type--app {
  color: #1e3a8a;
  background: rgba(59, 130, 246, 0.12);
}

.span-row__model {
  font-size: 0.7rem;
  color: var(--text-muted);
  background: var(--bg-hover);
  padding: 1px 6px;
  border-radius: 4px;
  flex-shrink: 0;
}

.span-row__agent-badge {
  font-size: 0.65rem;
  font-weight: 600;
  color: #0369a1;
  background: rgba(3, 105, 161, 0.08);
  padding: 1px 7px;
  border-radius: 4px;
  flex-shrink: 0;
  border: 1px solid rgba(3, 105, 161, 0.15);
  display: inline-flex;
  align-items: center;
  gap: 3px;
}

.span-row__agent {
  font-size: 0.7rem;
  font-weight: 600;
  color: #7c3aed;
  background: rgba(124, 58, 237, 0.08);
  padding: 1px 7px;
  border-radius: 4px;
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  gap: 3px;
}

.span-row__svc {
  font-size: 0.7rem;
  color: var(--text-muted);
  background: var(--bg-hover);
  padding: 1px 6px;
  border-radius: 4px;
  flex-shrink: 0;
}

.span-row__duration {
  color: var(--text-secondary);
  font-size: 0.8rem;
}

.span-row__tokens {
  color: var(--text-secondary);
  font-size: 0.78rem;
}

.span-status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}

.span-inspector {
  background: var(--bg-secondary);
  max-height: 640px;
  overflow-y: auto;
  overflow-x: hidden;
  min-width: 0;
}

.span-inspector__empty {
  padding: 20px;
  color: var(--text-secondary);
}

.span-inspector__empty-title {
  font-weight: 600;
  margin-bottom: 8px;
  color: var(--text-primary);
}

.span-inspector__empty-text {
  font-size: 0.85rem;
  line-height: 1.5;
}

/* ─── Detail panel ───────────────────────────────────────── */

.span-detail {
  border-top: 2px solid var(--accent);
  background: var(--bg-secondary);
  padding: 16px;
  min-width: 0;
  overflow: hidden;
}

.span-detail__header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
  min-width: 0;
}

.span-detail__type-badge {
  font-size: 0.65rem;
  font-weight: 700;
  padding: 3px 8px;
  border-radius: 4px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  flex-shrink: 0;
}

.span-detail__title {
  font-weight: 600;
  font-size: 0.95rem;
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.span-detail__close {
  background: none;
  border: 1px solid var(--border);
  color: var(--text-muted);
  cursor: pointer;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
  flex-shrink: 0;
}

.span-detail__close:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}

/* Summary chip row */
.span-detail__chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 16px;
}

.span-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.75rem;
  color: var(--text-primary);
  max-width: 100%;
  overflow: hidden;
}

.span-chip--muted {
  color: var(--text-muted);
}

.span-chip--level-error {
  background: rgba(220, 38, 38, 0.08);
  border-color: rgba(220, 38, 38, 0.3);
  color: #b91c1c;
}

.span-chip--level-warning {
  background: rgba(245, 158, 11, 0.1);
  border-color: rgba(245, 158, 11, 0.3);
  color: #b45309;
}

.span-chip--level-debug {
  background: rgba(148, 163, 184, 0.12);
  border-color: rgba(148, 163, 184, 0.3);
  color: #475569;
}

.span-chip--type-llm { border-color: rgba(249, 115, 22, 0.35); }
.span-chip--type-tool { border-color: rgba(234, 179, 8, 0.35); }
.span-chip--type-dispatch { border-color: rgba(124, 58, 237, 0.35); }
.span-chip--type-internal { border-color: rgba(20, 184, 166, 0.35); }
.span-chip--type-infra { border-color: rgba(148, 163, 184, 0.4); }
.span-chip--type-app { border-color: rgba(59, 130, 246, 0.35); }

.span-chip__dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}

.span-chip__label {
  color: var(--text-muted);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-weight: 600;
}

.span-chip__value {
  color: var(--text-primary);
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 160px;
}

.span-chip__sub {
  color: var(--text-muted);
  font-size: 0.7rem;
}

.span-detail__card {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 12px;
  margin-bottom: 10px;
}

.span-detail__section {
  min-width: 0;
}

.span-detail__section-title {
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.span-detail__section-title--summary {
  cursor: pointer;
  margin-bottom: 0;
}

details[open] .span-detail__section-title--summary {
  margin-bottom: 10px;
}

.span-detail__section-count {
  font-size: 0.7rem;
  color: var(--text-muted);
  background: var(--bg-hover);
  padding: 1px 8px;
  border-radius: 10px;
  text-transform: none;
  letter-spacing: 0;
  font-weight: 500;
}

.span-detail__value {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 10px;
  font-size: 0.78rem;
  font-family: 'SF Mono', 'Fira Code', monospace;
  color: var(--text-primary);
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 300px;
  overflow: auto;
  max-width: 100%;
  margin: 0;
}

.span-detail__value--pretty {
  max-height: 360px;
}

.span-detail__attrs {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.span-detail__attr-row {
  display: grid;
  grid-template-columns: minmax(140px, 40%) minmax(0, 1fr);
  gap: 12px;
  padding: 6px 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.78rem;
  min-width: 0;
}

.span-detail__attr-row:last-child {
  border-bottom: none;
}

.span-detail__attr-key {
  color: var(--accent);
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.span-detail__attr-val {
  color: var(--text-secondary);
  word-break: break-word;
  overflow: hidden;
}

/* Workflow topology */
.workflow-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 10px;
}

.workflow-tag {
  background: var(--accent-dim);
  color: var(--accent);
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 500;
  font-family: 'SF Mono', 'Fira Code', monospace;
}

.workflow-edges {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.workflow-col__title {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  margin-bottom: 6px;
}

.workflow-col__title--sub {
  margin-top: 4px;
}

.workflow-edge {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  background: var(--bg-secondary);
  border-radius: 6px;
  font-size: 0.75rem;
  font-family: 'SF Mono', 'Fira Code', monospace;
  color: var(--text-primary);
  flex-wrap: wrap;
}

.workflow-edge__node {
  color: var(--text-primary);
}

.workflow-edge__arrow {
  color: var(--text-muted);
  font-weight: 700;
}

/* Chat messages */
.chat-messages {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chat-msg {
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 8px 10px;
  background: var(--bg-secondary);
  border-left: 3px solid var(--text-muted);
}

.chat-msg--system {
  border-left-color: #94a3b8;
  background: rgba(148, 163, 184, 0.06);
}

.chat-msg--user {
  border-left-color: #3b82f6;
  background: rgba(59, 130, 246, 0.06);
}

.chat-msg--assistant {
  border-left-color: #10b981;
  background: rgba(16, 185, 129, 0.06);
}

.chat-msg--tool {
  border-left-color: #f59e0b;
  background: rgba(245, 158, 11, 0.06);
}

.chat-msg__header {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 700;
  color: var(--text-secondary);
  margin-bottom: 6px;
}

.chat-msg__role {
  color: var(--text-primary);
}

.chat-msg__name {
  color: var(--text-muted);
  text-transform: none;
  font-weight: 500;
  letter-spacing: 0;
}

.chat-msg__tool-id {
  margin-left: auto;
  color: var(--text-muted);
  text-transform: none;
  letter-spacing: 0;
  font-weight: 500;
  font-size: 0.7rem;
}

.chat-msg__content {
  font-size: 0.82rem;
  line-height: 1.5;
  color: var(--text-primary);
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 320px;
  overflow-y: auto;
}

.chat-msg__tools {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 8px;
}

.chat-tool {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 8px;
}

.chat-tool__header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 4px;
  flex-wrap: wrap;
}

.chat-tool__badge {
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  padding: 1px 6px;
  border-radius: 3px;
  background: rgba(245, 158, 11, 0.15);
  color: #b45309;
}

.chat-tool__name {
  font-weight: 600;
  font-size: 0.78rem;
  color: var(--text-primary);
}

.chat-tool__id {
  font-size: 0.68rem;
  color: var(--text-muted);
  margin-left: auto;
}

.chat-tool__args {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 6px 8px;
  font-size: 0.72rem;
  font-family: 'SF Mono', 'Fira Code', monospace;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 200px;
  overflow: auto;
  margin: 0;
  color: var(--text-secondary);
}

.tool-calls {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.tool-call-card {
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 10px;
  background: var(--bg-surface);
}

.tool-call-card__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 8px;
}

.tool-call-card__name {
  font-weight: 600;
  color: var(--text-primary);
}

.tool-call-card__id {
  font-size: 0.72rem;
  color: var(--text-muted);
}

.tool-call-card__label {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-muted);
  margin: 6px 0;
}

@media (max-width: 1200px) {
  .span-tree__layout {
    grid-template-columns: 1fr !important;
  }

  .span-tree__table {
    border-right: none;
    border-bottom: 1px solid var(--border);
  }

  .span-tree__resize-handle {
    display: none;
  }

  .span-inspector {
    max-height: none;
  }
}
</style>
