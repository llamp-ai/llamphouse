<script setup lang="ts">
import { ref, computed } from 'vue'
import type { Span } from '../api/client'

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

function getAttr(span: Span, key: string): string {
  return span.SpanAttributes?.[key] ?? ''
}

function truncate(text: string, max = 200): string {
  if (!text || text.length <= max) return text
  return text.slice(0, max) + '…'
}

function selectedSpan(): Span | null {
  if (!selectedSpanId.value) return null
  return props.spans.find(s => s.SpanId === selectedSpanId.value) ?? null
}

function prettyJson(value: string): string {
  try {
    return JSON.stringify(JSON.parse(value), null, 2)
  } catch {
    return value
  }
}
</script>

<template>
  <div class="span-tree">
    <template v-if="flatList.length === 0">
      <div class="empty-state">
        <div class="empty-state__icon"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></div>
        <div class="empty-state__title">No spans found</div>
      </div>
    </template>

    <template v-else>
      <div class="span-tree__header">
        <span>Span</span>
        <span>Duration</span>
        <span>Status</span>
      </div>

      <div
        v-for="node in flatList"
        :key="node.id"
        class="span-row"
        :class="{ 'span-row--selected': selectedSpanId === node.id }"
        :style="{ paddingLeft: (node.depth * 20 + 12) + 'px' }"
        @click="selectSpan(node.id)"
      >
        <div class="span-row__name">
          <button
            v-if="node.hasChildren"
            class="span-row__toggle"
            @click.stop="toggle(node.id)"
          >{{ collapsed.has(node.id) ? '▸' : '▾' }}</button>
          <span v-else class="span-row__dot">·</span>
          <span class="span-row__icon" :title="spanCategory(node.span.SpanName)">{{ spanIcon(node.span.SpanName) }}</span>
          <span class="span-row__label" :title="node.span.SpanName">{{ node.span.SpanName }}</span>
          <span v-if="spanCategory(node.span.SpanName)" class="span-row__tag">{{ spanCategory(node.span.SpanName) }}</span>
          <span v-if="getAttr(node.span, 'assistant.name') || getAttr(node.span, 'assistant.id')" class="span-row__agent-badge">{{ getAttr(node.span, 'assistant.name') || getAttr(node.span, 'assistant.id') }}</span>
          <span v-if="getAttr(node.span, 'dispatch.target_agent')" class="span-row__agent">→ {{ getAttr(node.span, 'dispatch.target_agent') }}</span>
          <span v-if="getAttr(node.span, 'gen_ai.request.model')" class="span-row__model">{{ getAttr(node.span, 'gen_ai.request.model') }}</span>
        </div>
        <div class="span-row__duration mono">{{ durationLabel(node.span.Duration) }}</div>
        <div class="span-row__status">
          <span class="span-status-dot" :style="{ background: statusColor(node.span.StatusCode) }"></span>
        </div>
      </div>

      <!-- Detail panel for selected span -->
      <div v-if="selectedSpan()" class="span-detail">
        <div class="span-detail__header">
          <span class="span-detail__icon">{{ spanIcon(selectedSpan()!.SpanName) }}</span>
          <span class="span-detail__title">{{ selectedSpan()!.SpanName }}</span>
          <button class="span-detail__close" @click="selectedSpanId = null">✕</button>
        </div>

        <div class="span-detail__meta">
          <div class="span-detail__meta-item">
            <span class="span-detail__label">Span ID</span>
            <span class="mono">{{ selectedSpan()!.SpanId }}</span>
          </div>
          <div class="span-detail__meta-item">
            <span class="span-detail__label">Duration</span>
            <span class="mono">{{ durationLabel(selectedSpan()!.Duration) }}</span>
          </div>
          <div v-if="getAttr(selectedSpan()!, 'assistant.name') || getAttr(selectedSpan()!, 'assistant.id')" class="span-detail__meta-item">
            <span class="span-detail__label">Agent</span>
            <span class="mono">{{ getAttr(selectedSpan()!, 'assistant.name') || getAttr(selectedSpan()!, 'assistant.id') }}</span>
          </div>
          <div v-if="getAttr(selectedSpan()!, 'gen_ai.request.model')" class="span-detail__meta-item">
            <span class="span-detail__label">Model</span>
            <span class="mono">{{ getAttr(selectedSpan()!, 'gen_ai.request.model') }}</span>
          </div>
          <div v-if="getAttr(selectedSpan()!, 'gen_ai.response.finish_reason')" class="span-detail__meta-item">
            <span class="span-detail__label">Finish Reason</span>
            <span class="mono">{{ getAttr(selectedSpan()!, 'gen_ai.response.finish_reason') }}</span>
          </div>
          <div v-if="getAttr(selectedSpan()!, 'dispatch.target_agent')" class="span-detail__meta-item">
            <span class="span-detail__label">Target Agent</span>
            <span class="mono">{{ getAttr(selectedSpan()!, 'dispatch.target_agent') }}</span>
          </div>
          <div v-if="getAttr(selectedSpan()!, 'dispatch.source_agent')" class="span-detail__meta-item">
            <span class="span-detail__label">Source Agent</span>
            <span class="mono">{{ getAttr(selectedSpan()!, 'dispatch.source_agent') }}</span>
          </div>
          <div v-if="getAttr(selectedSpan()!, 'dispatch.child_run')" class="span-detail__meta-item">
            <span class="span-detail__label">Child Run</span>
            <span class="mono">{{ getAttr(selectedSpan()!, 'dispatch.child_run') }}</span>
          </div>
          <div v-if="getAttr(selectedSpan()!, 'dispatch.response_chars')" class="span-detail__meta-item">
            <span class="span-detail__label">Response Chars</span>
            <span class="mono">{{ getAttr(selectedSpan()!, 'dispatch.response_chars') }}</span>
          </div>
        </div>

        <!-- Input -->
        <div v-if="getAttr(selectedSpan()!, 'input.value')" class="span-detail__section">
          <div class="span-detail__section-title">Input</div>
          <pre class="span-detail__value">{{ prettyJson(getAttr(selectedSpan()!, 'input.value')) }}</pre>
        </div>

        <!-- Output -->
        <div v-if="getAttr(selectedSpan()!, 'output.value')" class="span-detail__section">
          <div class="span-detail__section-title">Output</div>
          <pre class="span-detail__value">{{ prettyJson(getAttr(selectedSpan()!, 'output.value')) }}</pre>
        </div>

        <!-- All attributes -->
        <details class="span-detail__section">
          <summary class="span-detail__section-title" style="cursor: pointer;">All Attributes</summary>
          <div class="span-detail__attrs">
            <div v-for="(value, key) in selectedSpan()!.SpanAttributes" :key="key" class="span-detail__attr-row">
              <span class="span-detail__attr-key">{{ key }}</span>
              <span class="span-detail__attr-val mono">{{ truncate(value, 300) }}</span>
            </div>
            <div v-if="!selectedSpan()!.SpanAttributes || Object.keys(selectedSpan()!.SpanAttributes).length === 0" style="color: var(--text-muted); font-size: 0.8rem;">
              No attributes
            </div>
          </div>
        </details>
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

.span-tree__header {
  display: grid;
  grid-template-columns: 1fr 100px 60px;
  padding: 8px 12px;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
}

.span-row {
  display: grid;
  grid-template-columns: 1fr 100px 60px;
  align-items: center;
  padding: 6px 0;
  padding-right: 12px;
  border-bottom: 1px solid var(--border);
  font-size: 0.825rem;
  cursor: pointer;
  transition: background var(--transition);
}

.span-row:hover {
  background: var(--bg-hover);
}

.span-row--selected {
  background: var(--accent-dim);
  border-left: 2px solid var(--accent);
}

.span-row__name {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}

.span-row__toggle {
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 0.8rem;
  padding: 0;
  width: 16px;
  flex-shrink: 0;
}

.span-row__dot {
  width: 16px;
  text-align: center;
  color: var(--text-muted);
  flex-shrink: 0;
}

.span-row__icon {
  font-size: 0.6rem;
  font-weight: 700;
  flex-shrink: 0;
  width: 22px;
  height: 18px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-hover);
  color: var(--text-secondary);
  border-radius: 3px;
  letter-spacing: -0.02em;
  font-family: var(--font-mono);
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
}

.span-row__agent {
  font-size: 0.7rem;
  font-weight: 600;
  color: #7c3aed;
  background: rgba(124, 58, 237, 0.08);
  padding: 1px 7px;
  border-radius: 4px;
  flex-shrink: 0;
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

.span-status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}

/* ─── Detail panel ───────────────────────────────────────── */

.span-detail {
  border-top: 2px solid var(--accent);
  background: var(--bg-secondary);
  padding: 16px;
}

.span-detail__header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.span-detail__icon {
  font-size: 0.65rem;
  font-weight: 700;
  width: 24px;
  height: 20px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: var(--accent-dim);
  color: var(--accent);
  border-radius: 4px;
  font-family: var(--font-mono);
}

.span-detail__title {
  font-weight: 600;
  font-size: 0.95rem;
  flex: 1;
}

.span-detail__close {
  background: none;
  border: 1px solid var(--border);
  color: var(--text-muted);
  cursor: pointer;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
}

.span-detail__close:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}

.span-detail__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 16px;
  padding: 10px 12px;
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
}

.span-detail__meta-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.span-detail__label {
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
}

.span-detail__section {
  margin-bottom: 12px;
}

.span-detail__section-title {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 6px;
}

.span-detail__value {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 12px;
  font-size: 0.8rem;
  font-family: 'SF Mono', 'Fira Code', monospace;
  color: var(--text-primary);
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 300px;
  overflow-y: auto;
  margin: 0;
}

.span-detail__attrs {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 8px 12px;
}

.span-detail__attr-row {
  display: flex;
  gap: 12px;
  padding: 4px 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.8rem;
}

.span-detail__attr-row:last-child {
  border-bottom: none;
}

.span-detail__attr-key {
  color: var(--accent);
  min-width: 180px;
  flex-shrink: 0;
  font-weight: 500;
}

.span-detail__attr-val {
  color: var(--text-secondary);
  word-break: break-word;
}
</style>
