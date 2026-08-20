<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { compass, shortId } from '../api/client'
import type { Span } from '../api/client'
import SpanTree from '../components/SpanTree.vue'

const route = useRoute()
const runId = route.params.runId as string
const spans = ref<Span[]>([])
const loading = ref(true)
const error = ref('')
const showLlamphouseInternals = ref(false)

onMounted(async () => {
  try {
    spans.value = await compass.getRunTrace(runId)
  } catch (e: any) {
    error.value = e.message
  } finally {
    loading.value = false
  }
})

const rootSpan = computed<Span | undefined>(() =>
  spans.value.find(s => !s.ParentSpanId)
)

function isLlamphouseInternalSpan(span: Span): boolean {
  const n = span.SpanName || ''
  return n.startsWith('llamphouse.')
}

const filteredSpans = computed(() => {
  if (showLlamphouseInternals.value) return spans.value
  return spans.value.filter(s => !isLlamphouseInternalSpan(s))
})

const hiddenInternalCount = computed(() =>
  spans.value.filter(isLlamphouseInternalSpan).length
)

function rootAttr(key: string): string {
  return rootSpan.value?.SpanAttributes?.[key] ?? ''
}

/** Search all spans for the first non-empty value for one of the given keys. */
function firstSpanAttr(keys: string[]): string {
  for (const s of spans.value) {
    const attrs = s.SpanAttributes || {}
    for (const k of keys) {
      const v = attrs[k]
      if (v != null && v !== '') return String(v)
    }
  }
  return ''
}

// Linked entity IDs — traces have run.id + session.id (thread) attributes when
// created inside a LLAMPHouse run. If neither exists, the trace is standalone
// and the "Open …" buttons are hidden.
const linkedRunId = computed(() =>
  firstSpanAttr(['run.id', 'llamphouse.run_id']) || runId
)
const linkedThreadId = computed(() =>
  firstSpanAttr(['session.id', 'thread.id', 'llamphouse.thread_id'])
)
const linkedAssistantId = computed(() =>
  firstSpanAttr(['assistant.id', 'llamphouse.assistant_id'])
)
const linkedAssistantName = computed(() =>
  firstSpanAttr(['assistant.name', 'gen_ai.agent.name'])
)

function prettyJson(value: string): string {
  try { return JSON.stringify(JSON.parse(value), null, 2) }
  catch { return value }
}

function totalDuration(): string {
  if (!rootSpan.value) return '—'
  const ms = rootSpan.value.Duration / 1_000_000
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}
</script>

<template>
  <div>
    <div class="breadcrumbs">
      <router-link to="/traces">Traces</router-link>
      <span>›</span>
      <span>{{ shortId(runId) }}</span>
    </div>

    <div class="page-header">
      <div>
        <h1>Trace Detail</h1>
        <div class="page-header__subtitle">Debug spans, latency, and token usage for this run.</div>
      </div>
      <div class="flex gap-2 trace-header-actions">
        <router-link
          v-if="linkedThreadId && linkedRunId"
          :to="`/threads/${linkedThreadId}/runs/${linkedRunId}`"
          class="trace-link-btn"
          :title="`Open run ${shortId(linkedRunId)}`"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          Open Run
        </router-link>
        <router-link
          v-if="linkedThreadId"
          :to="`/threads/${linkedThreadId}`"
          class="trace-link-btn trace-link-btn--secondary"
          :title="`Open thread ${shortId(linkedThreadId)}`"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
          Open Thread
        </router-link>
        <router-link
          v-if="linkedAssistantId"
          :to="`/assistants`"
          class="trace-link-btn trace-link-btn--ghost"
          :title="linkedAssistantName || linkedAssistantId"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0116 0"/></svg>
          {{ linkedAssistantName || 'Assistant' }}
        </router-link>
        <span class="badge badge--info">{{ filteredSpans.length }} spans</span>
      </div>
    </div>

    <div v-if="loading" class="loading-center"><div class="spinner"></div></div>
    <div v-else-if="error" class="card" style="color: var(--error)">{{ error }}</div>

    <template v-else>
      <!-- Summary -->
      <div v-if="spans.length > 0" class="card mb-4">
        <div class="trace-summary">
          <div class="trace-summary__stat">
            <div class="section__title">Total Spans</div>
            <div class="trace-summary__value">{{ filteredSpans.length }}</div>
          </div>
          <div class="trace-summary__stat">
            <div class="section__title">Duration</div>
            <div class="trace-summary__value mono">{{ totalDuration() }}</div>
          </div>
          <div class="trace-summary__stat">
            <div class="section__title">Services</div>
            <div class="trace-summary__value">
              {{ new Set(spans.map(s => s.ServiceName)).size }}
            </div>
          </div>
          <div class="trace-summary__stat">
            <div class="section__title">Root Span</div>
            <div style="font-size: 0.85rem; font-weight: 500;">
              {{ rootSpan?.SpanName || '—' }}
            </div>
          </div>
        </div>

        <div class="trace-controls">
          <label class="trace-toggle">
            <input v-model="showLlamphouseInternals" type="checkbox">
            <span>Show LLAMPHouse internal spans</span>
          </label>
          <span v-if="!showLlamphouseInternals && hiddenInternalCount > 0" class="trace-muted">
            Hiding {{ hiddenInternalCount }} internal span{{ hiddenInternalCount === 1 ? '' : 's' }}
          </span>
        </div>
      </div>

      <!-- Root Input / Output -->
      <div v-if="rootAttr('input.value') || rootAttr('output.value')" class="trace-io mb-4">
        <div v-if="rootAttr('input.value')" class="trace-io__card">
          <div class="trace-io__label">Input</div>
          <pre class="trace-io__value">{{ prettyJson(rootAttr('input.value')) }}</pre>
        </div>
        <div v-if="rootAttr('output.value')" class="trace-io__card">
          <div class="trace-io__label">Output</div>
          <pre class="trace-io__value">{{ prettyJson(rootAttr('output.value')) }}</pre>
        </div>
      </div>

      <!-- Span tree -->
      <SpanTree :spans="filteredSpans" />

      <!-- Raw span attributes -->
      <div v-if="spans.length > 0" class="mt-4">
        <details>
          <summary style="cursor: pointer; color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 8px;">
            Raw span data
          </summary>
          <div class="json-view">{{ JSON.stringify(spans, null, 2) }}</div>
        </details>
      </div>
    </template>
  </div>
</template>

<style scoped>
.trace-summary {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.trace-summary__stat {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.trace-summary__value {
  font-size: 1.3rem;
  font-weight: 700;
}

.trace-controls {
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.trace-toggle {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--text-secondary);
  font-size: 0.85rem;
}

.trace-muted {
  color: var(--text-muted);
  font-size: 0.8rem;
}

.trace-io {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.trace-io__card {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 12px 16px;
  overflow: hidden;
}

.trace-io__label {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-muted);
  margin-bottom: 8px;
}

.trace-io__value {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 10px 12px;
  font-size: 0.8rem;
  font-family: var(--font-mono);
  color: var(--text-primary);
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 200px;
  overflow-y: auto;
  margin: 0;
}

.trace-header-actions {
  align-items: center;
  flex-wrap: wrap;
}

.trace-link-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 5px 11px;
  background: var(--accent);
  color: white;
  border: 1px solid var(--accent);
  border-radius: var(--radius-md);
  font-size: 0.8rem;
  font-weight: 600;
  text-decoration: none;
  transition: background var(--transition), border-color var(--transition), color var(--transition);
}

.trace-link-btn:hover {
  background: var(--accent-hover, var(--accent));
  filter: brightness(0.95);
}

.trace-link-btn--secondary {
  background: transparent;
  color: var(--accent);
  border-color: var(--accent);
}

.trace-link-btn--secondary:hover {
  background: var(--accent-dim);
  filter: none;
}

.trace-link-btn--ghost {
  background: transparent;
  color: var(--text-secondary);
  border-color: var(--border);
}

.trace-link-btn--ghost:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
  filter: none;
}
</style>
