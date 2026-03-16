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

function rootAttr(key: string): string {
  return rootSpan.value?.SpanAttributes?.[key] ?? ''
}

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
        <div class="page-header__subtitle mono">Run {{ runId }}</div>
      </div>
      <div class="flex gap-2">
        <span class="badge badge--info">{{ spans.length }} spans</span>
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
            <div class="trace-summary__value">{{ spans.length }}</div>
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
      <SpanTree :spans="spans" />

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
</style>
