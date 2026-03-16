<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { compass, shortId } from '../api/client'

interface TraceRow {
  TraceId: string
  SpanName: string
  run_id: string
  thread_id: string
  assistant_id: string
  duration_ms: number
  StatusCode: string
  Timestamp: string
  span_count: number
}

const router = useRouter()
const traces = ref<TraceRow[]>([])
const loading = ref(true)
const error = ref('')

onMounted(async () => {
  try {
    traces.value = await compass.listTraces(100) as any[]
  } catch (e: any) {
    error.value = e.message
  } finally {
    loading.value = false
  }
})

function goToTrace(row: TraceRow) {
  if (row.run_id) {
    router.push(`/traces/${row.run_id}`)
  }
}

function durationLabel(ms: number): string {
  if (!ms) return '—'
  if (ms < 1) return '<1ms'
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function spanIcon(name: string): string {
  if (name.includes('.openai') || name.includes('chat ')) return 'AI'
  if (name.includes('.streaming') || name.includes('.stream')) return 'ST'
  if (name.includes('.data_store')) return 'DB'
  if (name.includes('.queue')) return 'Q'
  if (name.includes('.worker')) return 'W'
  if (name.includes('.run')) return 'R'
  if (name.includes('.thread')) return 'TH'
  if (name.includes('.message')) return 'M'
  return 'TR'
}

function timeAgo(ts: string): string {
  if (!ts) return '—'
  const diff = Date.now() - new Date(ts).getTime()
  const sec = Math.floor(diff / 1000)
  if (sec < 60) return `${sec}s ago`
  const min = Math.floor(sec / 60)
  if (min < 60) return `${min}m ago`
  const hr = Math.floor(min / 60)
  if (hr < 24) return `${hr}h ago`
  const d = Math.floor(hr / 24)
  return `${d}d ago`
}
</script>

<template>
  <div>
    <div class="page-header">
      <div>
        <h1>Traces</h1>
        <div class="page-header__subtitle">Click a trace to view all spans</div>
      </div>
      <div class="flex gap-2">
        <span v-if="traces.length" class="badge badge--neutral">{{ traces.length }} traces</span>
      </div>
    </div>

    <div v-if="loading" class="loading-center"><div class="spinner"></div></div>
    <div v-else-if="error" class="card">
      <div class="empty-state">
        <div class="empty-state__icon"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></div>
        <div class="empty-state__title">Traces unavailable</div>
        <p style="color: var(--text-muted); margin-top: 8px; font-size: 0.85rem;">
          {{ error }}<br><br>
          Make sure ClickHouse and the OTel Collector are running.<br>
          Set <code class="mono">CLICKHOUSE_URL</code> in your environment.
        </p>
      </div>
    </div>

    <div v-else-if="traces.length === 0" class="card">
      <div class="empty-state">
        <div class="empty-state__icon"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></div>
        <div class="empty-state__title">No traces yet</div>
        <p style="color: var(--text-muted); margin-top: 8px; font-size: 0.85rem;">
          Run your agent with tracing enabled to see traces here.
        </p>
      </div>
    </div>

    <div v-else class="traces-list">
      <div class="traces-list__header">
        <span>Trace</span>
        <span>Spans</span>
        <span>Duration</span>
        <span>Status</span>
        <span>Time</span>
      </div>

      <div
        v-for="trace in traces"
        :key="trace.TraceId"
        class="trace-row"
        @click="goToTrace(trace)"
      >
        <div class="trace-row__name">
          <span class="trace-row__icon">{{ spanIcon(trace.SpanName) }}</span>
          <div class="trace-row__info">
            <span class="trace-row__label">{{ trace.SpanName }}</span>
            <span class="trace-row__id mono">{{ shortId(trace.run_id) }}</span>
          </div>
        </div>
        <div class="trace-row__count">
          <span class="trace-row__count-badge">{{ trace.span_count || '—' }}</span>
        </div>
        <div class="trace-row__duration mono">{{ durationLabel(trace.duration_ms) }}</div>
        <div class="trace-row__status">
          <span
            class="badge"
            :class="{
              'badge--success': trace.StatusCode === 'STATUS_CODE_OK',
              'badge--error': trace.StatusCode === 'STATUS_CODE_ERROR',
              'badge--neutral': trace.StatusCode !== 'STATUS_CODE_OK' && trace.StatusCode !== 'STATUS_CODE_ERROR',
            }"
          >
            {{ trace.StatusCode?.replace('STATUS_CODE_', '') || 'UNSET' }}
          </span>
        </div>
        <div class="trace-row__time" :title="new Date(trace.Timestamp).toLocaleString()">
          {{ timeAgo(trace.Timestamp) }}
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.traces-list {
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  overflow: hidden;
  background: var(--bg-surface);
  box-shadow: var(--shadow-sm);
}

.traces-list__header {
  display: grid;
  grid-template-columns: 1fr 80px 100px 100px 100px;
  padding: 10px 16px;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border);
}

.trace-row {
  display: grid;
  grid-template-columns: 1fr 80px 100px 100px 100px;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  transition: background var(--transition);
}

.trace-row:last-child {
  border-bottom: none;
}

.trace-row:hover {
  background: var(--bg-hover);
}

.trace-row__name {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.trace-row__icon {
  font-size: 0.6rem;
  font-weight: 700;
  flex-shrink: 0;
  width: 26px;
  height: 20px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-hover);
  color: var(--text-secondary);
  border-radius: 4px;
  font-family: var(--font-mono);
  letter-spacing: -0.02em;
}

.trace-row__info {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.trace-row__label {
  font-weight: 500;
  font-size: 0.875rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.trace-row__id {
  font-size: 0.75rem;
  color: var(--text-muted);
}

.trace-row__count-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 28px;
  padding: 2px 8px;
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-secondary);
  background: var(--bg-hover);
  border-radius: 12px;
}

.trace-row__duration {
  font-size: 0.825rem;
  color: var(--text-secondary);
}

.trace-row__time {
  font-size: 0.8rem;
  color: var(--text-muted);
}
</style>
