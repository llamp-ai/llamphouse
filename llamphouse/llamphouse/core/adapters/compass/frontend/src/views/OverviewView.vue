<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { compass, formatTs, shortId, statusBadge } from '../api/client'
import type { Overview, Thread, Run } from '../api/client'
import StatCard from '../components/StatCard.vue'
import DataTable from '../components/DataTable.vue'

const router = useRouter()
const data = ref<Overview | null>(null)
const recentThreads = ref<Thread[]>([])
const recentRuns = ref<(Run & { duration: number | null; tokens: number | null })[]>([])
const loading = ref(true)
const error = ref('')

onMounted(async () => {
  try {
    const [overview, threads] = await Promise.all([
      compass.overview(),
      compass.listThreads(10),
    ])
    data.value = overview
    recentThreads.value = threads.slice(0, 10)

    // Fetch runs from the most recent threads and enrich with derived fields
    const allRuns: (Run & { duration: number | null; tokens: number | null })[] = []
    for (const t of recentThreads.value.slice(0, 5)) {
      try {
        const runs = await compass.listRuns(t.id)
        for (const r of runs) {
          const endTs = r.completed_at ?? r.failed_at ?? null
          allRuns.push({
            ...r,
            duration: endTs != null && r.started_at != null ? endTs - r.started_at : null,
            tokens:   r.usage?.total_tokens ?? null,
          })
        }
      } catch { /* thread may have no runs */ }
    }
    allRuns.sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
    recentRuns.value = allRuns.slice(0, 10)
  } catch (e: any) {
    error.value = e.message
  } finally {
    loading.value = false
  }
})

// ── Column definitions ────────────────────────────────────────────────────────
const threadCols = [
  { key: 'id',         label: 'Thread ID', mono: true },
  { key: 'agent_name', label: 'Agent' },
  { key: 'created_at', label: 'Created' },
  { key: 'metadata',   label: 'Metadata' },
]

const runCols = [
  { key: 'id',           label: 'Run ID',   mono: true },
  { key: 'thread_id',    label: 'Thread',   mono: true },
  { key: 'status',       label: 'Status',   width: '110px' },
  { key: 'assistant_id', label: 'Agent',    mono: true },
  { key: 'model',        label: 'Model' },
  { key: 'created_at',   label: 'Created' },
  { key: 'completed_at', label: 'Completed' },
  { key: 'duration',     label: 'Duration' },
  { key: 'tokens',       label: 'Tokens' },
  { key: 'last_error',   label: 'Error' },
]

function fmtDuration(secs: number | null): string {
  if (secs == null) return '—'
  if (secs < 60) return `${secs.toFixed(2)}s`
  return `${Math.floor(secs / 60)}m ${(secs % 60).toFixed(2)}s`
}
</script>

<template>
  <div>
    <div class="page-header">
      <div>
        <h1>Overview</h1>
        <div class="page-header__subtitle">Compass Developer Dashboard</div>
      </div>
    </div>

    <div v-if="loading" class="loading-center"><div class="spinner"></div></div>

    <div v-else-if="error" class="card" style="color: var(--error)">{{ error }}</div>

    <template v-else-if="data">
      <!-- Stats -->
      <div class="grid-3 mb-4">
        <StatCard icon="A" label="Agents" :value="data.assistants" />
        <StatCard icon="T" label="Threads"    :value="data.threads" />
        <StatCard icon="R" label="Runs"       :value="data.runs" />
      </div>

      <!-- Recent Threads -->
      <div class="section">
        <div class="section__title">Recent Threads</div>
        <div class="card">
          <DataTable
            :columns="threadCols"
            :rows="recentThreads"
            :features="['columns']"
            table-id="overview-threads"
            clickable
            @row-click="(r) => router.push(`/threads/${r.id}`)"
          >
            <template #id="{ value }">{{ shortId(value) }}</template>
            <template #agent_name="{ value }">{{ value ?? '—' }}</template>
            <template #created_at="{ value }">{{ formatTs(value) }}</template>
            <template #metadata="{ row }">
              <span class="mono" style="color: var(--text-muted); font-size: 0.75rem;">
                {{ Object.keys(row.metadata || {}).length
                    ? JSON.stringify(row.metadata).slice(0, 60)
                    : '—' }}
              </span>
            </template>
          </DataTable>
        </div>
      </div>

      <!-- Recent Runs -->
      <div class="section">
        <div class="section__title">Recent Runs</div>
        <div class="card">
          <DataTable
            :columns="runCols"
            :rows="recentRuns"
            :features="['columns']"
            table-id="overview-runs"
            clickable
            @row-click="(r) => router.push(`/threads/${r.thread_id}/runs/${r.id}`)"
          >
            <template #id="{ value }">{{ shortId(value) }}</template>
            <template #thread_id="{ value }">{{ shortId(value) }}</template>
            <template #assistant_id="{ value }">{{ shortId(value) }}</template>
            <template #status="{ value }">
              <span class="badge" :class="statusBadge(value)">{{ value }}</span>
            </template>
            <template #created_at="{ value }">{{ formatTs(value) }}</template>
            <template #completed_at="{ value }">{{ value ? formatTs(value) : '—' }}</template>
            <template #duration="{ value }">{{ fmtDuration(value) }}</template>
            <template #tokens="{ value }">{{ value ?? '—' }}</template>
            <template #last_error="{ value }">
              <span v-if="value" style="color: var(--error); font-size: 0.78rem;">
                {{ typeof value === 'object' ? value.message ?? JSON.stringify(value) : value }}
              </span>
              <span v-else style="color: var(--text-muted)">—</span>
            </template>
          </DataTable>
        </div>
      </div>
    </template>
  </div>
</template>
