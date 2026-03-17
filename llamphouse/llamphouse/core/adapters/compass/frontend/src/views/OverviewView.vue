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
const recentRuns = ref<Run[]>([])
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

    // Fetch runs from the most recent threads
    const allRuns: Run[] = []
    for (const t of recentThreads.value.slice(0, 5)) {
      try {
        const runs = await compass.listRuns(t.id)
        allRuns.push(...runs)
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

const threadCols = [
  { key: 'id', label: 'Thread ID', mono: true },
  { key: 'created_at', label: 'Created' },
]

const runCols = [
  { key: 'id', label: 'Run ID', mono: true },
  { key: 'status', label: 'Status', width: '120px' },
  { key: 'model', label: 'Model' },
  { key: 'created_at', label: 'Created' },
]
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
            clickable
            @row-click="(r) => router.push(`/threads/${r.id}`)"
          >
            <template #id="{ value }">{{ shortId(value) }}</template>
            <template #created_at="{ value }">{{ formatTs(value) }}</template>
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
            clickable
            @row-click="(r) => router.push(`/threads/${r.thread_id}/runs/${r.id}`)"
          >
            <template #id="{ value }">{{ shortId(value) }}</template>
            <template #status="{ value }">
              <span class="badge" :class="statusBadge(value)">{{ value }}</span>
            </template>
            <template #created_at="{ value }">{{ formatTs(value) }}</template>
          </DataTable>
        </div>
      </div>
    </template>
  </div>
</template>
