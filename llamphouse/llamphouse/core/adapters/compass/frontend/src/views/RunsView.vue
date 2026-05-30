<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { compass, formatTs, shortId, statusBadge, durationMs } from '../api/client'
import type { Run } from '../api/client'
import DataTable from '../components/DataTable.vue'
import FilterBuilder from '../components/FilterBuilder.vue'
import type { FieldDef, FilterCondition } from '../components/FilterBuilder.vue'

const router = useRouter()
const runs = ref<Run[]>([])
const loading = ref(true)
const error = ref('')

onMounted(async () => {
  try {
    runs.value = await compass.listAllRuns(10_000)
  } catch (e: any) {
    error.value = e.message
  } finally {
    loading.value = false
  }
})

// ── FilterBuilder field definitions ──────────────────────────────────────────
const filterFields: FieldDef[] = [
  { key: 'id',           label: 'Run ID',    type: 'string' },
  { key: 'agent_name',   label: 'Agent',     type: 'string' },
  { key: 'status',       label: 'Status',    type: 'string' },
  { key: 'thread_id',    label: 'Thread ID', type: 'string' },
  { key: 'created_at',   label: 'Created',   type: 'date'   },
]

const activeConditions = ref<FilterCondition[]>([])

function matchCondition(run: Run, cond: FilterCondition): boolean {
  const fieldDef = filterFields.find((f) => f.key === cond.field)
  if (!fieldDef) return true

  let raw: any = run[cond.field as keyof Run]

  if (fieldDef.type === 'date') {
    const ts = typeof raw === 'number' ? raw * 1000 : 0
    const d1 = cond.value  ? new Date(cond.value).getTime()  : 0
    const d2 = cond.value2 ? new Date(cond.value2).getTime() : 0
    switch (cond.operator) {
      case 'is_after':   return d1 ? ts > d1 : true
      case 'is_before':  return d1 ? ts < d1 : true
      case 'is_on': {
        const a = new Date(ts).toDateString()
        const b = new Date(cond.value).toDateString()
        return a === b
      }
      case 'is_between': return (d1 && d2) ? ts >= d1 && ts <= d2 : true
    }
    return true
  }

  const s = raw == null ? '' : String(raw).toLowerCase()
  const q = (cond.value ?? '').toLowerCase()
  switch (cond.operator) {
    case 'contains':     return s.includes(q)
    case 'not_contains': return !s.includes(q)
    case 'equals':       return s === q
    case 'not_equals':   return s !== q
    case 'starts_with':  return s.startsWith(q)
    case 'ends_with':    return s.endsWith(q)
    case 'is_empty':     return s === ''
    case 'is_not_empty': return s !== ''
  }
  return true
}

const filtered = computed(() => {
  if (activeConditions.value.length === 0) return runs.value
  return runs.value.filter((r) =>
    activeConditions.value.every((c) => matchCondition(r, c)),
  )
})

const matchCount = computed(() => filtered.value.length)

const columns = [
  { key: 'id',         label: 'Run ID',   mono: true },
  { key: 'agent_name', label: 'Agent' },
  { key: 'thread_id',  label: 'Thread',   mono: true },
  { key: 'status',     label: 'Status',   width: '120px' },
  { key: 'duration',   label: 'Duration', width: '100px' },
  { key: 'tokens',     label: 'Tokens',   width: '80px' },
  { key: 'created_at', label: 'Created' },
]
</script>

<template>
  <div>
    <div class="page-header">
      <div>
        <h1>Runs</h1>
        <div class="page-header__subtitle">
          {{ matchCount }} / {{ runs.length }} runs
        </div>
      </div>
    </div>

    <div v-if="loading" class="loading-center"><div class="spinner"></div></div>
    <div v-else-if="error" class="card" style="color: var(--error)">{{ error }}</div>

    <template v-else>
      <FilterBuilder
        :fields="filterFields"
        @change="(c) => (activeConditions = c)"
      />

      <div class="card">
        <DataTable
          :columns="columns"
          :rows="filtered"
          :features="['sort', 'resize', 'columns']"
          table-id="runs"
          clickable
          @row-click="(r) => router.push(`/threads/${r.thread_id}/runs/${r.id}`)"
        >
          <template #id="{ value }">{{ shortId(value) }}</template>
          <template #agent_name="{ value }">{{ value ?? '—' }}</template>
          <template #thread_id="{ value }">{{ shortId(value) }}</template>
          <template #status="{ value }">
            <span class="badge" :class="statusBadge(value)">{{ value }}</span>
          </template>
          <template #duration="{ row }">
            {{ durationMs(row.started_at, row.completed_at || row.failed_at) }}
          </template>
          <template #tokens="{ row }">
            {{ row.usage?.total_tokens ?? '—' }}
          </template>
          <template #created_at="{ value }">{{ formatTs(value) }}</template>
        </DataTable>
      </div>
    </template>
  </div>
</template>
