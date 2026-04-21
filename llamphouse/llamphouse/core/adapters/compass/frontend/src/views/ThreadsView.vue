<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { compass, formatTs, shortId } from '../api/client'
import type { Thread } from '../api/client'
import DataTable from '../components/DataTable.vue'
import FilterBuilder from '../components/FilterBuilder.vue'
import type { FieldDef, FilterCondition } from '../components/FilterBuilder.vue'

const router = useRouter()
const threads = ref<Thread[]>([])
const loading = ref(true)
const error   = ref('')

// Load as many threads as the backend allows so filtering works across all data
onMounted(async () => {
  try {
    threads.value = await compass.listThreads(10_000)
  } catch (e: any) {
    error.value = e.message
  } finally {
    loading.value = false
  }
})

// ── FilterBuilder field definitions ──────────────────────────────────────────
const filterFields: FieldDef[] = [
  { key: 'id',         label: 'Thread ID',  type: 'string' },
  { key: 'agent_name', label: 'Agent',      type: 'string' },
  { key: 'created_at', label: 'Created',    type: 'date'   },
  { key: 'metadata',   label: 'Metadata',   type: 'string' },
]

const activeConditions = ref<FilterCondition[]>([])

// ── Per-condition match logic ─────────────────────────────────────────────────
function matchCondition(thread: Thread, cond: FilterCondition): boolean {
  const fieldDef = filterFields.find((f) => f.key === cond.field)
  if (!fieldDef) return true

  let raw: any = thread[cond.field as keyof Thread]

  // Stringify metadata for text search
  if (cond.field === 'metadata') {
    raw = JSON.stringify(raw ?? {})
  }

  // ── date ────────────────────────────────────────────
  if (fieldDef.type === 'date') {
    const ts = typeof raw === 'number' ? raw * 1000 : 0   // epoch → ms
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

  // ── string ──────────────────────────────────────────
  const s = raw == null ? '' : String(raw).toLowerCase()
  const q = (cond.value ?? '').toLowerCase()
  switch (cond.operator) {
    case 'contains':      return s.includes(q)
    case 'not_contains':  return !s.includes(q)
    case 'equals':        return s === q
    case 'not_equals':    return s !== q
    case 'starts_with':   return s.startsWith(q)
    case 'ends_with':     return s.endsWith(q)
    case 'is_empty':      return s === ''
    case 'is_not_empty':  return s !== ''
  }
  return true
}

const filtered = computed(() => {
  if (activeConditions.value.length === 0) return threads.value
  return threads.value.filter((t) =>
    activeConditions.value.every((c) => matchCondition(t, c)),
  )
})

const matchCount = computed(() => filtered.value.length)

// ── Table columns ─────────────────────────────────────────────────────────────
const columns = [
  { key: 'id',         label: 'Thread ID', mono: true },
  { key: 'agent_name', label: 'Agent' },
  { key: 'created_at', label: 'Created' },
  { key: 'metadata',   label: 'Metadata' },
]
</script>

<template>
  <div>
    <div class="page-header">
      <div>
        <h1>Threads</h1>
        <div class="page-header__subtitle">
          {{ matchCount }} / {{ threads.length }} threads
        </div>
      </div>
    </div>

    <div v-if="loading" class="loading-center"><div class="spinner"></div></div>
    <div v-else-if="error" class="card" style="color: var(--error)">{{ error }}</div>

    <template v-else>
      <!-- Advanced filter -->
      <FilterBuilder
        :fields="filterFields"
        @change="(c) => (activeConditions = c)"
      />

      <!-- Threads table (sort + resize + columns, no built-in per-column filter) -->
      <div class="card">
        <DataTable
          :columns="columns"
          :rows="filtered"
          :features="['sort', 'resize', 'columns']"
          table-id="threads"
          clickable
          @row-click="(r) => router.push(`/threads/${r.id}`)"
        >
          <template #id="{ value }">{{ shortId(value) }}</template>
          <template #agent_name="{ value }">{{ value ?? '—' }}</template>
          <template #created_at="{ value }">{{ formatTs(value) }}</template>
          <template #metadata="{ row }">
            <span class="mono" style="color: var(--text-muted); font-size: 0.75rem;">
              {{ Object.keys(row.metadata || {}).length
                  ? JSON.stringify(row.metadata).slice(0, 80)
                  : '—' }}
            </span>
          </template>
        </DataTable>
      </div>
    </template>
  </div>
</template>
