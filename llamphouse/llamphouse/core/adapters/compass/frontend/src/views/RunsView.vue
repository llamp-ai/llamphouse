<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { compass, formatTs, shortId, statusBadge, durationMs } from '../api/client'
import type { Run } from '../api/client'
import DataTable from '../components/DataTable.vue'
import FilterBuilder from '../components/FilterBuilder.vue'
import type { FieldDef, FilterCondition } from '../components/FilterBuilder.vue'

const router = useRouter()
const route  = useRoute()
const runs = ref<Run[]>([])
const loading = ref(true)
const error = ref('')

// ── Filterable fields (must match server allowlist) ─────────────────────────
const filterFields: FieldDef[] = [
  { key: 'id',         label: 'Run ID',    type: 'string' },
  { key: 'agent_id',   label: 'Agent ID',  type: 'string' },
  { key: 'thread_id',  label: 'Thread ID', type: 'string' },
  { key: 'status',     label: 'Status',    type: 'string' },
  { key: 'created_at', label: 'Created',   type: 'date'   },
]

// ── Filter state, seeded from the URL ───────────────────────────────────────
function readFiltersFromUrl(): FilterCondition[] {
  const raw = route.query.filters
  if (typeof raw !== 'string' || !raw) return []
  try {
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed
      .filter((f) => f && typeof f === 'object' && f.field && f.operator)
      .map((f, i) => ({
        id:       String(i + 1),
        field:    String(f.field),
        operator: String(f.operator),
        value:    f.value  ?? '',
        value2:   f.value2 ?? '',
      }))
  } catch {
    return []
  }
}

const activeFilters = ref<FilterCondition[]>(readFiltersFromUrl())

// ── Cursor stack ↔ URL ──────────────────────────────────────────────────────
function readCursorsFromUrl(): (string | undefined)[] {
  const raw = route.query.cursors
  if (typeof raw !== 'string' || !raw) return [undefined]
  try {
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return [undefined]
    return [undefined, ...parsed.filter((c) => typeof c === 'string')]
  } catch {
    return [undefined]
  }
}

function writeUrl(filters: FilterCondition[], cursors: (string | undefined)[]) {
  const slim = filters.map(({ field, operator, value, value2 }) => ({
    field, operator, value, value2,
  }))
  const tail = cursors.slice(1) as string[]
  const query = { ...route.query }
  if (slim.length === 0) delete query.filters
  else                   query.filters = JSON.stringify(slim)
  if (tail.length === 0) delete query.cursors
  else                   query.cursors = JSON.stringify(tail)
  router.replace({ query })
}

// ── Pagination ──────────────────────────────────────────────────────────────
const PAGE_SIZE = 50
const cursorStack = ref<(string | undefined)[]>(readCursorsFromUrl())
const hasMore = ref(false)
const total   = ref<number | null>(null)
const pageIndex = computed(() => cursorStack.value.length)

function filtersForRequest() {
  return activeFilters.value.map(({ field, operator, value, value2 }) => ({
    field, operator, value, value2,
  }))
}

async function loadPage(after: string | undefined) {
  loading.value = true
  error.value = ''
  try {
    const res = await compass.listAllRuns({
      limit:   PAGE_SIZE,
      after,
      filters: filtersForRequest(),
    })
    runs.value = res.data ?? []
    hasMore.value = !!res.has_more
    total.value   = typeof res.total === 'number' ? res.total : null
  } catch (e: any) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

function resetAndReload() {
  cursorStack.value = [undefined]
  writeUrl(activeFilters.value, cursorStack.value)
  return loadPage(undefined)
}

onMounted(() => loadPage(cursorStack.value[cursorStack.value.length - 1]))

function nextPage() {
  if (!hasMore.value || !runs.value.length) return
  const lastId = runs.value[runs.value.length - 1].id
  cursorStack.value.push(lastId)
  writeUrl(activeFilters.value, cursorStack.value)
  loadPage(lastId)
}

function prevPage() {
  if (cursorStack.value.length <= 1) return
  cursorStack.value.pop()
  writeUrl(activeFilters.value, cursorStack.value)
  const after = cursorStack.value[cursorStack.value.length - 1]
  loadPage(after)
}

function onApply(conditions: FilterCondition[]) {
  activeFilters.value = conditions
  cursorStack.value = [undefined]
  writeUrl(conditions, cursorStack.value)
  loadPage(undefined)
}

// Re-sync if the URL changes externally (e.g. Back from a detail page).
watch(
  () => [route.query.filters, route.query.cursors],
  ([newF, newC], [oldF, oldC]) => {
    if (newF === oldF && newC === oldC) return
    if (route.path !== '/runs' && route.name !== 'runs') return
    activeFilters.value = readFiltersFromUrl()
    cursorStack.value   = readCursorsFromUrl()
    loadPage(cursorStack.value[cursorStack.value.length - 1])
  },
)

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
          <span v-if="total !== null">{{ total.toLocaleString() }} total · </span>
          Page {{ pageIndex }} ({{ runs.length }} shown)
          <span v-if="activeFilters.length > 0">
            · {{ activeFilters.length }} filter{{ activeFilters.length === 1 ? '' : 's' }} applied
          </span>
        </div>
      </div>
    </div>

    <FilterBuilder
      :fields="filterFields"
      :model-value="activeFilters"
      @apply="onApply"
    />

    <div v-if="loading" class="loading-center"><div class="spinner"></div></div>
    <div v-else-if="error" class="card" style="color: var(--error)">{{ error }}</div>

    <template v-else>
      <div class="card">
        <DataTable
          :columns="columns"
          :rows="runs"
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

      <div class="pager">
        <button class="btn" :disabled="pageIndex <= 1 || loading" @click="prevPage">← Previous</button>
        <span class="pager__info">Page {{ pageIndex }}</span>
        <button class="btn" :disabled="!hasMore || loading" @click="nextPage">Next →</button>
      </div>
    </template>
  </div>
</template>

<style scoped>
.pager {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 0.75rem;
  margin-top: 1rem;
}
.pager__info {
  color: var(--text-muted);
  font-size: 0.85rem;
}
.btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
</style>
