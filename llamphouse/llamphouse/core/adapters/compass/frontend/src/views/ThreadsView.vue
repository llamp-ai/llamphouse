<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { compass, formatTs, shortId } from '../api/client'
import type { Thread } from '../api/client'
import DataTable from '../components/DataTable.vue'
import FilterBuilder from '../components/FilterBuilder.vue'
import type { FieldDef, FilterCondition } from '../components/FilterBuilder.vue'

const router = useRouter()
const route  = useRoute()
const threads = ref<Thread[]>([])
const loading = ref(true)
const error   = ref('')

// ── Filterable fields (must match server allowlist) ─────────────────────────
// `agent_id` lives on the Run table, not Thread — the server handles that
// via an EXISTS subquery (matches if any run uses this agent).
const filterFields: FieldDef[] = [
  { key: 'id',         label: 'Thread ID', type: 'string' },
  { key: 'agent_id',   label: 'Agent ID',  type: 'string' },
  { key: 'created_at', label: 'Created',   type: 'date'   },
  { key: 'metadata',   label: 'Metadata',  type: 'string' },
]

// ── Filter state, initialised from the URL ──────────────────────────────────
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
  // The stack always starts with `undefined` (first page). URL stores only
  // the cursors after that, as a JSON array of ids.
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
  // Strip local id; serialise only field/operator/value(s).
  const slim = filters.map(({ field, operator, value, value2 }) => ({
    field, operator, value, value2,
  }))
  const tail = cursors.slice(1) as string[]   // drop leading undefined
  const query = { ...route.query }
  if (slim.length === 0) delete query.filters
  else                   query.filters = JSON.stringify(slim)
  if (tail.length === 0) delete query.cursors
  else                   query.cursors = JSON.stringify(tail)
  // `replace` so each filter/page change rewrites the same history entry.
  // Clicking a row uses `push`, so Back from a detail page returns to the
  // exact same URL (filters + cursors intact).
  router.replace({ query })
}

// ── Pagination ──────────────────────────────────────────────────────────────
const PAGE_SIZE = 50
// Stack of "after" cursors for each page we've visited. First entry is
// undefined (first page). Top of stack is the cursor used for the current
// page.  Persisted in the URL so Back from a detail page restores it.
const cursorStack = ref<(string | undefined)[]>(readCursorsFromUrl())
const hasMore = ref(false)
const total   = ref<number | null>(null)
const pageIndex = computed(() => cursorStack.value.length)

function filtersForRequest() {
  // Strip the local `id` field — the server only needs field/operator/values.
  return activeFilters.value.map(({ field, operator, value, value2 }) => ({
    field, operator, value, value2,
  }))
}

async function loadPage(after: string | undefined) {
  loading.value = true
  error.value = ''
  try {
    const res = await compass.listThreads({
      limit:   PAGE_SIZE,
      after,
      filters: filtersForRequest(),
    })
    threads.value = res.data ?? []
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
  if (!hasMore.value || !threads.value.length) return
  const lastId = threads.value[threads.value.length - 1].id
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

// If the URL changes externally (e.g. Back from a detail page), re-sync.
watch(
  () => [route.query.filters, route.query.cursors],
  ([newF, newC], [oldF, oldC]) => {
    if (newF === oldF && newC === oldC) return
    if (route.path !== '/threads' && route.name !== 'threads') return
    activeFilters.value = readFiltersFromUrl()
    cursorStack.value   = readCursorsFromUrl()
    loadPage(cursorStack.value[cursorStack.value.length - 1])
  },
)

// ── Table columns ───────────────────────────────────────────────────────────
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
          <span v-if="total !== null">{{ total.toLocaleString() }} total · </span>
          Page {{ pageIndex }} ({{ threads.length }} shown)
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
          :rows="threads"
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
