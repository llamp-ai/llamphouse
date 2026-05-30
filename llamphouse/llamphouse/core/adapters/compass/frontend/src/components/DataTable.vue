<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { useTablePrefs } from '../composables/useTablePrefs'

export interface ColumnDef {
  key: string
  label: string
  mono?: boolean
  width?: string
}

const props = withDefaults(defineProps<{
  columns: ColumnDef[]
  rows?: Record<string, any>[]
  clickable?: boolean
  /**
   * Unique identifier used to persist column visibility, widths and sort
   * in localStorage. When omitted all state is ephemeral.
   */
  tableId?: string
  /**
   * Which interactive features to enable. Defaults to all four.
   * Pass a subset to limit the toolbar, e.g. `:features="['columns']"` for
   * column-picker only.
   */
  features?: Array<'sort' | 'filter' | 'resize' | 'columns'>
}>(), {
  rows: () => [],
  features: () => ['sort', 'filter', 'resize', 'columns'],
})

const emit = defineEmits<{
  rowClick: [row: Record<string, any>]
}>()

// ── Prefs (localStorage) ──────────────────────────────────────────────────────
const { hiddenCols, colWidths, sort, setWidth, toggleHidden, setSort, resetPrefs } =
  useTablePrefs(props.tableId)

// ── Filter (ephemeral) ────────────────────────────────────────────────────────
const filters     = ref<Record<string, string>>({})
const showFilters = ref(false)

function clearFilters() {
  filters.value = {}
}

const activeFilterCount = computed(() =>
  Object.values(filters.value).filter(Boolean).length,
)

// ── Feature flags ─────────────────────────────────────────────────────────────
const hasSort    = computed(() => (props.features as string[]).includes('sort'))
const hasFilter  = computed(() => (props.features as string[]).includes('filter'))
const hasResize  = computed(() => (props.features as string[]).includes('resize'))
const hasColumns = computed(() => (props.features as string[]).includes('columns'))
const hasToolbar = computed(() => hasFilter.value || hasColumns.value)

// ── Column picker ─────────────────────────────────────────────────────────────
const showColPicker = ref(false)

function onDocClick(e: MouseEvent) {
  if (!(e.target as HTMLElement).closest('.dt-col-picker')) {
    showColPicker.value = false
  }
}
onMounted(() => document.addEventListener('click', onDocClick, true))
onBeforeUnmount(() => document.removeEventListener('click', onDocClick, true))

// ── Derived columns / rows ────────────────────────────────────────────────────
const visibleCols = computed(() =>
  props.columns.filter((c) => !hiddenCols.value.includes(c.key)),
)

const processedRows = computed(() => {
  let rows = [...(props.rows ?? [])]

  // Per-column filter (case-insensitive substring)
  for (const [key, val] of Object.entries(filters.value)) {
    if (!val.trim()) continue
    const q = val.trim().toLowerCase()
    rows = rows.filter((row) => {
      const cell = row[key]
      if (cell == null) return false
      return String(cell).toLowerCase().includes(q)
    })
  }

  // Sort
  if (sort.value) {
    const { key, dir } = sort.value
    rows.sort((a, b) => {
      const av = a[key] ?? ''
      const bv = b[key] ?? ''
      const cmp = av < bv ? -1 : av > bv ? 1 : 0
      return dir === 'asc' ? cmp : -cmp
    })
  }

  return rows
})

// ── Column resize ─────────────────────────────────────────────────────────────
let _resizeKey    = ''
let _resizeStartX = 0
let _resizeStartW = 0

function onResizeStart(e: MouseEvent, colKey: string) {
  e.preventDefault()
  e.stopPropagation()
  const th = (e.target as HTMLElement).closest('th') as HTMLTableCellElement
  _resizeKey    = colKey
  _resizeStartX = e.clientX
  _resizeStartW = colWidths.value[colKey] ?? th.offsetWidth
  document.addEventListener('mousemove', onResizeMove)
  document.addEventListener('mouseup', onResizeEnd)
  document.body.style.userSelect = 'none'
  document.body.style.cursor = 'col-resize'
}

function onResizeMove(e: MouseEvent) {
  const delta = e.clientX - _resizeStartX
  const newW  = Math.max(60, _resizeStartW + delta)
  colWidths.value = { ...colWidths.value, [_resizeKey]: newW }
}

function onResizeEnd() {
  document.removeEventListener('mousemove', onResizeMove)
  document.removeEventListener('mouseup', onResizeEnd)
  document.body.style.userSelect = ''
  document.body.style.cursor = ''
  setWidth(_resizeKey, colWidths.value[_resizeKey] ?? _resizeStartW)
}

onBeforeUnmount(() => {
  document.removeEventListener('mousemove', onResizeMove)
  document.removeEventListener('mouseup', onResizeEnd)
})

// ── Helpers ───────────────────────────────────────────────────────────────────
function thStyle(col: ColumnDef) {
  const w = colWidths.value[col.key]
  if (w) return { width: `${w}px`, minWidth: `${w}px`, maxWidth: `${w}px` }
  if (col.width) return { width: col.width, minWidth: col.width }
  return {}
}

function sortIcon(key: string) {
  if (sort.value?.key !== key) return '↕'
  return sort.value.dir === 'asc' ? '↑' : '↓'
}
</script>

<template>
  <div class="dt-root">
    <!-- Toolbar -->
    <div v-if="hasToolbar" class="dt-toolbar">
      <!-- Filter toggle -->
      <button
        v-if="hasFilter"
        class="dt-btn"
        :class="{ 'dt-btn--active': showFilters || activeFilterCount > 0 }"
        @click="showFilters = !showFilters"
        title="Toggle per-column filters"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
          <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>
        </svg>
        Filter
        <span v-if="activeFilterCount > 0" class="dt-badge">{{ activeFilterCount }}</span>
      </button>

      <button
        v-if="hasFilter && activeFilterCount > 0"
        class="dt-btn dt-btn--ghost"
        @click="clearFilters"
        title="Clear all filters"
      >✕ Clear</button>

      <!-- Column picker -->
      <div v-if="hasColumns" class="dt-col-picker">
        <button
          class="dt-btn"
          :class="{ 'dt-btn--active': showColPicker }"
          @click.stop="showColPicker = !showColPicker"
          title="Show / hide columns"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="7" height="7" rx="1"/>
            <rect x="14" y="3" width="7" height="7" rx="1"/>
            <rect x="3" y="14" width="7" height="7" rx="1"/>
            <rect x="14" y="14" width="7" height="7" rx="1"/>
          </svg>
          Columns
          <span v-if="hiddenCols.length > 0" class="dt-badge">{{ hiddenCols.length }} hidden</span>
        </button>
        <div v-if="showColPicker" class="dt-col-picker__dropdown">
          <div class="dt-col-picker__title">Show / hide columns</div>
          <label
            v-for="col in columns"
            :key="col.key"
            class="dt-col-picker__item"
          >
            <input
              type="checkbox"
              :checked="!hiddenCols.includes(col.key)"
              :disabled="visibleCols.length === 1 && !hiddenCols.includes(col.key)"
              @change="toggleHidden(col.key)"
            />
            {{ col.label }}
          </label>
          <div class="dt-col-picker__footer">
            <button class="dt-btn dt-btn--ghost" style="font-size:0.75rem" @click="resetPrefs">Reset all</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Table -->
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th
              v-for="col in visibleCols"
              :key="col.key"
              :style="thStyle(col)"
              class="dt-th"
            >
              <div
                class="dt-th__inner"
                :class="{ 'dt-th__inner--sortable': hasSort }"
                @click="hasSort && setSort(col.key)"
              >
                <span class="dt-th__label">{{ col.label }}</span>
                <span
                  v-if="hasSort"
                  class="dt-sort-icon"
                  :class="{ 'dt-sort-icon--active': sort?.key === col.key }"
                >{{ sortIcon(col.key) }}</span>
              </div>
              <!-- Resize handle -->
              <div
                v-if="hasResize"
                class="dt-resize-handle"
                @mousedown="(e) => onResizeStart(e, col.key)"
              />
            </th>
          </tr>

          <!-- Filter row -->
          <tr v-if="hasFilter && showFilters" class="dt-filter-row">
            <th v-for="col in visibleCols" :key="col.key" class="dt-filter-cell">
              <input
                class="dt-filter-input"
                v-model="filters[col.key]"
                placeholder="filter…"
                @click.stop
              />
            </th>
          </tr>
        </thead>

        <tbody>
          <tr
            v-for="(row, i) in processedRows"
            :key="i"
            :class="{ clickable }"
            @click="clickable && emit('rowClick', row)"
          >
            <td
              v-for="col in visibleCols"
              :key="col.key"
              :class="{ mono: col.mono }"
              :style="thStyle(col)"
            >
              <slot :name="col.key" :row="row" :value="row[col.key]">
                {{ row[col.key] ?? '—' }}
              </slot>
            </td>
          </tr>
          <tr v-if="processedRows.length === 0">
            <td
              :colspan="visibleCols.length"
              style="text-align: center; color: var(--text-muted); padding: 32px;"
            >
              {{ activeFilterCount > 0 ? 'No rows match the active filters' : 'No data' }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
/* ── Root ────────────────────────────────────────── */
.dt-root {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

/* ── Toolbar ─────────────────────────────────────── */
.dt-toolbar {
  display: flex;
  align-items: center;
  gap: 6px;
}

.dt-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 10px;
  font-size: 0.78rem;
  font-weight: 500;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm, 4px);
  background: var(--bg-surface);
  color: var(--text-secondary);
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s, color 0.12s;
  white-space: nowrap;
}
.dt-btn:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}
.dt-btn--active {
  background: var(--accent-dim);
  border-color: var(--accent);
  color: var(--accent);
}
.dt-btn--ghost {
  background: none;
  border-color: transparent;
  color: var(--text-muted);
}
.dt-btn--ghost:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
  border-color: var(--border);
}

.dt-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.68rem;
  font-weight: 600;
  background: var(--accent);
  color: #fff;
  border-radius: 10px;
  padding: 0 5px;
  min-width: 16px;
  height: 16px;
  line-height: 1;
}

/* ── Column picker ───────────────────────────────── */
.dt-col-picker {
  position: relative;
}
.dt-col-picker__dropdown {
  position: absolute;
  top: calc(100% + 4px);
  left: 0;
  z-index: 200;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md, 6px);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.10);
  min-width: 180px;
  padding: 8px 0 4px;
}
.dt-col-picker__title {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  padding: 0 12px 6px;
}
.dt-col-picker__item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 12px;
  font-size: 0.82rem;
  color: var(--text-primary);
  cursor: pointer;
  user-select: none;
}
.dt-col-picker__item:hover {
  background: var(--bg-hover);
}
.dt-col-picker__item input {
  accent-color: var(--accent);
  cursor: pointer;
}
.dt-col-picker__footer {
  border-top: 1px solid var(--border);
  margin-top: 4px;
  padding: 4px 8px 2px;
}

/* ── Column headers ──────────────────────────────── */
.dt-th {
  position: relative;
  white-space: nowrap;
}
.dt-th__inner {
  display: flex;
  align-items: center;
  gap: 5px;
  cursor: default;
  padding-right: 8px; /* room for resize handle */
}
.dt-th__inner--sortable {
  cursor: pointer;
  user-select: none;
}
.dt-th__inner:hover .dt-sort-icon {
  opacity: 1;
}
.dt-th__label {
  flex: 1;
}
.dt-sort-icon {
  font-size: 0.7rem;
  opacity: 0.35;
  transition: opacity 0.12s;
}
.dt-sort-icon--active {
  opacity: 1;
  color: var(--accent);
}

/* ── Resize handle ───────────────────────────────── */
.dt-resize-handle {
  position: absolute;
  top: 0;
  right: 0;
  width: 5px;
  height: 100%;
  cursor: col-resize;
  z-index: 1;
  border-right: 2px solid transparent;
  transition: border-color 0.12s;
}
.dt-resize-handle:hover,
.dt-resize-handle:active {
  border-right-color: var(--accent);
}

/* ── Filter row ──────────────────────────────────── */
:deep(.dt-filter-row th) {
  padding: 4px 8px;
  background: var(--bg-secondary);
  font-weight: normal;
  text-transform: none;
  letter-spacing: normal;
}
.dt-filter-input {
  width: 100%;
  box-sizing: border-box;
  padding: 3px 7px;
  font-size: 0.78rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm, 4px);
  background: var(--bg-surface);
  color: var(--text-primary);
  outline: none;
}
.dt-filter-input:focus {
  border-color: var(--accent);
}
.dt-filter-input::placeholder {
  color: var(--text-muted);
}
</style>
