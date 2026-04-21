<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, nextTick, watch } from 'vue'
import { compass } from '../api/client'
import type { ChartDef, DashboardChart, QueryResult } from '../api/client'

// ── Props / Emits ─────────────────────────────────────────────────────────────
const props = defineProps<{
  chart: ChartDef
  layout: DashboardChart
  readonly?: boolean
}>()
const emit = defineEmits<{
  (e: 'update:chart', chart: ChartDef): void
  (e: 'update:layout', layout: DashboardChart): void
  (e: 'delete'): void
}>()

// ── Local state ───────────────────────────────────────────────────────────────
const localChart  = ref<ChartDef>({ ...props.chart, y_columns: [...(props.chart.y_columns ?? [])] })
const localLayout = ref<DashboardChart>({ ...props.layout })
const result = ref<QueryResult | null>(null)
const loading = ref(false)
const errMsg = ref('')
const showDrawer = ref(!props.readonly && !props.chart.sql)   // open on first mount when no SQL (not in readonly)
const editingTitle = ref(false)
const titleInput = ref<HTMLInputElement | null>(null)
const sqlRef = ref<HTMLTextAreaElement | null>(null)

// Keep local in sync when parent pushes a new chart prop (e.g. after library save)
watch(() => props.chart, (next) => {
  localChart.value = { ...next, y_columns: [...(next.y_columns ?? [])] }
})
watch(() => props.layout, (next) => {
  localLayout.value = { ...next }
})

// ── SQL Autosuggest ───────────────────────────────────────────────────────────
interface Suggestion { text: string; kind: 'keyword' | 'table' | 'column' }
const suggestions = ref<Suggestion[]>([])
const activeSuggestion = ref(0)
const showSuggestions = ref(false)

const SQL_KEYWORDS = [
  'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING',
  'JOIN', 'LEFT JOIN', 'INNER JOIN', 'RIGHT JOIN', 'FULL JOIN', 'CROSS JOIN',
  'ON', 'AS', 'AND', 'OR', 'NOT', 'IN', 'IS NULL', 'IS NOT NULL',
  'LIKE', 'BETWEEN', 'DISTINCT', 'LIMIT', 'OFFSET', 'WITH',
  'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'ROUND', 'CAST', 'COALESCE',
  'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'ASC', 'DESC',
]

const SCHEMA_TABLES: Record<string, string[]> = {
  threads:   ['id', 'created_at', 'metadata'],
  messages:  ['id', 'thread_id', 'role', 'status', 'assistant_id', 'run_id', 'created_at', 'completed_at', 'text'],
  runs:      ['id', 'thread_id', 'assistant_id', 'status', 'model', 'created_at', 'started_at', 'completed_at', 'failed_at', 'prompt_tokens', 'completion_tokens', 'total_tokens'],
  run_steps: ['id', 'run_id', 'thread_id', 'assistant_id', 'type', 'status', 'created_at', 'completed_at', 'prompt_tokens', 'completion_tokens', 'total_tokens'],
}
const ALL_COLUMNS = [...new Set(Object.values(SCHEMA_TABLES).flat())]

function onSqlInput() {
  const ta = sqlRef.value
  if (!ta) return
  const before = localChart.value.sql.slice(0, ta.selectionStart)
  const m = before.match(/\w+$/)
  const word = m ? m[0].toLowerCase() : ''
  if (word.length === 0) { showSuggestions.value = false; return }
  const hits: Suggestion[] = []
  for (const kw of SQL_KEYWORDS) {
    if (hits.length >= 12) break
    if (kw.toLowerCase().startsWith(word)) hits.push({ text: kw, kind: 'keyword' })
  }
  for (const tbl of Object.keys(SCHEMA_TABLES)) {
    if (hits.length >= 12) break
    if (tbl.startsWith(word)) hits.push({ text: tbl, kind: 'table' })
  }
  for (const col of ALL_COLUMNS) {
    if (hits.length >= 12) break
    if (col.startsWith(word) && !hits.find(h => h.text === col))
      hits.push({ text: col, kind: 'column' })
  }
  suggestions.value = hits
  activeSuggestion.value = 0
  showSuggestions.value = hits.length > 0
}

function applySuggestion(text: string) {
  const ta = sqlRef.value
  if (!ta) return
  const pos = ta.selectionStart
  const before = localChart.value.sql.slice(0, pos)
  const after = localChart.value.sql.slice(pos)
  const m = before.match(/\w+$/)
  const wordLen = m ? m[0].length : 0
  localChart.value.sql = before.slice(0, before.length - wordLen) + text + after
  showSuggestions.value = false
  suggestions.value = []
  nextTick(() => {
    ta.focus()
    const newPos = pos - wordLen + text.length
    ta.setSelectionRange(newPos, newPos)
  })
}

// ── Chart constants ───────────────────────────────────────────────────────────
// Margins (fixed; all other dims are measured from the DOM)
const ML = 54, MT = 14, MR = 18, MB = 66
const PALETTE = ['#5b67f5', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4', '#a855f7', '#14b8a6']

// Reactive canvas dimensions driven by ResizeObserver on the chart body.
// Safe from feedback loops because the SVG uses CSS height:100% (not a pixel
// :height attribute), so changing VH never causes the body to grow.
const bodyRef = ref<HTMLElement | null>(null)
const canvasW = ref(560)
const canvasH = ref(280)
const VW = computed(() => Math.max(canvasW.value, 200))
const VH = computed(() => Math.max(canvasH.value, 100))
const CW = computed(() => VW.value - ML - MR)
const CH = computed(() => VH.value - MT - MB)
let _resizeObs: ResizeObserver | null = null

// ── Auto-run on mount if SQL exists ──────────────────────────────────────────
onMounted(() => {
  if (widgetRef.value) {
    _resizeObs = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect
      if (rect) { canvasW.value = rect.width; canvasH.value = rect.height }
    })
    _resizeObs.observe(bodyRef.value!)
  }
  if (localChart.value.sql.trim()) runQuery()
})

// ── Query execution ───────────────────────────────────────────────────────────
async function runQuery() {
  if (!localChart.value.sql.trim()) return
  loading.value = true
  errMsg.value = ''
  try {
    const res = await compass.runQuery(localChart.value.sql)
    result.value = res
    // Auto-assign columns on first run
    if (res.columns.length && !localChart.value.x_column) {
      localChart.value.x_column = res.columns[0]
    }
    if (res.columns.length && !localChart.value.y_columns.length) {
      const numericCols = res.columns.filter((_c, ci) =>
        res.rows.some(r => r[ci] !== null && !isNaN(Number(r[ci])))
      )
      localChart.value.y_columns = numericCols.slice(0, 2)
      if (!localChart.value.y_columns.length && res.columns.length > 1) {
        localChart.value.y_columns = [res.columns[1]]
      }
    }
  } catch (e: any) {
    errMsg.value = e.message
  } finally {
    loading.value = false
  }
}

function onSqlKeydown(e: KeyboardEvent) {
  if (showSuggestions.value && suggestions.value.length) {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      activeSuggestion.value = (activeSuggestion.value + 1) % suggestions.value.length
      return
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault()
      activeSuggestion.value = (activeSuggestion.value - 1 + suggestions.value.length) % suggestions.value.length
      return
    }
    if (e.key === 'Tab' || (e.key === 'Enter' && !e.ctrlKey && !e.metaKey)) {
      e.preventDefault()
      applySuggestion(suggestions.value[activeSuggestion.value].text)
      return
    }
    if (e.key === 'Escape') {
      e.preventDefault()
      showSuggestions.value = false
      return
    }
  }
  if (e.key === 'Tab') {
    e.preventDefault()
    const ta = e.target as HTMLTextAreaElement
    const s = ta.selectionStart, end = ta.selectionEnd
    localChart.value.sql = localChart.value.sql.slice(0, s) + '  ' + localChart.value.sql.slice(end)
    nextTick(() => { ta.selectionStart = ta.selectionEnd = s + 2 })
  }
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) runQuery()
}

// ── Column selectors ──────────────────────────────────────────────────────────
const availableCols = computed(() => result.value?.columns ?? [])

function toggleYCol(col: string) {
  const idx = localChart.value.y_columns.indexOf(col)
  if (idx >= 0) {
    localChart.value.y_columns = localChart.value.y_columns.filter(c => c !== col)
  } else {
    localChart.value.y_columns = [...localChart.value.y_columns, col]
  }
}

function onConfigChange() {
  // chart config updated; the Save button in the drawer will persist
}

// ── Title editing ─────────────────────────────────────────────────────────────
function startEditTitle() {
  editingTitle.value = true
  nextTick(() => titleInput.value?.focus())
}

function commitTitle() {
  editingTitle.value = false
  if (!localChart.value.title.trim()) localChart.value.title = 'Chart'
  emit('update:chart', { ...localChart.value })
}

// ── Drawer save / discard ─────────────────────────────────────────────────────
let _savedChart: ChartDef | null = null

function openDrawer() {
  _savedChart = { ...localChart.value, y_columns: [...localChart.value.y_columns] }
  showDrawer.value = true
  nextTick(() => sqlRef.value?.focus())
}

function closeDrawer(save: boolean) {
  if (save) {
    showDrawer.value = false
    emit('update:chart', { ...localChart.value })
  } else {
    if (_savedChart) localChart.value = _savedChart
    showDrawer.value = false
  }
  _savedChart = null
}

// ── Chart type label ──────────────────────────────────────────────────────────
const chartTypeLabels: Record<string, string> = { table: 'Table', bar: 'Bar', line: 'Line', bignum: 'Big Number', pie: 'Pie' }

// ── Helper: get column index ──────────────────────────────────────────────────
function colIdx(col: string | null): number {
  if (!col || !result.value) return -1
  return result.value.columns.indexOf(col)
}

function numVal(v: any): number {
  const n = Number(v)
  return isNaN(n) ? 0 : n
}

function fmtLabel(v: any, maxLen = 14): string {
  const s = String(v ?? '')
  return s.length > maxLen ? s.slice(0, maxLen - 1) + '…' : s
}

function fmtNum(n: number): string {
  if (Math.abs(n) >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M'
  if (Math.abs(n) >= 1_000) return (n / 1_000).toFixed(1) + 'K'
  return Number.isInteger(n) ? String(n) : n.toFixed(2)
}

// ── Bar chart ─────────────────────────────────────────────────────────────────
interface BarItem { x: number; y: number; w: number; h: number; label: string; value: number }

const barData = computed((): BarItem[] => {
  if (!result.value || localChart.value.chart_type !== 'bar') return []
  const xi = colIdx(localChart.value.x_column)
  const yi = colIdx(localChart.value.y_columns[0])
  if (xi < 0 || yi < 0) return []
  const rows = result.value.rows.slice(0, 60)
  const vals = rows.map(r => numVal(r[yi]))
  const maxVal = Math.max(...vals, 0)
  if (rows.length === 0) return []
  const cw = CW.value, ch = CH.value
  const slotW = cw / rows.length
  const barW = Math.min(slotW * 0.72, 55)
  return rows.map((r, i) => {
    const h = maxVal > 0 ? (vals[i] / maxVal) * ch : 0
    return {
      x: ML + i * slotW + (slotW - barW) / 2,
      y: MT + ch - h,
      w: barW,
      h,
      label: fmtLabel(r[xi]),
      value: vals[i],
    }
  })
})

const barYTicks = computed(() => {
  if (!barData.value.length) return []
  const vals = barData.value.map(b => b.value)
  const maxVal = Math.max(...vals, 0)
  const ch = CH.value
  return [0, 0.25, 0.5, 0.75, 1].map(f => ({
    y: MT + (1 - f) * ch,
    label: fmtNum(maxVal * f),
  }))
})

const barRotateLabels = computed(() => barData.value.length > 8)

// ── Line chart ────────────────────────────────────────────────────────────────
interface LineSeries { col: string; color: string; polyline: string; pts: { x: number; y: number; label: string; value: number }[] }

const lineData = computed((): LineSeries[] => {
  if (!result.value || localChart.value.chart_type !== 'line') return []
  const xi = colIdx(localChart.value.x_column)
  if (xi < 0 || !localChart.value.y_columns.length) return []
  const rows = result.value.rows.slice(0, 200)
  const N = rows.length
  if (N === 0) return []

  // Find global y range across all series
  const allVals: number[] = []
  for (const col of localChart.value.y_columns) {
    const yi = colIdx(col)
    if (yi >= 0) rows.forEach(r => allVals.push(numVal(r[yi])))
  }
  const minV = Math.min(...allVals)
  const maxV = Math.max(...allVals)
  const range = maxV - minV || 1

  const cw = CW.value, ch = CH.value
  return localChart.value.y_columns.map((col, si) => {
    const yi = colIdx(col)
    if (yi < 0) return null as any
    const pts = rows.map((r, i) => {
      const v = numVal(r[yi])
      return {
        x: ML + (N > 1 ? (i / (N - 1)) : 0.5) * cw,
        y: MT + (1 - (v - minV) / range) * ch,
        label: fmtLabel(r[xi]),
        value: v,
      }
    })
    return {
      col,
      color: PALETTE[si % PALETTE.length],
      polyline: pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' '),
      pts,
    }
  }).filter(Boolean)
})

const lineYTicks = computed(() => {
  if (!lineData.value.length) return []
  const allVals = lineData.value.flatMap(s => s.pts.map(p => p.value))
  const minV = Math.min(...allVals)
  const maxV = Math.max(...allVals)
  const range = maxV - minV || 1
  const ch = CH.value
  return [0, 0.25, 0.5, 0.75, 1].map(f => ({
    y: MT + (1 - f) * ch,
    label: fmtNum(minV + range * f),
  }))
})

// ── Table: limit display rows ─────────────────────────────────────────────────
const tableRows = computed(() => result.value?.rows.slice(0, 200) ?? [])

// ── Big number ───────────────────────────────────────────────────────────────
const bigNumData = computed(() => {
  if (!result.value || localChart.value.chart_type !== 'bignum') return null
  const yi = colIdx(localChart.value.y_columns[0])
  if (yi < 0) return null
  const vals = result.value.rows.map(r => numVal(r[yi]))
  const total = vals.reduce((a, b) => a + b, 0)
  return { value: fmtNum(total), raw: total }
})

// ── Pie chart ─────────────────────────────────────────────────────────────────
interface PieSlice { label: string; value: number; pct: number; color: string; d: string; cx: number; cy: number; r: number; isFullCircle: boolean; labelX: number; labelY: number }

const pieData = computed((): PieSlice[] => {
  if (!result.value || localChart.value.chart_type !== 'pie') return []
  const xi = colIdx(localChart.value.x_column)
  const yi = colIdx(localChart.value.y_columns[0])
  if (xi < 0 || yi < 0) return []
  const rows = result.value.rows.slice(0, 20)
  const raw = rows.map(r => ({ label: fmtLabel(r[xi], 18), value: Math.abs(numVal(r[yi])) }))
  const total = raw.reduce((s, r) => s + r.value, 0) || 1
  const CX = VW.value / 2, CY = VH.value / 2, R = Math.min(CW.value, CH.value) / 2 - 10
  let angle = -Math.PI / 2
  return raw.map((item, i) => {
    const slice = (item.value / total) * Math.PI * 2
    const mid = angle + slice / 2
    const isFullCircle = raw.length === 1
    // For a full circle, offset end angle slightly so arc is not degenerate
    const effectiveSlice = isFullCircle ? Math.PI * 2 - 0.0001 : slice
    const x1 = CX + R * Math.cos(angle)
    const y1 = CY + R * Math.sin(angle)
    angle += effectiveSlice
    const x2 = CX + R * Math.cos(angle)
    const y2 = CY + R * Math.sin(angle)
    const large = effectiveSlice > Math.PI ? 1 : 0
    const lR = R * 0.72
    return {
      label: item.label,
      value: item.value,
      pct: Math.round((item.value / total) * 100),
      color: PALETTE[i % PALETTE.length],
      d: `M ${CX} ${CY} L ${x1.toFixed(2)} ${y1.toFixed(2)} A ${R} ${R} 0 ${large} 1 ${x2.toFixed(2)} ${y2.toFixed(2)} Z`,
      cx: CX, cy: CY, r: R,
      isFullCircle,
      labelX: CX + lR * Math.cos(mid),
      labelY: CY + lR * Math.sin(mid),
    }
  })
})

// ── X-axis label indices (up to 7 evenly-spaced, always incl. first & last) ──
const xLabelIndices = computed(() => {
  const N = lineData.value[0]?.pts.length ?? 0
  if (N === 0) return []
  if (N <= 7) return Array.from({ length: N }, (_, i) => i)
  const step = Math.ceil((N - 1) / 6)
  const idxs: number[] = []
  for (let i = 0; i < N; i += step) idxs.push(i)
  if (idxs[idxs.length - 1] !== N - 1) idxs.push(N - 1)
  return idxs
})

// ── Resize (height drag) ─────────────────────────────────────────────────────
const HEIGHT_SNAP = 80   // snap grid in px
const MIN_ROWS   = 2    // minimum snapped rows
const MAX_ROWS   = 12

function snapH(raw: number) {
  return Math.round(Math.max(MIN_ROWS * HEIGHT_SNAP, Math.min(MAX_ROWS * HEIGHT_SNAP, raw)) / HEIGHT_SNAP) * HEIGHT_SNAP
}

const resizing = ref(false)
let resizeStartY = 0
let resizeStartH = 0

const heightRows = computed(() => Math.round((localLayout.value.height_px ?? 280) / HEIGHT_SNAP))
const snapLines  = computed(() => Array.from({ length: MAX_ROWS - 1 }, (_, i) => (i + 1) * HEIGHT_SNAP))

function onResizeStart(e: MouseEvent) {
  resizing.value = true
  resizeStartY = e.clientY
  resizeStartH = localLayout.value.height_px ?? snapH(280)
  window.addEventListener('mousemove', onResizeMove)
  window.addEventListener('mouseup', onResizeEnd)
  e.preventDefault()
}

function onResizeMove(e: MouseEvent) {
  if (!resizing.value) return
  localLayout.value.height_px = snapH(resizeStartH + (e.clientY - resizeStartY))
}

function onResizeEnd() {
  if (!resizing.value) return
  resizing.value = false
  window.removeEventListener('mousemove', onResizeMove)
  window.removeEventListener('mouseup', onResizeEnd)
  emit('update:layout', { ...localLayout.value })
}

function setColSpan(_n: never) {} // removed – now handled by drag
const GRID_COLS = 4
const widgetRef = ref<HTMLElement | null>(null)

// ── Width resize (right-edge drag, snaps to column grid) ──────────────────────
const widthResizing = ref(false)
let widthResizeStartX = 0
let widthResizeStartCols = 2
let widthResizeStartW = 0

const colSnapLabel = computed(() => {
  const n = localLayout.value.col_span ?? 2
  return `${n} col${n > 1 ? 's' : ''}`
})

function onWidthResizeStart(e: MouseEvent) {
  widthResizing.value = true
  widthResizeStartX = e.clientX
  widthResizeStartCols = localLayout.value.col_span ?? 2
  widthResizeStartW = widgetRef.value?.offsetWidth ?? 400
  window.addEventListener('mousemove', onWidthResizeMove)
  window.addEventListener('mouseup', onWidthResizeEnd)
  e.preventDefault()
  e.stopPropagation()
}

function onWidthResizeMove(e: MouseEvent) {
  if (!widthResizing.value) return
  const singleColPx = widthResizeStartW / widthResizeStartCols
  const newRaw = widthResizeStartCols + (e.clientX - widthResizeStartX) / singleColPx
  localLayout.value.col_span = Math.round(Math.max(1, Math.min(GRID_COLS, newRaw))) as 1 | 2 | 3 | 4
}

function onWidthResizeEnd() {
  if (!widthResizing.value) return
  widthResizing.value = false
  window.removeEventListener('mousemove', onWidthResizeMove)
  window.removeEventListener('mouseup', onWidthResizeEnd)
  emit('update:layout', { ...localLayout.value })
}

onBeforeUnmount(() => {
  _resizeObs?.disconnect()
  window.removeEventListener('mousemove', onResizeMove)
  window.removeEventListener('mouseup', onResizeEnd)
  window.removeEventListener('mousemove', onWidthResizeMove)
  window.removeEventListener('mouseup', onWidthResizeEnd)
})
</script>

<template>
  <div ref="widgetRef" class="widget card" :class="{ 'widget--resizing-w': widthResizing }">
    <!-- ── Widget header ── -->
    <div class="widget__header">
      <div class="widget__title-row">
        <div v-if="!readonly" class="drag-grip" title="Drag to reorder"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="5" r="1" fill="currentColor"/><circle cx="9" cy="12" r="1" fill="currentColor"/><circle cx="9" cy="19" r="1" fill="currentColor"/><circle cx="15" cy="5" r="1" fill="currentColor"/><circle cx="15" cy="12" r="1" fill="currentColor"/><circle cx="15" cy="19" r="1" fill="currentColor"/></svg></div>
        <input
          v-if="editingTitle && !readonly"
          ref="titleInput"
          v-model="localChart.title"
          class="title-input"
          @blur="commitTitle"
          @keydown.enter="commitTitle"
          @keydown.esc="commitTitle"
        />
        <div v-else class="widget__title" :class="{ 'widget__title--static': readonly }" @click="!readonly && startEditTitle()" :title="readonly ? '' : 'Click to edit'">
          {{ localChart.title }}
        </div>

        <div class="widget__header-actions">
          <span v-if="result" class="duration-badge">{{ result.duration_ms }}ms</span>
          <template v-if="!readonly">
            <button class="icon-btn" :title="showDrawer ? 'Close editor' : 'Edit chart'" @click="showDrawer ? closeDrawer(false) : openDrawer()">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
            </button>
            <button class="icon-btn icon-btn--danger" title="Delete chart" @click="emit('delete')">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
          </template>
        </div>
      </div>

      <!-- ── SQL editor ── -->
      <div v-if="showDrawer" class="sql-editor">
        <div class="sql-wrap">
          <textarea
            ref="sqlRef"
            v-model="localChart.sql"
            class="sql-textarea"
            placeholder="SELECT status, COUNT(*) as cnt FROM runs GROUP BY status"
            rows="5"
            spellcheck="false"
            @keydown="onSqlKeydown"
            @input="onSqlInput"
            @blur="showSuggestions = false"
          />
          <div v-if="showSuggestions && suggestions.length" class="sql-suggestions">
            <div
              v-for="(s, i) in suggestions"
              :key="s.text"
              :class="['suggestion-item', { 'suggestion-item--active': i === activeSuggestion }]"
              @mousedown.prevent="applySuggestion(s.text)"
              @mouseover="activeSuggestion = i"
            >
              <span :class="['suggestion-kind', `suggestion-kind--${s.kind}`]">
                {{ s.kind === 'keyword' ? 'KW' : s.kind === 'table' ? 'TBL' : 'COL' }}
              </span>
              {{ s.text }}
            </div>
          </div>
        </div>
        <div class="sql-footer">
          <span class="sql-hint">Ctrl+Enter to run</span>
          <div style="display:flex;gap:6px">
            <button class="btn btn--ghost btn--sm" @click="closeDrawer(false)">Discard</button>
            <button class="btn btn--primary btn--sm" :disabled="loading || !localChart.sql.trim()" @click="runQuery">
              <svg v-if="loading" class="spin" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
              <svg v-else width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
              {{ loading ? 'Running…' : 'Run' }}
            </button>
            <button class="btn btn--save btn--sm" @click="closeDrawer(true)" title="Save chart definition to library">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>
              Save
            </button>
          </div>
        </div>

        <!-- Error -->
        <div v-if="errMsg" class="sql-error">{{ errMsg }}</div>

        <!-- Chart config (once there are results) -->
        <template v-if="result && availableCols.length">
          <div class="config-bar">
            <label class="config-label">
              Type
              <select v-model="localChart.chart_type" class="config-select" @change="onConfigChange">
                <option value="table">Table</option>
                <option value="bar">Bar</option>
                <option value="line">Line</option>
                <option value="bignum">Big Number</option>
                <option value="pie">Pie</option>
              </select>
            </label>

            <!-- X column: bar, line, pie -->
            <template v-if="localChart.chart_type === 'bar' || localChart.chart_type === 'line' || localChart.chart_type === 'pie'">
              <label class="config-label">
                X column
                <select v-model="localChart.x_column" class="config-select" @change="onConfigChange">
                  <option v-for="c in availableCols" :key="c" :value="c">{{ c }}</option>
                </select>
              </label>
            </template>

            <!-- Y column(s): bar, line, bignum, pie -->
            <template v-if="localChart.chart_type !== 'table'">
              <label class="config-label">
                {{ localChart.chart_type === 'line' ? 'Y columns' : 'Value column' }}
                <div class="y-cols-picker">
                  <label v-for="c in availableCols" :key="c" class="y-col-check">
                    <input
                      type="checkbox"
                      :checked="localChart.y_columns.includes(c)"
                      @change="toggleYCol(c)"
                    />
                    {{ c }}
                  </label>
                </div>
              </label>
            </template>
          </div>
        </template>
      </div>
    </div>

    <!-- ── Chart body ── -->
    <div ref="bodyRef" class="widget__body" :style="{ height: `${localLayout.height_px ?? snapH(280)}px` }">
      <!-- Height snap-grid ghost lines (only while dragging) -->
      <template v-if="resizing">
        <div v-for="line in snapLines" :key="line"
          class="snap-line"
          :style="{ top: `${line}px` }"
        />
      </template>
      <!-- Loading -->
      <div v-if="loading" class="chart-loading"><div class="spinner"></div></div>

      <!-- No results yet -->
      <div v-else-if="!result && !errMsg" class="chart-empty">
        Write a SQL query above and click Run
      </div>

      <!-- Error (no editor open) -->
      <div v-else-if="errMsg && !showDrawer" class="chart-error">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
        {{ errMsg }}
        <button class="btn-link" @click="openDrawer()">Edit SQL</button>
      </div>

      <template v-else-if="result">
        <!-- 0 rows -->
        <div v-if="result.rows.length === 0" class="chart-empty">Query returned no rows</div>

        <!-- Table chart -->
        <template v-else-if="localChart.chart_type === 'table'">
          <div class="table-wrap">
            <table class="result-table">
              <thead>
                <tr>
                  <th v-for="c in result.columns" :key="c">{{ c }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row, ri) in tableRows" :key="ri">
                  <td v-for="(cell, ci) in row" :key="ci">{{ cell ?? '—' }}</td>
                </tr>
              </tbody>
            </table>
            <div v-if="result.rows.length > 200" class="table-truncated">
              Showing 200 of {{ result.rows.length }} rows
            </div>
          </div>
        </template>

        <!-- Bar chart -->
        <template v-else-if="localChart.chart_type === 'bar'">
          <div v-if="!barData.length" class="chart-empty">Select X and Y columns to render the chart</div>
          <svg v-else :viewBox="`0 0 ${VW} ${VH}`" class="chart-svg" aria-label="Bar chart">
            <!-- Gridlines -->
            <g class="gridlines">
              <line v-for="t in barYTicks" :key="t.y"
                :x1="ML" :y1="t.y" :x2="ML + CW" :y2="t.y"
                stroke="var(--border)" stroke-width="1" />
            </g>
            <!-- Bars -->
            <g class="bars">
              <rect v-for="b in barData" :key="b.label"
                :x="b.x" :y="b.y" :width="b.w" :height="b.h"
                :fill="PALETTE[0]" rx="3" opacity="0.88" />
            </g>
            <!-- Y axis labels -->
            <g class="y-labels">
              <text v-for="t in barYTicks" :key="t.y"
                :x="ML - 6" :y="t.y + 4"
                text-anchor="end" font-size="10" fill="var(--text-muted)">
                {{ t.label }}
              </text>
            </g>
            <!-- X axis labels -->
            <g class="x-labels">
              <text v-for="b in barData" :key="b.label"
                :x="b.x + b.w / 2" :y="MT + CH + 14"
                text-anchor="end" font-size="10" fill="var(--text-muted)"
                :transform="barRotateLabels ? `rotate(-38, ${b.x + b.w / 2}, ${MT + CH + 14})` : ''">
                {{ b.label }}
              </text>
            </g>
            <!-- Y axis line -->
            <line :x1="ML" :y1="MT" :x2="ML" :y2="MT + CH + 2"
              stroke="var(--border)" stroke-width="1.5" />
            <!-- X axis line -->
            <line :x1="ML" :y1="MT + CH" :x2="ML + CW" :y2="MT + CH"
              stroke="var(--border)" stroke-width="1.5" />
          </svg>
        </template>

        <!-- Line chart -->
        <template v-else-if="localChart.chart_type === 'line'">
          <div v-if="!lineData.length" class="chart-empty">Select X and at least one Y column</div>
          <template v-else>
            <svg :viewBox="`0 0 ${VW} ${VH}`" class="chart-svg" aria-label="Line chart">
              <!-- Gridlines -->
              <g class="gridlines">
                <line v-for="t in lineYTicks" :key="t.y"
                  :x1="ML" :y1="t.y" :x2="ML + CW" :y2="t.y"
                  stroke="var(--border)" stroke-width="1" />
              </g>
              <!-- Series -->
              <g v-for="s in lineData" :key="s.col">
                <polyline
                  :points="s.polyline"
                  :stroke="s.color"
                  stroke-width="2"
                  fill="none"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                />
                <circle v-for="(p, pi) in s.pts" :key="pi"
                  :cx="p.x" :cy="p.y" r="3.5"
                  :fill="s.color" stroke="var(--bg-surface)" stroke-width="1.5" />
              </g>
              <!-- Y axis labels -->
              <g class="y-labels">
                <text v-for="t in lineYTicks" :key="t.y"
                  :x="ML - 6" :y="t.y + 4"
                  text-anchor="end" font-size="10" fill="var(--text-muted)">
                  {{ t.label }}
                </text>
              </g>
              <!-- X axis labels (first, mid, last) -->
              <g class="x-labels" v-if="lineData[0]?.pts.length">
                <text v-for="idx in xLabelIndices" :key="idx"
                  :x="lineData[0].pts[idx].x" :y="MT + CH + 16"
                  text-anchor="middle" font-size="10" fill="var(--text-muted)">
                  {{ lineData[0].pts[idx].label }}
                </text>
              </g>
              <!-- Axes -->
              <line :x1="ML" :y1="MT" :x2="ML" :y2="MT + CH + 2" stroke="var(--border)" stroke-width="1.5" />
              <line :x1="ML" :y1="MT + CH" :x2="ML + CW" :y2="MT + CH" stroke="var(--border)" stroke-width="1.5" />
            </svg>
            <!-- Legend for multiple series -->
            <div v-if="lineData.length > 1" class="chart-legend">
              <span v-for="s in lineData" :key="s.col" class="legend-item">
                <span class="legend-dot" :style="{ background: s.color }" />
                {{ s.col }}
              </span>
            </div>
          </template>
        </template>

        <!-- Big number chart -->
        <template v-else-if="localChart.chart_type === 'bignum'">
          <div v-if="!bigNumData" class="chart-empty">Select a numeric value column</div>
          <div v-else class="bignum-wrap">
            <div class="bignum-value">{{ bigNumData.value }}</div>
            <div class="bignum-label">{{ localChart.y_columns[0] ?? '' }}</div>
          </div>
        </template>

        <!-- Pie chart -->
        <template v-else-if="localChart.chart_type === 'pie'">
          <div v-if="!pieData.length" class="chart-empty">Select X (label) and value column</div>
          <template v-else>
            <svg :viewBox="`0 0 ${VW} ${VH}`" class="chart-svg" aria-label="Pie chart">
              <g v-for="s in pieData" :key="s.label">
                <!-- Full circle (single slice) -->
                <circle v-if="s.isFullCircle"
                  :cx="s.cx" :cy="s.cy" :r="s.r"
                  :fill="s.color" opacity="0.88" stroke="var(--bg-surface)" stroke-width="2" />
                <!-- Normal arc slice -->
                <path v-else :d="s.d" :fill="s.color" opacity="0.88" stroke="var(--bg-surface)" stroke-width="2" />
                <text
                  v-if="s.pct >= 5"
                  :x="s.labelX" :y="s.labelY"
                  text-anchor="middle" dominant-baseline="middle"
                  font-size="11" font-weight="600" fill="#fff">
                  {{ s.pct }}%
                </text>
              </g>
            </svg>
            <div class="chart-legend">
              <span v-for="s in pieData" :key="s.label" class="legend-item">
                <span class="legend-dot" :style="{ background: s.color }" />
                {{ s.label }} ({{ fmtNum(s.value) }})
              </span>
            </div>
          </template>
        </template>
      </template>
    </div>
    <template v-if="!readonly">
      <div
        class="resize-handle"
        :class="{ 'resize-handle--active': resizing }"
        draggable="false"
        @mousedown.prevent="onResizeStart"
      >
        <div class="resize-handle__bar"></div>
        <span v-if="resizing" class="resize-label">{{ heightRows }} rows</span>
      </div>
      <!-- Right-edge width resize -->
      <div
        class="resize-handle-right"
        :class="{ 'resize-handle-right--active': widthResizing }"
        draggable="false"
        @mousedown.prevent.stop="onWidthResizeStart"
      >
        <div class="resize-handle-right__bar"></div>
        <span v-if="widthResizing" class="resize-label resize-label--right">{{ colSnapLabel }}</span>
      </div>
    </template>
  </div>
</template>

<style scoped>
/* ── Card ──────────────────────────────────────────────────────────────────── */
.widget {
  padding: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* ── Header ─────────────────────────────────────────────────────────────────── */
.widget__header {
  padding: 14px 16px 0;
  border-bottom: 1px solid var(--border);
}

.widget__title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding-bottom: 10px;
}

.widget__title {
  font-weight: 600;
  font-size: 0.88rem;
  color: var(--text-primary);
  cursor: pointer;
  flex: 1;
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.widget__title:hover {
  color: var(--accent);
}

.widget__title--static {
  cursor: default;
}
.widget__title--static:hover {
  color: var(--text-primary);
}

.title-input {
  flex: 1;
  font-weight: 600;
  font-size: 0.88rem;
  border: none;
  border-bottom: 1.5px solid var(--accent);
  background: transparent;
  color: var(--text-primary);
  outline: none;
  min-width: 0;
}

.widget__header-actions {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
}

.duration-badge {
  font-size: 0.7rem;
  color: var(--text-muted);
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 1px 5px;
}

/* ── Icon buttons ───────────────────────────────────────────────────────────── */
.icon-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  border-radius: var(--radius-sm);
  border: none;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  transition: all var(--transition);
}
.icon-btn:hover { background: var(--bg-hover); color: var(--text-primary); }
.icon-btn--danger:hover { background: color-mix(in srgb, var(--error) 10%, transparent); color: var(--error); }

/* ── SQL editor ─────────────────────────────────────────────────────────────── */
.sql-editor {
  padding-bottom: 12px;
}

.sql-textarea {
  width: 100%;
  box-sizing: border-box;
  padding: 10px 12px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  color: var(--text-primary);
  font-family: 'Menlo', 'Consolas', 'Monaco', monospace;
  font-size: 0.78rem;
  line-height: 1.6;
  resize: vertical;
  outline: none;
  transition: border-color var(--transition);
  margin-top: 10px;
}
.sql-textarea:focus { border-color: var(--accent); }

.sql-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 8px;
}

.sql-hint { font-size: 0.7rem; color: var(--text-muted); }

.sql-error {
  margin-top: 8px;
  padding: 8px 10px;
  background: color-mix(in srgb, var(--error) 8%, transparent);
  border: 1px solid color-mix(in srgb, var(--error) 30%, transparent);
  border-radius: var(--radius-sm);
  font-size: 0.78rem;
  color: var(--error);
  white-space: pre-wrap;
}

/* ── Config bar ─────────────────────────────────────────────────────────────── */
.config-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
}

.config-label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 0.72rem;
  color: var(--text-muted);
  font-weight: 500;
}

.config-select {
  padding: 4px 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-secondary);
  color: var(--text-primary);
  font-size: 0.78rem;
  outline: none;
  cursor: pointer;
}
.config-select:focus { border-color: var(--accent); }

.y-cols-picker {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.y-col-check {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.78rem;
  color: var(--text-secondary);
  cursor: pointer;
  white-space: nowrap;
}

/* ── Widget body ────────────────────────────────────────────────────────────── */
.widget__body {
  /* height set via inline style (layout.height_px); overflow:hidden ensures
     the body never grows beyond that, breaking any ResizeObserver feedback. */
  overflow: hidden;
  display: flex;
  flex-direction: column;
  position: relative;
}

/* ── Snap grid ghost lines ─────────────────────────────────────────────────── */
.snap-line {
  position: absolute;
  left: 12px;
  right: 12px;
  height: 1px;
  background: color-mix(in srgb, var(--accent) 25%, transparent);
  pointer-events: none;
}

/* ── Resize handle ───────────────────────────────────────────────────────── */
.resize-handle {
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  cursor: ns-resize;
  user-select: none;
  flex-shrink: 0;
  opacity: 0;
  transition: opacity var(--transition);
  position: relative;
}
.widget:hover .resize-handle,
.resize-handle--active { opacity: 1; }

.resize-handle__bar {
  width: 36px;
  height: 3px;
  border-radius: 99px;
  background: var(--border);
  transition: background var(--transition);
}
.resize-handle:hover .resize-handle__bar,
.resize-handle--active .resize-handle__bar { background: var(--accent); }

.resize-label {
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--accent);
  background: color-mix(in srgb, var(--accent) 12%, var(--bg-surface));
  border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
  border-radius: 4px;
  padding: 1px 6px;
  pointer-events: none;
  white-space: nowrap;
}

/* ── Col-span segmented picker ──────────────────────────────────────────── */
/* removed – replaced by right-edge drag */

/* ── Drag grip ───────────────────────────────────────────────────────────── */
.drag-grip {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  color: var(--text-muted);
  cursor: grab;
  opacity: 0;
  transition: opacity var(--transition);
  margin-right: 4px;
}
.widget:hover .drag-grip { opacity: 1; }
.drag-grip:active { cursor: grabbing; }

/* ── Right-edge width resize ─────────────────────────────────────────────── */
.widget { position: relative; }
.widget--resizing-w { user-select: none; }

.resize-handle-right {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 18px; /* stop above height handle */
  width: 14px;
  cursor: ew-resize;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity var(--transition);
  z-index: 10;
}
.widget:hover .resize-handle-right,
.resize-handle-right--active { opacity: 1; }

.resize-handle-right__bar {
  width: 3px;
  height: 36px;
  border-radius: 99px;
  background: var(--border);
  transition: background var(--transition);
}
.resize-handle-right:hover .resize-handle-right__bar,
.resize-handle-right--active .resize-handle-right__bar { background: var(--accent); }

.resize-label--right {
  position: absolute;
  right: 18px;
  top: 50%;
  transform: translateY(-50%);
  white-space: nowrap;
}

.chart-loading,
.chart-empty {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.82rem;
  color: var(--text-muted);
}

.chart-error {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.82rem;
  color: var(--error);
}

.btn-link {
  background: none;
  border: none;
  color: var(--accent);
  font-size: 0.82rem;
  cursor: pointer;
  padding: 0;
  text-decoration: underline;
}

/* ── Table ──────────────────────────────────────────────────────────────────── */
.table-wrap {
  overflow: auto;
  flex: 1;
  height: 100%;
}

.result-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.78rem;
}

.result-table th,
.result-table td {
  padding: 5px 10px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}

.result-table th {
  background: var(--bg-secondary);
  color: var(--text-muted);
  font-weight: 600;
  position: sticky;
  top: 0;
}

.result-table td { color: var(--text-secondary); }
.result-table tr:last-child td { border-bottom: none; }

.table-truncated {
  text-align: center;
  font-size: 0.72rem;
  color: var(--text-muted);
  margin-top: 6px;
}

/* ── SVG chart ──────────────────────────────────────────────────────────────── */
.chart-svg {
  /* CSS drives the size; VW/VH viewBox scales content to fit.
     No :height attribute on the element — prevents ResizeObserver loops. */
  width: 100%;
  height: 100%;
  display: block;
  overflow: visible;
  flex: 1;
}

/* ── Legend ─────────────────────────────────────────────────────────────────── */
.chart-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 6px;
  font-size: 0.75rem;
  color: var(--text-secondary);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 5px;
}

.legend-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

/* ── Buttons ────────────────────────────────────────────────────────────────── */
.btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: var(--radius-md);
  font-size: 0.8rem;
  font-weight: 500;
  cursor: pointer;
  transition: all var(--transition);
  border: 1px solid transparent;
}

.btn--primary {
  background: var(--accent);
  color: #fff;
}
.btn--primary:hover:not(:disabled) { opacity: 0.88; }
.btn--primary:disabled { opacity: 0.45; cursor: not-allowed; }

.btn--ghost {
  background: transparent;
  color: var(--text-secondary);
  border-color: var(--border);
}
.btn--ghost:hover { background: var(--bg-hover); color: var(--text-primary); }

.btn--save {
  background: color-mix(in srgb, #22c55e 85%, transparent);
  color: #fff;
  border-color: transparent;
}
.btn--save:hover { opacity: 0.88; }

.btn--sm { padding: 5px 10px; font-size: 0.76rem; }

/* Spinner animation */
.spin {
  animation: spin 0.8s linear infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}

/* ── Big Number ─────────────────────────────────────────────────────────────── */
.bignum-wrap {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
}

.bignum-value {
  font-size: 3.8rem;
  font-weight: 800;
  color: var(--accent);
  line-height: 1;
  letter-spacing: -0.02em;
}

.bignum-label {
  font-size: 0.8rem;
  color: var(--text-muted);
  font-family: 'Menlo', 'Consolas', monospace;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

/* ── SQL autosuggest ────────────────────────────────────────────────────────── */
.sql-wrap { position: relative; }

.sql-suggestions {
  position: absolute;
  top: calc(100% - 3px);
  left: 0;
  right: 0;
  z-index: 200;
  background: var(--bg-secondary);
  border: 1px solid var(--accent);
  border-top: none;
  border-radius: 0 0 var(--radius-md) var(--radius-md);
  max-height: 210px;
  overflow-y: auto;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
  font-size: 0.79rem;
}

.suggestion-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  cursor: pointer;
  color: var(--text-secondary);
  font-family: 'Menlo', 'Consolas', monospace;
  transition: background var(--transition);
}
.suggestion-item:hover,
.suggestion-item--active {
  background: var(--bg-hover, rgba(255, 255, 255, 0.07));
  color: var(--text-primary);
}

.suggestion-kind {
  flex-shrink: 0;
  font-size: 0.62rem;
  font-weight: 700;
  padding: 1px 5px;
  border-radius: 3px;
  font-family: sans-serif;
  letter-spacing: 0.04em;
  min-width: 28px;
  text-align: center;
}
.suggestion-kind--keyword { background: color-mix(in srgb, #5b67f5 18%, transparent); color: #5b67f5; }
.suggestion-kind--table   { background: color-mix(in srgb, #22c55e 18%, transparent); color: #22c55e; }
.suggestion-kind--column  { background: color-mix(in srgb, #f59e0b 18%, transparent); color: #f59e0b; }
</style>
