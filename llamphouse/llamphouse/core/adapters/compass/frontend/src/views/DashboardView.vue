<script setup lang="ts">
import { ref, computed, onMounted, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { compass, formatTs } from '../api/client'
import type { Dashboard, ChartDef, DashboardChart } from '../api/client'
import ChartWidget from '../components/ChartWidget.vue'

const route = useRoute()
const router = useRouter()

const dashboard = ref<Dashboard | null>(null)
const chartLibrary = ref<ChartDef[]>([])   // global chart definitions
const loading = ref(true)
const error = ref('')
const saving = ref(false)
const saveError = ref('')
const editingTitle = ref(false)
const editingDesc = ref(false)
const titleRef = ref<HTMLInputElement | null>(null)
const descRef = ref<HTMLTextAreaElement | null>(null)
const showDeleteConfirm = ref(false)
const showAddChartModal = ref(false)

let _saveTimer: ReturnType<typeof setTimeout> | null = null

onMounted(async () => {
  try {
    const [db, charts] = await Promise.all([
      compass.getDashboard(route.params.dashboardId as string),
      compass.listCharts(),
    ])
    dashboard.value = db
    chartLibrary.value = charts
  } catch (e: any) {
    error.value = e.message
  } finally {
    loading.value = false
  }
})

// ── Resolve chart def from library ───────────────────────────────────────────

function resolveChart(slot: DashboardChart): ChartDef | null {
  return chartLibrary.value.find(c => c.id === slot.chart_id) ?? null
}

// ── Save helpers ──────────────────────────────────────────────────────────────

async function save() {
  if (!dashboard.value) return
  saving.value = true
  saveError.value = ''
  try {
    const updated = await compass.updateDashboard(dashboard.value.id, {
      title: dashboard.value.title,
      description: dashboard.value.description,
      charts: dashboard.value.charts,
    })
    dashboard.value = updated
  } catch (e: any) {
    saveError.value = e.message
  } finally {
    saving.value = false
  }
}

function debounceSave() {
  if (_saveTimer) clearTimeout(_saveTimer)
  _saveTimer = setTimeout(save, 600)
}

// ── Title / description inline edit ──────────────────────────────────────────

function startEditTitle() {
  editingTitle.value = true
  nextTick(() => titleRef.value?.focus())
}

function commitTitle() {
  editingTitle.value = false
  if (!dashboard.value?.title.trim()) dashboard.value!.title = 'Untitled Dashboard'
  debounceSave()
}

function startEditDesc() {
  editingDesc.value = true
  nextTick(() => descRef.value?.focus())
}

function commitDesc() {
  editingDesc.value = false
  debounceSave()
}

// ── Chart management ──────────────────────────────────────────────────────────

async function addNewChart() {
  // Create a brand-new chart def in the library, then add a slot on this dashboard
  const chart = await compass.createChart({ title: 'New Chart', sql: '', chart_type: 'table', x_column: null, y_columns: [] })
  chartLibrary.value.push(chart)
  const slot: DashboardChart = { chart_id: chart.id, col_span: 2, height_px: 280 }
  dashboard.value!.charts.push(slot)
  await save()
}

async function addExistingChart(chartId: string) {
  if (!dashboard.value) return
  // Avoid duplicates on this dashboard
  if (dashboard.value.charts.some(s => s.chart_id === chartId)) return
  const slot: DashboardChart = { chart_id: chartId, col_span: 2, height_px: 280 }
  dashboard.value.charts.push(slot)
  showAddChartModal.value = false
  await save()
}

async function onUpdateChart(updatedChart: ChartDef) {
  // Persist chart definition to the global library
  const saved = await compass.updateChart(updatedChart.id, updatedChart)
  const idx = chartLibrary.value.findIndex(c => c.id === saved.id)
  if (idx >= 0) chartLibrary.value[idx] = saved
  else chartLibrary.value.push(saved)
}

function onUpdateLayout(chartId: string, updatedLayout: DashboardChart) {
  if (!dashboard.value) return
  const idx = dashboard.value.charts.findIndex(s => s.chart_id === chartId)
  if (idx >= 0) dashboard.value.charts[idx] = updatedLayout
  debounceSave()
}

async function deleteChartSlot(chartId: string) {
  if (!dashboard.value) return
  dashboard.value.charts = dashboard.value.charts.filter(s => s.chart_id !== chartId)
  await save()
}

// ── Delete dashboard ──────────────────────────────────────────────────────────

async function deleteDashboard() {
  if (!dashboard.value) return
  try {
    await compass.deleteDashboard(dashboard.value.id)
    router.push('/dashboards')
  } catch (e: any) {
    saveError.value = e.message
  }
}

// ── Drag-to-reorder ───────────────────────────────────────────────────────────
const dragFromIdx = ref(-1)
const dragOverIdx = ref(-1)

function onChartDragStart(idx: number, e: DragEvent) {
  dragFromIdx.value = idx
  e.dataTransfer!.effectAllowed = 'move'
  e.dataTransfer!.setDragImage(e.currentTarget as HTMLElement, 20, 20)
}

function onChartDragOver(idx: number) {
  if (dragFromIdx.value < 0 || dragFromIdx.value === idx) return
  dragOverIdx.value = idx
}

function onChartDrop(idx: number) {
  if (dragFromIdx.value < 0 || dragFromIdx.value === idx) {
    onChartDragEnd(); return
  }
  const charts = [...dashboard.value!.charts]
  const [moved] = charts.splice(dragFromIdx.value, 1)
  charts.splice(idx, 0, moved)
  dashboard.value!.charts = charts
  onChartDragEnd()
  save()
}

function onChartDragEnd() {
  dragFromIdx.value = -1
  dragOverIdx.value = -1
}

// ── Charts not on this dashboard (for "add existing" modal) ──────────────────
const availableToAdd = computed(() => {
  const onDashboard = new Set(dashboard.value?.charts.map(s => s.chart_id) ?? [])
  return chartLibrary.value.filter(c => !onDashboard.has(c.id))
})

// Schema reference shown in the collapsible panel
const schemaInfo = [
  { name: 'threads',   cols: ['id', 'created_at', 'metadata'] },
  { name: 'messages',  cols: ['id', 'thread_id', 'role', 'status', 'assistant_id', 'run_id', 'created_at', 'completed_at', 'text'] },
  { name: 'runs',      cols: ['id', 'thread_id', 'assistant_id', 'status', 'model', 'created_at', 'started_at', 'completed_at', 'failed_at', 'prompt_tokens', 'completion_tokens', 'total_tokens'] },
  { name: 'run_steps', cols: ['id', 'run_id', 'thread_id', 'assistant_id', 'type', 'status', 'created_at', 'completed_at', 'prompt_tokens', 'completion_tokens', 'total_tokens'] },
]
</script>

<template>
  <div>
    <div v-if="loading" class="loading-center"><div class="spinner"></div></div>

    <div v-else-if="error" class="card" style="color: var(--error)">{{ error }}</div>

    <template v-else-if="dashboard">
      <!-- Header -->
      <div class="db-header">
        <div class="db-header__left">
          <router-link to="/dashboards" class="breadcrumb-link">Dashboards</router-link>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--text-muted)"><polyline points="9 18 15 12 9 6"/></svg>

          <!-- Inline title -->
          <input
            v-if="editingTitle"
            ref="titleRef"
            v-model="dashboard.title"
            class="title-input"
            @blur="commitTitle"
            @keydown.enter="commitTitle"
            @keydown.esc="commitTitle"
          />
          <h1 v-else class="db-title" @click="startEditTitle" title="Click to edit">
            {{ dashboard.title }}
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="edit-icon"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
          </h1>
        </div>

        <div class="db-header__right">
          <span v-if="saving" class="save-indicator">Saving…</span>
          <span v-else-if="saveError" class="save-indicator save-indicator--error">{{ saveError }}</span>
          <span v-else-if="dashboard.updated_at" class="save-indicator">Saved {{ formatTs(dashboard.updated_at) }}</span>
          <a
            :href="`/compass/dashboards/${dashboard.id}/present`"
            target="_blank"
            rel="noopener"
            class="btn btn--present"
            title="Open stakeholder view in new tab"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2"/><polyline points="8 21 12 17 16 21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
            Present
          </a>
          <button class="btn btn--danger-ghost" @click="showDeleteConfirm = true">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/></svg>
            Delete
          </button>
        </div>
      </div>

      <!-- Description -->
      <textarea
        v-if="editingDesc"
        ref="descRef"
        v-model="dashboard.description"
        class="desc-input"
        placeholder="Add a description…"
        rows="2"
        @blur="commitDesc"
        @keydown.esc="commitDesc"
      />
      <div
        v-else
        class="db-desc"
        :class="{ 'db-desc--placeholder': !dashboard.description }"
        @click="startEditDesc"
      >
        {{ dashboard.description || 'Add a description…' }}
      </div>

      <!-- Schema reference -->
      <details class="schema-ref">
        <summary class="schema-ref__toggle">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
          Schema reference
          <svg class="schema-chevron" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
        </summary>
        <div class="schema-ref__body">
          <div class="schema-table" v-for="t in schemaInfo" :key="t.name">
            <div class="schema-table__header">
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg>
              <span class="schema-table__name">{{ t.name }}</span>
            </div>
            <div class="schema-table__cols">
              <span class="schema-col" v-for="col in t.cols" :key="col">{{ col }}</span>
            </div>
          </div>
        </div>
      </details>

      <!-- Charts grid -->
      <div v-if="dashboard.charts.length" class="charts-grid">
        <div
          v-for="(slot, idx) in dashboard.charts"
          :key="slot.chart_id"
          :style="{ gridColumn: `span ${slot.col_span ?? 2}` }"
          :class="['chart-cell', { 'chart-cell--drag-over': dragOverIdx === idx }]"
          draggable="true"
          @dragstart="onChartDragStart(idx, $event)"
          @dragover.prevent="onChartDragOver(idx)"
          @drop.prevent="onChartDrop(idx)"
          @dragend="onChartDragEnd"
          @dragleave="dragOverIdx = -1"
        >
          <ChartWidget
            v-if="resolveChart(slot)"
            :chart="resolveChart(slot)!"
            :layout="slot"
            @update:chart="onUpdateChart"
            @update:layout="onUpdateLayout(slot.chart_id, $event)"
            @delete="deleteChartSlot(slot.chart_id)"
          />
          <div v-else class="chart-missing card">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
            Chart not found (id: {{ slot.chart_id.slice(0, 8) }}…)
            <button class="btn-link" @click="deleteChartSlot(slot.chart_id)">Remove</button>
          </div>
        </div>
      </div>

      <div v-else class="charts-empty">
        <p>No charts yet — add your first chart to visualise data from this dashboard.</p>
      </div>

      <!-- Add chart row -->
      <div class="add-chart-row">
        <button class="btn btn--primary" @click="addNewChart">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          New Chart
        </button>
        <button class="btn btn--outline" :disabled="availableToAdd.length === 0" @click="showAddChartModal = true" title="Reuse a chart from another dashboard">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><polyline points="16 6 12 2 8 6"/><line x1="12" y1="2" x2="12" y2="15"/></svg>
          Add from Library
        </button>
      </div>

      <!-- Add from Library modal -->
      <div v-if="showAddChartModal" class="modal-overlay" @click.self="showAddChartModal = false">
        <div class="modal card modal--library">
          <div class="modal__title">Add Chart from Library</div>
          <div class="library-list">
            <div
              v-for="c in availableToAdd"
              :key="c.id"
              class="library-item"
              @click="addExistingChart(c.id)"
            >
              <div class="library-item__title">{{ c.title }}</div>
              <div class="library-item__meta">
                <span class="badge badge--neutral">{{ c.chart_type }}</span>
                <span class="library-item__sql">{{ c.sql ? c.sql.slice(0, 60) + (c.sql.length > 60 ? '…' : '') : 'No SQL' }}</span>
              </div>
            </div>
            <div v-if="availableToAdd.length === 0" class="library-empty">
              All charts in the library are already on this dashboard.
            </div>
          </div>
          <div class="modal__actions">
            <button class="btn btn--ghost" @click="showAddChartModal = false">Cancel</button>
          </div>
        </div>
      </div>

      <!-- Delete confirmation modal -->
      <div v-if="showDeleteConfirm" class="modal-overlay" @click.self="showDeleteConfirm = false">
        <div class="modal card">
          <div class="modal__title">Delete Dashboard?</div>
          <div class="modal__body">This will permanently delete <strong>{{ dashboard.title }}</strong> and all its charts.</div>
          <div class="modal__actions">
            <button class="btn btn--ghost" @click="showDeleteConfirm = false">Cancel</button>
            <button class="btn btn--danger" @click="deleteDashboard">Delete</button>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>



<style scoped>
.db-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 6px;
}

.db-header__left {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.db-header__right {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}

.breadcrumb-link {
  font-size: 0.82rem;
  color: var(--text-muted);
  text-decoration: none;
  white-space: nowrap;
}

.breadcrumb-link:hover { color: var(--text-primary); }

.db-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 1.4rem;
  font-weight: 700;
  cursor: pointer;
  white-space: nowrap;
}

.edit-icon {
  opacity: 0;
  color: var(--text-muted);
  transition: opacity var(--transition);
  flex-shrink: 0;
}
.db-title:hover .edit-icon { opacity: 1; }

.title-input {
  font-size: 1.4rem;
  font-weight: 700;
  border: none;
  border-bottom: 2px solid var(--accent);
  background: transparent;
  color: var(--text-primary);
  outline: none;
  width: 320px;
}

.save-indicator {
  font-size: 0.75rem;
  color: var(--text-muted);
}

.save-indicator--error { color: var(--error); }

.db-desc {
  font-size: 0.83rem;
  color: var(--text-secondary);
  margin-bottom: 20px;
  cursor: pointer;
  padding: 4px 0;
  border-bottom: 1px dashed transparent;
  min-height: 20px;
  transition: border-color var(--transition);
}
.db-desc:hover { border-bottom-color: var(--border); }
.db-desc--placeholder { color: var(--text-muted); font-style: italic; }

.desc-input {
  width: 100%;
  font-size: 0.83rem;
  color: var(--text-secondary);
  background: transparent;
  border: none;
  border-bottom: 2px solid var(--accent);
  outline: none;
  resize: none;
  margin-bottom: 20px;
  padding: 4px 0;
  box-sizing: border-box;
}

.schema-ref {
  margin-bottom: 24px;
  font-size: 0.78rem;
}

.schema-ref__toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--text-muted);
  cursor: pointer;
  user-select: none;
  list-style: none;
  padding: 5px 0;
  font-size: 0.78rem;
  font-weight: 500;
}
.schema-ref__toggle::-webkit-details-marker { display: none; }
.schema-ref__toggle:hover { color: var(--text-secondary); }

.schema-chevron { transition: transform 0.15s ease; flex-shrink: 0; }
details[open] .schema-chevron { transform: rotate(180deg); }

.schema-ref__body {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
  gap: 10px;
  margin-top: 10px;
  padding: 14px;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
}

.schema-table {
  padding: 10px 12px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm, 6px);
}

.schema-table__header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
  color: var(--text-muted);
}

.schema-table__name {
  font-weight: 700;
  font-size: 0.82rem;
  color: var(--accent);
  font-family: 'Menlo', 'Consolas', monospace;
}

.schema-table__cols {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.schema-col {
  display: inline-block;
  padding: 2px 7px;
  background: color-mix(in srgb, var(--accent) 10%, transparent);
  color: var(--text-secondary);
  border-radius: 99px;
  font-family: 'Menlo', 'Consolas', monospace;
  font-size: 0.71rem;
  white-space: nowrap;
  cursor: default;
}

.charts-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
  margin-bottom: 20px;
}

.chart-cell {
  min-width: 0; /* prevent overflow in narrow cells */
  transition: opacity 0.15s;
}
.chart-cell--drag-over {
  outline: 2px dashed var(--accent);
  outline-offset: 3px;
  border-radius: var(--radius-md);
  opacity: 0.7;
}

.charts-empty {
  padding: 40px;
  text-align: center;
  color: var(--text-muted);
  font-size: 0.85rem;
  margin-bottom: 20px;
}

.add-chart-row {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-bottom: 40px;
}

.chart-missing {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 16px;
  color: var(--text-muted);
  font-size: 0.82rem;
}

/* Library modal */
.modal--library {
  width: 520px;
  max-width: 96vw;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
}

.library-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 4px;
}

.library-item {
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all var(--transition);
  background: var(--bg-secondary);
}
.library-item:hover {
  border-color: var(--accent);
  background: var(--accent-dim, color-mix(in srgb, var(--accent) 6%, transparent));
}

.library-item__title {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.library-item__meta {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.library-item__sql {
  font-size: 0.72rem;
  color: var(--text-muted);
  font-family: 'Menlo', 'Consolas', monospace;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 300px;
}

.library-empty {
  padding: 20px;
  text-align: center;
  font-size: 0.82rem;
  color: var(--text-muted);
}

/* Buttons */
.btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 14px;
  border-radius: var(--radius-md);
  font-size: 0.82rem;
  font-weight: 500;
  cursor: pointer;
  transition: all var(--transition);
  border: 1px solid transparent;
}

.btn--danger-ghost {
  background: transparent;
  color: var(--error);
  border-color: transparent;
}
.btn--danger-ghost:hover { background: color-mix(in srgb, var(--error) 10%, transparent); border-color: var(--error); }

.btn--danger {
  background: var(--error);
  color: #fff;
}
.btn--danger:hover { opacity: 0.88; }

.btn--ghost {
  background: transparent;
  color: var(--text-secondary);
  border-color: var(--border);
}
.btn--ghost:hover { background: var(--bg-hover); color: var(--text-primary); }

.btn--outline {
  background: transparent;
  color: var(--accent);
  border-color: var(--accent);
}
.btn--outline:hover:not(:disabled) { background: var(--accent-dim); }
.btn--outline:disabled { opacity: 0.4; cursor: not-allowed; }

.btn--primary {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}
.btn--primary:hover { opacity: 0.88; }

.btn--present {
  background: transparent;
  color: var(--text-secondary);
  border-color: var(--border);
  text-decoration: none;
}
.btn--present:hover {
  color: var(--accent);
  border-color: var(--accent);
  background: color-mix(in srgb, var(--accent) 6%, transparent);
}

/* Modal */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal {
  width: 380px;
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.modal__title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.modal__body {
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.modal__actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}
</style>
