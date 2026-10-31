<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { compass } from '../api/client'
import type { Dashboard, ChartDef, DashboardChart } from '../api/client'
import ChartWidget from '../components/ChartWidget.vue'

const route = useRoute()
const router = useRouter()

const dashboard = ref<Dashboard | null>(null)
const chartLibrary = ref<ChartDef[]>([])
const loading = ref(true)
const error = ref('')
const lastRefreshed = ref<Date | null>(null)
const refreshedAgo = ref('')

// ── Auto-refresh state ────────────────────────────────────────────────────────
type RefreshInterval = 0 | 30 | 60 | 300
const refreshOptions: { label: string; value: RefreshInterval }[] = [
  { label: 'Off',  value: 0 },
  { label: '30s',  value: 30 },
  { label: '1 min', value: 60 },
  { label: '5 min', value: 300 },
]
const selectedRefresh = ref<RefreshInterval>(0)
let _refreshTimer: ReturnType<typeof setInterval> | null = null
let _agoTimer: ReturnType<typeof setInterval> | null = null

// ── Load data ─────────────────────────────────────────────────────────────────
async function load() {
  loading.value = true
  error.value = ''
  try {
    const [db, charts] = await Promise.all([
      compass.getDashboard(route.params.dashboardId as string),
      compass.listCharts(),
    ])
    dashboard.value = db
    chartLibrary.value = charts
    lastRefreshed.value = new Date()
    updateAgo()
  } catch (e: any) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

// ── "Last refreshed X ago" tick ───────────────────────────────────────────────
function updateAgo() {
  if (!lastRefreshed.value) return
  const secs = Math.round((Date.now() - lastRefreshed.value.getTime()) / 1000)
  if (secs < 10) refreshedAgo.value = 'just now'
  else if (secs < 60) refreshedAgo.value = `${secs}s ago`
  else if (secs < 3600) refreshedAgo.value = `${Math.floor(secs / 60)}m ago`
  else refreshedAgo.value = `${Math.floor(secs / 3600)}h ago`
}

// ── Auto-refresh ──────────────────────────────────────────────────────────────
function applyRefresh(val: RefreshInterval) {
  selectedRefresh.value = val
  if (_refreshTimer) clearInterval(_refreshTimer)
  if (val > 0) {
    _refreshTimer = setInterval(load, val * 1000)
  }
}

// ── Resolve chart def ─────────────────────────────────────────────────────────
function resolveChart(slot: DashboardChart): ChartDef | null {
  return chartLibrary.value.find(c => c.id === slot.chart_id) ?? null
}

// ── Computed: edit URL ────────────────────────────────────────────────────────
const editUrl = computed(() =>
  `/compass/dashboards/${route.params.dashboardId}`
)

onMounted(() => {
  load()
  _agoTimer = setInterval(updateAgo, 5000)
})

onBeforeUnmount(() => {
  if (_refreshTimer) clearInterval(_refreshTimer)
  if (_agoTimer) clearInterval(_agoTimer)
})
</script>

<template>
  <div class="present-shell">
    <!-- ── Top bar ──────────────────────────────────────────────────────────── -->
    <div class="topbar">
      <div class="topbar__left">
        <div class="topbar__logo">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><path d="M14 17h7m-3.5-3.5v7"/></svg>
        </div>
        <div class="topbar__title-group">
          <div v-if="loading && !dashboard" class="topbar__title topbar__title--loading">Loading…</div>
          <div v-else-if="dashboard" class="topbar__title">{{ dashboard.title }}</div>
          <div v-if="dashboard?.description" class="topbar__desc">{{ dashboard.description }}</div>
        </div>
      </div>

      <div class="topbar__right">
        <!-- Refreshed ago -->
        <span v-if="lastRefreshed" class="refreshed-ago">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
          {{ refreshedAgo }}
        </span>

        <!-- Refresh interval picker -->
        <div class="refresh-picker">
          <button
            v-for="opt in refreshOptions"
            :key="opt.value"
            :class="['refresh-btn', { 'refresh-btn--active': selectedRefresh === opt.value }]"
            @click="applyRefresh(opt.value)"
          >{{ opt.label }}</button>
        </div>

        <!-- Manual refresh -->
        <button class="icon-action" title="Refresh now" :class="{ 'icon-action--spinning': loading }" @click="load">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
        </button>

        <!-- Edit link (opens editor in same tab) -->
        <a :href="editUrl" class="edit-link" title="Edit dashboard">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
          Edit
        </a>
      </div>
    </div>

    <!-- ── Body ────────────────────────────────────────────────────────────── -->
    <div class="present-body">
      <div v-if="loading && !dashboard" class="loading-center">
        <div class="spinner"></div>
      </div>

      <div v-else-if="error" class="error-msg">{{ error }}</div>

      <template v-else-if="dashboard">
        <!-- Empty state -->
        <div v-if="!dashboard.charts.length" class="empty-state">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
          <p>This dashboard has no charts yet.</p>
          <a :href="editUrl" class="edit-link">Open editor to add charts →</a>
        </div>

        <!-- Charts grid -->
        <div v-else class="charts-grid">
          <div
            v-for="slot in dashboard.charts"
            :key="slot.chart_id"
            :style="{ gridColumn: `span ${slot.col_span ?? 2}` }"
            class="chart-cell"
          >
            <ChartWidget
              v-if="resolveChart(slot)"
              :chart="resolveChart(slot)!"
              :layout="slot"
              :readonly="true"
            />
            <div v-else class="chart-missing card">
              Chart not found
            </div>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
/* ── Shell ──────────────────────────────────────────────────────────────────── */
.present-shell {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--bg-secondary);
  overflow: hidden;
}

/* ── Top bar ────────────────────────────────────────────────────────────────── */
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 0 24px;
  height: 56px;
  flex-shrink: 0;
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border);
}

.topbar__left {
  display: flex;
  align-items: center;
  gap: 14px;
  min-width: 0;
}

.topbar__logo {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  background: var(--accent);
  color: #fff;
  border-radius: var(--radius-md);
  flex-shrink: 0;
}

.topbar__title-group {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.topbar__title {
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.topbar__title--loading {
  color: var(--text-muted);
  font-weight: 400;
}

.topbar__desc {
  font-size: 0.75rem;
  color: var(--text-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 480px;
}

.topbar__right {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}

/* ── Refreshed-ago ──────────────────────────────────────────────────────────── */
.refreshed-ago {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.75rem;
  color: var(--text-muted);
  white-space: nowrap;
}

/* ── Refresh interval picker ───────────────────────────────────────────────── */
.refresh-picker {
  display: flex;
  gap: 2px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 2px;
}

.refresh-btn {
  padding: 3px 10px;
  font-size: 0.72rem;
  font-weight: 500;
  border: none;
  border-radius: calc(var(--radius-md) - 2px);
  cursor: pointer;
  background: transparent;
  color: var(--text-secondary);
  transition: all var(--transition);
  white-space: nowrap;
}

.refresh-btn:hover { color: var(--text-primary); background: var(--bg-hover); }

.refresh-btn--active {
  background: var(--accent);
  color: #fff;
}

/* ── Manual refresh icon ────────────────────────────────────────────────────── */
.icon-action {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
  background: transparent;
  color: var(--text-secondary);
  cursor: pointer;
  transition: all var(--transition);
}
.icon-action:hover { color: var(--text-primary); border-color: var(--text-muted); }
.icon-action--spinning svg {
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}

/* ── Edit link ──────────────────────────────────────────────────────────────── */
.edit-link {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.78rem;
  color: var(--text-muted);
  text-decoration: none;
  padding: 4px 10px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  transition: all var(--transition);
}
.edit-link:hover {
  color: var(--accent);
  border-color: var(--accent);
}

/* ── Body / scroll area ─────────────────────────────────────────────────────── */
.present-body {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
}

/* ── Charts grid (same 4-col structure) ─────────────────────────────────────── */
.charts-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
}

.chart-cell {
  min-width: 0;
}

/* ── States ─────────────────────────────────────────────────────────────────── */
.loading-center {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 200px;
}

.error-msg {
  text-align: center;
  color: var(--error);
  padding: 40px;
  font-size: 0.85rem;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  height: 300px;
  color: var(--text-muted);
  font-size: 0.85rem;
}

.empty-state svg {
  opacity: 0.3;
}

.chart-missing {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  color: var(--text-muted);
  font-size: 0.82rem;
}

/* ── Spinner ─────────────────────────────────────────────────────────────────── */
.spinner {
  width: 28px;
  height: 28px;
  border: 3px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
</style>
