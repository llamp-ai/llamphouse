<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { compass, formatTs } from '../api/client'
import type { Dashboard } from '../api/client'

const router = useRouter()
const dashboards = ref<Dashboard[]>([])
const loading = ref(true)
const error = ref('')
const creating = ref(false)
const newTitle = ref('')
const showCreate = ref(false)

onMounted(async () => {
  try {
    dashboards.value = await compass.listDashboards()
  } catch (e: any) {
    error.value = e.message
  } finally {
    loading.value = false
  }
})

async function createDashboard() {
  if (!newTitle.value.trim()) return
  creating.value = true
  try {
    const d = await compass.createDashboard(newTitle.value.trim())
    router.push(`/dashboards/${d.id}`)
  } catch (e: any) {
    error.value = e.message
    creating.value = false
  }
}

function startCreate() {
  newTitle.value = ''
  showCreate.value = true
}
</script>

<template>
  <div>
    <div class="page-header">
      <div>
        <h1>Dashboards</h1>
        <div class="page-header__subtitle">Custom charts powered by SQL queries</div>
      </div>
      <button class="btn btn--primary" @click="startCreate">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        New Dashboard
      </button>
    </div>

    <!-- Create dialog -->
    <div v-if="showCreate" class="create-overlay" @click.self="showCreate = false">
      <div class="create-modal card">
        <div class="create-modal__title">New Dashboard</div>
        <input
          v-model="newTitle"
          class="input"
          placeholder="Dashboard title"
          autofocus
          @keydown.enter="createDashboard"
          @keydown.esc="showCreate = false"
        />
        <div class="create-modal__actions">
          <button class="btn btn--ghost" @click="showCreate = false">Cancel</button>
          <button class="btn btn--primary" :disabled="!newTitle.trim() || creating" @click="createDashboard">
            {{ creating ? 'Creating…' : 'Create' }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="loading" class="loading-center"><div class="spinner"></div></div>

    <div v-else-if="error" class="card" style="color: var(--error)">{{ error }}</div>

    <template v-else>
      <!-- Empty state -->
      <div v-if="!dashboards.length" class="empty-state card">
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="color: var(--text-muted)"><rect x="2" y="3" width="9" height="9" rx="1"/><rect x="13" y="3" width="9" height="9" rx="1"/><rect x="2" y="14" width="9" height="7" rx="1"/><rect x="13" y="14" width="9" height="7" rx="1"/></svg>
        <div class="empty-state__text">No dashboards yet</div>
        <div class="empty-state__sub">Create a dashboard to visualise your data with SQL queries</div>
        <button class="btn btn--primary" @click="startCreate">Create your first dashboard</button>
      </div>

      <!-- Dashboard grid -->
      <div v-else class="dashboard-grid">
        <div
          v-for="d in dashboards"
          :key="d.id"
          class="dashboard-card card"
          @click="router.push(`/dashboards/${d.id}`)"
        >
          <div class="dashboard-card__icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="9" height="9" rx="1"/><rect x="13" y="3" width="9" height="9" rx="1"/><rect x="2" y="14" width="9" height="7" rx="1"/><rect x="13" y="14" width="9" height="7" rx="1"/></svg>
          </div>
          <div class="dashboard-card__body">
            <div class="dashboard-card__title">{{ d.title }}</div>
            <div v-if="d.description" class="dashboard-card__desc">{{ d.description }}</div>
            <div class="dashboard-card__meta">
              <span>{{ d.charts.length }} chart{{ d.charts.length !== 1 ? 's' : '' }}</span>
              <span>Updated {{ formatTs(d.updated_at) }}</span>
            </div>
          </div>
          <svg class="dashboard-card__arrow" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 16px;
  margin-top: 24px;
}

.dashboard-card {
  display: flex;
  align-items: flex-start;
  gap: 14px;
  cursor: pointer;
  transition: all var(--transition);
  padding: 20px;
}

.dashboard-card:hover {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent), 0 4px 16px rgba(0, 0, 0, 0.08);
}

.dashboard-card__icon {
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  border-radius: var(--radius-md);
  background: var(--accent-dim);
  color: var(--accent);
  display: flex;
  align-items: center;
  justify-content: center;
}

.dashboard-card__body {
  flex: 1;
  min-width: 0;
}

.dashboard-card__title {
  font-weight: 600;
  font-size: 0.9rem;
  color: var(--text-primary);
  margin-bottom: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.dashboard-card__desc {
  font-size: 0.78rem;
  color: var(--text-muted);
  margin-bottom: 8px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.dashboard-card__meta {
  display: flex;
  gap: 12px;
  font-size: 0.72rem;
  color: var(--text-muted);
}

.dashboard-card__arrow {
  flex-shrink: 0;
  color: var(--text-muted);
  margin-top: 2px;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 60px 40px;
  text-align: center;
  margin-top: 24px;
}

.empty-state__text {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.empty-state__sub {
  font-size: 0.83rem;
  color: var(--text-muted);
  max-width: 360px;
  margin-bottom: 8px;
}

/* Create modal */
.create-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.create-modal {
  width: 380px;
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.create-modal__title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.create-modal__actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}

.input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--bg-secondary);
  color: var(--text-primary);
  font-size: 0.875rem;
  outline: none;
  box-sizing: border-box;
  transition: border-color var(--transition);
}

.input:focus {
  border-color: var(--accent);
}

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

.btn--primary {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}

.btn--primary:hover:not(:disabled) {
  opacity: 0.88;
}

.btn--primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn--ghost {
  background: transparent;
  color: var(--text-secondary);
  border-color: var(--border);
}

.btn--ghost:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}

.page-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 4px;
}
</style>
