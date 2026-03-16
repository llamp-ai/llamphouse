<script setup lang="ts">
import { ref, computed } from 'vue'
import { compass, shortId, statusBadge, formatTs, durationMs } from '../api/client'

const runIdsInput = ref('')
const loading = ref(false)
const error = ref('')
const result = ref<any>(null)

async function doCompare() {
  const ids = runIdsInput.value
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean)

  if (ids.length < 2) {
    error.value = 'Enter at least 2 run IDs separated by commas.'
    return
  }

  loading.value = true
  error.value = ''
  try {
    result.value = await compass.compareRuns(ids)
  } catch (e: any) {
    error.value = e.message
    result.value = null
  } finally {
    loading.value = false
  }
}

const runs = computed(() => result.value?.runs || [])

function tokenTotal(run: any): number {
  return run.usage?.total_tokens ?? 0
}
</script>

<template>
  <div>
    <div class="page-header">
      <div>
        <h1>Compare Runs</h1>
        <div class="page-header__subtitle">Side-by-side run comparison</div>
      </div>
    </div>

    <!-- Input -->
    <div class="card mb-4">
      <div class="flex gap-3 items-center">
        <input
          class="input"
          v-model="runIdsInput"
          placeholder="Paste run IDs separated by commas…"
          @keydown.enter="doCompare"
        />
        <button class="btn btn--primary" :disabled="loading" @click="doCompare">
          {{ loading ? 'Loading…' : 'Compare' }}
        </button>
      </div>
      <div v-if="error" style="color: var(--error); font-size: 0.8rem; margin-top: 8px;">
        {{ error }}
      </div>
    </div>

    <!-- Results -->
    <template v-if="runs.length > 0">
      <div class="compare-grid" :style="{ gridTemplateColumns: `repeat(${runs.length}, 1fr)` }">
        <div v-for="run in runs" :key="run.id" class="card compare-col">
          <!-- Header -->
          <div class="compare-col__header">
            <span class="badge" :class="statusBadge(run.status)">{{ run.status }}</span>
            <span class="mono" style="font-size: 0.75rem;">{{ shortId(run.id) }}</span>
          </div>

          <!-- Meta -->
          <div class="compare-section">
            <div class="compare-meta">
              <div><span class="label">Model</span> {{ run.model }}</div>
              <div><span class="label">Created</span> {{ formatTs(run.created_at) }}</div>
              <div><span class="label">Duration</span> {{ durationMs(run.started_at, run.completed_at || run.failed_at) }}</div>
              <div><span class="label">Tokens</span> {{ tokenTotal(run) }}</div>
            </div>
          </div>

          <!-- Instructions diff -->
          <div class="compare-section">
            <div class="section__title">Instructions</div>
            <div class="json-view" style="max-height: 160px; overflow-y: auto;">
              {{ run.instructions || '(none)' }}
            </div>
          </div>

          <!-- Tools -->
          <div class="compare-section">
            <div class="section__title">Tools ({{ run.tools?.length ?? 0 }})</div>
            <div class="json-view" style="max-height: 120px; overflow-y: auto;">
              {{ JSON.stringify(run.tools || [], null, 2) }}
            </div>
          </div>

          <!-- Error -->
          <div v-if="run.last_error" class="compare-section">
            <div class="section__title" style="color: var(--error)">Error</div>
            <div class="json-view" style="color: var(--error)">{{ JSON.stringify(run.last_error, null, 2) }}</div>
          </div>
        </div>
      </div>
    </template>

    <!-- Empty state -->
    <div v-else-if="!loading && !error" class="empty-state">
      <div class="empty-state__icon"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg></div>
      <div class="empty-state__title">Enter run IDs to compare</div>
      <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 4px;">
        Paste two or more run IDs above to see a side-by-side comparison of model, tokens, config, and more.
      </p>
    </div>
  </div>
</template>

<style scoped>
.compare-grid {
  display: grid;
  gap: 16px;
}

.compare-col {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.compare-col__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}

.compare-section {
  margin-bottom: 14px;
}

.compare-meta > div {
  font-size: 0.825rem;
  padding: 4px 0;
  display: flex;
  justify-content: space-between;
}

.label {
  color: var(--text-muted);
  font-size: 0.75rem;
}
</style>
