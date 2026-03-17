<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { compass, formatTs, shortId } from '../api/client'
import type { Thread } from '../api/client'
import DataTable from '../components/DataTable.vue'

const router = useRouter()
const threads = ref<Thread[]>([])
const loading = ref(true)
const error = ref('')
const search = ref('')

onMounted(async () => {
  try {
    threads.value = await compass.listThreads(100)
  } catch (e: any) {
    error.value = e.message
  } finally {
    loading.value = false
  }
})

const filtered = computed(() => {
  if (!search.value) return threads.value
  const q = search.value.toLowerCase()
  return threads.value.filter((t) =>
    t.id.toLowerCase().includes(q) ||
    (t.agent_name ?? '').toLowerCase().includes(q) ||
    (t.agent_id ?? '').toLowerCase().includes(q)
  )
})

const columns = [
  { key: 'id', label: 'Thread ID', mono: true },
  { key: 'agent_name', label: 'Agent' },
  { key: 'created_at', label: 'Created' },
  { key: 'meta', label: 'Metadata' },
]
</script>

<template>
  <div>
    <div class="page-header">
      <div>
        <h1>Threads</h1>
        <div class="page-header__subtitle">{{ threads.length }} total</div>
      </div>
      <input class="input" style="max-width: 280px" v-model="search" placeholder="Search by ID…" />
    </div>

    <div v-if="loading" class="loading-center"><div class="spinner"></div></div>
    <div v-else-if="error" class="card" style="color: var(--error)">{{ error }}</div>

    <div v-else class="card">
      <DataTable
        :columns="columns"
        :rows="filtered"
        clickable
        @row-click="(r) => router.push(`/threads/${r.id}`)"
      >
        <template #id="{ value }">{{ shortId(value) }}</template>
        <template #agent_name="{ value }">{{ value ?? '—' }}</template>
        <template #created_at="{ value }">{{ formatTs(value) }}</template>
        <template #meta="{ row }">
          <span class="mono" style="color: var(--text-muted); font-size: 0.75rem;">
            {{ Object.keys(row.metadata || {}).length ? JSON.stringify(row.metadata).slice(0, 60) : '—' }}
          </span>
        </template>
      </DataTable>
    </div>
  </div>
</template>
