<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { compass, formatTs, shortId, statusBadge, durationMs } from '../api/client'
import type { Message, Run } from '../api/client'
import MessageBubble from '../components/MessageBubble.vue'
import DataTable from '../components/DataTable.vue'

const route = useRoute()
const router = useRouter()
const threadId = route.params.threadId as string

const messages = ref<Message[]>([])
const runs = ref<Run[]>([])
const loading = ref(true)
const error = ref('')
const tab = ref<'messages' | 'runs'>('messages')

onMounted(async () => {
  try {
    const [m, r] = await Promise.all([
      compass.listMessages(threadId),
      compass.listRuns(threadId),
    ])
    messages.value = m
    runs.value = r
  } catch (e: any) {
    error.value = e.message
  } finally {
    loading.value = false
  }
})

const runCols = [
  { key: 'id', label: 'Run ID', mono: true },
  { key: 'assistant_id', label: 'Agent', mono: true },
  { key: 'status', label: 'Status', width: '120px' },
  { key: 'model', label: 'Model' },
  { key: 'duration', label: 'Duration', width: '100px' },
  { key: 'tokens', label: 'Tokens', width: '80px' },
  { key: 'created_at', label: 'Created' },
]
</script>

<template>
  <div>
    <div class="breadcrumbs">
      <router-link to="/threads">Threads</router-link>
      <span>›</span>
      <span>{{ shortId(threadId) }}</span>
    </div>

    <div class="page-header">
      <div>
        <h1>Thread Detail</h1>
        <div class="page-header__subtitle mono">{{ threadId }}</div>
      </div>
    </div>

    <div v-if="loading" class="loading-center"><div class="spinner"></div></div>
    <div v-else-if="error" class="card" style="color: var(--error)">{{ error }}</div>

    <template v-else>
      <div class="tabs">
        <div class="tab" :class="{ 'tab--active': tab === 'messages' }" @click="tab = 'messages'">
          Messages ({{ messages.length }})
        </div>
        <div class="tab" :class="{ 'tab--active': tab === 'runs' }" @click="tab = 'runs'">
          Runs ({{ runs.length }})
        </div>
      </div>

      <!-- Messages tab -->
      <div v-if="tab === 'messages'" class="message-list">
        <div v-if="messages.length === 0" class="empty-state">
          <div class="empty-state__icon"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg></div>
          <div class="empty-state__title">No messages</div>
        </div>
        <MessageBubble v-for="msg in messages" :key="msg.id" :message="msg" />
      </div>

      <!-- Runs tab -->
      <div v-if="tab === 'runs'" class="card">
        <DataTable
          :columns="runCols"
          :rows="runs"
          clickable
          @row-click="(r) => router.push(`/threads/${threadId}/runs/${r.id}`)"
        >
          <template #id="{ value }">{{ shortId(value) }}</template>
          <template #assistant_id="{ value }">{{ shortId(value) }}</template>
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
    </template>
  </div>
</template>

<style scoped>
.message-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
</style>
