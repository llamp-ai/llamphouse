<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { v2, ago, durationMs, shortId } from '../api/client'
import type { AgentView, SessionList, SessionRow } from '../api/client'
import IdLink from '../components/IdLink.vue'
import Hint from '../components/Hint.vue'

const router = useRouter()
const route = useRoute()
const data = ref<SessionList | null>(null)
const agents = ref<AgentView[]>([])
const agent = ref(String(route.query.agent || ''))
const state = ref(String(route.query.state || ''))  // '' | 'compacting' | 'idle'
const error = ref('')
let timer: number | undefined

async function load() {
  try {
    data.value = await v2.sessions({ agent: agent.value || undefined, limit: 200,
      revising: state.value === 'compacting' ? true : state.value === 'idle' ? false : undefined })
    error.value = ''
  } catch (e: any) { error.value = e.message }
  clearTimeout(timer)
  timer = window.setTimeout(load, data.value?.revising ? 1500 : 5000)  // faster while something is compacted
}
onMounted(async () => { await load(); agents.value = await v2.agents().catch(() => []) })
onUnmounted(() => clearTimeout(timer))

const rows = computed<SessionRow[]>(() => data.value?.sessions ?? [])
function since(s: SessionRow) { return s.working?.since ? durationMs(s.working.since, new Date().toISOString()) : '' }
function last(s: SessionRow) {
  const c = s.last
  if (!c) return '—'
  if (c.outcome === 'completed') return `${c.before} → ${c.after}`
  return c.outcome
}
</script>
<template>
  <div>
    <div class="page-header">
      <h1>Sessions</h1>
      <p class="page-header__subtitle">Each agent's working memory on a thread, and its compactions. Those being compacted come first.
        <span class="live-dot" /> <Hint text="A session provider compacts a session in the background (a revision): the run keeps going on the session as it was, and what is added meanwhile survives. A revision is a job on the run queue: a worker slot does it, like a run." topic="sessions" /></p>
    </div>

    <div v-if="data" class="grid-3 mb-4">
      <div class="callout" :class="{ 'callout--info': data.revising }">
        <b>{{ data.revising }}</b> being compacted now
      </div>
      <div class="callout" :class="{ 'callout--attention': data.waiting_for_worker }">
        <b>{{ data.waiting_for_worker }}</b> waiting for a worker
        <Hint v-if="data.waiting_for_worker" text="Asked for, not picked up yet: every slot that takes revisions is busy. It is picked up when one frees, or by another worker process." />
      </div>
      <div class="callout"><div>
        <template v-if="data.revision_workers_here"><b>{{ data.revision_workers_here }}</b> revision slot(s), <b>{{ data.workers_here }}</b> run slot(s)</template>
        <template v-else><b>{{ data.workers_here }}</b> slot(s) for runs and revisions</template> in this process
        <div class="callout__hint">{{ data.queued?.revisions ?? 0 }} revision(s) and {{ data.queued?.runs ?? 0 }} run(s) queued in the deployment
          <Hint text="Each lane's queue is shared by every worker process. Scale workers on llamphouse_jobs_queued (GET /metrics); give revisions slots of their own with revision_workers." /></div>
      </div></div>
    </div>

    <div class="toolbar">
      <select v-model="agent" class="input" style="width:180px" @change="load"><option value="">All agents</option>
        <option v-for="a in agents" :key="a.name" :value="a.name">{{ a.name }}</option></select>
      <select v-model="state" class="input" style="width:200px" @change="load">
        <option value="">Any state</option><option value="compacting">Being compacted</option><option value="idle">Idle</option></select>
    </div>
    <div v-if="error" class="error-text">{{ error }}</div>

    <div class="table-wrap card" style="padding:0">
      <table>
        <thead><tr><th>Session</th><th>Agent</th><th>Provider</th><th>Thread</th><th>State</th>
          <th>Seen / kept <Hint text="Seen: what the agent gets from ctx.session.messages() (the provider's view). Kept: what the session holds." /></th>
          <th>Compactions</th><th>Last</th><th>Changed</th></tr></thead>
        <tbody>
          <tr v-for="s in rows" :key="s.id" style="cursor:pointer" @click="router.push(`/sessions/${s.id}`)">
            <td class="mono small">{{ shortId(s.id) }}</td>
            <td>{{ s.agent }}</td>
            <td class="small">{{ s.provider }}</td>
            <td @click.stop><IdLink :id="s.thread" kind="thread" /></td>
            <td>
              <span v-if="s.error" class="error-text small">{{ s.error }}</span>
              <span v-else-if="s.working?.worker" class="pill pill--info"><span class="live-dot" /> compacting · {{ since(s) }}</span>
              <span v-else-if="s.working" class="pill pill--warn">waiting for a worker · {{ since(s) }}</span>
              <span v-else class="pill">idle</span>
            </td>
            <td class="mono small">{{ s.seen }} / {{ s.kept }}</td>
            <td class="mono small">{{ s.revisions }}</td>
            <td class="mono small" :class="{ 'error-text': s.last?.outcome === 'failed' }">{{ last(s) }}</td>
            <td class="muted small">{{ ago(s.updated_at) }}</td>
          </tr>
          <tr v-if="!rows.length"><td colspan="9" class="muted">No sessions yet: a session is made when an agent's run first reads it.</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
