<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { v2, ago } from '../api/client'
import type { AgentView, ThreadRow } from '../api/client'
import Badge from '../components/Badge.vue'
import IdLink from '../components/IdLink.vue'
import Hint from '../components/Hint.vue'

const router = useRouter()
const route = useRoute()
const rows = ref<ThreadRow[]>([])
const agents = ref<AgentView[]>([])
const agent = ref(String(route.query.agent || ''))
const labels = ref(String(route.query.labels || ''))
const top = ref(route.query.all === undefined)
const error = ref('')

async function load() {
  error.value = ''
  const metadata: Record<string, string> = {}
  for (const pair of labels.value.split(',').map(s => s.trim()).filter(Boolean)) {
    const [k, ...v] = pair.split('=')
    metadata[k.trim()] = v.join('=').trim()
  }
  try { rows.value = await v2.threads({ agent: agent.value || undefined, top: top.value, limit: 200, metadata }) }
  catch (e: any) { error.value = e.message }
}
onMounted(async () => { await load(); agents.value = await v2.agents().catch(() => []) })
</script>
<template>
  <div>
    <div class="page-header"><h1>Threads</h1>
      <p class="page-header__subtitle">Conversations: a thread has one agent and at most one active run.
        <Hint text="Everything is a thread with one agent. Forks and threads of other agents are threads too - created by runs." topic="threads" /></p></div>
    <form class="toolbar" @submit.prevent="load">
      <select v-model="agent" class="input" style="width:180px" @change="load">
        <option value="">All agents</option><option v-for="a in agents" :key="a.name" :value="a.name">{{ a.name }}</option>
      </select>
      <input v-model="labels" class="input" style="width:260px" placeholder="labels: org=acme, board=b1" />
      <label class="small"><input v-model="top" type="checkbox" @change="load" /> conversations only (not forks and calls)</label>
      <button class="btn btn--sm">Filter</button>
    </form>
    <div v-if="error" class="error-text">{{ error }}</div>
    <div class="table-wrap card" style="padding:0">
      <table>
        <thead><tr><th>Thread</th><th>Agent</th><th>Now</th><th>Labels</th><th>Created</th></tr></thead>
        <tbody>
          <tr v-for="t in rows" :key="t.id" class="clickable" @click="router.push(`/threads/${t.id}`)">
            <td><IdLink :id="t.id" kind="thread" /> <span v-if="t.forked_from" class="pill">fork</span><span v-else-if="t.created_by" class="pill">call</span></td>
            <td>{{ t.agent }}</td>
            <td><Badge v-if="t.active_run" :status="t.active_run.status" /><span v-else class="muted small">{{ t.closed ? 'closed' : 'idle' }}</span></td>
            <td><span v-for="(v, k) in t.metadata" :key="k" class="pill" style="margin-right:4px">{{ k }}={{ v }}</span></td>
            <td class="muted small">{{ ago(t.created_at) }}</td>
          </tr>
          <tr v-if="!rows.length"><td colspan="5" class="muted">No threads. Start one from an agent's <b>Try it</b>.</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
<style scoped>.clickable { cursor: pointer; }</style>
