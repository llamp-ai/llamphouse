<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { v2, ago, durationMs } from '../api/client'
import type { AgentView, RunRow } from '../api/client'
import Badge from '../components/Badge.vue'
import IdLink from '../components/IdLink.vue'

const router = useRouter()
const route = useRoute()
const rows = ref<RunRow[]>([])
const agents = ref<AgentView[]>([])
const agent = ref(String(route.query.agent || ''))
const status = ref(String(route.query.status || ''))
const picked = ref<string[]>([])
const error = ref('')
const STATUSES = ['queued', 'running', 'waiting', 'input_required', 'auth_required', 'completed', 'failed', 'cancelled', 'rejected']
let timer: number | undefined

async function load() {
  try { rows.value = await v2.runs({ agent: agent.value || undefined, status: status.value || undefined, limit: 200 }); error.value = '' }
  catch (e: any) { error.value = e.message }
}
onMounted(async () => { await load(); agents.value = await v2.agents().catch(() => []); timer = window.setInterval(load, 4000) })
onUnmounted(() => clearInterval(timer))
function toggle(id: string) { picked.value = picked.value.includes(id) ? picked.value.filter(x => x !== id) : [...picked.value, id] }
</script>
<template>
  <div>
    <div class="page-header"><h1>Runs</h1><p class="page-header__subtitle">Every run, newest first. Pick two or more to compare them. <span class="live-dot" /></p></div>
    <div class="toolbar">
      <select v-model="agent" class="input" style="width:180px" @change="load"><option value="">All agents</option>
        <option v-for="a in agents" :key="a.name" :value="a.name">{{ a.name }}</option></select>
      <select v-model="status" class="input" style="width:180px" @change="load"><option value="">Any status</option>
        <option v-for="s in STATUSES" :key="s" :value="s">{{ s.replace('_', ' ') }}</option></select>
      <span class="ml-auto" />
      <button class="btn btn--sm" :disabled="picked.length < 2" @click="router.push(`/compare?runs=${picked.join(',')}`)">Compare {{ picked.length || '' }}</button>
    </div>
    <div v-if="error" class="error-text">{{ error }}</div>
    <div class="table-wrap card" style="padding:0">
      <table>
        <thead><tr><th /><th>Run</th><th>Agent</th><th>Status</th><th>Thread</th><th>Started</th><th>Took</th></tr></thead>
        <tbody>
          <tr v-for="r in rows" :key="r.id" style="cursor:pointer" @click="router.push(`/runs/${r.id}`)">
            <td @click.stop><input type="checkbox" :checked="picked.includes(r.id)" @change="toggle(r.id)" /></td>
            <td><IdLink :id="r.id" kind="run" /></td><td>{{ r.agent }}</td><td><Badge :status="r.status" /></td>
            <td><IdLink :id="r.thread_id" kind="thread" /></td><td class="muted small">{{ ago(r.created_at) }}</td>
            <td class="mono small">{{ durationMs(r.created_at, r.ended_at) }}</td>
          </tr>
          <tr v-if="!rows.length"><td colspan="7" class="muted">No runs here.</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
