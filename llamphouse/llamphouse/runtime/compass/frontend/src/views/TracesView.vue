<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { v2, formatTs } from '../api/client'
import type { AgentView, TraceRow } from '../api/client'
import IdLink from '../components/IdLink.vue'

const router = useRouter()
const rows = ref<TraceRow[]>([])
const tracing = ref(true)
const agents = ref<AgentView[]>([])
const agent = ref('')
const error = ref('')
async function load() {
  try { const r = await v2.traces(agent.value || undefined); rows.value = r.traces; tracing.value = r.tracing; error.value = r.error || '' }
  catch (e: any) { error.value = e.message }
}
onMounted(async () => { await load(); agents.value = await v2.agents().catch(() => []) })
</script>
<template>
  <div>
    <div class="page-header"><h1>Traces</h1><p class="page-header__subtitle">A trace per execution of a run (llamphouse.run), with the spans your agent code makes inside it.</p></div>
    <div v-if="!tracing" class="callout"><span class="callout__dot" /><div>No tracing store. Add one to the runtime file:
      <span class="mono">tracing: {in_memory: {}}</span> (or postgres, clickhouse), or set TRACING_STORE.</div></div>
    <div class="toolbar">
      <select v-model="agent" class="input" style="width:180px" @change="load"><option value="">All agents</option>
        <option v-for="a in agents" :key="a.name" :value="a.name">{{ a.name }}</option></select>
    </div>
    <div v-if="error" class="error-text">{{ error }}</div>
    <div class="table-wrap card" style="padding:0"><table>
      <thead><tr><th>Span</th><th>Run</th><th>Agent</th><th>Spans</th><th>Duration</th><th>Status</th><th>When</th></tr></thead>
      <tbody>
        <tr v-for="t in rows" :key="t.TraceId" style="cursor:pointer" @click="router.push(`/traces/${t.run_id}`)">
          <td class="mono small">{{ t.SpanName }}</td><td><IdLink :id="t.run_id" kind="run" /></td><td>{{ t.agent }}</td>
          <td>{{ t.span_count }}</td><td class="mono small">{{ t.duration_ms.toFixed(1) }}ms</td>
          <td><span class="badge" :class="t.StatusCode.includes('ERROR') ? 'badge--error' : t.StatusCode.includes('OK') ? 'badge--success' : 'badge--neutral'">{{ t.StatusCode.replace('STATUS_CODE_', '') }}</span></td>
          <td class="muted small">{{ formatTs(t.Timestamp) }}</td>
        </tr>
        <tr v-if="!rows.length && tracing"><td colspan="7" class="muted">No traces yet.</td></tr>
      </tbody></table></div>
  </div>
</template>
