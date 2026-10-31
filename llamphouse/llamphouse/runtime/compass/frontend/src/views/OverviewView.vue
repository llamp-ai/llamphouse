<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { v2, ago } from '../api/client'
import type { Info, Overview } from '../api/client'
import StatCard from '../components/StatCard.vue'
import MiniBars from '../components/MiniBars.vue'
import Badge from '../components/Badge.vue'
import IdLink from '../components/IdLink.vue'
import Hint from '../components/Hint.vue'

const data = ref<Overview | null>(null)
const info = ref<Info | null>(null)
const hours = ref(24)
const error = ref('')
let timer: number | undefined

async function load() {
  try {
    data.value = await v2.overview(hours.value)
    info.value ??= await v2.info()
    error.value = ''
  } catch (e: any) { error.value = e.message }
}
onMounted(() => { load(); timer = window.setInterval(load, 5000) })
onUnmounted(() => clearInterval(timer))
const n = (s: string) => data.value?.by_status?.[s] ?? 0
</script>

<template>
  <div>
    <div class="page-header flex items-center justify-between">
      <div>
        <h1>Overview</h1>
        <p class="page-header__subtitle">What your agents did in the last
          <select v-model.number="hours" class="input" style="width:auto;display:inline-block;padding:2px 6px" @change="load">
            <option :value="1">hour</option><option :value="24">24 hours</option><option :value="72">3 days</option>
          </select>
          <span class="live-dot" style="margin-left:8px" title="Refreshes every 5 seconds" /></p>
      </div>
    </div>
    <div v-if="error" class="callout callout--error"><span class="callout__dot" />{{ error }}</div>
    <template v-if="data">
      <div class="grid-4 mb-4">
        <StatCard icon="🧵" label="Conversations" :value="data.threads" sub="threads people and apps talk to" />
        <StatCard icon="▶" label="Runs" :value="data.runs" :sub="`${data.active} active now`" />
        <StatCard icon="✋" label="Waiting for someone" :value="n('input_required') + n('auth_required')" variant="warning" sub="asked a question" />
        <StatCard icon="✕" label="Failed" :value="n('failed') + n('rejected')" :variant="n('failed') ? 'error' : 'default'" />
      </div>
      <div class="card mb-4">
        <div class="section__title">Runs per hour, by status</div>
        <MiniBars :points="data.timeline" :hours="hours" />
        <div class="row-actions small mt-2">
          <span v-for="(c, s) in data.by_status" :key="s"><Badge :status="String(s)" /> {{ c }}</span>
        </div>
      </div>
      <div class="grid-2 mb-4">
        <div class="card">
          <div class="section__title">Needs someone <Hint text="Runs that asked a question (ctx.ask) or for credentials: they hold no worker while they wait." topic="asking" /></div>
          <div v-if="!data.attention.length" class="muted small">Nobody is being waited for.</div>
          <div v-for="r in data.attention" :key="r.id" class="row-actions" style="padding:4px 0">
            <IdLink :id="r.id" kind="run" /> <span class="muted small">{{ r.agent }}</span> <Badge :status="r.status" />
            <span class="ml-auto muted small">{{ ago(r.created_at) }}</span>
          </div>
        </div>
        <div class="card">
          <div class="section__title">Failed recently</div>
          <div v-if="!data.failures.length" class="muted small">Nothing failed.</div>
          <div v-for="r in data.failures" :key="r.id" class="row-actions" style="padding:4px 0">
            <IdLink :id="r.id" kind="run" /> <span class="muted small">{{ r.agent }}</span>
            <span class="ml-auto muted small">{{ ago(r.ended_at || r.created_at) }}</span>
          </div>
        </div>
      </div>
      <div class="card mb-4">
        <div class="section__title">Agents</div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Agent</th><th>Runs</th><th>Statuses</th><th>p50</th><th>p95</th></tr></thead>
            <tbody>
              <tr v-for="(a, name) in data.by_agent" :key="name">
                <td><router-link :to="`/agents/${name}`">{{ name }}</router-link></td>
                <td>{{ a.runs }}</td>
                <td><span v-for="(c, s) in a.statuses" :key="s" style="margin-right:6px"><Badge :status="String(s)" /> {{ c }}</span></td>
                <td class="mono">{{ a.p50_ms != null ? (a.p50_ms / 1000).toFixed(2) + 's' : '—' }}</td>
                <td class="mono">{{ a.p95_ms != null ? (a.p95_ms / 1000).toFixed(2) + 's' : '—' }}</td>
              </tr>
              <tr v-if="!Object.keys(data.by_agent).length"><td colspan="5" class="muted">No runs yet: try an agent from <router-link to="/agents">Agents</router-link>.</td></tr>
            </tbody>
          </table>
        </div>
      </div>
      <div v-if="info" class="card">
        <div class="section__title">This deployment</div>
        <dl class="kv">
          <dt>Version</dt><dd>{{ info.version }}</dd>
          <dt>This process</dt><dd><span class="mono">{{ info.engine }}</span> · {{ info.workers }} worker(s) · {{ info.role }}</dd>
          <template v-for="(v, k) in info.components" :key="k"><dt>{{ String(k).replace('_', ' ') }}</dt><dd>{{ v || '—' }}</dd></template>
          <dt>Auth</dt><dd>{{ info.auth.join(', ') || 'none: everyone is anonymous' }} · access: {{ info.access }}</dd>
          <dt>Adapters</dt><dd>{{ info.adapters.map(a => `${a.kind} ${a.prefix}`).join(' · ') || 'the native API only (/v2)' }}</dd>
          <dt>You</dt><dd>{{ info.you.subject }}<span v-if="info.you.tenant"> · tenant {{ info.you.tenant }}</span><span v-if="info.you.admin" class="pill" style="margin-left:6px">admin</span></dd>
        </dl>
      </div>
    </template>
    <div v-else-if="!error" class="loading-center"><div class="spinner" /></div>
  </div>
</template>
