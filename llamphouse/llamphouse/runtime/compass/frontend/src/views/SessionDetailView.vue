<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { v2, ago, durationMs, formatTs } from '../api/client'
import type { SessionDetail } from '../api/client'
import IdLink from '../components/IdLink.vue'
import Hint from '../components/Hint.vue'
import JsonView from '../components/JsonView.vue'
import MessageView from '../components/MessageView.vue'

const route = useRoute()
const id = String(route.params.sessionId)
const s = ref<SessionDetail | null>(null)
const error = ref('')
const tab = ref<'seen' | 'kept' | 'log'>('seen')
let timer: number | undefined

async function load() {
  try { s.value = await v2.sessionDetail(id); error.value = '' } catch (e: any) { error.value = e.message }
  clearTimeout(timer)
  timer = window.setTimeout(load, s.value?.revising ? 1500 : 6000)
}
onMounted(load)
onUnmounted(() => clearTimeout(timer))

const biggest = computed(() => Math.max(1, ...(s.value?.compactions ?? []).map(c => c.before ?? 0)))
const shown = computed(() => (tab.value === 'kept' ? s.value?.messages : s.value?.view) ?? [])
function took(start: string | null, end: string | null) { return end ? durationMs(start, end) : durationMs(start, new Date().toISOString()) }
</script>
<template>
  <div>
    <div class="breadcrumbs"><router-link to="/sessions">Sessions</router-link><span>/</span><span class="mono">{{ id }}</span></div>
    <div v-if="error" class="error-text">{{ error }}</div>
    <template v-if="s">
      <div class="page-header">
        <h1>{{ s.agent }} <span class="pill">session</span></h1>
        <p class="page-header__subtitle">
          on thread <IdLink :id="s.thread" kind="thread" /> · provider <b>{{ s.provider }}</b> · version {{ s.version }}
          <template v-if="s.runs.length"> · used by {{ s.runs.length }} run(s)</template>
          <template v-if="s.forked_from"> · seeded from <span class="mono">{{ s.forked_from }}</span></template>
        </p>
      </div>

      <div v-if="s.working" class="callout mb-4" :class="s.working.worker ? 'callout--info' : 'callout--attention'">
        <div><template v-if="s.working.worker">
          <span class="live-dot" /> <b>Being compacted</b> for {{ took(s.working.since, null) }} by
          <span class="mono">{{ s.working.worker }}</span>{{ s.working.here ? ' (this process)' : '' }}.
          If it isn't done by {{ formatTs(s.working.due) }}, another worker takes it over.
        </template>
        <template v-else>
          <b>Waiting for a worker</b> for {{ took(s.working.since, null) }}: the compaction was asked for, and every worker slot is busy.
          <Hint text="Revisions are jobs on the run queue, done by the same worker slots as runs. More worker processes (or --workers) pick them up sooner." />
        </template>
        <div class="callout__hint">asked with <span class="mono">{{ JSON.stringify(s.working.options) }}</span></div></div>
      </div>
      <div v-else class="callout mb-4">Idle: no compaction in flight.</div>

      <div class="grid-4 mb-4">
        <div class="card"><div class="section__title">Seen</div><b>{{ s.seen }}</b> <span class="muted small">messages the agent gets</span></div>
        <div class="card"><div class="section__title">Kept</div><b>{{ s.kept }}</b> <span class="muted small">messages it holds</span></div>
        <div class="card"><div class="section__title">Compactions</div><b>{{ s.revisions }}</b> <span class="muted small">done</span></div>
        <div class="card"><div class="section__title">Last</div>
          <b v-if="s.last?.outcome === 'completed'">{{ s.last.before }} → {{ s.last.after }}</b>
          <b v-else>{{ s.last?.outcome ?? '—' }}</b> <span class="muted small">{{ ago(s.last?.ended_at) }}</span></div>
      </div>

      <div class="card mb-4">
        <div class="section__title">Compactions <Hint text="Each revision the provider was asked for: when, why (its options), how long it took, and what it did - the messages before and after. What came in meanwhile is kept after the revised part." topic="sessions" /></div>
        <table v-if="s.compactions.length">
          <thead><tr><th>#</th><th>Asked</th><th>Reason</th><th>Took</th><th>Outcome</th><th style="width:40%">Before → after</th></tr></thead>
          <tbody>
            <tr v-for="(c, i) in s.compactions" :key="i">
              <td class="mono small">{{ i + 1 }}</td>
              <td class="small">{{ formatTs(c.asked_at) }}<div class="muted">at position {{ c.at_position ?? '—' }}</div></td>
              <td class="mono small">{{ c.options?.reason ?? JSON.stringify(c.options) }}</td>
              <td class="mono small">{{ took(c.asked_at, c.ended_at) }}</td>
              <td><span class="pill" :class="{ 'pill--info': c.outcome === 'working', 'pill--warn': c.outcome === 'declined', 'error-text': c.outcome === 'failed' }">{{ c.outcome }}</span>
                <div v-if="c.error" class="error-text small">{{ c.error }}</div></td>
              <td>
                <template v-if="c.outcome === 'completed'">
                  <div class="bar"><div class="bar__before" :style="{ width: `${100 * (c.before ?? 0) / biggest}%` }">{{ c.before }}</div></div>
                  <div class="bar"><div class="bar__after" :style="{ width: `${Math.max(4, 100 * (c.after ?? 0) / biggest)}%` }">{{ c.after }}</div></div>
                </template>
                <span v-else class="muted small">—</span>
              </td>
            </tr>
          </tbody>
        </table>
        <div v-else class="muted small">Not compacted yet.</div>
      </div>

      <div class="tabs">
        <button class="tab" :class="{ 'tab--active': tab === 'seen' }" @click="tab = 'seen'">What it sees<span class="muted">{{ s.seen }}</span></button>
        <button class="tab" :class="{ 'tab--active': tab === 'kept' }" @click="tab = 'kept'">As kept<span class="muted">{{ s.kept }}</span></button>
        <button class="tab" :class="{ 'tab--active': tab === 'log' }" @click="tab = 'log'">Log<span class="muted">{{ s.log.length }}</span></button>
      </div>
      <div v-if="tab === 'log'" class="card"><JsonView :value="s.log" /></div>
      <div v-else class="card">
        <div v-if="s.view_error" class="error-text small mb-2">The provider's view failed: {{ s.view_error }}</div>
        <div v-for="m in shown" :key="m.id" class="bubble" :class="{ 'bubble--agent': m.own }" style="max-width:92%">
          <div class="bubble__who"><b>{{ m.sender.name || m.sender.id }}</b><span>{{ m.own ? 'own' : m.sender.kind }}</span></div>
          <MessageView :parts="m.parts" />
        </div>
        <div v-if="!shown.length" class="muted small">Empty.</div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.bar { background: var(--bg-secondary); border-radius: 4px; margin: 2px 0; }
.bar__before, .bar__after { font-size: 0.72rem; padding: 1px 6px; border-radius: 4px; color: #fff; white-space: nowrap; }
.bar__before { background: var(--text-muted, #9ca3af); }
.bar__after { background: var(--accent); }
</style>
