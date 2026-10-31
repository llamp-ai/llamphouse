<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { v2, follow, ago, formatTs, textOf } from '../api/client'
import type { Debug, Run, Thread, ThreadEvent, TreeNode } from '../api/client'
import Badge from '../components/Badge.vue'
import IdLink from '../components/IdLink.vue'
import MessageView from '../components/MessageView.vue'
import JsonView from '../components/JsonView.vue'
import Composer from '../components/Composer.vue'
import Why from '../components/Why.vue'
import TreeView from '../components/TreeView.vue'
import Hint from '../components/Hint.vue'

const route = useRoute()
const id = String(route.params.threadId)
const thread = ref<Thread | null>(null)
const events = ref<ThreadEvent[]>([])
const runs = ref<Run[]>([])
const tree = ref<{ root: string; tree: TreeNode } | null>(null)
const activeDebug = ref<Debug | null>(null)
const tab = ref(String(route.query.tab || 'conversation'))
const error = ref('')
const live = ref(false)
const open = ref<Set<number>>(new Set())
let stream: AbortController | null = null
let refreshing = 0

async function refresh() {
  const mine = ++refreshing
  try {
    const [t, r] = await Promise.all([v2.thread(id), v2.threadRuns(id)])
    if (mine !== refreshing) return
    thread.value = t
    runs.value = r
    activeDebug.value = t.active_run ? await v2.debug(t.active_run.id).catch(() => null) : null
  } catch (e: any) { error.value = e.message }
}
let soon: number | undefined
function refreshSoon() { clearTimeout(soon); soon = window.setTimeout(refresh, 150) }

onMounted(async () => {
  await refresh()
  stream = new AbortController()
  live.value = true
  follow(`/v2/threads/${id}/events/stream?after=0`, stream.signal, (ev: ThreadEvent) => {
    if (typeof ev !== 'object' || ev.position == null) return
    if (!events.value.length || ev.position > events.value[events.value.length - 1].position) events.value.push(ev)
    refreshSoon()
  }).catch(() => { live.value = false })
})
onUnmounted(() => stream?.abort())
watch(tab, async t => { if (t === 'tree' && !tree.value) tree.value = await v2.threadTree(id).catch(() => null) })

const messages = computed(() => events.value.filter(e => e.message || e.kind.startsWith('run.')))
function bubbleClass(e: ThreadEvent) {
  const k = e.message?.sender.kind
  return { 'bubble--agent': k === 'agent', 'bubble--system': k === 'system' || k === 'trigger', 'bubble--to': !!e.to }
}
function lifecycle(e: ThreadEvent): string {
  const what = e.kind.replace('run.', '').replace('_', ' ')
  if (e.kind === 'run.spawned') return `${e.run} started ${e.data?.thread}`
  if (e.kind === 'run.input_required') return `${e.run} asked: ${e.data?.reason ?? ''}`
  return `${e.run ?? ''} ${what}${e.data?.error ? ': ' + e.data.error : ''}`
}
function summary(e: ThreadEvent): string {
  if (e.message) return `${e.message.sender.id}: ${textOf(e.message)}`
  if (e.kind === 'state.changed') return (e.data?.changes ?? []).map((c: any) => `${c.key}${c.deleted ? ' ✕' : ' = ' + JSON.stringify(c.value)}`).join(', ')
  return e.data ? JSON.stringify(e.data).slice(0, 160) : ''
}
function toggle(p: number) { open.value.has(p) ? open.value.delete(p) : open.value.add(p); open.value = new Set(open.value) }

// state, at a position
const at = ref<number | null>(null)
const state = ref<Record<string, { value: any; version: number }>>({})
const statePositions = computed(() => events.value.filter(e => e.kind === 'state.changed').map(e => e.position))
async function loadState() { state.value = await v2.state(id, at.value ?? undefined).catch(() => ({})) }
watch([tab, at], () => { if (tab.value === 'state') loadState() })
watch(() => events.value.length, () => { if (tab.value === 'state' && at.value == null) loadState() })
</script>

<template>
  <div v-if="thread">
    <div class="breadcrumbs"><router-link to="/threads">Threads</router-link><span>/</span><span class="mono">{{ id }}</span></div>
    <div class="page-header flex items-center justify-between">
      <div>
        <h1 class="flex items-center gap-2">{{ thread.agent }}
          <Badge v-if="thread.active_run" :status="thread.active_run.status" /><span v-else class="pill">{{ thread.closed ? 'closed' : 'idle' }}</span>
          <span v-if="live" class="live-dot" title="Live" /></h1>
        <p class="page-header__subtitle">
          <IdLink :id="id" full /> · created {{ ago(thread.created_at) }}
          <template v-if="thread.forked_from"> · fork of <IdLink :id="thread.forked_from.thread" kind="thread" /> at #{{ thread.forked_from.position }}</template>
          <template v-if="thread.created_by"> · made by run <IdLink :id="thread.created_by" kind="run" /></template>
        </p>
        <div class="row-actions mt-2"><span v-for="(v, k) in thread.metadata" :key="k" class="pill">{{ k }}={{ v }}</span></div>
      </div>
    </div>
    <div v-if="error" class="error-text">{{ error }}</div>
    <div class="split">
      <div>
        <div class="tabs">
          <button v-for="t in ['conversation', 'timeline', 'state', 'tree', 'runs']" :key="t" class="tab" :class="{ 'tab--active': tab === t }" @click="tab = t">
            {{ t[0].toUpperCase() + t.slice(1) }}<span v-if="t === 'runs'" class="muted"> {{ runs.length }}</span></button>
        </div>

        <div v-if="tab === 'conversation'" class="card">
          <div v-if="!messages.length" class="muted">Nothing yet.</div>
          <template v-for="e in messages" :key="e.position">
            <div v-if="e.message" class="bubble" :class="bubbleClass(e)">
              <div class="bubble__who"><b>{{ e.message.sender.name || e.message.sender.id }}</b><span>{{ e.message.sender.kind }}</span>
                <span v-if="e.to">→ <IdLink :id="e.to" kind="run" /> (its inbox)</span>
                <span v-if="e.run && e.message.sender.kind === 'agent'"><IdLink :id="e.run" kind="run" /></span><span class="muted">#{{ e.position }}</span></div>
              <MessageView :parts="e.message.parts" />
            </div>
            <div v-else class="lifecycle small muted"><span>#{{ e.position }}</span>
              <router-link v-if="e.run" :to="`/runs/${e.run}`">{{ lifecycle(e) }}</router-link><span v-else>{{ lifecycle(e) }}</span></div>
          </template>
        </div>

        <div v-if="tab === 'timeline'" class="card" style="padding:8px">
          <p class="small muted" style="padding:4px">Every event on the thread, in order: its record. A fork shows its parent's history up to the fork, then its own.
            <Hint text="Thread events have positions: gapless, in commit order. Text deltas are run events, not here." topic="events" /></p>
          <template v-for="e in events" :key="e.position">
            <div class="timeline-row" style="cursor:pointer" @click="toggle(e.position)">
              <span class="mono muted">#{{ e.position }}</span>
              <span><span class="pill">{{ e.kind }}</span></span>
              <span>{{ summary(e) }} <span v-if="e.run" class="muted small">· <IdLink :id="e.run" kind="run" /></span></span>
            </div>
            <div v-if="open.has(e.position)" style="padding:6px 60px"><div class="small muted">{{ formatTs(e.at) }}</div><JsonView :value="e" open /></div>
          </template>
        </div>

        <div v-if="tab === 'state'" class="card">
          <p class="small muted">The thread's state (ctx.state): every write an event, so you can look at it as it was at any position.
            <Hint text="state.set / update / batch are thread events: a fork starts from the state at the fork; at_start() shows a run what it began with." topic="state" /></p>
          <div class="toolbar mt-2">
            <label class="small">as of</label>
            <select v-model="at" class="input" style="width:auto">
              <option :value="null">now</option>
              <option v-for="p in statePositions" :key="p" :value="p">#{{ p }}</option>
            </select>
          </div>
          <div v-if="!Object.keys(state).length" class="muted small">Empty.</div>
          <dl class="kv"><template v-for="(v, k) in state" :key="k"><dt class="mono">{{ k }} <span class="pill">v{{ v.version }}</span></dt><dd><JsonView :value="v.value" /></dd></template></dl>
        </div>

        <div v-if="tab === 'tree'" class="card">
          <p class="small muted mb-2">Threads its runs created - forks (the same agent, from a point in this thread) and threads of other agents - and theirs.</p>
          <TreeView v-if="tree" :node="tree.tree" kind="thread" root />
          <div v-else class="spinner" />
        </div>

        <div v-if="tab === 'runs'" class="card" style="padding:0">
          <table><thead><tr><th>Run</th><th>Status</th><th>Input</th><th>Activations</th></tr></thead>
            <tbody><tr v-for="r in runs" :key="r.id"><td><IdLink :id="r.id" kind="run" /></td><td><Badge :status="r.status" /></td>
              <td class="small">{{ textOf(r.input).slice(0, 80) }}</td><td>{{ r.activations }}</td></tr></tbody></table>
        </div>
      </div>

      <div class="sticky">
        <Composer v-if="!thread.closed" :thread="thread" @done="refresh" />
        <div v-if="activeDebug" class="card mt-4">
          <div class="section__title flex items-center justify-between">The active run <IdLink :id="activeDebug.id" kind="run" /></div>
          <Why :items="activeDebug.why" />
        </div>
      </div>
    </div>
  </div>
  <div v-else-if="error" class="error-text">{{ error }}</div>
  <div v-else class="loading-center"><div class="spinner" /></div>
</template>
<style scoped>
.lifecycle { display: flex; gap: 8px; justify-content: center; padding: 4px 0; border-top: 1px dashed var(--border); margin: 6px 0; }
</style>
