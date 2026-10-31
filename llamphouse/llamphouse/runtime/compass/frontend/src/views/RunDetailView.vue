<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { v2, follow, ago, durationMs, formatTs, textOf } from '../api/client'
import type { Debug, Run, SessionView, Span, Thread } from '../api/client'
import Badge from '../components/Badge.vue'
import IdLink from '../components/IdLink.vue'
import JsonView from '../components/JsonView.vue'
import MessageView from '../components/MessageView.vue'
import Why from '../components/Why.vue'
import TreeView from '../components/TreeView.vue'
import Composer from '../components/Composer.vue'
import SpanTree from '../components/SpanTree.vue'
import Hint from '../components/Hint.vue'

const route = useRoute()
const id = String(route.params.runId)
const run = ref<Run | null>(null)
const debug = ref<Debug | null>(null)
const thread = ref<Thread | null>(null)
const session = ref<SessionView | null>(null)
const spans = ref<Span[] | null>(null)
const tracing = ref(true)
const tab = ref(String(route.query.tab || 'story'))
const error = ref('')
let timer: number | undefined

async function load() {
  try {
    const [r, d] = await Promise.all([v2.run(id), v2.debug(id)])
    run.value = r
    debug.value = d
    thread.value = await v2.thread(r.thread)
    error.value = ''
  } catch (e: any) { error.value = e.message }
}
onMounted(() => { load(); timer = window.setInterval(() => { if (run.value && !['completed', 'failed', 'cancelled', 'rejected'].includes(run.value.status)) load() }, 2500) })
onUnmounted(() => { clearInterval(timer); liveStream?.abort() })

watch(tab, async t => {
  if (t === 'session' && !session.value) session.value = await v2.session(id).catch(() => null)
  if (t === 'trace' && spans.value === null) { const tr = await v2.trace(id).catch(() => null); spans.value = tr?.spans ?? []; tracing.value = !!tr?.tracing }
  if (t === 'live') startLive()
}, { immediate: true })

// the run's stream: stored events, then live (text deltas too)
interface LiveItem { seq: number; kind: string; text?: string; data?: any }
const liveItems = ref<LiveItem[]>([])
const liveDone = ref('')
let liveStream: AbortController | null = null
function startLive() {
  if (liveStream) return
  liveStream = new AbortController()
  const byMessage = new Map<string, LiveItem>()
  follow(`/v2/runs/${id}/events/stream?after=0`, liveStream.signal, (ev: any, event: string) => {
    if (event === 'end') { liveDone.value = ev.status; return }
    const kind = ev.kind ?? event
    if (kind === 'message.started' || kind === 'message.delta') {
      let item = byMessage.get(ev.message_id)
      if (!item) { item = { seq: ev.seq, kind: 'message', text: '' }; byMessage.set(ev.message_id, item); liveItems.value.push(item) }
      if (ev.delta) item.text += ev.delta
      liveItems.value = [...liveItems.value]
    } else if (kind === 'message.completed') {
      const item = byMessage.get(ev.message?.id)
      if (item) { item.text = textOf(ev.message); liveItems.value = [...liveItems.value] }
      else liveItems.value.push({ seq: ev.seq, kind: 'message', text: textOf(ev.message) })
    } else liveItems.value.push({ seq: ev.seq, kind, data: ev })
  }).catch(() => {})
}
const STORY_ICON: Record<string, string> = { input: '➜', created: '●', said: '💬', inbox: '✉', asked: '?', answered: '✓', spawned: '⑂', state: '≡', ended: '■', run: '·', post: '◆' }
const ended = computed(() => !!run.value && ['completed', 'failed', 'cancelled', 'rejected'].includes(run.value.status))
</script>

<template>
  <div v-if="run && debug">
    <div class="breadcrumbs"><router-link to="/runs">Runs</router-link><span>/</span><span class="mono">{{ id }}</span></div>
    <div class="page-header">
      <h1 class="flex items-center gap-2">{{ run.agent }} <Badge :status="run.status" /></h1>
      <p class="page-header__subtitle">
        <IdLink :id="id" full /> on thread <IdLink :id="run.thread" kind="thread" />
        <template v-if="debug.parent"> · started by <IdLink :id="debug.parent" kind="run" /></template>
        · {{ ago(debug.created_at) }}<template v-if="debug.ended_at"> · took {{ durationMs(debug.created_at, debug.ended_at) }}</template>
        · caller {{ debug.caller?.subject }}<template v-if="debug.caller?.tenant"> ({{ debug.caller.tenant }})</template>
      </p>
    </div>
    <div v-if="error" class="error-text">{{ error }}</div>
    <div class="split">
      <div>
        <Why :items="debug.why" />
        <div class="tabs mt-4">
          <button v-for="t in ['story', 'live', 'session', 'steps', 'waiting', 'trace', 'tree', 'raw']" :key="t" class="tab" :class="{ 'tab--active': tab === t }" @click="tab = t">
            {{ { story: 'Story', live: 'Live', session: 'Session', steps: `Steps ${debug.steps.length}`, waiting: 'Waiting on', trace: 'Trace', tree: 'Tree', raw: 'Raw' }[t] }}</button>
        </div>

        <div v-if="tab === 'story'" class="card">
          <p class="small muted mb-2">What happened to this run, from its thread's record.</p>
          <div v-for="(s, i) in debug.story" :key="i" class="timeline-row" style="grid-template-columns: 48px 28px 1fr">
            <span class="mono muted">#{{ s.position }}</span><span>{{ STORY_ICON[s.kind] ?? '·' }}</span>
            <span>{{ s.text }}
              <router-link v-if="s.kind === 'spawned' && s.thread" :to="`/threads/${s.thread}`" class="small"> open thread</router-link>
              <router-link v-if="s.kind === 'spawned' && s.run" :to="`/runs/${s.run}`" class="small"> · its run</router-link>
              <span class="muted small" style="margin-left:6px">{{ s.at ? formatTs(s.at) : '' }}</span></span>
          </div>
          <div v-if="debug.output != null" class="mt-4"><div class="section__title">Output</div><JsonView :value="debug.output" /></div>
          <div v-if="run.error" class="mt-4 callout callout--error"><span class="callout__dot" /><div class="mono small">{{ run.error }}</div></div>
        </div>

        <div v-if="tab === 'live'" class="card">
          <p class="small muted mb-2">Its stream as it happens: messages being written (text deltas), progress, custom and state events.
            <Hint text="Run events: numbered per run, kept (except deltas), and published live to whichever API process serves you." topic="events" /></p>
          <div v-if="!liveItems.length" class="muted small">Nothing yet{{ ended ? ' (text deltas are live only: see Story for the messages)' : '…' }}</div>
          <div v-for="item in liveItems" :key="item.seq + item.kind" class="mb-2">
            <div v-if="item.kind === 'message'" class="bubble bubble--agent" style="max-width:100%">{{ item.text }}</div>
            <div v-else class="small"><span class="pill">{{ item.kind }}</span> <span class="mono">{{ JSON.stringify(item.data).slice(0, 200) }}</span></div>
          </div>
          <div v-if="liveDone" class="small muted">The run ended: {{ liveDone }}.</div>
        </div>

        <div v-if="tab === 'session'" class="card">
          <template v-if="session">
            <p class="small muted mb-2">Its working memory: what <b>{{ session.agent }}</b> sees (ctx.session.messages()) - session {{ session.id }}, version {{ session.version }},
              {{ session.revisions }} revision(s){{ session.revising ? ', revising now' : '' }}, provider {{ debug.session?.provider }}.
              <Hint text="The session is built from the thread (minus messages addressed to other runs), plus entries the agent added (session.add); a provider revises it in the background." topic="sessions" /></p>
            <div v-for="m in session.messages" :key="m.id" class="bubble" :class="{ 'bubble--agent': m.own }" style="max-width:92%">
              <div class="bubble__who"><b>{{ m.sender.name || m.sender.id }}</b><span>{{ m.own ? 'own' : 'other' }}</span></div>
              <MessageView :parts="m.parts" />
            </div>
            <div v-if="session.log.length" class="mt-4"><div class="section__title">Its log</div>
              <div v-for="(l, i) in session.log" :key="i" class="small"><span class="pill">{{ l.kind }}</span> <span class="mono">{{ JSON.stringify(l).slice(0, 160) }}</span></div></div>
          </template>
          <div v-else class="muted small">No session.</div>
        </div>

        <div v-if="tab === 'steps'" class="card">
          <p class="small muted mb-2">What it recorded, by call key. When it is executed again - after waiting, a restart(), or on another worker after a crash - these return what they returned, instead of happening again.
            <Hint text="Durability: every send, step, state read and write, fork and wait is recorded under name#n. Code between them runs again; what they did doesn't." topic="durability" /></p>
          <div v-if="!debug.steps.length" class="muted small">Nothing recorded yet.</div>
          <div class="table-wrap" v-else><table><thead><tr><th>Key</th><th>Recorded</th></tr></thead>
            <tbody><tr v-for="s in debug.steps" :key="s.key"><td class="mono">{{ s.key }}</td><td><JsonView :value="s.value" /></td></tr></tbody></table></div>
        </div>

        <div v-if="tab === 'waiting'" class="card">
          <dl class="kv mb-4">
            <dt>Executions</dt><dd>{{ debug.activations }} <span class="muted small">(generation {{ debug.generation }}: restarts)</span></dd>
            <dt>Lease</dt><dd>{{ debug.lease.owner ? `${debug.lease.owner} until ${formatTs(debug.lease.until)}` : 'none (not running)' }}</dd>
            <dt>Wakes at</dt><dd>{{ debug.wake_at ? formatTs(debug.wake_at) : '—' }}</dd>
            <dt>Rung again at</dt><dd>{{ debug.ring_at ? formatTs(debug.ring_at) : '—' }}</dd>
            <dt>Depth</dt><dd>{{ debug.depth }} <span v-if="debug.root" class="muted small">under <IdLink :id="debug.root" kind="run" /></span></dd>
            <dt>Run state</dt><dd><JsonView :value="debug.run_state" /></dd>
          </dl>
          <div class="section__title">Its subscriptions <Hint text="What it listens to: its inbox, threads it created. A cursor per thread: how far it has read." topic="inbox" /></div>
          <div v-for="w in debug.waits_on" :key="w.key" class="card mb-2" :style="w.waiting ? 'border-color: var(--info)' : ''">
            <div class="row-actions"><b class="mono">{{ w.key }}</b><span v-if="w.waiting" class="pill">waiting on this</span></div>
            <div class="small mt-2">read up to: <span v-for="(c, t) in w.cursors" :key="t" class="pill" style="margin-right:4px">{{ t }} #{{ c }}</span></div>
            <JsonView :value="w.filters" />
          </div>
        </div>

        <div v-if="tab === 'trace'" class="card">
          <div v-if="spans === null" class="spinner" />
          <div v-else-if="!tracing" class="muted small">No tracing store: set one in the runtime file (tracing: {in_memory: {}}) or TRACING_STORE.</div>
          <div v-else-if="!spans.length" class="muted small">No spans for this run.</div>
          <SpanTree v-else :spans="spans" />
        </div>

        <div v-if="tab === 'tree'" class="card">
          <p class="small muted mb-2">The runs started from the same run the app started: forks and calls, all the way down.</p>
          <TreeView :node="debug.tree" kind="run" root />
        </div>

        <div v-if="tab === 'raw'" class="card">
          <div class="section__title">The run (GET /v2/runs/{{ id }})</div><JsonView :value="run" open />
          <div class="section__title mt-4">Its request</div><JsonView :value="debug.request" />
          <div class="section__title mt-4">Its input</div><MessageView :parts="run.input.parts" />
        </div>
      </div>
      <div class="sticky">
        <Composer v-if="thread && thread.active_run?.id === id" :thread="thread" @done="load" />
        <div class="card mt-4">
          <div class="section__title">Input</div>
          <MessageView :parts="run.input.parts" />
          <div class="small muted mt-2">from {{ run.input.sender.id }} ({{ run.input.sender.kind }})</div>
        </div>
        <div v-if="run.unread.length" class="card mt-4">
          <div class="section__title">Sent to it, not read <Hint text="Messages to a run enter its session when it reads them (ctx.next()). Unread ones stay on the thread." topic="inbox" /></div>
          <div v-for="m in run.unread" :key="m.id" class="small">{{ m.sender.id }}: {{ textOf(m) }}</div>
        </div>
        <div v-if="debug.children.length || debug.created.length" class="card mt-4">
          <div class="section__title">It started</div>
          <div v-for="c in debug.created" :key="c" class="small">thread <IdLink :id="c" kind="thread" /></div>
          <div v-for="c in debug.children" :key="c" class="small">run <IdLink :id="c" kind="run" /></div>
        </div>
      </div>
    </div>
  </div>
  <div v-else-if="error" class="error-text">{{ error }}</div>
  <div v-else class="loading-center"><div class="spinner" /></div>
</template>
