<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { compass, formatTs, shortId, statusBadge, durationMs } from '../api/client'
import type { Run, RunStep, Span, FlowData, FlowNode, FlowEdge, Message } from '../api/client'
import SpanTree from '../components/SpanTree.vue'
import MessageBubble from '../components/MessageBubble.vue'

const route = useRoute()
const router = useRouter()
const threadId = ref(route.params.threadId as string)
const runId = ref(route.params.runId as string)

const run = ref<Run | null>(null)
const steps = ref<RunStep[]>([])
const config = ref<any>(null)
const spans = ref<Span[]>([])
const flow = ref<FlowData>({ nodes: [], edges: [], has_flow: false })
const messages = ref<Message[]>([])

// Per-section loading flags so each panel can render its own state as soon
// as its data arrives, instead of blocking on the slowest request.
const loading = ref({
  run:      true,
  steps:    true,
  config:   true,
  spans:    true,
  flow:     true,
  messages: true,
})
const error = ref('')
const tab = ref<'io' | 'details' | 'steps' | 'config' | 'trace' | 'flow'>('io')

function fetchData() {
  // Reset state before re-fetching (used by the route watcher).
  error.value = ''
  run.value = null
  steps.value = []
  config.value = null
  spans.value = []
  flow.value = { nodes: [], edges: [], has_flow: false }
  messages.value = []
  loading.value = {
    run: true, steps: true, config: true, spans: true, flow: true, messages: true,
  }

  // Fire all requests in parallel.  Each settles independently and updates
  // its own slice of state + loading flag — the page chrome renders right
  // away and each section fills in as its response lands.

  compass.listRuns(threadId.value)
    .then((runs) => { run.value = runs.find((r) => r.id === runId.value) || null })
    .catch((e: any) => { error.value = e.message })
    .finally(() => { loading.value.run = false })

  compass.listRunSteps(threadId.value, runId.value)
    .then((v) => { steps.value = v })
    .catch(() => {})
    .finally(() => { loading.value.steps = false })

  compass.getRunConfig(threadId.value, runId.value)
    .then((v) => { config.value = v })
    .catch(() => {})
    .finally(() => { loading.value.config = false })

  compass.getRunTrace(runId.value)
    .then((v) => { spans.value = v })
    .catch(() => {})
    .finally(() => { loading.value.spans = false })

  compass.getRunFlow(runId.value)
    .then((v) => { flow.value = v })
    .catch(() => {})
    .finally(() => { loading.value.flow = false })

  compass.listMessages(threadId.value)
    .then((v) => { messages.value = v })
    .catch(() => {})
    .finally(() => { loading.value.messages = false })
}

onMounted(fetchData)

watch(
  () => [route.params.threadId, route.params.runId],
  ([newThread, newRun]) => {
    if (newThread && newRun && (newThread !== threadId.value || newRun !== runId.value)) {
      threadId.value = newThread as string
      runId.value = newRun as string
      fetchData()
    }
  },
)

const runMeta = computed(() => {
  if (!run.value) return []
  const r = run.value
  return [
    { label: 'Status', value: r.status },
    { label: 'Model', value: r.model },
    { label: 'Agent', value: r.assistant_id },
    { label: 'Created', value: formatTs(r.created_at) },
    { label: 'Started', value: formatTs(r.started_at ?? 0) },
    { label: 'Completed', value: formatTs(r.completed_at ?? 0) },
    { label: 'Duration', value: durationMs(r.started_at, r.completed_at ?? r.failed_at) },
    { label: 'Prompt tokens', value: r.usage?.prompt_tokens ?? '—' },
    { label: 'Completion tokens', value: r.usage?.completion_tokens ?? '—' },
    { label: 'Total tokens', value: r.usage?.total_tokens ?? '—' },
  ]
})

/* ── I/O message split ─────────────────────────────────── */

// A message qualifies as "output" of this run if either:
//   (a) its run_id explicitly matches, OR
//   (b) it has no run_id stamped AND it was created during this run's
//       lifespan (started_at .. completed_at/failed_at) and it's not from
//       the user/system.
// This handles legacy / agent-inserted messages that weren't tagged with
// run_id by `context.insert_message`.
function isOutputMessage(m: Message): boolean {
  if (m.run_id && m.run_id === runId.value) return true
  if (m.run_id) return false   // belongs to a different run

  if (!run.value) return false
  if (m.role === 'user' || m.role === 'system') return false
  const start = run.value.started_at ?? run.value.created_at
  const end   = run.value.completed_at ?? run.value.failed_at ?? Infinity
  if (!start) return false
  return m.created_at >= start && m.created_at <= end
}

const outputMessages = computed(() => messages.value.filter(isOutputMessage))

// Earliest created_at of any output message (seconds epoch)
const firstOutputAt = computed(() =>
  outputMessages.value.length
    ? Math.min(...outputMessages.value.map((m) => m.created_at))
    : Infinity,
)

// Last user/system message strictly before the first output → the prompt
const inputMessage = computed<Message | null>(() => {
  const candidates = messages.value.filter(
    (m) => (m.role === 'user' || m.role === 'system') && m.created_at < firstOutputAt.value,
  )
  return candidates.length ? candidates[candidates.length - 1] : null
})

// Everything before the input message → context/history
const contextMessages = computed(() => {
  const cutoff = inputMessage.value?.created_at ?? firstOutputAt.value
  return messages.value.filter(
    (m) => m.created_at < cutoff && !isOutputMessage(m),
  )
})

const hasIo = computed(() => outputMessages.value.length > 0 || inputMessage.value !== null)

const LANE_W = 200
const LANE_GAP = 60
const NODE_W = 172
const NODE_H = 68
const ROW_H = 96
const HEADER_H = 44
const PAD = 24

interface LayoutNode extends FlowNode {
  x: number
  y: number
  lane: number
  row: number
}

interface Lane {
  agentId: string
  agentName: string
  x: number
}

interface Spine {
  x: number
  y1: number
  y2: number
}

interface ThreadGroup {
  x: number
  y1: number
  y2: number
  threadId: string
}

interface LayoutEdge extends FlowEdge {
  /** SVG path `d` attribute, pre-computed once during layout. */
  path: string
  /** Midpoint for the sequence badge. */
  midX: number
  midY: number
  /** Pre-computed styling so the template avoids function calls. */
  color: string
  dash: string
  markerEnd: string
}

const flowLayout = computed(() => {
  const { nodes, edges, has_flow } = flow.value
  if (!nodes.length || !has_flow) {
    return {
      nodes: [] as LayoutNode[],
      edges: [] as LayoutEdge[],
      lanes: [] as Lane[],
      spines: [] as Spine[],
      threadGroups: [] as ThreadGroup[],
      width: 0,
      height: 0,
    }
  }

  const nodeMap = new Map(nodes.map(n => [n.id, n]))

  // Find root
  const root = nodes.find(n => n.is_root) || nodes[0]

  // Sort edges by backend-provided sequence (chronological)
  const sorted = [...edges].sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0))

  // Build lane order: root agent first, then others by first appearance
  const seen = new Set<string>([root.agent_id])
  const agentOrder = [root.agent_id]
  const agentNames: Record<string, string> = { [root.agent_id]: root.agent_name }
  for (const e of sorted) {
    const child = nodeMap.get(e.target)
    if (child && !seen.has(child.agent_id)) {
      seen.add(child.agent_id)
      agentOrder.push(child.agent_id)
    }
    if (child) agentNames[child.agent_id] = child.agent_name
  }

  // Lane index lookup
  const laneOf: Record<string, number> = {}
  agentOrder.forEach((a, i) => { laneOf[a] = i })

  // Row assignments: root = row 0, children ordered by edge sequence
  const rowOf: Record<string, number> = { [root.id]: 0 }
  let nextRow = 1
  for (const e of sorted) {
    if (!(e.target in rowOf)) rowOf[e.target] = nextRow++
  }
  for (const n of nodes) {
    if (!(n.id in rowOf)) rowOf[n.id] = nextRow++
  }

  const maxRow = Math.max(...Object.values(rowOf))

  // Lane positions
  const lanes: Lane[] = agentOrder.map((aid, i) => ({
    agentId: aid,
    agentName: agentNames[aid] || aid,
    x: PAD + i * (LANE_W + LANE_GAP),
  }))

  // Layout nodes with position
  const layoutNodes: LayoutNode[] = nodes.map(n => {
    const li = laneOf[n.agent_id] ?? 0
    const r = rowOf[n.id] ?? 0
    const lx = PAD + li * (LANE_W + LANE_GAP)
    return {
      ...n,
      x: lx + (LANE_W - NODE_W) / 2,
      y: PAD + HEADER_H + r * ROW_H,
      lane: li,
      row: r,
    }
  })

  const layoutNodeMap = new Map(layoutNodes.map(n => [n.id, n]))

  // Spines: vertical dashed line from each parent to its last child
  const spines: Spine[] = []
  const childrenByParent = new Map<string, LayoutNode[]>()
  for (const e of sorted) {
    const child = layoutNodeMap.get(e.target)
    if (!child) continue
    const list = childrenByParent.get(e.source) || []
    list.push(child)
    childrenByParent.set(e.source, list)
  }
  for (const [pid, children] of childrenByParent) {
    const parent = layoutNodeMap.get(pid)
    if (!parent || children.length === 0) continue
    const last = children[children.length - 1]
    spines.push({
      x: parent.x + NODE_W / 2,
      y1: parent.y + NODE_H,
      y2: last.y + NODE_H / 2,
    })
  }

  // Thread groups: bracket connecting nodes in the same lane that share a thread
  const threadGroups: ThreadGroup[] = []
  const byKey = new Map<string, LayoutNode[]>()
  for (const n of layoutNodes) {
    if (!n.thread_id || n.is_root) continue
    const k = `${n.agent_id}:${n.thread_id}`
    const arr = byKey.get(k) || []
    arr.push(n)
    byKey.set(k, arr)
  }
  for (const [, group] of byKey) {
    if (group.length < 2) continue
    group.sort((a, b) => a.row - b.row)
    threadGroups.push({
      x: group[0].x + NODE_W + 12,
      y1: group[0].y + NODE_H / 2,
      y2: group[group.length - 1].y + NODE_H / 2,
      threadId: group[0].thread_id!,
    })
  }

  const width = PAD * 2 + agentOrder.length * LANE_W + (agentOrder.length - 1) * LANE_GAP
  const height = PAD + HEADER_H + (maxRow + 1) * ROW_H + PAD

  // ── Pre-compute edge geometry & styling ────────────────────────────────────
  // This is the hot path for big graphs: doing it here (memoised by the
  // computed) instead of in template-time functions turns rendering into
  // pure data binding — O(E) once, not O(E·N) per render.
  const layoutEdges: LayoutEdge[] = []
  for (const e of sorted) {
    const src = layoutNodeMap.get(e.source)
    const tgt = layoutNodeMap.get(e.target)
    if (!src || !tgt) continue

    const spineX = src.x + NODE_W / 2
    const tgtCY = tgt.y + NODE_H / 2

    let path = ''
    let midX = 0
    let midY = 0

    if (src.lane === tgt.lane) {
      // Same lane: straight vertical from parent bottom to child top.
      path = `M ${spineX} ${src.y + NODE_H} L ${spineX} ${tgt.y}`
      midX = spineX
      midY = (src.y + NODE_H + tgt.y) / 2
    } else if (tgt.lane > src.lane) {
      // Cross-lane right: horizontal from spine to target's left edge.
      path = `M ${spineX} ${tgtCY} L ${tgt.x} ${tgtCY}`
      midX = (spineX + tgt.x) / 2
      midY = tgtCY
    } else {
      // Cross-lane left: horizontal from spine to target's right edge.
      path = `M ${spineX} ${tgtCY} L ${tgt.x + NODE_W} ${tgtCY}`
      midX = (spineX + tgt.x + NODE_W) / 2
      midY = tgtCY
    }

    const isHandover = e.type === 'handover'
    layoutEdges.push({
      ...e,
      path,
      midX,
      midY,
      color:     isHandover ? '#7c3aed' : '#64748b',
      dash:      isHandover ? 'none' : '6,4',
      markerEnd: isHandover ? 'url(#arrow-handover)' : 'url(#arrow-call)',
    })
  }

  return { nodes: layoutNodes, edges: layoutEdges, lanes, spines, threadGroups, width, height }
})

function flowStatusColor(status: string): string {
  switch (status) {
    case 'completed': return '#16a34a'
    case 'in_progress':
    case 'queued': return '#2563eb'
    case 'failed':
    case 'cancelled': return '#dc2626'
    default: return '#6b7280'
  }
}

function flowDuration(ms: number | null): string {
  if (ms == null) return ''
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function navigateToRun(node: LayoutNode) {
  if (node.thread_id) {
    router.push(`/threads/${node.thread_id}/runs/${node.id}`)
  }
}
</script>

<template>
  <div>
    <div class="breadcrumbs">
      <router-link to="/threads">Threads</router-link>
      <span>›</span>
      <router-link :to="`/threads/${threadId}`">{{ shortId(threadId) }}</router-link>
      <span>›</span>
      <span>Run {{ shortId(runId) }}</span>
    </div>

    <div class="page-header">
      <div>
        <h1>
          Run Detail
          <span v-if="run" class="badge ml-auto" :class="statusBadge(run.status)" style="margin-left: 12px; vertical-align: middle;">
            {{ run.status }}
          </span>
        </h1>
        <div class="page-header__subtitle mono">{{ runId }}</div>
      </div>
      <router-link :to="`/traces/${runId}`" class="btn btn--sm">
        View Full Trace
      </router-link>
    </div>

    <div v-if="error" class="card" style="color: var(--error)">{{ error }}</div>

    <template v-else>
      <div class="tabs">
        <div class="tab" :class="{ 'tab--active': tab === 'io' }" @click="tab = 'io'">
          Input / Output
          <span v-if="loading.messages || loading.run" class="spinner spinner--tiny"></span>
        </div>
        <div class="tab" :class="{ 'tab--active': tab === 'details' }" @click="tab = 'details'">Details</div>
        <div class="tab" :class="{ 'tab--active': tab === 'steps' }" @click="tab = 'steps'">
          Steps <span v-if="!loading.steps">({{ steps.length }})</span>
          <span v-else class="spinner spinner--tiny"></span>
        </div>
        <div class="tab" :class="{ 'tab--active': tab === 'flow' }" @click="tab = 'flow'">
          Flow <span v-if="!loading.flow">({{ flow.nodes.length }})</span>
          <span v-else class="spinner spinner--tiny"></span>
        </div>
        <div class="tab" :class="{ 'tab--active': tab === 'config' }" @click="tab = 'config'">Config</div>
        <div class="tab" :class="{ 'tab--active': tab === 'trace' }" @click="tab = 'trace'">
          Trace <span v-if="!loading.spans">({{ spans.length }})</span>
          <span v-else class="spinner spinner--tiny"></span>
        </div>
      </div>

      <!-- I/O tab -->
      <div v-if="tab === 'io'">
        <div v-if="loading.messages || loading.run" class="loading-center"><div class="spinner"></div></div>
        <div v-else-if="!hasIo" class="empty-state">
          <div class="empty-state__icon"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg></div>
          <div class="empty-state__title">No messages for this run</div>
        </div>
        <div v-else class="io-layout">

          <!-- Context (earlier messages) -->
          <template v-if="contextMessages.length">
            <div class="io-section-label io-section-label--context">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
              Context ({{ contextMessages.length }} earlier message{{ contextMessages.length !== 1 ? 's' : '' }})
            </div>
            <div class="io-context">
              <MessageBubble
                v-for="msg in contextMessages"
                :key="msg.id"
                :message="msg"
                :thread-id="threadId"
                class="io-context__bubble"
              />
            </div>
          </template>

          <!-- Input -->
          <template v-if="inputMessage">
            <div class="io-section-label io-section-label--input">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
              Input
            </div>
            <MessageBubble :message="inputMessage" :thread-id="threadId" />
          </template>

          <!-- Output -->
          <template v-if="outputMessages.length">
            <div class="io-section-label io-section-label--output">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="5 12 12 5 19 12"/><line x1="12" y1="19" x2="12" y2="5"/></svg>
              Output
            </div>
            <div class="io-output">
              <MessageBubble
                v-for="msg in outputMessages"
                :key="msg.id"
                :message="msg"
                :thread-id="threadId"
              />
            </div>
          </template>

        </div>
      </div>

      <!-- Details tab -->
      <div v-if="tab === 'details'" class="card">
        <div v-if="loading.run" class="loading-center"><div class="spinner"></div></div>
        <template v-else>
        <div class="detail-grid">
          <div v-for="item in runMeta" :key="item.label" class="detail-row">
            <span class="detail-row__label">{{ item.label }}</span>
            <span class="detail-row__value">
              <template v-if="item.label === 'Status'">
                <span class="badge" :class="statusBadge(item.value as string)">{{ item.value }}</span>
              </template>
              <template v-else-if="item.label === 'Agent'">
                <span class="mono">{{ shortId(item.value as string) }}</span>
              </template>
              <template v-else>{{ item.value }}</template>
            </span>
          </div>
        </div>

        <div v-if="run?.last_error" class="mt-4">
          <div class="section__title">Error</div>
          <div class="json-view" style="color: var(--error)">{{ JSON.stringify(run.last_error, null, 2) }}</div>
        </div>

        <div v-if="run?.instructions" class="mt-4">
          <div class="section__title">System Prompt</div>
          <div class="json-view">{{ run.instructions }}</div>
        </div>

        <div v-if="run?.tools && run.tools.length" class="mt-4">
          <div class="section__title">Skills</div>
          <div class="json-view">{{ JSON.stringify(run.tools, null, 2) }}</div>
        </div>
        </template>
      </div>

      <!-- Steps tab -->
      <div v-if="tab === 'steps'">
        <div v-if="loading.steps" class="loading-center"><div class="spinner"></div></div>
        <div v-else-if="steps.length === 0" class="empty-state">
          <div class="empty-state__icon"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg></div>
          <div class="empty-state__title">No run steps</div>
        </div>
        <div v-else class="steps-list">
          <div v-for="step in steps" :key="step.id" class="card step-card">
            <div class="flex items-center justify-between mb-2">
              <div class="flex items-center gap-2">
                <span class="badge" :class="statusBadge(step.status)">{{ step.status }}</span>
                <span class="badge badge--neutral">{{ step.type }}</span>
              </div>
              <span class="mono" style="font-size: 0.7rem; color: var(--text-muted)">
                {{ shortId(step.id) }}
              </span>
            </div>
            <div class="json-view" style="max-height: 200px; overflow-y: auto;">
              {{ JSON.stringify(step.step_details, null, 2) }}
            </div>
            <div v-if="step.usage" class="mt-2" style="font-size: 0.75rem; color: var(--text-muted)">
              Tokens: {{ step.usage.total_tokens }} ({{ step.usage.prompt_tokens }}p / {{ step.usage.completion_tokens }}c)
            </div>
          </div>
        </div>
      </div>

      <!-- Config tab -->
      <div v-if="tab === 'config'" class="card">
        <div v-if="loading.config" class="loading-center"><div class="spinner"></div></div>
        <template v-else-if="config">
          <div class="json-view">{{ JSON.stringify(config, null, 2) }}</div>
        </template>
        <div v-else class="empty-state">
          <div class="empty-state__icon"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg></div>
          <div class="empty-state__title">No config snapshot</div>
        </div>
      </div>

      <!-- Trace tab -->
      <div v-if="tab === 'trace'">
        <div v-if="loading.spans" class="loading-center"><div class="spinner"></div></div>
        <SpanTree v-else :spans="spans" />
      </div>

      <!-- Flow tab -->
      <div v-if="tab === 'flow'" class="card flow-card">
        <div v-if="loading.flow" class="loading-center"><div class="spinner"></div></div>
        <div v-else-if="!flow.has_flow" class="empty-state">
          <div class="empty-state__icon">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="6" cy="6" r="3"/>
              <circle cx="18" cy="18" r="3"/>
              <line x1="8.5" y1="8.5" x2="15.5" y2="15.5"/>
            </svg>
          </div>
          <div class="empty-state__title">No agent flow for this run</div>
          <div class="empty-state__subtitle">
            This run did not call or hand over to other agents.
          </div>
        </div>
        <template v-else>
        <div class="flow-legend">
          <span class="flow-legend__item">
            <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4" stroke="#64748b" stroke-width="2" stroke-dasharray="6,4"/></svg>
            call_agent
          </span>
          <span class="flow-legend__item">
            <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4" stroke="#7c3aed" stroke-width="2"/></svg>
            handover
          </span>
          <span class="flow-legend__item">
            <svg width="16" height="16"><circle cx="8" cy="8" r="7" fill="white" stroke="#94a3b8" stroke-width="1"/><text x="8" y="11.5" text-anchor="middle" font-size="8" fill="#475569" font-weight="600">1</text></svg>
            sequence
          </span>
          <span class="flow-legend__item">
            <svg width="20" height="14"><line x1="2" y1="2" x2="2" y2="12" stroke="#8b5cf6" stroke-width="2"/><line x1="2" y1="2" x2="8" y2="2" stroke="#8b5cf6" stroke-width="2"/><line x1="2" y1="12" x2="8" y2="12" stroke="#8b5cf6" stroke-width="2"/></svg>
            same thread
          </span>
        </div>
        <div class="flow-canvas">
          <svg
            :width="flowLayout.width"
            :height="flowLayout.height"
            :viewBox="`0 0 ${flowLayout.width} ${flowLayout.height}`"
          >
            <defs>
              <marker id="arrow-call" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
                <path d="M0,0 L8,3 L0,6" fill="#64748b"/>
              </marker>
              <marker id="arrow-handover" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
                <path d="M0,0 L8,3 L0,6" fill="#7c3aed"/>
              </marker>
            </defs>

            <!-- Lane backgrounds -->
            <rect
              v-for="(lane, i) in flowLayout.lanes" :key="'lane-bg-' + lane.agentId"
              :x="lane.x" :y="PAD"
              :width="LANE_W" :height="flowLayout.height - PAD * 2"
              :rx="8" :ry="8"
              :fill="i % 2 === 0 ? '#f8fafc' : '#f1f5f9'"
            />

            <!-- Lane headers -->
            <text
              v-for="lane in flowLayout.lanes" :key="'lane-hdr-' + lane.agentId"
              :x="lane.x + LANE_W / 2" :y="PAD + HEADER_H / 2 + 5"
              text-anchor="middle" class="flow-lane-header"
            >{{ lane.agentName }}</text>

            <!-- Separator line under headers -->
            <line
              :x1="PAD" :y1="PAD + HEADER_H - 2"
              :x2="flowLayout.width - PAD" :y2="PAD + HEADER_H - 2"
              stroke="#e2e8f0" stroke-width="1"
            />

            <!-- Spines (vertical dashed timeline from parent to children) -->
            <line
              v-for="(spine, i) in flowLayout.spines" :key="'spine-' + i"
              :x1="spine.x" :y1="spine.y1" :x2="spine.x" :y2="spine.y2"
              stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="4,4"
            />

            <!-- Thread groups (bracket connecting same-thread nodes) -->
            <g v-for="(tg, i) in flowLayout.threadGroups" :key="'tg-' + i">
              <line :x1="tg.x" :y1="tg.y1" :x2="tg.x" :y2="tg.y2" stroke="#8b5cf6" stroke-width="2"/>
              <line :x1="tg.x" :y1="tg.y1" :x2="tg.x - 6" :y2="tg.y1" stroke="#8b5cf6" stroke-width="2"/>
              <line :x1="tg.x" :y1="tg.y2" :x2="tg.x - 6" :y2="tg.y2" stroke="#8b5cf6" stroke-width="2"/>
              <text
                :x="tg.x + 6" :y="(tg.y1 + tg.y2) / 2 + 3"
                class="flow-thread-label"
              >same thread</text>
            </g>

            <!-- Edges (arrows with sequence badges) — geometry pre-computed
                 in flowLayout, so each binding is O(1). -->
            <g v-for="edge in flowLayout.edges" :key="`edge-${edge.source}-${edge.target}`">
              <path
                :d="edge.path"
                fill="none"
                :stroke="edge.color"
                :stroke-dasharray="edge.dash"
                stroke-width="2"
                :marker-end="edge.markerEnd"
              />
              <!-- Sequence badge -->
              <circle
                :cx="edge.midX" :cy="edge.midY"
                r="10" fill="white" stroke="#94a3b8" stroke-width="1"
              />
              <text
                :x="edge.midX" :y="edge.midY + 3.5"
                text-anchor="middle" class="flow-seq-label"
              >{{ edge.sequence }}</text>
            </g>

            <!-- Nodes -->
            <g
              v-for="node in flowLayout.nodes"
              :key="node.id"
              class="flow-node"
              :class="{ 'flow-node--current': node.id === runId }"
              @click="navigateToRun(node)"
              style="cursor: pointer"
            >
              <rect
                :x="node.x"
                :y="node.y"
                :width="NODE_W"
                :height="NODE_H"
                rx="8"
                ry="8"
                :stroke="node.id === runId ? '#2563eb' : '#d1d5db'"
                :stroke-width="node.id === runId ? 2.5 : 1.5"
                fill="white"
              />
              <!-- Status dot -->
              <circle
                :cx="node.x + 14"
                :cy="node.y + 20"
                r="5"
                :fill="flowStatusColor(node.status)"
              />
              <!-- Agent name -->
              <text
                :x="node.x + 26"
                :y="node.y + 24"
                class="flow-node__name"
              >{{ node.agent_name }}</text>
              <!-- Status + duration -->
              <text
                :x="node.x + 26"
                :y="node.y + 42"
                class="flow-node__detail"
              >{{ node.status }}{{ node.duration_ms != null ? ' · ' + flowDuration(node.duration_ms) : '' }}</text>
              <!-- Dispatch type badge -->
              <text
                v-if="node.dispatch_type && !node.is_root"
                :x="node.x + NODE_W - 10"
                :y="node.y + 58"
                class="flow-node__dispatch"
                text-anchor="end"
              >{{ node.dispatch_type === 'handover' ? 'handover' : 'call' }}</text>
              <!-- Root badge -->
              <text
                v-if="node.is_root"
                :x="node.x + NODE_W - 10"
                :y="node.y + 16"
                class="flow-node__root"
                text-anchor="end"
              >root</text>
            </g>
          </svg>
        </div>
        </template>
      </div>
    </template>
  </div>
</template>

<style scoped>
/* ── I/O layout ─────────────────────────────────────────── */
.io-layout {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.io-section-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 4px 2px;
  margin-top: 8px;
}

.io-section-label--context {
  color: var(--text-muted);
}

.io-section-label--input {
  color: var(--accent);
}

.io-section-label--output {
  color: var(--success);
}

.io-context {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px 14px;
  border-left: 3px solid var(--border);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  opacity: 0.65;
}

.io-context__bubble {
  font-size: 0.8rem;
}

.io-output {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

/* ── Details grid ──────────────────────────────────────── */
.detail-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
}

.detail-row {
  display: flex;
  justify-content: space-between;
  padding: 10px 0;
  border-bottom: 1px solid var(--border);
}

.detail-row__label {
  font-size: 0.8rem;
  color: var(--text-secondary);
  font-weight: 500;
}

.detail-row__value {
  font-size: 0.875rem;
  text-align: right;
}

.steps-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.step-card {
  padding: 16px;
}

/* ── Flow graph ─────────────────────── */

.flow-card {
  padding: 20px;
}

.flow-legend {
  display: flex;
  gap: 24px;
  margin-bottom: 16px;
  font-size: 0.78rem;
  color: var(--text-secondary);
}

.flow-legend__item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.flow-canvas {
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: #fafbfc;
  padding: 8px;
}

.flow-lane-header {
  font-size: 11px;
  font-weight: 700;
  fill: var(--text-secondary, #475569);
  text-transform: uppercase;
  letter-spacing: 0.6px;
}

.flow-node__name {
  font-size: 13px;
  font-weight: 600;
  fill: var(--text-primary, #1e293b);
}

.flow-node__detail {
  font-size: 11px;
  fill: var(--text-secondary, #64748b);
}

.flow-node__dispatch {
  font-size: 9px;
  fill: var(--text-secondary, #94a3b8);
  font-weight: 500;
}

.flow-node__root {
  font-size: 9px;
  fill: #2563eb;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.flow-seq-label {
  font-size: 9px;
  font-weight: 600;
  fill: #475569;
}

.flow-thread-label {
  font-size: 9px;
  fill: #8b5cf6;
  font-weight: 500;
}

.flow-node:hover rect {
  stroke: #2563eb;
  filter: drop-shadow(0 1px 4px rgba(37, 99, 235, 0.15));
}

.flow-node--current rect {
  fill: #eff6ff;
}
</style>
