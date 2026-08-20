<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { compass, formatTs, shortId, statusBadge, durationMs } from '../api/client'
import type { Run, RunStep, FlowData, FlowNode, FlowEdge, Message } from '../api/client'
import MessageBubble from '../components/MessageBubble.vue'

const route = useRoute()
const router = useRouter()
const threadId = ref(route.params.threadId as string)
const runId = ref(route.params.runId as string)

const run = ref<Run | null>(null)
const steps = ref<RunStep[]>([])
const config = ref<any>(null)
const flow = ref<FlowData>({ nodes: [], edges: [], has_flow: false })
const messages = ref<Message[]>([])
const loading = ref(true)
const error = ref('')
const tab = ref<'io' | 'details' | 'steps' | 'config' | 'flow'>('io')

// Per-run step cache for the workflow inspector side panel.
// Keyed by run_id. The current run is preloaded from `steps` once it loads.
const runStepsByRun = ref<Record<string, RunStep[]>>({})
const selectedNodeId = ref<string | null>(null)
const loadingStepsFor = ref<Set<string>>(new Set())

// Selected step state. Kept separate per context so a click in the side
// panel mini-graph doesn't blow away the user's selection in the Steps tab
// (and vice-versa).
const selectedStepId = ref<string | null>(null)        // Steps tab
const selectedPanelStepId = ref<string | null>(null)   // Workflow side panel

function selectStep(step: RunStep) {
  selectedStepId.value = step.id
}

function selectPanelStep(step: RunStep) {
  selectedPanelStepId.value = step.id
  // If the clicked step belongs to the current run, jump to the Steps tab
  // and surface its details there.
  if (steps.value.some(s => s.id === step.id)) {
    selectedStepId.value = step.id
    tab.value = 'steps'
  } else if (selectedNode.value?.thread_id) {
    // Otherwise navigate to the run that owns the step, deep-linking to the
    // Steps tab with this step pre-selected.
    router.push({
      path: `/threads/${selectedNode.value.thread_id}/runs/${selectedNode.value.id}`,
      query: { tab: 'steps', step: step.id },
    })
  }
}

async function ensureStepsLoaded(node: { id: string; thread_id?: string | null }) {
  if (node.id in runStepsByRun.value) return
  if (!node.thread_id) {
    runStepsByRun.value = { ...runStepsByRun.value, [node.id]: [] }
    return
  }
  const next = new Set(loadingStepsFor.value)
  next.add(node.id)
  loadingStepsFor.value = next
  try {
    const list = await compass.listRunSteps(node.thread_id, node.id)
    runStepsByRun.value = { ...runStepsByRun.value, [node.id]: list }
  } catch {
    runStepsByRun.value = { ...runStepsByRun.value, [node.id]: [] }
  } finally {
    const cleared = new Set(loadingStepsFor.value)
    cleared.delete(node.id)
    loadingStepsFor.value = cleared
  }
}

async function selectRunNode(node: { id: string; thread_id?: string | null }) {
  selectedNodeId.value = node.id
  await ensureStepsLoaded(node)
}

function clearSelection() {
  selectedNodeId.value = null
}

async function fetchData() {
  loading.value = true
  error.value = ''
  run.value = null
  steps.value = []
  config.value = null
  flow.value = { nodes: [], edges: [], has_flow: false }
  messages.value = []
  runStepsByRun.value = {}
  selectedNodeId.value = null
  loadingStepsFor.value = new Set()
  selectedStepId.value = null
  selectedPanelStepId.value = null

  try {
    const runs = await compass.listRuns(threadId.value)
    run.value = runs.find((r) => r.id === runId.value) || null

    const [s, c, f, m] = await Promise.allSettled([
      compass.listRunSteps(threadId.value, runId.value),
      compass.getRunConfig(threadId.value, runId.value),
      compass.getRunFlow(runId.value),
      compass.listMessages(threadId.value),
    ])

    if (s.status === 'fulfilled') steps.value = s.value
    if (c.status === 'fulfilled') config.value = c.value
    if (f.status === 'fulfilled') flow.value = f.value
    if (m.status === 'fulfilled') messages.value = m.value

    // Workflow inspector: seed cache for the current run and auto-select it
    // so the side panel opens with its @step graph immediately visible.
    runStepsByRun.value = { ...runStepsByRun.value, [runId.value]: steps.value }
    selectedNodeId.value = runId.value

    // Honour deep-link query params (?tab=steps&step=<id>) so cross-run
    // navigation from the workflow side panel lands on the right step.
    const qTab = route.query.tab as string | undefined
    const qStep = route.query.step as string | undefined
    if (qTab === 'steps' || qTab === 'io' || qTab === 'details' || qTab === 'config' || qTab === 'flow') {
      tab.value = qTab
    } else if (qTab === 'trace') {
      router.replace(`/traces/${runId.value}`)
    }
    if (qStep && steps.value.some(s => s.id === qStep)) {
      selectedStepId.value = qStep
    } else {
      selectedStepId.value = steps.value.length ? steps.value[0].id : null
    }
  } catch (e: any) {
    error.value = e.message
  } finally {
    loading.value = false
  }
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
    { label: 'Created', value: formatTs(r.created_at) },
    { label: 'Started', value: formatTs(r.started_at ?? 0) },
    { label: 'Completed', value: formatTs(r.completed_at ?? 0) },
    { label: 'Duration', value: durationMs(r.started_at, r.completed_at ?? r.failed_at) },
  ]
})

// ── Steps summary (Details tab) ──────────────────────
interface StepStatusBucket {
  status: string
  count: number
  color: string
}

const stepStatusSummary = computed<StepStatusBucket[]>(() => {
  const buckets = new Map<string, number>()
  for (const s of steps.value) {
    buckets.set(s.status, (buckets.get(s.status) ?? 0) + 1)
  }
  // Stable, intuitive ordering
  const order = ['completed', 'in_progress', 'queued', 'failed', 'cancelled', 'expired']
  const known = order
    .filter(k => buckets.has(k))
    .map(k => ({ status: k, count: buckets.get(k)!, color: flowStatusColor(k) }))
  const others = [...buckets.keys()]
    .filter(k => !order.includes(k))
    .map(k => ({ status: k, count: buckets.get(k)!, color: flowStatusColor(k) }))
  return [...known, ...others]
})

// ── Workflow summary (Details tab, only when has_flow) ──
const workflowSummary = computed(() => {
  if (!flow.value.has_flow || flow.value.nodes.length <= 1) return null
  const nodes = flow.value.nodes
  const buckets = new Map<string, number>()
  for (const n of nodes) buckets.set(n.status, (buckets.get(n.status) ?? 0) + 1)
  const order = ['completed', 'in_progress', 'queued', 'failed', 'cancelled']
  const byStatus = order
    .filter(k => buckets.has(k))
    .map(k => ({ status: k, count: buckets.get(k)!, color: flowStatusColor(k) }))
  return {
    total: nodes.length,
    handovers: flow.value.edges.filter(e => e.type === 'handover').length,
    calls: flow.value.edges.filter(e => e.type !== 'handover').length,
    byStatus,
  }
})

/* ── I/O message split ─────────────────────────────────── */

// Messages whose run_id matches this run → output
const outputMessages = computed(() =>
  messages.value.filter((m) => m.run_id === runId.value),
)

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
    (m) => m.created_at < cutoff && m.run_id !== runId.value,
  )
})

const hasIo = computed(() => outputMessages.value.length > 0 || inputMessage.value !== null)

const LANE_W = 200
const LANE_GAP = 60
const NODE_W = 172
const NODE_H = 68
// Max characters that fit on the agent-name line at 13px/600 weight inside a
// NODE_W-wide box (text starts at x+26 and we leave a small right margin).
const NODE_NAME_MAX_CHARS = 18

function truncateNodeName(name: string | null | undefined): string {
  const s = (name ?? '').toString()
  if (s.length <= NODE_NAME_MAX_CHARS) return s
  return s.slice(0, NODE_NAME_MAX_CHARS - 1) + '…'
}
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

const flowLayout = computed(() => {
  const { nodes, edges, has_flow } = flow.value
  if (!nodes.length || !has_flow) {
    return {
      nodes: [] as LayoutNode[],
      edges: [] as FlowEdge[],
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

  return { nodes: layoutNodes, edges: sorted, lanes, spines, threadGroups, width, height }
})

function flowNodeById(id: string): LayoutNode | undefined {
  return flowLayout.value.nodes.find(n => n.id === id)
}

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

function flowEdgeDash(type: string): string {
  return type === 'handover' ? 'none' : '6,4'
}

function flowEdgeColor(type: string): string {
  return type === 'handover' ? '#7c3aed' : '#64748b'
}

function flowDuration(ms: number | null): string {
  if (ms == null) return ''
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function flowEdgePath(edge: FlowEdge): string {
  const src = flowNodeById(edge.source)
  const tgt = flowNodeById(edge.target)
  if (!src || !tgt) return ''

  const spineX = src.x + NODE_W / 2
  const tgtCY = tgt.y + NODE_H / 2

  // Same lane: straight down from parent bottom to child top
  if (src.lane === tgt.lane) {
    return `M ${spineX} ${src.y + NODE_H} L ${spineX} ${tgt.y}`
  }

  // Cross-lane: horizontal from spine to target node edge
  if (tgt.lane > src.lane) {
    return `M ${spineX} ${tgtCY} L ${tgt.x} ${tgtCY}`
  }
  return `M ${spineX} ${tgtCY} L ${tgt.x + NODE_W} ${tgtCY}`
}

function flowEdgeMid(edge: FlowEdge): { x: number; y: number } {
  const src = flowNodeById(edge.source)
  const tgt = flowNodeById(edge.target)
  if (!src || !tgt) return { x: 0, y: 0 }

  const spineX = src.x + NODE_W / 2

  if (src.lane === tgt.lane) {
    return { x: spineX, y: (src.y + NODE_H + tgt.y) / 2 }
  }

  const tgtEdgeX = tgt.lane > src.lane ? tgt.x : tgt.x + NODE_W
  return { x: (spineX + tgtEdgeX) / 2, y: tgt.y + NODE_H / 2 }
}

function navigateToRun(node: LayoutNode) {
  if (node.thread_id) {
    router.push(`/threads/${node.thread_id}/runs/${node.id}`)
  }
}

// ── Side-panel data ───────────────────────────────────

const selectedNode = computed<LayoutNode | null>(() => {
  if (!selectedNodeId.value) return null
  return flowLayout.value.nodes.find(n => n.id === selectedNodeId.value) || null
})

const selectedNodeSteps = computed<RunStep[]>(() => {
  if (!selectedNodeId.value) return []
  return runStepsByRun.value[selectedNodeId.value] || []
})

const selectedNodeLoading = computed(() =>
  selectedNodeId.value ? loadingStepsFor.value.has(selectedNodeId.value) : false,
)

// Mini step-flow diagram inside the side panel.
const STEP_NODE_W = 252
const STEP_NODE_H = 44
const STEP_GAP = 18
const STEP_PAD = 12

// Larger nodes for the Steps tab graph.
const TAB_STEP_NODE_W = 360
const TAB_STEP_NODE_H = 56
const TAB_STEP_GAP = 22
const TAB_STEP_PAD = 16

interface StepLayoutNode {
  step: RunStep
  x: number
  y: number
}

interface StepGraph {
  nodes: StepLayoutNode[]
  width: number
  height: number
  nodeW: number
  nodeH: number
}

function buildStepLayout(items: RunStep[], nodeW: number, nodeH: number, gap: number, pad: number): StepGraph {
  if (!items.length) {
    return { nodes: [], width: nodeW + pad * 2, height: 0, nodeW, nodeH }
  }
  const nodes: StepLayoutNode[] = items.map((step, i) => ({
    step,
    x: pad,
    y: pad + i * (nodeH + gap),
  }))
  const height = pad * 2 + items.length * nodeH + (items.length - 1) * gap
  return { nodes, width: nodeW + pad * 2, height, nodeW, nodeH }
}

const stepGraphLayout = computed(() =>
  buildStepLayout(selectedNodeSteps.value, STEP_NODE_W, STEP_NODE_H, STEP_GAP, STEP_PAD),
)

const stepsTabGraphLayout = computed(() =>
  buildStepLayout(steps.value, TAB_STEP_NODE_W, TAB_STEP_NODE_H, TAB_STEP_GAP, TAB_STEP_PAD),
)

// Selected-step lookups. The Steps tab always shows steps from the current run;
// the side panel shows steps from the currently selected workflow node.
const selectedStepInTab = computed<RunStep | null>(() => {
  if (!selectedStepId.value) return null
  return steps.value.find(s => s.id === selectedStepId.value) || null
})

const selectedStepInPanel = computed<RunStep | null>(() => {
  if (!selectedPanelStepId.value) return null
  return selectedNodeSteps.value.find(s => s.id === selectedPanelStepId.value) || null
})

function stepLabel(step: RunStep): string {
  const meta = stepMetadata(step)
  if (meta.step_type === 'langgraph_node' && typeof meta.node_name === 'string') {
    return meta.node_name
  }
  const d = step.step_details
  if (!d) return step.type
  if (d.type === 'step') {
    // Strip "ClassName." prefix that the @step decorator records via __qualname__
    // so the label shows just the step's own name (e.g. "validate_destination").
    const raw = d.name || 'step'
    const dot = raw.lastIndexOf('.')
    return dot >= 0 ? raw.slice(dot + 1) : raw
  }
  if (d.type === 'tool_calls') {
    const calls = (d.tool_calls || [])
    const name = calls[0]?.function?.name || calls[0]?.type || 'tool'
    return calls.length > 1 ? `${name} (+${calls.length - 1})` : name
  }
  if (d.type === 'message_creation') return 'message'
  return step.type
}

// Full label (incl. ClassName prefix) for tooltips.
function stepLabelFull(step: RunStep): string {
  const d = step.step_details
  if (d && d.type === 'step') return d.name || 'step'
  return stepLabel(step)
}

// Visual truncation for SVG labels (SVG text doesn't support CSS ellipsis).
function truncateLabel(text: string, max: number): string {
  const s = (text ?? '').toString()
  return s.length <= max ? s : s.slice(0, max - 1) + '…'
}

function stepBadgeColor(step: RunStep): string {
  const meta = stepMetadata(step)
  if (meta.step_type === 'langgraph_node' || meta.framework === 'langgraph') return '#7c3aed' // violet
  const t = step.step_details?.type ?? step.type
  switch (t) {
    case 'step': return '#0d9488' // teal
    case 'tool_calls': return '#d97706' // amber
    case 'message_creation': return '#2563eb' // blue
    default: return '#64748b'
  }
}

function stepBadgeLetter(step: RunStep): string {
  const meta = stepMetadata(step)
  if (meta.step_type === 'langgraph_node' || meta.framework === 'langgraph') return 'L'
  const t = step.step_details?.type ?? step.type
  switch (t) {
    case 'step': return 'S'
    case 'tool_calls': return 'T'
    case 'message_creation': return 'M'
    default: return '?'
  }
}

function stepTypeName(step: RunStep): string {
  const meta = stepMetadata(step)
  if (meta.step_type === 'langgraph_node' || meta.framework === 'langgraph') return 'langgraph node'
  const t = step.step_details?.type ?? step.type
  if (t === 'step') return '@step'
  if (t === 'tool_calls') return 'tool call'
  if (t === 'message_creation') return 'message'
  return String(t)
}

function stepMetadata(step: RunStep): Record<string, any> {
  const meta = (step as any)?.metadata
  return meta && typeof meta === 'object' ? meta : {}
}

function isLangGraphStep(step: RunStep | null): boolean {
  if (!step) return false
  const meta = stepMetadata(step)
  return meta.step_type === 'langgraph_node' || meta.framework === 'langgraph'
}

function hasValue(v: any): boolean {
  return !(v === null || v === undefined || v === '')
}

function prettyValue(v: any): string {
  if (!hasValue(v)) return '—'
  if (typeof v === 'string') return v
  try {
    return JSON.stringify(v, null, 2)
  } catch {
    return String(v)
  }
}

function langGraphNodeName(step: RunStep): string {
  const meta = stepMetadata(step)
  if (typeof meta.node_name === 'string' && meta.node_name) return meta.node_name
  const d = step.step_details
  if (d?.type === 'step' && typeof d.name === 'string') return d.name
  return stepLabel(step)
}

function toolCallsForStep(step: RunStep): Array<any> {
  const d = step.step_details
  if (!d || d.type !== 'tool_calls' || !Array.isArray(d.tool_calls)) return []
  return d.tool_calls
}

// When the user selects a workflow node, auto-select the first step of that
// node's run once its steps have loaded so the panel's detail section is
// populated immediately.
watch(selectedNodeSteps, (next) => {
  if (!next.length) {
    selectedPanelStepId.value = null
    return
  }
  const stillValid = selectedPanelStepId.value && next.some(s => s.id === selectedPanelStepId.value)
  if (!stillValid) selectedPanelStepId.value = next[0].id
})
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

    <div v-if="loading" class="loading-center"><div class="spinner"></div></div>
    <div v-else-if="error" class="card" style="color: var(--error)">{{ error }}</div>

    <template v-else>
      <div class="tabs">
        <div class="tab" :class="{ 'tab--active': tab === 'io' }" @click="tab = 'io'">Input / Output</div>
        <div class="tab" :class="{ 'tab--active': tab === 'details' }" @click="tab = 'details'">Details</div>
        <div class="tab" :class="{ 'tab--active': tab === 'steps' }" @click="tab = 'steps'">Steps ({{ steps.length }})</div>
        <div v-if="flow.has_flow" class="tab" :class="{ 'tab--active': tab === 'flow' }" @click="tab = 'flow'">Workflow</div>
        <div class="tab" :class="{ 'tab--active': tab === 'config' }" @click="tab = 'config'">Config</div>
      </div>

      <!-- I/O tab -->
      <div v-if="tab === 'io'">
        <div v-if="!hasIo" class="empty-state">
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
      <div v-if="tab === 'details'" class="details-grid">
        <!-- Status hero -->
        <div class="card details-hero">
          <div class="details-hero__status">
            <span
              class="details-hero__dot"
              :style="{ background: flowStatusColor(run?.status ?? '') }"
            ></span>
            <div>
              <div class="details-hero__label">Status</div>
              <div class="details-hero__value">{{ run?.status ?? '—' }}</div>
            </div>
          </div>
          <div class="details-hero__divider"></div>
          <div class="details-hero__metric">
            <div class="details-hero__label">Duration</div>
            <div class="details-hero__value">
              {{ durationMs(run?.started_at ?? null, run?.completed_at ?? run?.failed_at ?? null) || '—' }}
            </div>
          </div>
          <div class="details-hero__divider"></div>
          <div class="details-hero__metric">
            <div class="details-hero__label">Steps</div>
            <div class="details-hero__value">{{ steps.length }}</div>
          </div>
          <div v-if="run?.usage?.total_tokens != null" class="details-hero__divider"></div>
          <div v-if="run?.usage?.total_tokens != null" class="details-hero__metric">
            <div class="details-hero__label">Tokens</div>
            <div class="details-hero__value">{{ run.usage.total_tokens }}</div>
          </div>
        </div>

        <!-- Steps breakdown -->
        <div v-if="steps.length" class="card">
          <div class="section__title">Steps by status</div>
          <div class="status-chips">
            <div
              v-for="b in stepStatusSummary"
              :key="b.status"
              class="status-chip"
            >
              <span class="status-chip__dot" :style="{ background: b.color }"></span>
              <span class="status-chip__count">{{ b.count }}</span>
              <span class="status-chip__label">{{ b.status }}</span>
            </div>
          </div>
          <!-- Stacked bar -->
          <div class="status-bar">
            <div
              v-for="b in stepStatusSummary"
              :key="'bar-' + b.status"
              class="status-bar__seg"
              :style="{ flex: b.count, background: b.color }"
              :title="`${b.count} ${b.status}`"
            ></div>
          </div>
        </div>

        <!-- Workflow summary (only multi-agent runs) -->
        <div v-if="workflowSummary" class="card">
          <div class="section__title">Workflow</div>
          <div class="workflow-summary">
            <div class="workflow-summary__metric">
              <div class="workflow-summary__value">{{ workflowSummary.total }}</div>
              <div class="workflow-summary__label">agents</div>
            </div>
            <div class="workflow-summary__metric">
              <div class="workflow-summary__value">{{ workflowSummary.calls }}</div>
              <div class="workflow-summary__label">call_agent</div>
            </div>
            <div class="workflow-summary__metric">
              <div class="workflow-summary__value">{{ workflowSummary.handovers }}</div>
              <div class="workflow-summary__label">handovers</div>
            </div>
          </div>
          <div class="status-chips" style="margin-top: 12px">
            <div
              v-for="b in workflowSummary.byStatus"
              :key="'wf-' + b.status"
              class="status-chip"
            >
              <span class="status-chip__dot" :style="{ background: b.color }"></span>
              <span class="status-chip__count">{{ b.count }}</span>
              <span class="status-chip__label">{{ b.status }}</span>
            </div>
          </div>
        </div>

        <!-- Timeline -->
        <div class="card">
          <div class="section__title">Timeline</div>
          <div class="detail-grid">
            <div v-for="item in runMeta" :key="item.label" class="detail-row">
              <span class="detail-row__label">{{ item.label }}</span>
              <span class="detail-row__value">{{ item.value || '—' }}</span>
            </div>
          </div>
        </div>

        <!-- Error (only if present) -->
        <div v-if="run?.last_error" class="card">
          <div class="section__title" style="color: var(--error)">Error</div>
          <div class="json-view" style="color: var(--error)">{{ JSON.stringify(run.last_error, null, 2) }}</div>
        </div>
      </div>

      <!-- Steps tab -->
      <div v-if="tab === 'steps'">
        <div v-if="steps.length === 0" class="empty-state">
          <div class="empty-state__icon"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg></div>
          <div class="empty-state__title">No run steps</div>
        </div>
        <div v-else class="steps-layout">
          <!-- Step flow diagram -->
          <div class="steps-graph card">
            <svg
              :width="stepsTabGraphLayout.width"
              :height="stepsTabGraphLayout.height"
              :viewBox="`0 0 ${stepsTabGraphLayout.width} ${stepsTabGraphLayout.height}`"
            >
              <defs>
                <marker id="step-arrow-tab" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
                  <path d="M0,0 L8,3 L0,6" fill="#94a3b8"/>
                </marker>
              </defs>
              <template v-for="(sn, i) in stepsTabGraphLayout.nodes" :key="'tab-edge-' + sn.step.id">
                <line
                  v-if="i < stepsTabGraphLayout.nodes.length - 1"
                  :x1="sn.x + stepsTabGraphLayout.nodeW / 2"
                  :y1="sn.y + stepsTabGraphLayout.nodeH"
                  :x2="sn.x + stepsTabGraphLayout.nodeW / 2"
                  :y2="stepsTabGraphLayout.nodes[i + 1].y"
                  stroke="#cbd5e1"
                  stroke-width="1.5"
                  marker-end="url(#step-arrow-tab)"
                />
              </template>
              <g
                v-for="sn in stepsTabGraphLayout.nodes"
                :key="sn.step.id"
                class="step-node"
                :class="{ 'step-node--selected': sn.step.id === selectedStepId }"
                @click="selectStep(sn.step)"
                style="cursor: pointer"
              >
                <rect
                  :x="sn.x"
                  :y="sn.y"
                  :width="stepsTabGraphLayout.nodeW"
                  :height="stepsTabGraphLayout.nodeH"
                  rx="8"
                  ry="8"
                  fill="white"
                  :stroke="sn.step.id === selectedStepId ? '#2563eb' : '#e2e8f0'"
                  :stroke-width="sn.step.id === selectedStepId ? 2.5 : 1.5"
                />
                <rect
                  :x="sn.x + 12"
                  :y="sn.y + 12"
                  width="32"
                  height="32"
                  rx="6"
                  ry="6"
                  :fill="stepBadgeColor(sn.step)"
                />
                <text
                  :x="sn.x + 28"
                  :y="sn.y + 33"
                  text-anchor="middle"
                  class="step-node__letter step-node__letter--lg"
                >{{ stepBadgeLetter(sn.step) }}</text>
                <text
                  :x="sn.x + 56"
                  :y="sn.y + 26"
                  class="step-node__label step-node__label--lg"
                >{{ truncateLabel(stepLabel(sn.step), 36) }}<title>{{ stepLabelFull(sn.step) }}</title></text>
                <text
                  :x="sn.x + 56"
                  :y="sn.y + 44"
                  class="step-node__type"
                >{{ stepTypeName(sn.step) }} · {{ sn.step.status }}</text>
                <circle
                  :cx="sn.x + stepsTabGraphLayout.nodeW - 16"
                  :cy="sn.y + stepsTabGraphLayout.nodeH / 2"
                  r="5"
                  :fill="flowStatusColor(sn.step.status)"
                />
              </g>
            </svg>
          </div>

          <!-- Step detail panel -->
          <aside class="step-detail card">
            <template v-if="selectedStepInTab">
              <header class="step-detail__header">
                <div class="step-detail__title-row">
                  <span
                    class="step-detail__type-badge"
                    :style="{ background: stepBadgeColor(selectedStepInTab) }"
                  >{{ stepBadgeLetter(selectedStepInTab) }}</span>
                  <h3 class="step-detail__title">{{ stepLabel(selectedStepInTab) }}</h3>
                </div>
                <div class="step-detail__meta">
                  <span class="badge" :class="statusBadge(selectedStepInTab.status)">{{ selectedStepInTab.status }}</span>
                  <span class="badge badge--neutral">{{ stepTypeName(selectedStepInTab) }}</span>
                </div>
                <div class="step-detail__id mono">{{ selectedStepInTab.id }}</div>
              </header>

              <div class="step-detail__section-title">Created</div>
              <div class="step-detail__value">{{ formatTs(selectedStepInTab.created_at) }}</div>

              <template v-if="selectedStepInTab.usage">
                <div class="step-detail__section-title">Tokens</div>
                <div class="step-detail__value">
                  {{ selectedStepInTab.usage.total_tokens }}
                  <span class="step-detail__muted">
                    ({{ selectedStepInTab.usage.prompt_tokens }}p / {{ selectedStepInTab.usage.completion_tokens }}c)
                  </span>
                </div>
              </template>

              <div class="step-detail__section-title">Details</div>

              <template v-if="isLangGraphStep(selectedStepInTab)">
                <div class="step-structured">
                  <div class="step-kv-grid">
                    <div>
                      <div class="step-kv__label">Node</div>
                      <div class="step-kv__value">{{ langGraphNodeName(selectedStepInTab) }}</div>
                    </div>
                    <div>
                      <div class="step-kv__label">Event</div>
                      <div class="step-kv__value">{{ stepMetadata(selectedStepInTab).event_name || '—' }}</div>
                    </div>
                    <div>
                      <div class="step-kv__label">Source</div>
                      <div class="step-kv__value">{{ stepMetadata(selectedStepInTab).event_source || '—' }}</div>
                    </div>
                  </div>

                  <template v-if="hasValue(stepMetadata(selectedStepInTab).state)">
                    <div class="step-detail__section-title">State Snapshot (metadata)</div>
                    <pre class="json-view">{{ prettyValue(stepMetadata(selectedStepInTab).state) }}</pre>
                  </template>

                  <template v-if="hasValue(selectedStepInTab.step_details?.input)">
                    <div class="step-detail__section-title">Node Input</div>
                    <pre class="json-view">{{ prettyValue(selectedStepInTab.step_details.input) }}</pre>
                  </template>

                  <template v-if="hasValue(selectedStepInTab.step_details?.output)">
                    <div class="step-detail__section-title">Node Output</div>
                    <pre class="json-view">{{ prettyValue(selectedStepInTab.step_details.output) }}</pre>
                  </template>
                </div>
              </template>

              <template v-else-if="selectedStepInTab.step_details?.type === 'step'">
                <div class="step-structured">
                  <div class="step-kv-grid">
                    <div>
                      <div class="step-kv__label">Step Name</div>
                      <div class="step-kv__value">{{ selectedStepInTab.step_details.name || stepLabel(selectedStepInTab) }}</div>
                    </div>
                  </div>
                  <template v-if="hasValue(selectedStepInTab.step_details.input)">
                    <div class="step-detail__section-title">Input</div>
                    <pre class="json-view">{{ prettyValue(selectedStepInTab.step_details.input) }}</pre>
                  </template>
                  <template v-if="hasValue(selectedStepInTab.step_details.output)">
                    <div class="step-detail__section-title">Output</div>
                    <pre class="json-view">{{ prettyValue(selectedStepInTab.step_details.output) }}</pre>
                  </template>
                </div>
              </template>

              <template v-else-if="selectedStepInTab.step_details?.type === 'tool_calls'">
                <div class="step-structured">
                  <div class="step-kv-grid">
                    <div>
                      <div class="step-kv__label">Tool Calls</div>
                      <div class="step-kv__value">{{ toolCallsForStep(selectedStepInTab).length }}</div>
                    </div>
                  </div>
                  <div v-for="(tc, idx) in toolCallsForStep(selectedStepInTab)" :key="tc.id || idx" class="tool-call-card">
                    <div class="tool-call-card__title">
                      {{ tc.function?.name || tc.type || `tool_call_${idx + 1}` }}
                    </div>
                    <template v-if="hasValue(tc.function?.arguments)">
                      <div class="step-detail__section-title">Arguments</div>
                      <pre class="json-view">{{ prettyValue(tc.function.arguments) }}</pre>
                    </template>
                    <template v-if="hasValue(tc)">
                      <details class="raw-toggle">
                        <summary>Raw tool call</summary>
                        <pre class="json-view">{{ prettyValue(tc) }}</pre>
                      </details>
                    </template>
                  </div>
                </div>
              </template>

              <template v-else-if="selectedStepInTab.step_details?.type === 'message_creation'">
                <div class="step-structured">
                  <div class="step-kv-grid">
                    <div>
                      <div class="step-kv__label">Message ID</div>
                      <div class="step-kv__value mono">{{ selectedStepInTab.step_details.message_creation?.message_id || '—' }}</div>
                    </div>
                  </div>
                </div>
              </template>

              <details class="raw-toggle" open>
                <summary>Raw step_details</summary>
                <div class="json-view">{{ JSON.stringify(selectedStepInTab.step_details, null, 2) }}</div>
              </details>

              <template v-if="selectedStepInTab.last_error">
                <div class="step-detail__section-title">Error</div>
                <div class="json-view" style="color: var(--error)">{{ JSON.stringify(selectedStepInTab.last_error, null, 2) }}</div>
              </template>
            </template>
            <div v-else class="step-detail__empty">Select a step to see its details.</div>
          </aside>
        </div>
      </div>

      <!-- Config tab -->
      <div v-if="tab === 'config'" class="card">
        <template v-if="config">
          <div class="json-view">{{ JSON.stringify(config, null, 2) }}</div>
        </template>
        <div v-else class="empty-state">
          <div class="empty-state__icon"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg></div>
          <div class="empty-state__title">No config snapshot</div>
        </div>
      </div>

      <!-- Flow tab -->
      <div v-if="tab === 'flow'" class="card flow-card">
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
        <div class="flow-body">
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

            <!-- Edges (arrows with sequence badges) -->
            <g v-for="edge in flowLayout.edges" :key="`edge-${edge.source}-${edge.target}`">
              <path
                :d="flowEdgePath(edge)"
                fill="none"
                :stroke="flowEdgeColor(edge.type)"
                :stroke-dasharray="flowEdgeDash(edge.type)"
                stroke-width="2"
                :marker-end="edge.type === 'handover' ? 'url(#arrow-handover)' : 'url(#arrow-call)'"
              />
              <!-- Sequence badge -->
              <circle
                :cx="flowEdgeMid(edge).x" :cy="flowEdgeMid(edge).y"
                r="10" fill="white" stroke="#94a3b8" stroke-width="1"
              />
              <text
                :x="flowEdgeMid(edge).x" :y="flowEdgeMid(edge).y + 3.5"
                text-anchor="middle" class="flow-seq-label"
              >{{ edge.sequence }}</text>
            </g>

            <!-- Nodes -->
            <g
              v-for="node in flowLayout.nodes"
              :key="node.id"
              class="flow-node"
              :class="{
                'flow-node--current': node.id === runId,
                'flow-node--selected': node.id === selectedNodeId,
              }"
              @click="selectRunNode(node)"
              style="cursor: pointer"
            >
              <rect
                :x="node.x"
                :y="node.y"
                :width="NODE_W"
                :height="NODE_H"
                rx="8"
                ry="8"
                :stroke="node.id === selectedNodeId ? '#2563eb' : (node.id === runId ? '#60a5fa' : '#d1d5db')"
                :stroke-width="node.id === selectedNodeId ? 2.5 : 1.5"
                fill="white"
              />
              <!-- Status dot -->
              <circle
                :cx="node.x + 14"
                :cy="node.y + 20"
                r="5"
                :fill="flowStatusColor(node.status)"
              />
              <!-- Agent name (truncated to fit node width) -->
              <text
                :x="node.x + 26"
                :y="node.y + 24"
                class="flow-node__name"
              >{{ truncateNodeName(node.agent_name) }}<title>{{ node.agent_name }}</title></text>
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

        <!-- Side panel: agent details + step flow diagram -->
        <aside v-if="selectedNode" class="flow-panel">
          <header class="flow-panel__header">
            <div class="flow-panel__title-row">
              <span
                class="flow-panel__status-dot"
                :style="{ background: flowStatusColor(selectedNode.status) }"
              ></span>
              <h3 class="flow-panel__title">{{ selectedNode.agent_name }}</h3>
              <button class="flow-panel__close" @click="clearSelection" aria-label="Close">×</button>
            </div>
            <div class="flow-panel__meta">
              <span class="badge" :class="statusBadge(selectedNode.status)">{{ selectedNode.status }}</span>
              <span v-if="selectedNode.duration_ms != null" class="flow-panel__meta-item">
                {{ flowDuration(selectedNode.duration_ms) }}
              </span>
              <span v-if="selectedNode.dispatch_type && !selectedNode.is_root" class="flow-panel__meta-item">
                {{ selectedNode.dispatch_type === 'handover' ? 'handover' : 'call_agent' }}
              </span>
              <span v-if="selectedNode.is_root" class="flow-panel__meta-item flow-panel__meta-item--accent">root</span>
            </div>
            <div class="flow-panel__id mono">{{ shortId(selectedNode.id) }}</div>
            <button
              v-if="selectedNode.thread_id && selectedNode.id !== runId"
              class="btn btn--sm flow-panel__open-btn"
              @click="navigateToRun(selectedNode)"
            >Open run detail →</button>
          </header>

          <div class="flow-panel__section-title">Steps</div>

          <div v-if="selectedNodeLoading" class="flow-panel__empty">Loading steps…</div>
          <div v-else-if="selectedNodeSteps.length === 0" class="flow-panel__empty">
            No steps recorded for this run.
          </div>
          <div v-else class="flow-panel__graph">
            <svg
              :width="stepGraphLayout.width"
              :height="stepGraphLayout.height"
              :viewBox="`0 0 ${stepGraphLayout.width} ${stepGraphLayout.height}`"
            >
              <defs>
                <marker id="step-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
                  <path d="M0,0 L8,3 L0,6" fill="#94a3b8"/>
                </marker>
              </defs>

              <!-- Connectors between consecutive steps -->
              <template v-for="(sn, i) in stepGraphLayout.nodes" :key="'edge-' + sn.step.id">
                <line
                  v-if="i < stepGraphLayout.nodes.length - 1"
                  :x1="sn.x + STEP_NODE_W / 2"
                  :y1="sn.y + STEP_NODE_H"
                  :x2="sn.x + STEP_NODE_W / 2"
                  :y2="stepGraphLayout.nodes[i + 1].y"
                  stroke="#cbd5e1"
                  stroke-width="1.5"
                  marker-end="url(#step-arrow)"
                />
              </template>

              <!-- Step nodes -->
              <g
                v-for="sn in stepGraphLayout.nodes"
                :key="sn.step.id"
                class="step-node"
                :class="{ 'step-node--selected': sn.step.id === selectedPanelStepId }"
                @click="selectPanelStep(sn.step)"
                style="cursor: pointer"
              >
                <rect
                  :x="sn.x"
                  :y="sn.y"
                  :width="STEP_NODE_W"
                  :height="STEP_NODE_H"
                  rx="6"
                  ry="6"
                  fill="white"
                  :stroke="sn.step.id === selectedPanelStepId ? '#2563eb' : '#e2e8f0'"
                  :stroke-width="sn.step.id === selectedPanelStepId ? 2 : 1.5"
                />
                <!-- Type badge -->
                <rect
                  :x="sn.x + 8"
                  :y="sn.y + 8"
                  width="20"
                  height="20"
                  rx="4"
                  ry="4"
                  :fill="stepBadgeColor(sn.step)"
                />
                <text
                  :x="sn.x + 18"
                  :y="sn.y + 22"
                  text-anchor="middle"
                  class="step-node__letter"
                >{{ stepBadgeLetter(sn.step) }}</text>
                <!-- Label -->
                <text
                  :x="sn.x + 36"
                  :y="sn.y + 20"
                  class="step-node__label"
                >{{ truncateLabel(stepLabel(sn.step), 26) }}<title>{{ stepLabelFull(sn.step) }}</title></text>
                <!-- Type meta line -->
                <text
                  :x="sn.x + 36"
                  :y="sn.y + 35"
                  class="step-node__type"
                >{{ stepTypeName(sn.step) }} · {{ sn.step.status }}</text>
                <!-- Status dot -->
                <circle
                  :cx="sn.x + STEP_NODE_W - 14"
                  :cy="sn.y + STEP_NODE_H / 2"
                  r="4.5"
                  :fill="flowStatusColor(sn.step.status)"
                />
              </g>
            </svg>
          </div>
        </aside>
        </div>
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

/* ── Details tab — flair ────────────────────────────── */
.details-grid {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.details-hero {
  display: flex;
  align-items: center;
  gap: 24px;
  padding: 20px 24px;
}

.details-hero__status {
  display: flex;
  align-items: center;
  gap: 12px;
}

.details-hero__dot {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  flex-shrink: 0;
  box-shadow: 0 0 0 4px rgba(0, 0, 0, 0.04);
}

.details-hero__metric {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.details-hero__label {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted, #94a3b8);
}

.details-hero__value {
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--text-primary, #1e293b);
  text-transform: capitalize;
}

.details-hero__divider {
  width: 1px;
  align-self: stretch;
  background: var(--border);
}

.status-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 6px;
}

.status-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  background: var(--bg-surface, #f8fafc);
  border: 1px solid var(--border);
  border-radius: 999px;
  font-size: 0.78rem;
}

.status-chip__dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-chip__count {
  font-weight: 700;
  color: var(--text-primary, #1e293b);
}

.status-chip__label {
  color: var(--text-secondary, #64748b);
  text-transform: capitalize;
}

.status-bar {
  display: flex;
  height: 8px;
  border-radius: 4px;
  overflow: hidden;
  margin-top: 12px;
  background: var(--border);
}

.step-structured {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.step-kv-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 8px;
}

.step-kv__label {
  font-size: 0.68rem;
  font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.step-kv__value {
  font-size: 0.85rem;
  color: var(--text-primary);
  margin-top: 2px;
}

.tool-call-card {
  border: 1px solid var(--border);
  background: var(--bg-muted);
  border-radius: var(--radius-sm);
  padding: 10px;
}

.tool-call-card__title {
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 6px;
}

.raw-toggle {
  border-top: 1px dashed var(--border);
  margin-top: 8px;
  padding-top: 8px;
}

.raw-toggle summary {
  cursor: pointer;
  font-size: 0.75rem;
  color: var(--text-secondary);
  user-select: none;
  margin-bottom: 6px;
}

.status-bar__seg {
  height: 100%;
  transition: flex 0.2s ease;
}

.workflow-summary {
  display: flex;
  gap: 24px;
  margin-top: 6px;
}

.workflow-summary__metric {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}

.workflow-summary__value {
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--text-primary, #1e293b);
  line-height: 1.1;
}

.workflow-summary__label {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted, #94a3b8);
  margin-top: 2px;
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
  flex: 1 1 auto;
  min-width: 0;
}

.flow-body {
  display: flex;
  align-items: stretch;
  gap: 16px;
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

.flow-node--selected rect {
  filter: drop-shadow(0 2px 6px rgba(37, 99, 235, 0.25));
}

/* ── Workflow inspector side panel ─────────────────────── */
.flow-panel {
  flex: 0 0 320px;
  max-width: 360px;
  background: white;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 14px 18px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-self: flex-start;
  position: sticky;
  top: 16px;
  max-height: calc(100vh - 80px);
  overflow-y: auto;
}

.flow-panel__header {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}

.flow-panel__title-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.flow-panel__status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}

.flow-panel__title {
  font-size: 0.95rem;
  font-weight: 600;
  margin: 0;
  flex: 1;
  color: var(--text-primary, #1e293b);
}

.flow-panel__close {
  background: transparent;
  border: none;
  font-size: 1.4rem;
  line-height: 1;
  color: var(--text-muted, #94a3b8);
  cursor: pointer;
  padding: 0 4px;
}

.flow-panel__close:hover {
  color: var(--text-primary, #1e293b);
}

.flow-panel__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
  font-size: 0.75rem;
  color: var(--text-secondary, #64748b);
}

.flow-panel__meta-item--accent {
  color: #2563eb;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-size: 0.7rem;
}

.flow-panel__id {
  font-size: 0.7rem;
  color: var(--text-muted, #94a3b8);
}

.flow-panel__open-btn {
  align-self: flex-start;
}

.flow-panel__section-title {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary, #64748b);
  margin-top: 4px;
}

.flow-panel__empty {
  font-size: 0.8rem;
  color: var(--text-muted, #94a3b8);
  font-style: italic;
  padding: 10px 0;
}

.flow-panel__graph {
  margin: 0 -4px;
}

.step-node__letter {
  font-size: 11px;
  font-weight: 700;
  fill: white;
  user-select: none;
}

.step-node__label {
  font-size: 12px;
  font-weight: 600;
  fill: var(--text-primary, #1e293b);
}

.step-node__type {
  font-size: 10px;
  fill: var(--text-secondary, #94a3b8);
}

.step-node:hover rect:first-of-type {
  stroke: #cbd5e1;
}

.step-node--selected rect:first-of-type {
  filter: drop-shadow(0 2px 6px rgba(37, 99, 235, 0.18));
}

.step-node__letter--lg {
  font-size: 14px;
}

.step-node__label--lg {
  font-size: 14px;
}

/* ── Steps tab two-column layout ───────────────────────── */
.steps-layout {
  display: flex;
  align-items: flex-start;
  gap: 16px;
}

.steps-graph {
  flex: 0 0 auto;
  padding: 12px;
  background: #fafbfc;
  overflow-x: auto;
}

.step-detail {
  flex: 1 1 auto;
  min-width: 0;
  padding: 18px;
  position: sticky;
  top: 16px;
  max-height: calc(100vh - 80px);
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.step-detail__header {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}

.step-detail__title-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.step-detail__type-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 6px;
  color: white;
  font-size: 0.75rem;
  font-weight: 700;
  flex-shrink: 0;
}

.step-detail__title {
  font-size: 1rem;
  font-weight: 600;
  margin: 0;
  color: var(--text-primary, #1e293b);
}

.step-detail__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}

.step-detail__id {
  font-size: 0.7rem;
  color: var(--text-muted, #94a3b8);
  word-break: break-all;
}

.step-detail__section-title {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary, #64748b);
  margin-top: 6px;
}

.step-detail__value {
  font-size: 0.85rem;
  color: var(--text-primary, #1e293b);
}

.step-detail__muted {
  color: var(--text-muted, #94a3b8);
  font-size: 0.78rem;
}

.step-detail__empty {
  font-size: 0.85rem;
  color: var(--text-muted, #94a3b8);
  font-style: italic;
}

/* ── Workflow side-panel: step detail sub-section ──────── */
.panel-step-detail {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.panel-step-detail__title-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.panel-step-detail__title {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-primary, #1e293b);
}

.panel-step-detail__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  align-items: center;
}

.panel-step-detail__id {
  font-size: 0.68rem;
  color: var(--text-muted, #94a3b8);
  word-break: break-all;
}

.panel-step-detail__tokens {
  font-size: 0.72rem;
  color: var(--text-muted, #94a3b8);
}

.panel-step-detail__json {
  max-height: 220px;
  overflow-y: auto;
  font-size: 0.72rem;
}
</style>
