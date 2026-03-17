<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { compass, formatTs, shortId, statusBadge, durationMs } from '../api/client'
import type { Run, RunStep, Span, FlowData, FlowNode, FlowEdge } from '../api/client'
import SpanTree from '../components/SpanTree.vue'

const route = useRoute()
const router = useRouter()
const threadId = ref(route.params.threadId as string)
const runId = ref(route.params.runId as string)

const run = ref<Run | null>(null)
const steps = ref<RunStep[]>([])
const config = ref<any>(null)
const spans = ref<Span[]>([])
const flow = ref<FlowData>({ nodes: [], edges: [], has_flow: false })
const loading = ref(true)
const error = ref('')
const tab = ref<'details' | 'steps' | 'config' | 'trace' | 'flow'>('details')

async function fetchData() {
  loading.value = true
  error.value = ''
  run.value = null
  steps.value = []
  config.value = null
  spans.value = []
  flow.value = { nodes: [], edges: [], has_flow: false }

  try {
    const runs = await compass.listRuns(threadId.value)
    run.value = runs.find((r) => r.id === runId.value) || null

    const [s, c, t, f] = await Promise.allSettled([
      compass.listRunSteps(threadId.value, runId.value),
      compass.getRunConfig(threadId.value, runId.value),
      compass.getRunTrace(runId.value),
      compass.getRunFlow(runId.value),
    ])

    if (s.status === 'fulfilled') steps.value = s.value
    if (c.status === 'fulfilled') config.value = c.value
    if (t.status === 'fulfilled') spans.value = t.value
    if (f.status === 'fulfilled') flow.value = f.value
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

/* ── Swim-lane flow layout ─────────────────────────────── */

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
        <div class="tab" :class="{ 'tab--active': tab === 'details' }" @click="tab = 'details'">Details</div>
        <div class="tab" :class="{ 'tab--active': tab === 'steps' }" @click="tab = 'steps'">Steps ({{ steps.length }})</div>
        <div v-if="flow.has_flow" class="tab" :class="{ 'tab--active': tab === 'flow' }" @click="tab = 'flow'">Flow</div>
        <div class="tab" :class="{ 'tab--active': tab === 'config' }" @click="tab = 'config'">Config</div>
        <div class="tab" :class="{ 'tab--active': tab === 'trace' }" @click="tab = 'trace'">Trace ({{ spans.length }})</div>
      </div>

      <!-- Details tab -->
      <div v-if="tab === 'details'" class="card">
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
      </div>

      <!-- Steps tab -->
      <div v-if="tab === 'steps'">
        <div v-if="steps.length === 0" class="empty-state">
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
        <template v-if="config">
          <div class="json-view">{{ JSON.stringify(config, null, 2) }}</div>
        </template>
        <div v-else class="empty-state">
          <div class="empty-state__icon"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg></div>
          <div class="empty-state__title">No config snapshot</div>
        </div>
      </div>

      <!-- Trace tab -->
      <div v-if="tab === 'trace'">
        <SpanTree :spans="spans" />
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
      </div>
    </template>
  </div>
</template>

<style scoped>
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
