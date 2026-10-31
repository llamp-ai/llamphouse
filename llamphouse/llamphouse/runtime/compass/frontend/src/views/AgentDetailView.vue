<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { v2, until, copy } from '../api/client'
import type { AgentView } from '../api/client'
import SchemaForm from '../components/SchemaForm.vue'
import JsonView from '../components/JsonView.vue'
import Badge from '../components/Badge.vue'
import Hint from '../components/Hint.vue'

const route = useRoute()
const router = useRouter()
const name = String(route.params.name)
const agent = ref<AgentView | null>(null)
const tab = ref(String(route.query.tab || 'try'))
const error = ref('')

// try it
const input = ref('')
const inputData = ref<Record<string, any>>({})
const labels = ref('')
const starting = ref(false)
// config
const config = ref<Record<string, any>>({})
const saving = ref(false)
const saved = ref('')

async function load() {
  try {
    agent.value = await v2.agent(name)
    config.value = { ...(agent.value.config?.effective ?? {}) }
  } catch (e: any) { error.value = e.message }
}
onMounted(load)
const isWorkflow = computed(() => !!agent.value?.input_schema)

async function tryIt() {
  error.value = ''
  starting.value = true
  try {
    const metadata = labels.value.trim() ? JSON.parse(labels.value) : undefined
    const thread = await v2.newThread({ agent: name, metadata })
    await v2.start(thread.id, isWorkflow.value ? { data: inputData.value, sender: { kind: 'person', id: 'developer' } }
      : { text: input.value, sender: { kind: 'person', id: 'developer' } })
    router.push(`/threads/${thread.id}`)
  } catch (e: any) { error.value = e.message } finally { starting.value = false }
}

async function saveConfig() {
  if (!agent.value?.config) return
  saving.value = true
  saved.value = ''
  error.value = ''
  try {
    const file = agent.value.config.file ?? {}
    const changed = Object.fromEntries(Object.entries(config.value).filter(([k, v]) => JSON.stringify(v) !== JSON.stringify(file[k])))
    await v2.setConfig(name, changed, agent.value.config.version)
    saved.value = 'Saved: every process uses it from its next run.'
    await load()
  } catch (e: any) { error.value = e.message } finally { saving.value = false }
}
async function resetConfig() {
  if (!agent.value?.config) return
  await v2.setConfig(name, {}, agent.value.config.version)
  saved.value = "Back to the agent file's config."
  await load()
}
async function fire(id: string) {
  try { await v2.fire(id); saved.value = 'Fired: see Runs.'; await load() } catch (e: any) { error.value = e.message }
}
</script>

<template>
  <div v-if="agent">
    <div class="breadcrumbs"><router-link to="/agents">Agents</router-link><span>/</span><span>{{ agent.name }}</span></div>
    <div class="page-header">
      <h1>{{ agent.display_name }} <span class="pill">{{ agent.type }}</span></h1>
      <p class="page-header__subtitle">{{ agent.description || 'No description: give its class a docstring.' }}</p>
    </div>
    <div class="tabs">
      <button v-for="t in ['try', 'config', 'triggers', 'about']" :key="t" class="tab" :class="{ 'tab--active': tab === t }" @click="tab = t">
        {{ { try: 'Try it', config: 'Config', triggers: 'Triggers', about: 'About' }[t] }}</button>
    </div>
    <div v-if="error" class="callout callout--error"><span class="callout__dot" />{{ error }}</div>
    <div v-if="saved" class="callout callout--done"><span class="callout__dot" />{{ saved }}</div>

    <div v-if="tab === 'try'" class="card">
      <p class="small muted mb-2">A new thread of <b>{{ agent.name }}</b>, and a run on it with this input. You'll follow it live.
        <Hint text="POST /v2/threads, then POST /v2/threads/{id}/runs: the two primitives this does." topic="primitives" /></p>
      <SchemaForm v-if="isWorkflow" v-model="inputData" :schema="agent.input_schema" />
      <textarea v-else v-model="input" class="input" rows="4" placeholder="What do you want to ask it?" />
      <details class="mt-2"><summary class="small muted">Thread labels (metadata)</summary>
        <textarea v-model="labels" class="input mono mt-2" rows="2" placeholder='{"org": "acme"}' /></details>
      <div class="row-actions mt-4"><span class="ml-auto" /><button class="btn btn--primary" :disabled="starting" @click="tryIt">Start</button></div>
    </div>

    <div v-if="tab === 'config' && agent.config" class="split">
      <div class="card">
        <SchemaForm v-model="config" :schema="agent.config.schema" :base="agent.config.file" />
        <div class="row-actions mt-4">
          <span class="small muted">version {{ agent.config.version }}<span v-if="agent.config.updated_by"> · by {{ agent.config.updated_by }}</span></span>
          <span class="ml-auto" />
          <button class="btn" :disabled="!Object.keys(agent.config.changed).length" @click="resetConfig">Reset to the file's</button>
          <button class="btn btn--primary" :disabled="saving" @click="saveConfig">Save for every process</button>
        </div>
      </div>
      <div class="card">
        <div class="section__title">Where a run's config comes from <Hint text="The file's, then what was changed here, then the thread's own (engine.thread(config=...))." topic="config" /></div>
        <ol class="small" style="padding-left:18px">
          <li>its agent file (<span class="mono">{{ agent.source || 'the engine' }}</span>)</li>
          <li>changed here, at runtime: <JsonView :value="agent.config.changed" /></li>
          <li>a thread's own config, when it was made with one</li>
        </ol>
      </div>
    </div>
    <div v-else-if="tab === 'config'" class="muted">This agent has no config.</div>

    <div v-if="tab === 'triggers'">
      <div class="card mb-4">
        <div class="section__title">Webhooks <Hint text="A POST from another system starts (or reaches) a run. Declared in the agent file's triggers." topic="triggers" /></div>
        <div v-if="!agent.webhooks.length" class="muted small">None. Add one in the agent file: <span class="mono">triggers: [{webhook: {path: /hooks/x, secret_env: X}}]</span></div>
        <div v-for="w in agent.webhooks" :key="w.path" class="mb-4">
          <div class="row-actions"><b class="mono">POST {{ w.path }}</b><span class="pill">{{ w.auth }}</span><span class="pill">busy: {{ w.busy }}</span>
            <span v-if="w.label" class="pill">thread per {{ w.label }}</span><span v-if="w.idempotency" class="pill">idempotent on {{ w.idempotency.key }}</span></div>
          <div class="code-block mt-2">{{ w.curl }}</div><button class="btn btn--sm mt-2" @click="copy(w.curl)">Copy</button>
        </div>
      </div>
      <div class="card">
        <div class="section__title">Schedules</div>
        <div v-if="!agent.schedules.length" class="muted small">None. Add one: <span class="mono">triggers: [{schedule: {cron: "0 7 * * MON-FRI"}}]</span></div>
        <div class="table-wrap" v-else>
          <table><thead><tr><th>Schedule</th><th>When</th><th>Next</th><th /></tr></thead>
            <tbody><tr v-for="s in agent.schedules" :key="s.id">
              <td class="mono">{{ s.id }}</td>
              <td>{{ s.data.cron ? `cron ${s.data.cron} (${s.data.timezone})` : s.data.every ? `every ${s.data.every}s` : `once at ${s.data.at}` }}</td>
              <td>{{ s.next_at ? 'in ' + until(s.next_at) : 'never again' }}</td>
              <td><button class="btn btn--sm" @click="fire(s.id)">Fire now</button></td>
            </tr></tbody></table>
        </div>
      </div>
    </div>

    <div v-if="tab === 'about'" class="grid-2">
      <div class="card"><dl class="kv">
        <dt>Name</dt><dd class="mono">{{ agent.name }}</dd>
        <dt>Class</dt><dd class="mono">{{ agent.class }}</dd>
        <dt>Version</dt><dd>{{ agent.version || '—' }}</dd>
        <dt>Requires</dt><dd>{{ agent.requires.join(', ') || 'nothing' }}</dd>
        <dt>May call</dt><dd>{{ agent.permissions.agents === '*' ? 'any agent' : (agent.permissions.agents as string[]).join(', ') || 'no agent' }}</dd>
        <dt>Files</dt><dd>{{ agent.permissions.files }}</dd>
        <dt>Session</dt><dd>{{ agent.session_provider }}</dd>
        <dt>Today</dt><dd><template v-if="agent.stats">{{ agent.stats.runs }} run(s) <span v-for="(c, s) in agent.stats.statuses" :key="s" style="margin-left:4px"><Badge :status="String(s)" /> {{ c }}</span></template><span v-else>no runs</span></dd>
      </dl></div>
      <div class="card">
        <div class="section__title">Input</div><JsonView :value="agent.input_schema || 'a message (any text, data or files)'" />
        <div class="section__title mt-4">Output</div><JsonView :value="agent.output_schema || 'none declared (Any)'" />
      </div>
    </div>
  </div>
  <div v-else-if="error" class="error-text">{{ error }}</div>
  <div v-else class="loading-center"><div class="spinner" /></div>
</template>
