<script setup lang="ts">
/* The thread's primitives, as its state allows them - and what each one means.
   Idle: start a run, or append for the record. Active: send to the run's inbox, answer its question,
   signal it, cancel it. Nothing here decides for you: it's what the API offers. */
import { computed, ref } from 'vue'
import type { Thread } from '../api/client'
import { v2 } from '../api/client'
import Hint from './Hint.vue'

const props = defineProps<{ thread: Thread }>()
const emit = defineEmits<{ done: [] }>()
const text = ref('')
const asData = ref(false)
const who = ref(localStorage.getItem('llamphouse.compass.as') || 'developer')
const signalKey = ref('')
const error = ref('')
const busy = ref(false)

const active = computed(() => props.thread.active_run)
const asked = computed(() => active.value && ['input_required', 'auth_required'].includes(active.value.status))
const calls = computed(() => (active.value?.pending?.prompt ?? []).filter((p: any) => p.kind === 'tool_call'))

function body() {
  localStorage.setItem('llamphouse.compass.as', who.value)
  const sender = { kind: 'person', id: who.value || 'developer' }
  if (asData.value) {
    try { return { data: JSON.parse(text.value || 'null'), sender } } catch { throw new Error('the data is not JSON') }
  }
  return { text: text.value, sender }
}
async function act(fn: () => Promise<any>, keep = false) {
  error.value = ''
  busy.value = true
  try { await fn(); if (!keep) text.value = ''; emit('done') } catch (e: any) { error.value = e.message } finally { busy.value = false }
}
const start = () => act(() => v2.start(props.thread.id, body()))
const append = () => act(() => v2.append(props.thread.id, body()))
const send = () => act(() => v2.send(active.value!.id, body()))
const answer = () => act(() => v2.answer(active.value!.id, body()))
const signal = () => act(() => v2.signal(active.value!.id, signalKey.value, asData.value ? JSON.parse(text.value || 'null') : text.value))
const cancel = () => act(() => v2.cancel(active.value!.id), true)
function keydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); active.value ? (asked.value ? answer() : send()) : start() }
}
</script>
<template>
  <div class="composer card">
    <div class="composer__state small">
      <template v-if="!active">The thread is idle: <b>start</b> a run with a message, or <b>append</b> one for the record.
        <Hint text="Nothing starts a run by itself: a run starts when you say so. Appending writes on the thread only." topic="primitives" /></template>
      <template v-else-if="calls.length">The run waits for its client to run <b>{{ calls.map((c: any) => c.name).join(', ') }}</b>: answer with the result.</template>
      <template v-else-if="asked">The run asked: <b>{{ active.pending?.reason }}</b> - answer it.
        <Hint text="ask() parks the run (input_required) without holding a worker; the answer resumes it from where it stopped." topic="asking" /></template>
      <template v-else>A run is {{ active.status }}: a message goes to its <b>inbox</b> (it reads it with ctx.next()).
        <Hint text="While a run is active the thread takes no message and no new run - only messages sent to that run." topic="inbox" /></template>
    </div>
    <textarea v-model="text" class="input" rows="3" :placeholder="asData ? '{&quot;json&quot;: true}' : 'A message…  (⌘↵ to send)'" @keydown="keydown" />
    <div class="row-actions" style="margin-top:8px">
      <label class="small muted">as <input v-model="who" class="input" style="width:110px;display:inline-block;padding:3px 6px" /></label>
      <label class="small muted"><input v-model="asData" type="checkbox" /> JSON</label>
      <span class="ml-auto" />
      <template v-if="!active">
        <button class="btn btn--sm" :disabled="busy" title="On the thread, for the record: no run starts" @click="append">Append</button>
        <button class="btn btn--sm btn--primary" :disabled="busy" @click="start">Start run</button>
      </template>
      <template v-else>
        <button class="btn btn--sm btn--danger" :disabled="busy" @click="cancel">Cancel run</button>
        <input v-model="signalKey" class="input" placeholder="signal" style="width:90px;padding:3px 6px" />
        <button class="btn btn--sm" :disabled="busy || !signalKey" title="ctx.run.wait_for_signal(key) receives it" @click="signal">Signal</button>
        <button v-if="!asked" class="btn btn--sm btn--primary" :disabled="busy" @click="send">Send to run</button>
        <button v-else class="btn btn--sm btn--primary" :disabled="busy" @click="answer">Answer</button>
      </template>
    </div>
    <div v-if="error" class="error-text mt-2">{{ error }}</div>
  </div>
</template>
<style scoped>
.composer { padding: 14px; }
.composer__state { margin-bottom: 8px; color: var(--text-secondary); }
</style>
