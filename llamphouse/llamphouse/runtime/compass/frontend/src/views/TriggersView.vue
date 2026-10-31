<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { v2, until, formatTs, copy } from '../api/client'
import type { Schedule, Webhook } from '../api/client'
import Hint from '../components/Hint.vue'

const hooks = ref<Webhook[]>([])
const schedules = ref<Schedule[]>([])
const note = ref('')
const error = ref('')
async function load() {
  try { [hooks.value, schedules.value] = await Promise.all([v2.webhooks(), v2.schedules()]) } catch (e: any) { error.value = e.message }
}
onMounted(load)
async function fire(id: string) {
  try { const r: any = await v2.fire(id); note.value = r.note || `Fired ${id}: see Runs.`; await load() } catch (e: any) { error.value = e.message }
}
</script>
<template>
  <div>
    <div class="page-header"><h1>Triggers</h1><p class="page-header__subtitle">What starts agents besides people: other systems (webhooks) and the clock (schedules).
      <Hint text="Declared in agent files' triggers:. Schedules are kept by the store and fired once by whichever worker sees them due." topic="triggers" /></p></div>
    <div v-if="error" class="error-text">{{ error }}</div>
    <div v-if="note" class="callout callout--done"><span class="callout__dot" />{{ note }}</div>
    <div class="card mb-4">
      <div class="section__title">Webhooks</div>
      <div v-if="!hooks.length" class="muted small">None declared.</div>
      <div v-for="w in hooks" :key="w.path" class="mb-4">
        <div class="row-actions"><b class="mono">POST {{ w.path }}</b> → <router-link :to="`/agents/${w.agent}`">{{ w.agent }}</router-link>
          <span class="pill">{{ w.auth }}</span><span class="pill">busy: {{ w.busy }}</span>
          <span v-if="w.label" class="pill">thread per {{ w.label }}</span><span v-if="w.idempotency" class="pill">idempotent on {{ w.idempotency.key }}</span></div>
        <div class="code-block mt-2">{{ w.curl }}</div><button class="btn btn--sm mt-2" @click="copy(w.curl)">Copy</button>
      </div>
    </div>
    <div class="card">
      <div class="section__title">Schedules</div>
      <div v-if="!schedules.length" class="muted small">None. Agent files declare them; agents schedule runs with the Scheduler capability.</div>
      <div class="table-wrap" v-else><table><thead><tr><th>Schedule</th><th>Agent</th><th>When</th><th>Next</th><th /></tr></thead>
        <tbody><tr v-for="s in schedules" :key="s.id">
          <td class="mono small">{{ s.id }}</td><td>{{ s.agent }}</td>
          <td>{{ s.data.cron ? `cron ${s.data.cron} (${s.data.timezone})` : s.data.every ? `every ${s.data.every}s` : `once` }}
            <span v-if="s.data.scheduled_by" class="muted small"> · by run <router-link :to="`/runs/${s.data.scheduled_by}`">{{ s.data.scheduled_by }}</router-link></span></td>
          <td :title="formatTs(s.next_at)">{{ s.next_at ? 'in ' + until(s.next_at) : 'never again' }}</td>
          <td><button class="btn btn--sm" @click="fire(s.id)">Fire now</button></td></tr></tbody></table></div>
    </div>
  </div>
</template>
