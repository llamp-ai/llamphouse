<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { v2, ago, formatTs } from '../api/client'
import type { Delivery, Subscription } from '../api/client'
import JsonView from '../components/JsonView.vue'
import Hint from '../components/Hint.vue'

const subs = ref<Subscription[]>([])
const picked = ref<Subscription | null>(null)
const deliveries = ref<Delivery[]>([])
const url = ref('')
const filter = ref('{"statuses": ["completed", "failed"]}')
const secret = ref('')
const error = ref('')
async function load() { try { subs.value = await v2.subscriptions() } catch (e: any) { error.value = e.message } }
onMounted(load)
async function pick(s: Subscription) { picked.value = s; deliveries.value = await v2.deliveries(s.id).catch(() => []) }
async function create() {
  error.value = ''
  try { await v2.subscribe({ url: url.value, filter: JSON.parse(filter.value || '{}'), secret: secret.value || undefined }); url.value = ''; await load() }
  catch (e: any) { error.value = e.message }
}
async function remove(s: Subscription) { await v2.unsubscribe(s.id); picked.value = null; await load() }
async function retry(d: Delivery) { await v2.retry(picked.value!.id, d.id); await pick(picked.value!) }
</script>
<template>
  <div>
    <div class="page-header"><h1>Subscriptions</h1><p class="page-header__subtitle">URLs that hear about runs: every matching status change is POSTed, again until it arrives.
      <Hint text="Deliveries are written with the change they report (an outbox) and sent by the workers, signed with the secret (X-Llamphouse-Signature)." topic="subscriptions" /></p></div>
    <div v-if="error" class="error-text mb-2">{{ error }}</div>
    <div class="split">
      <div>
        <div class="table-wrap card" style="padding:0"><table>
          <thead><tr><th>URL</th><th>Filter</th><th>Format</th><th /></tr></thead>
          <tbody>
            <tr v-for="s in subs" :key="s.id" style="cursor:pointer" :style="picked?.id === s.id ? 'background:var(--bg-hover)' : ''" @click="pick(s)">
              <td class="mono small">{{ s.url }}</td><td class="mono small">{{ JSON.stringify(s.filter) }}</td><td>{{ s.format }}</td>
              <td><button class="btn btn--sm btn--danger" @click.stop="remove(s)">Remove</button></td></tr>
            <tr v-if="!subs.length"><td colspan="4" class="muted">None yet.</td></tr>
          </tbody></table></div>
        <div v-if="picked" class="card mt-4">
          <div class="section__title">Deliveries to {{ picked.url }}</div>
          <div v-if="!deliveries.length" class="muted small">None yet.</div>
          <div v-for="d in deliveries" :key="d.id" class="mb-2">
            <div class="row-actions small">
              <span class="badge" :class="d.delivered_at ? 'badge--success' : d.dead ? 'badge--error' : 'badge--warning'">{{ d.delivered_at ? 'delivered' : d.dead ? 'dead' : 'retrying' }}</span>
              <span class="mono">{{ d.id }}</span><span>{{ d.attempts }} attempt(s)</span><span class="muted">{{ ago(d.created_at) }}</span>
              <span v-if="d.last_error" class="error-text">{{ d.last_error }}</span>
              <span v-if="!d.delivered_at" class="muted">next {{ formatTs(d.next_at) }}</span>
              <button v-if="!d.delivered_at" class="btn btn--sm" @click="retry(d)">Retry now</button>
            </div>
            <JsonView :value="d.payload" />
          </div>
        </div>
      </div>
      <form class="card" @submit.prevent="create">
        <div class="section__title">New subscription</div>
        <label class="small">URL</label><input v-model="url" class="input mb-2" placeholder="https://example.com/hooks/llamphouse" />
        <label class="small">Filter (agent, thread, run, statuses, metadata)</label><textarea v-model="filter" class="input mono mb-2" rows="3" />
        <label class="small">Secret (signs each delivery)</label><input v-model="secret" class="input mb-2" type="password" />
        <button class="btn btn--primary">Subscribe</button>
      </form>
    </div>
  </div>
</template>
