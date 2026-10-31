<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { v2, durationMs, textOf } from '../api/client'
import type { Debug, Run } from '../api/client'
import Badge from '../components/Badge.vue'
import IdLink from '../components/IdLink.vue'
import JsonView from '../components/JsonView.vue'

const route = useRoute()
const ids = String(route.query.runs || '').split(',').filter(Boolean)
const runs = ref<{ run: Run; debug: Debug }[]>([])
const error = ref('')
onMounted(async () => {
  try { runs.value = await Promise.all(ids.map(async id => ({ run: await v2.run(id), debug: await v2.debug(id) }))) }
  catch (e: any) { error.value = e.message }
})
</script>
<template>
  <div>
    <div class="page-header"><h1>Compare runs</h1><p class="page-header__subtitle">Side by side: input, how it went, what it recorded, what came out.</p></div>
    <div v-if="error" class="error-text">{{ error }}</div>
    <div class="compare" :style="`grid-template-columns: repeat(${runs.length || 1}, minmax(280px, 1fr))`">
      <div v-for="{ run, debug } in runs" :key="run.id" class="card">
        <div class="row-actions"><IdLink :id="run.id" kind="run" /><Badge :status="run.status" /></div>
        <dl class="kv mt-2">
          <dt>Agent</dt><dd>{{ run.agent }}</dd>
          <dt>Took</dt><dd>{{ durationMs(debug.created_at, debug.ended_at) }}</dd>
          <dt>Executions</dt><dd>{{ debug.activations }}</dd>
          <dt>Steps</dt><dd>{{ debug.steps.length }}</dd>
        </dl>
        <div class="section__title mt-4">Input</div><div class="small">{{ textOf(run.input) }}</div>
        <div class="section__title mt-4">Story</div>
        <div v-for="(s, i) in debug.story" :key="i" class="small">#{{ s.position }} {{ s.text }}</div>
        <div class="section__title mt-4">Output</div><JsonView :value="debug.output" />
        <div v-if="run.error" class="error-text mt-2">{{ run.error }}</div>
      </div>
    </div>
  </div>
</template>
<style scoped>.compare { display: grid; gap: 16px; overflow-x: auto; }</style>
