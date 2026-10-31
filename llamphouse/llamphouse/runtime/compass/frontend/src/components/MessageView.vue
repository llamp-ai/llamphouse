<script setup lang="ts">
/* A message's parts as they are: text, data, files, artifacts, tool calls and results. */
import type { Part } from '../api/client'
import { fileUrl } from '../api/client'
import JsonView from './JsonView.vue'
defineProps<{ parts: Part[] }>()
</script>
<template>
  <div class="parts">
    <template v-for="(p, i) in parts" :key="i">
      <div v-if="p.kind === 'text'" class="part-text">{{ p.text }}</div>
      <div v-else-if="p.kind === 'data'"><span class="pill">data</span><JsonView :value="p.data" /></div>
      <div v-else-if="p.kind === 'file'">
        <span class="pill">file</span>
        <a v-if="p.uri" :href="fileUrl(p.uri)" target="_blank">{{ p.name || p.uri }}</a>
        <span v-else>{{ p.name || 'inline bytes' }}</span>
        <span class="muted small"> {{ p.mime_type }}</span>
      </div>
      <div v-else-if="p.kind === 'artifact'" class="artifact">
        <span class="pill">artifact</span> <b>{{ p.name }}</b> <span class="muted small">{{ p.description }}</span>
        <MessageView :parts="p.parts || []" />
      </div>
      <div v-else-if="p.kind === 'tool_call'"><span class="pill">calls {{ p.name }}</span><JsonView :value="p.arguments" /></div>
      <div v-else-if="p.kind === 'tool_result'"><span class="pill">result of {{ p.call_id }}</span><MessageView :parts="p.output || []" /></div>
      <div v-else><span class="pill">{{ p.kind }}</span><JsonView :value="p" /></div>
    </template>
  </div>
</template>
<style scoped>
.part-text { white-space: pre-wrap; }
.artifact { border-left: 3px solid var(--accent); padding-left: 8px; margin: 4px 0; }
.parts > * + * { margin-top: 4px; }
</style>
