<script setup lang="ts">
/* An id that links where it belongs (thread, run) and copies on a click of its icon. */
import { computed } from 'vue'
import { shortId, copy } from '../api/client'
const props = defineProps<{ id: string | null | undefined; kind?: 'thread' | 'run' | 'agent' | 'trace'; full?: boolean }>()
const to = computed(() => {
  if (!props.id) return null
  if (props.kind === 'thread') return `/threads/${props.id}`
  if (props.kind === 'run') return `/runs/${props.id}`
  if (props.kind === 'agent') return `/agents/${props.id}`
  if (props.kind === 'trace') return `/traces/${props.id}`
  return null
})
</script>
<template>
  <span v-if="id" class="idlink">
    <router-link v-if="to" :to="to" class="mono" :title="id">{{ full ? id : shortId(id) }}</router-link>
    <span v-else class="mono" :title="id">{{ full ? id : shortId(id) }}</span>
    <button class="idlink__copy" title="Copy" @click.stop.prevent="copy(id!)">⧉</button>
  </span>
  <span v-else class="muted">—</span>
</template>
<style scoped>
.idlink { display: inline-flex; align-items: center; gap: 2px; }
.idlink__copy { border: 0; background: none; color: var(--text-muted); cursor: pointer; font-size: 11px; padding: 0 2px; }
.idlink__copy:hover { color: var(--accent); }
</style>
