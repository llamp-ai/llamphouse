<script setup lang="ts">
/* Runs per hour, stacked by status: a small SVG chart, no library. */
import { computed } from 'vue'
const props = defineProps<{ points: { at: string; status: string; count: number }[]; hours: number }>()
const COLORS: Record<string, string> = { completed: 'var(--success)', failed: 'var(--error)', rejected: 'var(--error)',
  cancelled: 'var(--text-muted)', input_required: 'var(--warning)', auth_required: 'var(--warning)' }
const buckets = computed(() => {
  const now = new Date(); now.setMinutes(0, 0, 0)
  const out: { at: Date; parts: { status: string; count: number }[]; total: number }[] = []
  for (let i = Math.min(props.hours, 72) - 1; i >= 0; i--) out.push({ at: new Date(now.getTime() - i * 3600e3), parts: [], total: 0 })
  for (const p of props.points) {
    const t = new Date(p.at).getTime()
    const b = out.find(b => b.at.getTime() === t)
    if (b) { b.parts.push({ status: p.status, count: p.count }); b.total += p.count }
  }
  return out
})
const max = computed(() => Math.max(1, ...buckets.value.map(b => b.total)))
</script>
<template>
  <svg class="minibars" :viewBox="`0 0 ${buckets.length * 10} 60`" preserveAspectRatio="none">
    <g v-for="(b, i) in buckets" :key="i">
      <title>{{ b.at.toLocaleString() }}: {{ b.total }} run(s) {{ b.parts.map(p => `${p.count} ${p.status}`).join(', ') }}</title>
      <template v-for="(p, j) in b.parts" :key="j">
        <rect :x="i * 10 + 1" :width="8"
              :y="60 - (b.parts.slice(0, j + 1).reduce((s, x) => s + x.count, 0) / max) * 58"
              :height="(p.count / max) * 58" :style="`fill: ${COLORS[p.status] || 'var(--info)'}`" />
      </template>
      <rect v-if="!b.total" :x="i * 10 + 1" :width="8" y="58" height="2" style="fill: var(--border)" />
    </g>
  </svg>
</template>
<style scoped>.minibars { width: 100%; height: 90px; display: block; }</style>
