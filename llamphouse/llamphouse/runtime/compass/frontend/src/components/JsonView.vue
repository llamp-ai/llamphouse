<script setup lang="ts">
/* JSON, folded when it is long; values Compass cut ({$cut, $length}) say so. */
import { computed, ref } from 'vue'
const props = defineProps<{ value: any; open?: boolean; max?: number }>()
const expanded = ref(!!props.open)
const text = computed(() => {
  const v = props.value
  if (v && typeof v === 'object' && '$cut' in v) return `${v.$cut}…  (${v.$length} characters, cut)`
  return typeof v === 'string' ? v : JSON.stringify(v, null, 2)
})
const long = computed(() => text.value.length > (props.max ?? 240))
</script>
<template>
  <div class="jsonview">
    <pre class="json-view" :class="{ 'jsonview--short': long && !expanded }">{{ text }}</pre>
    <button v-if="long" class="btn btn--sm" @click="expanded = !expanded">{{ expanded ? 'Less' : 'More' }}</button>
  </div>
</template>
<style scoped>
.jsonview--short { max-height: 120px; overflow: hidden; mask-image: linear-gradient(to bottom, black 70%, transparent); }
.json-view { margin: 0 0 4px; }
</style>
