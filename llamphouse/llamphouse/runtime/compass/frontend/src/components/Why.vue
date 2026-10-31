<script setup lang="ts">
/* Why a run is where it is, in words - Compass's "explain" lines. */
import type { Why } from '../api/client'
import { copy } from '../api/client'
defineProps<{ items: Why[] }>()
</script>
<template>
  <div>
    <div v-for="(w, i) in items" :key="i" class="callout" :class="`callout--${w.level}`">
      <span class="callout__dot" />
      <div>
        <div>{{ w.text }}</div>
        <div v-if="w.hint" class="callout__hint">{{ w.hint }}
          <button v-if="w.hint.includes('/v2/') || w.hint.startsWith('POST')" class="btn btn--sm" @click="copy(w.hint)">copy</button>
        </div>
      </div>
    </div>
  </div>
</template>
