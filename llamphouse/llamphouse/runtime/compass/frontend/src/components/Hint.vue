<script setup lang="ts">
/* A "?" that explains a concept - and links to Learn. */
defineProps<{ text: string; topic?: string }>()
</script>
<template>
  <router-link v-if="topic" :to="`/learn#${topic}`" class="hint" :title="text">?</router-link>
  <span v-else class="hint" :title="text">?</span>
</template>
