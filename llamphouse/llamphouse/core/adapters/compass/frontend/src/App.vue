<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import AppSidebar from './components/AppSidebar.vue'

const route = useRoute()
const isFullscreen = computed(() => !!route.meta.fullscreen)
</script>

<template>
  <div v-if="isFullscreen" class="app-fullscreen">
    <router-view />
  </div>
  <div v-else class="app-layout">
    <AppSidebar />
    <main class="app-main">
      <router-view />
    </main>
  </div>
</template>

<style scoped>
.app-layout {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

.app-main {
  flex: 1;
  overflow-y: auto;
  padding: 28px 32px;
  background: var(--bg-secondary);
}

.app-fullscreen {
  height: 100vh;
  overflow: hidden;
  background: var(--bg-secondary);
}
</style>
