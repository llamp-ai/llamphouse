<script setup lang="ts">
import AppSidebar from './components/AppSidebar.vue'
</script>

<template>
  <div class="app-layout">
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
</style>
