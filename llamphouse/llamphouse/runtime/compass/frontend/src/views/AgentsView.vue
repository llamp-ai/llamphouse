<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { v2 } from '../api/client'
import type { AgentView } from '../api/client'
import Badge from '../components/Badge.vue'

const agents = ref<AgentView[]>([])
const error = ref('')
onMounted(async () => { try { agents.value = await v2.agents() } catch (e: any) { error.value = e.message } })
</script>
<template>
  <div>
    <div class="page-header"><h1>Agents</h1>
      <p class="page-header__subtitle">What this deployment serves: open one to see its config, try it, or see what starts it.</p></div>
    <div v-if="error" class="error-text">{{ error }}</div>
    <div class="grid-3">
      <router-link v-for="a in agents" :key="a.name" :to="`/agents/${a.name}`" class="card agent-card">
        <div class="flex items-center justify-between"><b>{{ a.display_name }}</b><span class="pill">{{ a.type }}</span></div>
        <div class="mono small muted">{{ a.name }}</div>
        <p class="small mt-2" style="color:var(--text-secondary)">{{ a.description || 'No description: give its class a docstring.' }}</p>
        <div class="row-actions small mt-2">
          <span v-if="a.stats">{{ a.stats.runs }} run(s) today</span>
          <span v-for="(c, s) in a.stats?.statuses || {}" :key="s"><Badge :status="String(s)" /> {{ c }}</span>
          <span v-if="a.schedules.length" class="pill">⏰ {{ a.schedules.length }}</span>
          <span v-if="a.webhooks.length" class="pill">⚡ {{ a.webhooks.length }}</span>
        </div>
      </router-link>
    </div>
  </div>
</template>
<style scoped>
.agent-card { color: var(--text-primary); display: block; transition: border-color var(--transition); }
.agent-card:hover { border-color: var(--accent); color: var(--text-primary); }
</style>
