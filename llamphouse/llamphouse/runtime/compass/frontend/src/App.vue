<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AppSidebar from './components/AppSidebar.vue'
import { session, setKey, v2 } from './api/client'

const route = useRoute()
const router = useRouter()
const isFullscreen = computed(() => !!route.meta.fullscreen)
const search = ref('')
const searchError = ref('')
const keyInput = ref('')

/* paste any id - thread, run, response, session - and go there */
async function jump() {
  const id = search.value.trim()
  if (!id) return
  searchError.value = ''
  try {
    const found = await v2.lookup(id)
    router.push(found.kind === 'thread' ? `/threads/${found.id}` : found.kind === 'run' ? `/runs/${found.id}`
      : `/runs/${found.runs?.[0] ?? ''}`)
    search.value = ''
  } catch {
    searchError.value = `Nothing is called ${id}`
  }
}
const theme = ref(localStorage.getItem('llamphouse.compass.theme') || 'system')
function setTheme(t: string) {
  theme.value = t
  localStorage.setItem('llamphouse.compass.theme', t)
  if (t === 'system') document.documentElement.removeAttribute('data-theme')
  else document.documentElement.setAttribute('data-theme', t)
}
setTheme(theme.value)
</script>

<template>
  <div v-if="isFullscreen" class="app-fullscreen"><router-view /></div>
  <div v-else class="app-layout">
    <AppSidebar />
    <div class="app-column">
      <header class="topbar">
        <form class="topbar__search" @submit.prevent="jump">
          <input v-model="search" class="input" placeholder="Jump to a thread, run, response or session id…" />
        </form>
        <span v-if="searchError" class="error-text">{{ searchError }}</span>
        <span class="ml-auto" />
        <select class="input" style="width:auto" :value="theme" title="Theme" @change="setTheme(($event.target as HTMLSelectElement).value)">
          <option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option>
        </select>
        <button v-if="session.key" class="btn btn--sm" title="Forget the API key in this browser" @click="setKey('')">Sign out</button>
      </header>
      <main class="app-main"><router-view :key="route.fullPath" /></main>
    </div>
    <div v-if="session.needsKey" class="modal">
      <form class="card modal__card" @submit.prevent="setKey(keyInput)">
        <h3>An API key, please</h3>
        <p class="muted small">This deployment asks who you are. The key stays in this browser.</p>
        <input v-model="keyInput" class="input mt-2" type="password" placeholder="API key" autofocus />
        <div class="row-actions mt-4"><span class="ml-auto" /><button class="btn btn--primary">Use it</button></div>
      </form>
    </div>
  </div>
</template>

<style scoped>
.app-layout { display: flex; height: 100vh; overflow: hidden; }
.app-column { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.topbar { display: flex; align-items: center; gap: 10px; padding: 10px 32px; border-bottom: 1px solid var(--border);
  background: var(--bg-primary); }
.topbar__search { width: min(520px, 50%); }
.app-main { flex: 1; overflow-y: auto; padding: 24px 32px; background: var(--bg-secondary); }
.app-fullscreen { height: 100vh; overflow: hidden; background: var(--bg-secondary); }
.modal { position: fixed; inset: 0; background: rgba(0, 0, 0, 0.4); display: flex; align-items: center; justify-content: center; z-index: 50; }
.modal__card { width: 380px; }
</style>
