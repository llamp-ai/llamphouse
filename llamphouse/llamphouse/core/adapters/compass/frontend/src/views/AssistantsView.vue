<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { compass, shortId } from '../api/client'
import type { Assistant, ConfigParam } from '../api/client'
import ConfigForm from '../components/ConfigForm.vue'

const assistants = ref<Assistant[]>([])
const loading = ref(true)
const error = ref('')

const selected = ref<string | null>(null)
const configParams = ref<ConfigParam[]>([])
const configDefaults = ref<Record<string, any>>({})
const configLoading = ref(false)

onMounted(async () => {
  try {
    assistants.value = await compass.listAssistants()
  } catch (e: any) {
    error.value = e.message
  } finally {
    loading.value = false
  }
})

async function selectAssistant(id: string) {
  selected.value = id
  configLoading.value = true
  try {
    const raw = await compass.getAssistantConfig(id)
    configParams.value = raw.params ?? []
    configDefaults.value = raw.defaults ?? {}
  } catch {
    configParams.value = []
    configDefaults.value = {}
  } finally {
    configLoading.value = false
  }
}

async function saveConfig(values: Record<string, any>) {
  if (!selected.value) return
  const res = await compass.setAssistantConfig(selected.value, values)
  if (res?.defaults) {
    configDefaults.value = res.defaults
  }
}
</script>

<template>
  <div>
    <div class="page-header">
      <div>
        <h1>Agents</h1>
        <div class="page-header__subtitle">{{ assistants.length }} registered</div>
      </div>
    </div>

    <div v-if="loading" class="loading-center"><div class="spinner"></div></div>
    <div v-else-if="error" class="card" style="color: var(--error)">{{ error }}</div>

    <template v-else>
      <div class="assistant-layout">
        <!-- List -->
        <div class="assistant-list">
          <div
            v-for="a in assistants"
            :key="a.id"
            class="assistant-item"
            :class="{ 'assistant-item--active': selected === a.id }"
            @click="selectAssistant(a.id)"
          >
            <div class="assistant-item__name">{{ a.name || shortId(a.id) }}</div>
            <div class="assistant-item__model">{{ a.model }}</div>
            <div class="assistant-item__id mono">{{ shortId(a.id) }}</div>
          </div>

          <div v-if="assistants.length === 0" class="empty-state">
            <div class="empty-state__icon"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a4 4 0 0 1 4 4v2a4 4 0 0 1-8 0V6a4 4 0 0 1 4-4z"/><path d="M16 14H8a4 4 0 0 0-4 4v2h16v-2a4 4 0 0 0-4-4z"/></svg></div>
            <div class="empty-state__title">No agents</div>
          </div>
        </div>

        <!-- Detail -->
        <div class="assistant-detail">
          <template v-if="!selected">
            <div class="empty-state">
              <div class="empty-state__icon"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg></div>
              <div class="empty-state__title">Select an agent</div>
            </div>
          </template>

          <template v-else>
            <div class="card mb-4">
              <div class="section__title">Details</div>
              <div class="detail-grid">
                <div class="detail-item">
                  <span class="detail-label">ID</span>
                  <span class="mono">{{ selected }}</span>
                </div>
                <div class="detail-item">
                  <span class="detail-label">Name</span>
                  <span>{{ assistants.find(a => a.id === selected)?.name || '—' }}</span>
                </div>
                <div class="detail-item">
                  <span class="detail-label">Model</span>
                  <span>{{ assistants.find(a => a.id === selected)?.model || '—' }}</span>
                </div>
                <div class="detail-item">
                  <span class="detail-label">Skills</span>
                  <span>{{ assistants.find(a => a.id === selected)?.skills?.length ?? 0 }}</span>
                </div>
              </div>

              <div v-if="assistants.find(a => a.id === selected)?.description" class="mt-4">
                <div class="section__title">Description</div>
                <div class="json-view">{{ assistants.find(a => a.id === selected)?.description }}</div>
              </div>

              <div v-if="assistants.find(a => a.id === selected)?.skills?.length" class="mt-4">
                <div class="section__title">Skills</div>
                <div class="skills-list">
                  <div v-for="skill in assistants.find(a => a.id === selected)?.skills" :key="skill.id" class="skill-item">
                    <span class="skill-name">{{ skill.name }}</span>
                    <span v-if="skill.description" class="skill-desc">{{ skill.description }}</span>
                  </div>
                </div>
              </div>
            </div>

            <div class="card">
              <div class="section__title">Config Store</div>
              <div v-if="configLoading" class="loading-center"><div class="spinner"></div></div>
              <ConfigForm v-else :params="configParams" :defaults="configDefaults" @save="saveConfig" />
            </div>
          </template>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.assistant-layout {
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 20px;
  align-items: start;
}

.assistant-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.assistant-item {
  padding: 12px 14px;
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
  background: var(--bg-surface);
  cursor: pointer;
  transition: all var(--transition);
}

.assistant-item:hover {
  border-color: var(--border-light);
  background: var(--bg-hover);
}

.assistant-item--active {
  border-color: var(--accent);
  background: var(--accent-dim);
}

.assistant-item__name {
  font-weight: 600;
  font-size: 0.9rem;
  margin-bottom: 2px;
}

.assistant-item__model {
  font-size: 0.75rem;
  color: var(--text-secondary);
}

.assistant-item__id {
  font-size: 0.7rem;
  color: var(--text-muted);
  margin-top: 4px;
}

.detail-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.detail-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.detail-label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-muted);
}

.skills-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.skill-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  background: var(--bg-surface);
}

.skill-name {
  font-weight: 600;
  font-size: 0.85rem;
  color: var(--text-primary);
}

.skill-desc {
  font-size: 0.78rem;
  color: var(--text-secondary);
  line-height: 1.4;
}
</style>
