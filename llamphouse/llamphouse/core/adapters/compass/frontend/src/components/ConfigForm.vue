<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import type { ConfigParam } from '../api/client'

const props = defineProps<{
  params: ConfigParam[]
  defaults: Record<string, any>
}>()

const emit = defineEmits<{
  save: [values: Record<string, any>]
}>()

/* Local editable copy of values, keyed by param key */
const values = ref<Record<string, any>>({})
const dirty = ref<Set<string>>(new Set())
const saving = ref(false)

watch(
  () => props.defaults,
  (d) => {
    const v: Record<string, any> = {}
    for (const p of props.params) {
      v[p.key] = d[p.key] ?? p.default
    }
    values.value = v
    dirty.value = new Set()
  },
  { immediate: true },
)

function markDirty(key: string) {
  dirty.value.add(key)
}

const hasDirty = computed(() => dirty.value.size > 0)

async function saveAll() {
  saving.value = true
  emit('save', { ...values.value })
  setTimeout(() => {
    dirty.value = new Set()
    saving.value = false
  }, 400)
}

function reset(key: string) {
  values.value[key] = props.defaults[key] ?? props.params.find(p => p.key === key)?.default
  dirty.value.delete(key)
}

function typeBadgeClass(type: string): string {
  switch (type) {
    case 'number': return 'badge--info'
    case 'boolean': return 'badge--success'
    case 'select': return 'badge--warning'
    case 'prompt': return 'badge--error'
    case 'string': return 'badge--neutral'
    default: return 'badge--neutral'
  }
}

function varChip(name: string): string {
  return `\u007B\u007B${name}\u007D\u007D`
}
</script>

<template>
  <div class="config-form">
    <div v-if="params.length === 0" class="empty-state">
      <div class="empty-state__icon"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg></div>
      <div class="empty-state__title">No config parameters defined</div>
    </div>

    <div v-for="param in params" :key="param.key" class="config-param">
      <!-- Header row: label + type badge -->
      <div class="config-param__header">
        <div class="config-param__label">{{ param.label || param.key }}</div>
        <span class="badge" :class="typeBadgeClass(param.type)">{{ param.type }}</span>
      </div>

      <!-- Description -->
      <div v-if="param.description" class="config-param__desc">{{ param.description }}</div>

      <!-- Key -->
      <div class="config-param__key mono">{{ param.key }}</div>

      <!-- Editor — switches by param type -->
      <div class="config-param__editor">
        <!-- NUMBER -->
        <template v-if="param.type === 'number'">
          <div class="number-editor">
            <input
              type="range"
              class="range-input"
              :min="param.min"
              :max="param.max"
              :step="param.step"
              :value="values[param.key]"
              @input="values[param.key] = Number(($event.target as HTMLInputElement).value); markDirty(param.key)"
            />
            <input
              type="number"
              class="input number-input"
              :min="param.min"
              :max="param.max"
              :step="param.step"
              v-model.number="values[param.key]"
              @input="markDirty(param.key)"
            />
            <span class="number-range-label">{{ param.min }} – {{ param.max }}</span>
          </div>
        </template>

        <!-- BOOLEAN -->
        <template v-else-if="param.type === 'boolean'">
          <label class="toggle">
            <input
              type="checkbox"
              :checked="values[param.key]"
              @change="values[param.key] = ($event.target as HTMLInputElement).checked; markDirty(param.key)"
            />
            <span class="toggle__slider"></span>
            <span class="toggle__text">{{ values[param.key] ? 'On' : 'Off' }}</span>
          </label>
        </template>

        <!-- SELECT -->
        <template v-else-if="param.type === 'select'">
          <select
            class="input select-input"
            :value="values[param.key]"
            @change="values[param.key] = ($event.target as HTMLSelectElement).value; markDirty(param.key)"
          >
            <option v-for="opt in param.options" :key="opt" :value="opt">{{ opt }}</option>
          </select>
        </template>

        <!-- PROMPT -->
        <template v-else-if="param.type === 'prompt'">
          <textarea
            class="input prompt-textarea"
            v-model="values[param.key]"
            @input="markDirty(param.key)"
            rows="4"
          ></textarea>
          <div v-if="param.variables?.length" class="prompt-variables">
            <span class="prompt-variables__label">Variables:</span>
            <span
              v-for="v in param.variables"
              :key="v"
              class="prompt-var-chip"
            >{{ varChip(v) }}</span>
          </div>
        </template>

        <!-- STRING (default) -->
        <template v-else>
          <input
            type="text"
            class="input"
            v-model="values[param.key]"
            @input="markDirty(param.key)"
            :maxlength="(param as any).max_length || undefined"
          />
          <div v-if="(param as any).max_length" class="string-counter">
            {{ (values[param.key] || '').length }} / {{ (param as any).max_length }}
          </div>
        </template>
      </div>

      <!-- Per-field reset -->
      <button
        v-if="dirty.has(param.key)"
        class="btn btn--xs btn--ghost"
        @click="reset(param.key)"
      >Reset</button>
    </div>

    <!-- Global save -->
    <div v-if="params.length > 0" class="config-form__actions">
      <button
        class="btn btn--primary"
        :disabled="!hasDirty || saving"
        @click="saveAll"
      >
        {{ saving ? 'Saving…' : 'Save Changes' }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.config-form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.config-param {
  padding: 14px 16px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--bg-surface);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.config-param__header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.config-param__label {
  font-weight: 600;
  font-size: 0.9rem;
}

.config-param__desc {
  font-size: 0.78rem;
  color: var(--text-secondary);
  line-height: 1.4;
}

.config-param__key {
  font-size: 0.7rem;
  color: var(--text-muted);
}

.config-param__editor {
  margin-top: 4px;
}

/* Number editor */
.number-editor {
  display: flex;
  align-items: center;
  gap: 12px;
}

.range-input {
  flex: 1;
  accent-color: var(--accent);
  height: 6px;
}

.number-input {
  width: 80px;
  text-align: center;
  font-family: var(--font-mono);
  font-size: 0.85rem;
}

.number-range-label {
  font-size: 0.7rem;
  color: var(--text-muted);
  white-space: nowrap;
}

/* Toggle / Boolean */
.toggle {
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  user-select: none;
}

.toggle input {
  display: none;
}

.toggle__slider {
  width: 40px;
  height: 22px;
  background: var(--border);
  border-radius: 11px;
  position: relative;
  transition: background 0.2s;
}

.toggle__slider::after {
  content: '';
  position: absolute;
  top: 3px;
  left: 3px;
  width: 16px;
  height: 16px;
  background: var(--text-primary);
  border-radius: 50%;
  transition: transform 0.2s;
}

.toggle input:checked + .toggle__slider {
  background: var(--accent);
}

.toggle input:checked + .toggle__slider::after {
  transform: translateX(18px);
}

.toggle__text {
  font-size: 0.8rem;
  color: var(--text-secondary);
  font-weight: 500;
}

/* Select */
.select-input {
  width: 100%;
  max-width: 320px;
  appearance: auto;
}

/* Prompt / textarea */
.prompt-textarea {
  width: 100%;
  resize: vertical;
  min-height: 80px;
  font-family: var(--font-mono);
  font-size: 0.82rem;
  line-height: 1.5;
}

.prompt-variables {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 6px;
}

.prompt-variables__label {
  font-size: 0.72rem;
  color: var(--text-muted);
}

.prompt-var-chip {
  display: inline-block;
  padding: 2px 8px;
  font-size: 0.72rem;
  font-family: var(--font-mono);
  background: var(--accent-dim);
  color: var(--accent);
  border-radius: var(--radius-sm);
  border: 1px solid var(--accent);
}

/* String counter */
.string-counter {
  font-size: 0.7rem;
  color: var(--text-muted);
  margin-top: 4px;
  text-align: right;
}

/* Actions bar */
.config-form__actions {
  display: flex;
  justify-content: flex-end;
  padding-top: 8px;
  border-top: 1px solid var(--border);
}

/* Small ghost button for reset */
.btn--xs {
  padding: 2px 8px;
  font-size: 0.7rem;
}

.btn--ghost {
  background: transparent;
  color: var(--text-muted);
  border: 1px solid var(--border);
}

.btn--ghost:hover {
  color: var(--text-secondary);
  border-color: var(--border-light);
}
</style>
