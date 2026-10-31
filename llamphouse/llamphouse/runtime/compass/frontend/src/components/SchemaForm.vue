<script setup lang="ts">
/* A form for a JSON Schema object (an agent's Config, a workflow's Input): a field per property, JSON for
   anything nested. The value is what the form holds; `changed` marks values that differ from `base`. */
import { computed } from 'vue'
const props = defineProps<{ schema: any; modelValue: Record<string, any>; base?: Record<string, any>; disabled?: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [value: Record<string, any>] }>()

function resolve(p: any): any {
  if (p?.$ref && props.schema?.$defs) return props.schema.$defs[p.$ref.split('/').pop()] ?? p
  if (p?.anyOf) return { ...p, ...(p.anyOf.find((x: any) => x.type !== 'null') ?? {}) }
  return p ?? {}
}
const fields = computed(() => Object.entries(props.schema?.properties ?? {}).map(([name, raw]) => {
  const p = resolve(raw)
  const kind = p.enum ? 'enum' : p.type === 'boolean' ? 'boolean' : (p.type === 'integer' || p.type === 'number') ? 'number'
    : p.type === 'string' ? ((p.maxLength ?? 0) > 200 || /prompt|instructions|template/i.test(name) ? 'text' : 'string') : 'json'
  return { name, p, kind, required: (props.schema?.required ?? []).includes(name) }
}))
function set(name: string, value: any) { emit('update:modelValue', { ...props.modelValue, [name]: value }) }
function setJson(name: string, text: string) { try { set(name, JSON.parse(text)) } catch { /* keep typing */ } }
const changed = (name: string) => props.base && JSON.stringify(props.base[name]) !== JSON.stringify(props.modelValue[name])
</script>
<template>
  <div class="schema-form">
    <div v-if="!fields.length" class="muted small">Nothing to configure: no fields.</div>
    <label v-for="f in fields" :key="f.name" class="field">
      <span class="field__label">{{ f.p.title || f.name }}<span v-if="f.required" class="muted"> *</span>
        <span v-if="changed(f.name)" class="pill" style="margin-left:6px">changed</span></span>
      <span v-if="f.p.description" class="field__desc">{{ f.p.description }}</span>
      <select v-if="f.kind === 'enum'" class="input" :disabled="disabled" :value="modelValue[f.name]" @change="set(f.name, ($event.target as HTMLSelectElement).value)">
        <option v-for="o in f.p.enum" :key="o" :value="o">{{ o }}</option>
      </select>
      <input v-else-if="f.kind === 'boolean'" type="checkbox" :disabled="disabled" :checked="!!modelValue[f.name]" @change="set(f.name, ($event.target as HTMLInputElement).checked)" />
      <input v-else-if="f.kind === 'number'" type="number" class="input" :disabled="disabled" :value="modelValue[f.name]" :step="f.p.type === 'integer' ? 1 : 'any'" @input="set(f.name, Number(($event.target as HTMLInputElement).value))" />
      <textarea v-else-if="f.kind === 'text'" class="input" rows="5" :disabled="disabled" :value="modelValue[f.name]" @input="set(f.name, ($event.target as HTMLTextAreaElement).value)" />
      <input v-else-if="f.kind === 'string'" class="input" :disabled="disabled" :value="modelValue[f.name]" @input="set(f.name, ($event.target as HTMLInputElement).value)" />
      <textarea v-else class="input mono" rows="4" :disabled="disabled" :value="JSON.stringify(modelValue[f.name] ?? null, null, 2)" @input="setJson(f.name, ($event.target as HTMLTextAreaElement).value)" />
    </label>
  </div>
</template>
<style scoped>
.schema-form { display: flex; flex-direction: column; gap: 12px; }
.field { display: flex; flex-direction: column; gap: 4px; }
.field__label { font-weight: 600; font-size: 0.875rem; }
.field__desc { color: var(--text-secondary); font-size: 0.8rem; }
</style>
