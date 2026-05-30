<script setup lang="ts">
import { ref } from 'vue'

// ── Public types ──────────────────────────────────────────────────────────────
export interface FieldDef {
  key: string
  label: string
  /** Controls the available operators and input widget. */
  type: 'string' | 'date' | 'number' | 'select'
  options?: string[]   // only used when type === 'select'
}

export interface FilterCondition {
  id: string
  field: string
  operator: string
  value: string
  value2: string   // second bound for 'between' operators
}

// ── Operator sets ─────────────────────────────────────────────────────────────
const STRING_OPS = [
  { value: 'contains',      label: 'contains' },
  { value: 'not_contains',  label: 'does not contain' },
  { value: 'equals',        label: 'equals' },
  { value: 'not_equals',    label: 'not equals' },
  { value: 'starts_with',   label: 'starts with' },
  { value: 'ends_with',     label: 'ends with' },
  { value: 'is_empty',      label: 'is empty' },
  { value: 'is_not_empty',  label: 'is not empty' },
]

const DATE_OPS = [
  { value: 'is_after',   label: 'is after' },
  { value: 'is_before',  label: 'is before' },
  { value: 'is_on',      label: 'is on' },
  { value: 'is_between', label: 'is between' },
]

const NUMBER_OPS = [
  { value: 'eq',      label: '=' },
  { value: 'neq',     label: '≠' },
  { value: 'gt',      label: '>' },
  { value: 'gte',     label: '≥' },
  { value: 'lt',      label: '<' },
  { value: 'lte',     label: '≤' },
  { value: 'between', label: 'between' },
]

const SELECT_OPS = [
  { value: 'equals',     label: 'is' },
  { value: 'not_equals', label: 'is not' },
]

// ── Props / emits ─────────────────────────────────────────────────────────────
const props = defineProps<{ fields: FieldDef[] }>()
const emit  = defineEmits<{ change: [conditions: FilterCondition[]] }>()

// ── State ─────────────────────────────────────────────────────────────────────
const conditions = ref<FilterCondition[]>([])
let _idCtr = 0

// ── Helpers ───────────────────────────────────────────────────────────────────
function fieldFor(key: string) {
  return props.fields.find((f) => f.key === key)
}

function opsFor(fieldKey: string) {
  const t = fieldFor(fieldKey)?.type ?? 'string'
  if (t === 'date')   return DATE_OPS
  if (t === 'number') return NUMBER_OPS
  if (t === 'select') return SELECT_OPS
  return STRING_OPS
}

function defaultOp(fieldKey: string): string {
  return opsFor(fieldKey)[0].value
}

function inputType(fieldKey: string): string {
  const t = fieldFor(fieldKey)?.type ?? 'string'
  if (t === 'date')   return 'date'
  if (t === 'number') return 'number'
  return 'text'
}

function needsValue(op: string) {
  return op !== 'is_empty' && op !== 'is_not_empty'
}

function isBetween(op: string) {
  return op === 'is_between' || op === 'between'
}

// ── Mutations ─────────────────────────────────────────────────────────────────
function push() {
  emit('change', [...conditions.value])
}

function add() {
  const first = props.fields[0]
  conditions.value.push({
    id:       String(++_idCtr),
    field:    first.key,
    operator: defaultOp(first.key),
    value:    '',
    value2:   '',
  })
  push()
}

function remove(id: string) {
  conditions.value = conditions.value.filter((c) => c.id !== id)
  push()
}

function clearAll() {
  conditions.value = []
  push()
}

function onFieldChange(cond: FilterCondition, key: string) {
  cond.field    = key
  cond.operator = defaultOp(key)
  cond.value    = ''
  cond.value2   = ''
  push()
}

function onOpChange(cond: FilterCondition, op: string) {
  cond.operator = op
  push()
}
</script>

<template>
  <div class="fb-root">
    <!-- Active conditions -->
    <div
      v-for="cond in conditions"
      :key="cond.id"
      class="fb-row"
    >
      <!-- Field select -->
      <select
        class="fb-select"
        :value="cond.field"
        @change="(e) => onFieldChange(cond, (e.target as HTMLSelectElement).value)"
      >
        <option v-for="f in fields" :key="f.key" :value="f.key">{{ f.label }}</option>
      </select>

      <!-- Operator select -->
      <select
        class="fb-select"
        :value="cond.operator"
        @change="(e) => onOpChange(cond, (e.target as HTMLSelectElement).value)"
      >
        <option v-for="op in opsFor(cond.field)" :key="op.value" :value="op.value">
          {{ op.label }}
        </option>
      </select>

      <!-- Value input(s) -->
      <template v-if="needsValue(cond.operator)">
        <!-- select-type field: render a <select> for options -->
        <select
          v-if="fieldFor(cond.field)?.type === 'select'"
          class="fb-select fb-select--value"
          v-model="cond.value"
          @change="push"
        >
          <option value="">— choose —</option>
          <option v-for="opt in fieldFor(cond.field)?.options ?? []" :key="opt" :value="opt">
            {{ opt }}
          </option>
        </select>

        <!-- other types: text / date / number input -->
        <template v-else>
          <input
            class="fb-input"
            :type="inputType(cond.field)"
            v-model="cond.value"
            @input="push"
            placeholder="value"
          />
          <template v-if="isBetween(cond.operator)">
            <span class="fb-sep">and</span>
            <input
              class="fb-input"
              :type="inputType(cond.field)"
              v-model="cond.value2"
              @input="push"
              placeholder="value"
            />
          </template>
        </template>
      </template>

      <!-- Remove button -->
      <button class="fb-remove" @click="remove(cond.id)" title="Remove filter">✕</button>
    </div>

    <!-- Toolbar -->
    <div class="fb-toolbar">
      <button class="fb-add" @click="add">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <line x1="12" y1="5" x2="12" y2="19" />
          <line x1="5" y1="12" x2="19" y2="12" />
        </svg>
        Add filter
      </button>
      <button v-if="conditions.length > 0" class="fb-clear" @click="clearAll">
        Clear all
      </button>
    </div>
  </div>
</template>

<style scoped>
.fb-root {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.fb-row {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

/* ── Controls ──────────────────────────────────────── */
.fb-select,
.fb-input {
  padding: 4px 8px;
  font-size: 0.8rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm, 4px);
  background: var(--bg-surface);
  color: var(--text-primary);
  outline: none;
  height: 28px;
}

.fb-select { cursor: pointer; }
.fb-select:focus,
.fb-input:focus  { border-color: var(--accent); }

.fb-select--value { min-width: 130px; }

.fb-input {
  min-width: 150px;
}
.fb-input[type='date'],
.fb-input[type='number'] { min-width: 140px; }

.fb-input::placeholder { color: var(--text-muted); }

.fb-sep {
  font-size: 0.78rem;
  color: var(--text-muted);
  white-space: nowrap;
}

.fb-remove {
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 0.75rem;
  padding: 3px 6px;
  border-radius: var(--radius-sm, 4px);
  line-height: 1;
  transition: background 0.1s, color 0.1s;
}
.fb-remove:hover {
  background: var(--bg-hover);
  color: var(--error, #e84040);
}

/* ── Toolbar ───────────────────────────────────────── */
.fb-toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
}

.fb-add {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 10px;
  font-size: 0.78rem;
  font-weight: 500;
  border: 1px dashed var(--border);
  border-radius: var(--radius-sm, 4px);
  background: none;
  color: var(--text-secondary);
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s, color 0.12s;
}
.fb-add:hover {
  background: var(--bg-hover);
  border-color: var(--accent);
  color: var(--accent);
}

.fb-clear {
  background: none;
  border: none;
  font-size: 0.78rem;
  color: var(--text-muted);
  cursor: pointer;
  padding: 4px 8px;
  border-radius: var(--radius-sm, 4px);
  transition: background 0.1s, color 0.1s;
}
.fb-clear:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}
</style>
