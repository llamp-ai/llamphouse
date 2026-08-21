<script setup lang="ts">
/**
 * Reusable filter builder.
 *
 * Edits live in a *draft* state.  Nothing is committed to the parent until the
 * user clicks **Apply** — at which point the current draft is emitted as the
 * canonical `apply` event.  This lets consumers fetch server-side without
 * thrashing the network on every keystroke, and gives users a clear "Apply /
 * Reset" affordance.
 *
 * Events:
 *   - `apply`  — user clicked Apply.  Payload: current conditions.
 *   - `change` — draft changed (every keystroke).  Useful if a consumer wants
 *                live-preview client-side filtering; can usually be ignored.
 */
import { ref, computed, watch } from 'vue'

// ── Public types ─────────────────────────────────────────────────────────────
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

// ── Operator sets ────────────────────────────────────────────────────────────
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

// ── Props / emits ────────────────────────────────────────────────────────────
const props = withDefaults(
  defineProps<{
    fields: FieldDef[]
    /** Pre-populate filters on mount / when parent resets. */
    modelValue?: FilterCondition[]
    /** Show the row of pre-defined fields as quick-add buttons. */
    quickAdd?: boolean
  }>(),
  { modelValue: () => [], quickAdd: true },
)

const emit = defineEmits<{
  apply:  [conditions: FilterCondition[]]
  change: [conditions: FilterCondition[]]
}>()

// ── State: draft (editing) vs applied (last emitted) ─────────────────────────
let _idCtr = 0
const newId = () => String(++_idCtr)

function clone(list: FilterCondition[]): FilterCondition[] {
  return list.map((c) => ({ ...c }))
}

const draft   = ref<FilterCondition[]>(clone(props.modelValue))
const applied = ref<FilterCondition[]>(clone(props.modelValue))

// Keep draft in sync if the parent overwrites modelValue (e.g. reset).
watch(
  () => props.modelValue,
  (v) => {
    draft.value   = clone(v)
    applied.value = clone(v)
  },
)

// "Dirty" means draft differs from applied.
const dirty = computed(
  () => JSON.stringify(draft.value) !== JSON.stringify(applied.value),
)

// ── Helpers ──────────────────────────────────────────────────────────────────
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

function onChange() {
  emit('change', clone(draft.value))
}

// ── Mutations on the draft ───────────────────────────────────────────────────
function addCondition(fieldKey?: string) {
  const f = fieldKey ? fieldFor(fieldKey) : props.fields[0]
  if (!f) return
  draft.value.push({
    id:       newId(),
    field:    f.key,
    operator: defaultOp(f.key),
    value:    '',
    value2:   '',
  })
  onChange()
}

function removeCondition(id: string) {
  draft.value = draft.value.filter((c) => c.id !== id)
  onChange()
}

function clearDraft() {
  draft.value = []
  onChange()
}

function onFieldChange(cond: FilterCondition, key: string) {
  cond.field    = key
  cond.operator = defaultOp(key)
  cond.value    = ''
  cond.value2   = ''
  onChange()
}

function onOpChange(cond: FilterCondition, op: string) {
  cond.operator = op
  onChange()
}

// ── Apply / Reset ────────────────────────────────────────────────────────────
function apply() {
  applied.value = clone(draft.value)
  emit('apply', clone(applied.value))
}

function reset() {
  draft.value = clone(applied.value)
  emit('change', clone(draft.value))
}

function clearAndApply() {
  draft.value = []
  apply()
}
</script>

<template>
  <div class="fb">
    <!-- Header row: title + dirty indicator + actions -->
    <div class="fb__header">
      <div class="fb__title">
        <span class="fb__chip">Filters</span>
        <span v-if="applied.length > 0" class="fb__count">{{ applied.length }} active</span>
        <span v-if="dirty" class="fb__pending">● unsaved changes</span>
      </div>
      <div class="fb__actions">
        <button
          class="fb__btn fb__btn--ghost"
          :disabled="!dirty"
          @click="reset"
          title="Discard unsaved changes"
        >Reset</button>
        <button
          v-if="applied.length > 0"
          class="fb__btn fb__btn--ghost"
          @click="clearAndApply"
          title="Remove all filters"
        >Clear</button>
        <button
          class="fb__btn fb__btn--primary"
          :disabled="!dirty"
          @click="apply"
        >Apply</button>
      </div>
    </div>

    <!-- Active draft conditions -->
    <div v-if="draft.length > 0" class="fb__rows">
      <div v-for="cond in draft" :key="cond.id" class="fb__row">
        <select
          class="fb__select"
          :value="cond.field"
          @change="(e) => onFieldChange(cond, (e.target as HTMLSelectElement).value)"
        >
          <option v-for="f in fields" :key="f.key" :value="f.key">{{ f.label }}</option>
        </select>

        <select
          class="fb__select"
          :value="cond.operator"
          @change="(e) => onOpChange(cond, (e.target as HTMLSelectElement).value)"
        >
          <option v-for="op in opsFor(cond.field)" :key="op.value" :value="op.value">
            {{ op.label }}
          </option>
        </select>

        <template v-if="needsValue(cond.operator)">
          <select
            v-if="fieldFor(cond.field)?.type === 'select'"
            class="fb__select fb__select--value"
            v-model="cond.value"
            @change="onChange"
          >
            <option value="">— choose —</option>
            <option v-for="opt in fieldFor(cond.field)?.options ?? []" :key="opt" :value="opt">
              {{ opt }}
            </option>
          </select>
          <template v-else>
            <input
              class="fb__input"
              :type="inputType(cond.field)"
              v-model="cond.value"
              @input="onChange"
              placeholder="value"
            />
            <template v-if="isBetween(cond.operator)">
              <span class="fb__sep">and</span>
              <input
                class="fb__input"
                :type="inputType(cond.field)"
                v-model="cond.value2"
                @input="onChange"
                placeholder="value"
              />
            </template>
          </template>
        </template>

        <button class="fb__icon-btn" @click="removeCondition(cond.id)" title="Remove">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
      </div>
    </div>

    <!-- Quick-add row: pre-defined field chips + custom add -->
    <div class="fb__toolbar">
      <template v-if="quickAdd">
        <span class="fb__toolbar-label">Add:</span>
        <button
          v-for="f in fields"
          :key="f.key"
          class="fb__chip-btn"
          @click="addCondition(f.key)"
          :title="`Filter by ${f.label}`"
        >
          + {{ f.label }}
        </button>
      </template>
      <button v-else class="fb__chip-btn" @click="addCondition()">+ Add filter</button>
    </div>
  </div>
</template>

<style scoped>
.fb {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius, 8px);
  background: var(--bg-elevated, var(--bg-surface));
  margin-bottom: 1rem;
}

/* ── Header ─────────────────────────────────────────── */
.fb__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.fb__title {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 0.82rem;
  color: var(--text-secondary);
}
.fb__chip {
  display: inline-flex;
  align-items: center;
  padding: 2px 9px;
  border-radius: 999px;
  background: var(--bg-hover);
  color: var(--text-primary);
  font-weight: 600;
  font-size: 0.74rem;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}
.fb__count {
  color: var(--text-muted);
  font-size: 0.78rem;
}
.fb__pending {
  color: var(--accent, #4a90e2);
  font-size: 0.74rem;
  font-weight: 500;
}

.fb__actions {
  display: flex;
  align-items: center;
  gap: 6px;
}
.fb__btn {
  font-size: 0.78rem;
  font-weight: 500;
  padding: 5px 12px;
  border-radius: var(--radius-sm, 4px);
  border: 1px solid transparent;
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s, color 0.12s, opacity 0.12s;
}
.fb__btn--primary {
  background: var(--accent, #4a90e2);
  color: white;
  border-color: var(--accent, #4a90e2);
}
.fb__btn--primary:hover:not(:disabled) {
  filter: brightness(1.08);
}
.fb__btn--ghost {
  background: transparent;
  color: var(--text-secondary);
  border-color: var(--border);
}
.fb__btn--ghost:hover:not(:disabled) {
  background: var(--bg-hover);
  color: var(--text-primary);
}
.fb__btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* ── Condition rows ─────────────────────────────────── */
.fb__rows {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.fb__row {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  padding: 6px 8px;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm, 4px);
}
.fb__select,
.fb__input {
  padding: 4px 8px;
  font-size: 0.8rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm, 4px);
  background: var(--bg-surface);
  color: var(--text-primary);
  outline: none;
  height: 28px;
}
.fb__select { cursor: pointer; }
.fb__select:focus,
.fb__input:focus { border-color: var(--accent); }
.fb__select--value { min-width: 130px; }
.fb__input { min-width: 150px; }
.fb__input[type='date'],
.fb__input[type='number'] { min-width: 140px; }
.fb__input::placeholder { color: var(--text-muted); }
.fb__sep {
  font-size: 0.78rem;
  color: var(--text-muted);
  white-space: nowrap;
}

.fb__icon-btn {
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  padding: 4px;
  border-radius: var(--radius-sm, 4px);
  display: inline-flex;
  align-items: center;
  margin-left: auto;
  transition: background 0.1s, color 0.1s;
}
.fb__icon-btn:hover {
  background: var(--bg-hover);
  color: var(--error, #e84040);
}

/* ── Quick-add toolbar ──────────────────────────────── */
.fb__toolbar {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.fb__toolbar-label {
  font-size: 0.74rem;
  color: var(--text-muted);
  margin-right: 2px;
}
.fb__chip-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  font-size: 0.76rem;
  font-weight: 500;
  border: 1px dashed var(--border);
  border-radius: 999px;
  background: none;
  color: var(--text-secondary);
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s, color 0.12s;
}
.fb__chip-btn:hover {
  background: var(--bg-hover);
  border-color: var(--accent);
  color: var(--accent);
}
</style>
