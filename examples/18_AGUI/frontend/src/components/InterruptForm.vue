<script setup lang="ts">
// What a waiting run asks, as a form built from the interrupt's
// responseSchema (a JSON Schema object): one boolean field becomes
// approve / reject buttons, other fields inputs.
import type { Interrupt } from "@ag-ui/client";
import { computed, reactive } from "vue";

const props = defineProps<{ interrupt: Interrupt; busy: boolean }>();
const emit = defineEmits<{ answer: [payload: unknown]; dismiss: [] }>();

type Field = { key: string; type: string; title: string; required: boolean; options?: string[] };

const schema = computed(() => (props.interrupt.responseSchema ?? {}) as Record<string, any>);
const fields = computed<Field[]>(() => {
  const required: string[] = schema.value.required ?? [];
  return Object.entries<Record<string, any>>(schema.value.properties ?? {}).map(([key, spec]) => ({
    key,
    type: spec.enum ? "enum" : spec.type ?? "string",
    title: spec.title ?? key,
    required: required.includes(key),
    options: spec.enum,
  }));
});
// The yes/no field, when the question is one: answered by its buttons.
const decision = computed(() => fields.value.find((f) => f.type === "boolean" && f.required));
const others = computed(() => fields.value.filter((f) => f !== decision.value));
const values = reactive<Record<string, any>>({});

function payload(extra: Record<string, unknown> = {}) {
  if (!fields.value.length) return values.answer ?? "";
  const out: Record<string, unknown> = {};
  for (const f of others.value) {
    const v = values[f.key];
    if (v === undefined || v === "") continue;
    out[f.key] = f.type === "number" || f.type === "integer" ? Number(v) : v;
  }
  return { ...out, ...extra };
}
</script>

<template>
  <div class="interrupt">
    <div class="interrupt-head">
      <span class="badge">Needs your input</span>
      <span v-if="interrupt.expiresAt" class="muted">answer before {{ new Date(interrupt.expiresAt).toLocaleString() }}</span>
    </div>
    <p class="interrupt-question">{{ interrupt.message ?? "The agent is waiting for an answer." }}</p>

    <form @submit.prevent="emit('answer', decision ? payload({ [decision.key]: true }) : payload())">
      <textarea v-if="!fields.length" v-model="values.answer" rows="2" placeholder="Your answer" />
      <label v-for="f in others" :key="f.key" class="field">
        <span>{{ f.title }}<em v-if="f.required"> *</em></span>
        <select v-if="f.type === 'enum'" v-model="values[f.key]" :required="f.required">
          <option v-for="o in f.options" :key="o" :value="o">{{ o }}</option>
        </select>
        <input v-else-if="f.type === 'boolean'" v-model="values[f.key]" type="checkbox" />
        <input v-else-if="f.type === 'number' || f.type === 'integer'" v-model="values[f.key]" type="number" :required="f.required" />
        <input v-else v-model="values[f.key]" type="text" :required="f.required" />
      </label>

      <div class="actions">
        <template v-if="decision">
          <button type="submit" class="primary" :disabled="busy">Approve</button>
          <button type="button" :disabled="busy" @click="emit('answer', payload({ [decision.key]: false }))">Reject</button>
        </template>
        <button v-else type="submit" class="primary" :disabled="busy">Send</button>
        <button type="button" class="link" :disabled="busy" @click="emit('dismiss')">Cancel the run</button>
      </div>
    </form>
  </div>
</template>
