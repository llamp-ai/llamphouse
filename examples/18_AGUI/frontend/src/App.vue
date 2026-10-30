<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import InterruptForm from "./components/InterruptForm.vue";
import { useAgent } from "./useAgent";

const agent = useAgent("/ag-ui/agents/trip-planner");
const draft = ref("");
const scroller = ref<HTMLElement | null>(null);
const cities = ["Paris", "Lyon", "Berlin", "Amsterdam", "Milan", "Barcelona"];
const suggestions = ["Find me a train to Berlin", "I need to be in Milan tomorrow", "Any trains to Amsterdam?"];

const chat = computed(() =>
  agent.messages.value
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({ id: m.id, role: m.role, text: typeof m.content === "string" ? m.content : "" }))
    .filter((m) => m.text),
);
const trip = computed(() => agent.state.value);

async function submit(text = draft.value) {
  draft.value = "";
  await agent.send(text);
}

watch(
  () => [chat.value.length, chat.value.at(-1)?.text, agent.interrupts.value.length],
  async () => {
    await nextTick();
    scroller.value?.scrollTo({ top: scroller.value.scrollHeight, behavior: "smooth" });
  },
);
</script>

<template>
  <header class="top">
    <div>
      <h1>Trip planner</h1>
      <p class="muted">A LLAMPHouse agent over AG-UI · thread <code>{{ agent.threadId.value }}</code></p>
    </div>
    <button class="link" :disabled="agent.running.value" @click="agent.reset()">New conversation</button>
  </header>

  <main class="layout">
    <section class="panel chat">
      <div ref="scroller" class="messages">
        <div v-if="!chat.length" class="empty">
          <p>Ask for a train. The agent searches, shares the options, and asks you before it books.</p>
          <div class="suggestions">
            <button v-for="s in suggestions" :key="s" @click="submit(s)">{{ s }}</button>
          </div>
        </div>
        <div v-for="m in chat" :key="m.id" class="message" :class="m.role">
          <div class="bubble">{{ m.text }}</div>
        </div>
        <div v-if="agent.running.value && chat.at(-1)?.role === 'user'" class="message assistant">
          <div class="bubble typing"><span /><span /><span /></div>
        </div>
        <InterruptForm
          v-for="i in agent.interrupts.value"
          :key="i.id"
          :interrupt="i"
          :busy="agent.running.value"
          @answer="(payload) => agent.answer(i, payload)"
          @dismiss="agent.dismiss(i)"
        />
        <p v-if="agent.error.value" class="error">{{ agent.error.value }}</p>
      </div>

      <form class="composer" @submit.prevent="submit()">
        <input
          v-model="draft"
          :disabled="agent.running.value || agent.waiting.value"
          :placeholder="agent.waiting.value ? 'Answer the question above first' : 'Where to?'"
        />
        <button class="primary" :disabled="!draft.trim() || agent.running.value || agent.waiting.value">Send</button>
      </form>
    </section>

    <aside class="side">
      <section class="panel">
        <h2>Trip <span class="muted small">shared state</span></h2>
        <label class="field">
          <span>Default destination</span>
          <select :value="trip.city" :disabled="agent.running.value" @change="agent.setState({ city: ($event.target as HTMLSelectElement).value })">
            <option v-for="c in cities" :key="c" :value="c">{{ c }}</option>
          </select>
        </label>
        <p class="muted small">Sent to the agent with the next message; it uses it when you name no city.</p>

        <table v-if="trip.options?.length" class="options">
          <thead><tr><th>Train</th><th>Departs</th><th>Price</th></tr></thead>
          <tbody>
            <tr v-for="t in trip.options" :key="t.train" :class="{ chosen: t.train === trip.chosen?.train }">
              <td>{{ t.train }}</td><td>{{ t.departs }}</td><td>€{{ t.price_eur }}</td>
            </tr>
          </tbody>
        </table>
        <p v-if="trip.booking" class="booked">Booked · <code>{{ trip.booking }}</code></p>
      </section>

      <section class="panel">
        <h2>Activity <span class="muted small">steps of the run</span></h2>
        <p v-if="!agent.activity.value.length" class="muted small">Nothing yet.</p>
        <ol class="activity">
          <li v-for="a in agent.activity.value" :key="a.id" :class="a.status">
            <span class="dot" />{{ a.label }}
          </li>
        </ol>
      </section>
    </aside>
  </main>
</template>
