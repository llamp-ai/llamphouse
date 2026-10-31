// An AG-UI agent as Vue state: the conversation, the shared state, what the
// run is doing, and what it waits for. Built on @ag-ui/client's HttpAgent,
// the client every AG-UI front-end uses (CopilotKit's too).
import { HttpAgent, type AgentSubscriber, type Interrupt, type Message } from "@ag-ui/client";
import { computed, ref, shallowRef } from "vue";

export type Activity = {
  id: string;
  label: string;
  status: "running" | "done" | "waiting" | "failed" | "cancelled";
};

const STORAGE_KEY = "llamphouse-ag-ui-example";

// Kept across reloads, interrupts too: a run may wait for days.
type Saved = { threadId: string; messages: Message[]; state: Record<string, unknown>; interrupts?: Interrupt[] };

function load(): Saved | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Saved) : null;
  } catch {
    return null;
  }
}

function newId(prefix: string): string {
  return `${prefix}-${crypto.randomUUID().slice(0, 8)}`;
}

export function useAgent(url: string) {
  const messages = shallowRef<Message[]>([]);
  const state = ref<Record<string, any>>({});
  const activity = ref<Activity[]>([]);
  const interrupts = shallowRef<Interrupt[]>([]);
  const running = ref(false);
  const error = ref<string | null>(null);
  const threadId = ref("");
  let agent: HttpAgent;

  function save() {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ threadId: threadId.value, messages: agent.messages, state: agent.state, interrupts: interrupts.value }),
      );
    } catch {
      /* private mode: the conversation just isn't kept */
    }
  }

  function connect(saved: Saved | null) {
    threadId.value = saved?.threadId ?? newId("thread");
    agent = new HttpAgent({
      url,
      threadId: threadId.value,
      initialMessages: saved?.messages ?? [],
      initialState: saved?.state ?? { city: "Paris" },
    });
    messages.value = [...agent.messages];
    state.value = { ...agent.state };
    activity.value = [];
    interrupts.value = saved?.interrupts ?? [];
    error.value = null;
  }

  // What the run is doing: each `progress` event (the agent's `Progress`) starts an entry and
  // finishes the one before it.
  function progress(label: string) {
    finishRunning();
    activity.value.push({ id: newId("act"), label, status: "running" });
  }

  function finishRunning() {
    for (const a of activity.value) if (a.status === "running") a.status = "done";
  }

  const subscriber: AgentSubscriber = {
    onMessagesChanged: ({ messages: m }) => {
      messages.value = [...m];
    },
    onStateChanged: ({ state: s }) => {
      state.value = { ...(s as Record<string, any>) };
    },
    // llamphouse sends the agent's Progress events as CUSTOM events named "progress".
    onCustomEvent: ({ event }) => {
      if (event.name === "progress") progress(String((event.value as { label?: unknown })?.label ?? ""));
    },
    onRunFinishedEvent: (params) => {
      finishRunning();
      if (params.outcome === "interrupt") {
        interrupts.value = [...params.interrupts];
        activity.value.push({ id: newId("act"), label: "Waiting for you", status: "waiting" });
      } else if (params.outcome === "cancelled") {
        activity.value.push({ id: newId("act"), label: "Run cancelled", status: "cancelled" });
      }
    },
    onRunErrorEvent: ({ event }) => {
      finishRunning();
      error.value = event.message;
      activity.value.push({ id: newId("act"), label: "Run failed", status: "failed" });
    },
  };

  async function run(parameters: Parameters<HttpAgent["runAgent"]>[0] = {}) {
    running.value = true;
    error.value = null;
    try {
      await agent.runAgent(parameters, subscriber);
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e);
    } finally {
      running.value = false;
      messages.value = [...agent.messages];
      save();
    }
  }

  async function send(text: string) {
    if (!text.trim() || running.value || interrupts.value.length) return;
    activity.value = [];
    agent.addMessage({ id: newId("msg"), role: "user", content: text.trim() });
    messages.value = [...agent.messages];
    await run();
  }

  // Answer what the run waits for: the same run carries on.
  async function answer(interrupt: Interrupt, payload: unknown) {
    interrupts.value = interrupts.value.filter((i) => i.id !== interrupt.id);
    await run({ resume: [{ interruptId: interrupt.id, status: "resolved", payload }] });
  }

  async function dismiss(interrupt: Interrupt) {
    interrupts.value = interrupts.value.filter((i) => i.id !== interrupt.id);
    await run({ resume: [{ interruptId: interrupt.id, status: "cancelled" }] });
  }

  // The front-end's half of the shared state: sent with the next run.
  function setState(patch: Record<string, unknown>) {
    agent.setState({ ...agent.state, ...patch });
    state.value = { ...agent.state };
    save();
  }

  function reset() {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* nothing kept */
    }
    connect(null);
  }

  connect(load());

  return {
    messages,
    state,
    activity,
    interrupts,
    running,
    error,
    threadId,
    waiting: computed(() => interrupts.value.length > 0),
    send,
    answer,
    dismiss,
    setState,
    reset,
  };
}
