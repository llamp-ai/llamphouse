<script setup lang="ts">
/* The ideas behind llamphouse 2.0, each with where to see it in your own deployment. */
const topics = [
  { id: 'threads', title: 'Threads: everything is a thread with one agent',
    text: 'A thread is a conversation with one agent: people, apps and other agents write on it. At most one run is active on a thread at a time. Forks and threads of other agents are threads too - created by runs.',
    code: 'thread = await engine.thread("desk", metadata={"org": "acme"})', see: '/threads' },
  { id: 'primitives', title: 'Primitives: append, start, send',
    text: 'Nothing starts a run by itself. Append writes on an idle thread for the record; start begins a run with a message; send reaches the active run\'s inbox. What a message means while a run is busy (queue it, steer, refuse) is your app\'s decision - the adapters each follow their protocol\'s convention.',
    code: 'POST /v2/threads/{id}/messages   # append\nPOST /v2/threads/{id}/runs       # start\nPOST /v2/runs/{id}/messages      # send to the run', see: '/threads' },
  { id: 'inbox', title: 'The inbox: messages to a running run',
    text: 'A message sent to a run is on the thread at once (addressed to it), and enters its session when it reads it with ctx.next(). A run waiting on its inbox holds no worker.',
    code: 'while (ev := await ctx.next(timeout=30)) is not None:\n    ...  # ev.message: what someone sent it', see: '/runs?status=waiting' },
  { id: 'asking', title: 'Asking: input_required',
    text: 'ctx.ask("Which month?") parks the run (input_required), holding no worker, until an answer comes - the run then carries on from where it stopped. Client tools work the same way: the client runs the tool and sends the result.',
    code: 'month = await ctx.ask("For which month?")', see: '/runs?status=input_required' },
  { id: 'forks', title: 'Forks, calls, handovers',
    text: 'ctx.fork() makes a new thread of the same agent from here - parallel work that looks like a main thread to itself. ctx.call("other") makes a thread of another agent. Nothing comes back by itself: the creator listens (ctx.next(*forks)) and reads. ctx.handover("other") gives the thread to another agent.',
    code: 'research = await ctx.fork(DataPart(data={"city": "Rome"}))\nweather = await ctx.call("weather", "Rome in May")\nev = await ctx.next(research, weather)', see: '/threads?all=1' },
  { id: 'state', title: 'Thread state',
    text: 'ctx.state is the thread\'s state: versioned keys, written as state.changed events - so it is there as it was at any position, a fork starts from the state at the fork, and batch() writes several keys at once.',
    code: 'async with ctx.state.batch() as s:\n    s.set("cities", ["rome"])\n    s.set("month", "may")', see: '/threads' },
  { id: 'sessions', title: 'Sessions: what the agent sees',
    text: 'A session is an agent\'s working memory on a thread: the thread\'s messages (not those addressed to other runs) plus entries it added (session.add - tool calls, notes). A provider shapes it (a rolling window, a summary) and revises it in the background.',
    code: 'view = await ctx.session.messages()\nawait ctx.session.add(ToolCallPart(...))\nawait ctx.session.revise()', see: '/runs' },
  { id: 'durability', title: 'Durability: steps and replay',
    text: 'Everything a run does - sends, steps, state, forks, waits - is recorded under a call key. When it is executed again (after waiting, a restart, or on another worker after a crash), recorded calls return what they returned; only code between them runs again. Open a run\'s Steps tab to see its record.',
    code: '@step\nasync def quote(self, ctx, city: str) -> Quote:\n    return await prices.get(city)  # done once, whatever happens', see: '/runs' },
  { id: 'events', title: 'Thread events and run events',
    text: 'Thread events are the record: positioned, in commit order (messages, runs starting and ending, state). Run events are a run\'s stream: progress, messages being written (deltas, live only), custom events.',
    code: 'GET /v2/threads/{id}/events/stream   # the record, live\nGET /v2/runs/{id}/events/stream      # a run\'s stream, deltas too', see: '/threads' },
  { id: 'config', title: 'Config',
    text: 'An agent\'s Config (a pydantic model) comes from its agent file, then from changes made at runtime (every process, from its next run), then from a thread\'s own config - one class, many instances.',
    code: 'class Persona(Config):\n    tone: str = "plain"\n\nclass Assistant(Agent[Persona]): ...', see: '/agents' },
  { id: 'triggers', title: 'Triggers and the scheduler',
    text: 'Webhooks (a POST from another system) and schedules (cron, every, once) start runs, declared in the agent file. Agents schedule runs themselves with the Scheduler capability.',
    code: 'triggers:\n  - webhook: {path: /hooks/orders, secret_env: ORDERS_SECRET, thread: {label: $.order.id}}\n  - schedule: {cron: "0 7 * * MON-FRI", input: Daily digest}', see: '/triggers' },
  { id: 'subscriptions', title: 'Subscriptions',
    text: 'A URL that hears about runs: each matching status change is written to an outbox with the change itself and POSTed by the workers until it arrives. A2A push notifications are subscriptions too.',
    code: 'POST /v2/subscriptions {"url": "https://…", "filter": {"statuses": ["completed"]}, "secret": "…"}', see: '/subscriptions' },
]
</script>
<template>
  <div>
    <div class="page-header"><h1>Learn</h1><p class="page-header__subtitle">The ideas behind llamphouse 2.0 - and where to see each in your own deployment.</p></div>
    <nav class="toc card mb-4"><a v-for="t in topics" :key="t.id" :href="`#${t.id}`">{{ t.title.split(':')[0] }}</a></nav>
    <section v-for="t in topics" :id="t.id" :key="t.id" class="card mb-4">
      <h3>{{ t.title }}</h3>
      <p class="mt-2" style="color: var(--text-secondary)">{{ t.text }}</p>
      <pre class="code-block mt-2">{{ t.code }}</pre>
      <router-link class="btn btn--sm mt-2" :to="t.see">See it in your data →</router-link>
    </section>
  </div>
</template>
<style scoped>
.toc { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 0.875rem; }
section { scroll-margin-top: 16px; }
</style>
