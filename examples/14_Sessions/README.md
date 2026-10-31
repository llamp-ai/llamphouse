# Sessions

A session is an agent's working memory on a thread: what it hands its model. The thread keeps every message;
the agent's **session provider** decides what the agent sees of it (`view`), and when and how to revise it (a
summary, compaction - in the background, never on a read). Here one agent class, `Mirror`, runs as three agents
with three providers, and every turn it says what its session holds. No model needed.

| agent    | provider                         | sees                                                        |
|----------|----------------------------------|-------------------------------------------------------------|
| `window` | `rolling_window: {size: 4}`      | the latest 4 messages (the default provider keeps 50)       |
| `summary`| `summarizing: {max_messages: 6, keep: 2}` | past 6, older messages become one summary - your function's |
| `facts`  | `providers.py:Facts` (our own)   | a note with the numbers the person mentioned, and the latest 2 |

## Run it

```bash
llamphouse up --memory --agents agents.yaml     # http://127.0.0.1:8000/compass
python client.py                                # the same five messages to each agent
python client.py facts                          # one of them
```

```
======== summary (thread thread_40371582…)
...
ada:   What do you remember about our plans?
       I see 5 messages:
         session: Summary: Hi, I'm planning a team; Our budget is 1500 EUR; Somewhere warm, in March.
         me: I see 5 messages:
         Ada: Lisbon or Valencia, probably.
         me: I see 7 messages:
         Ada: What do you remember about our plans?
[the session as kept: 6 messages, 1 revision(s); ...]
```

The summary comes from `providers.summarize`: a stand-in that keeps the first words of what the person said.
With `OPENAI_API_KEY` set (and `pip install -r requirements.txt`), a model writes it instead (Responses API,
the model from `SUMMARY_MODEL` in `agents.yaml`).

## What to try, what to look at

- The `summary` agent sees 7 messages on turn 4: the revision starts when the session passes 6, as a job in the
  background, and the next turn sees its result. The agent never waits for it (unless it calls
  `await ctx.session.settled()`).
- `window` and `facts` keep all 10 messages; they only shape what `ctx.session.messages()` returns. `summary`
  rewrote its session: 6 messages, 1 revision.
- In Compass, open a thread, then a run, and its **Session** tab: the session as kept (its version, revisions,
  provider) and its log (entries, revisions). The Agents page shows each agent's session provider.
- `is_own(message)`: the agent's own replies (`me`) - the model's assistant role.

## The code

The agent only reads its session:

```python
session = await ctx.session.messages()      # the provider's view
for m in session:
    who = "me" if ctx.session.is_own(m) else m.sender.name or m.sender.id
```

A provider of your own overrides what it needs - here only `view`, which must be cheap (no model calls):

```python
class Facts(SessionProvider):
    def view(self, session: SessionInfo) -> Sequence[Message]:
        facts = [n for m in session.messages if m.sender.kind == SenderKind.PERSON
                 for n in re.findall(r"\d[\d,.]*", m.text or "")]
        note = Message.create(Sender.system("facts"), "Numbers mentioned: " + ", ".join(facts))
        return [note, *session.messages[-self.keep:]]
```

To revise in the background, return `Revise(...)` from `on_event`, `on_entry` or `after_run` (cheap: they
only decide), and write the new session in `async def revise(...)` - that is where a model call goes. Agent
code can ask for one too: `await ctx.session.revise(reason="...")`.

The agent file picks the provider per agent; the runtime file sets the default for all
(`sessions: {rolling_window: {size: 50}}`):

```yaml
session: {rolling_window: {size: 4}}
session: {summarizing: {summarize: {import: providers:summarize}, max_messages: 6, keep: 2}}
session: {providers.py:Facts: {keep: 2}}           # your own class, by path or module:Class
```
