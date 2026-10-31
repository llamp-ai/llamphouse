# 06 Handover

One conversation, passed between agents. A front desk (`triage`) answers general questions; when you ask
about a refund it hands the thread to a billing specialist (`billing`), which carries on the same
conversation - it starts with the message the front desk was answering, asks which order, and pays the
refund. Ask billing something else and it hands the thread back. No model: keyword rules stand in for one,
so it runs without keys.

## Run it

```bash
llamphouse up --memory --agents agents.yaml   # http://127.0.0.1:8000/compass
python client.py                              # in a second terminal
```

```
you: I was charged twice for my lamp - can I get a refund?
  triage: That's one for our billing team - passing you over now.
  [triage hands the thread to billing]
  billing asks: Which order is it about? (e.g. A-1042)
  (the thread is now billing's)

you: A-1042
  billing: Done: a refund of 25.00 EUR for your desk lamp (A-1042) is on its way. Anything else?
```

## What to try

- In Compass, start a thread of `triage` and talk to it yourself: after a refund question, every message
  goes to `billing` until billing hands it back.
- Ask for a refund with the order number in the message (*"Refund for A-2077 please"*): billing doesn't
  need to ask.
- Remove `billing` from triage's `permissions.agents` in `agents.yaml`: the handover fails the run
  (`not_permitted`).

## In Compass

Open the thread: its runs alternate between `triage` and `billing`, all on one thread. A triage run that
handed over ends `completed`, with `handover: billing` on its end; the billing run after it has the same
input. The thread's state keeps `order`, whoever has the thread.

## The key idea

```python
class Triage(Agent):
    async def run(self, ctx) -> None:
        if about_billing(ctx.input.text):
            await ctx.send("That's one for our billing team - passing you over now.")
            await ctx.handover("billing", session=ctx.session)  # this run ends here
        ...
```

`ctx.handover(agent)` ends the run and gives the thread to the other agent: a run of it starts on the same
thread, with the same input, and the next messages start its runs. `session=ctx.session` gives it a copy
of the front desk's session; without it, billing's session is built from the thread, as for any new run.
Whom an agent may hand over to is in its agent file (`permissions.agents`).
