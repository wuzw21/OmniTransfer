# Problem

OmniTransfer studies the replay-time grounding problem in GUI record-and-replay
systems.

The central observation is simple: recording an action is not the hard part.
The hard part is replaying that action after the interface has changed. A
recorded coordinate, selector, resource id, text label, or XPath may no longer
identify the intended UI target when the app version, device size, language,
list content, dialog state, or layout changes. Record-and-replay therefore
needs a target-relocation layer: given what was operated on during recording,
find the corresponding target on the current replay screen.

## Task: Replay-Time Target Relocation

Given a recorded source interaction:

```text
source screen / UI graph G_s
source grounded element e_s
recorded operation a_s
source local context C_s
```

and a target replay observation:

```text
target screen / UI graph G_t
target candidates C_t = {c_1, ..., c_n}
optional retrieved history H
```

predict the replay target:

```text
target grounded element e_t / bbox_t / point_t
```

The recorded operation is reused by the executor:

```text
a_s + e_t -> concrete replay action
```

The model does not decide whether the user wants to click, input, or swipe.
That decision was already made when the source trace was recorded. OmniTransfer
only decides where the recorded operation should be grounded on the target
screen.

## Probabilistic Formulation

For each target candidate `c_i`, the grounder assigns a score conditioned on the
recorded source target and the target replay screen:

```text
s_i = f_theta(G_s, e_s, C_s, G_t, c_i, H)
P(c_i | G_s, e_s, C_s, G_t, H) = softmax_i(s_i)
e_t = argmax_i P(c_i | ...)
```

When the correct replay target is absent or ambiguous, the candidate set may
include a NULL target:

```text
C_t^+ = C_t union {NULL}
e_t = argmax_{c_i in C_t^+} P(c_i | ...)
```

This turns replay safety into a grounding decision rather than a forced click.

## Why This Is Not Ordinary GUI Grounding

Most GUI grounding work studies:

```text
instruction + screenshot -> coordinate / region
```

Record-and-replay grounding is different:

```text
recorded source element + source context + target replay screen
  -> corresponding target element
```

The source interaction provides a strong conditioning signal that generic
instruction grounding ignores: the original element, its local row/container,
neighboring labels, affordances, and previous successful execution context.
The problem is therefore closer to UI-to-UI correspondence than to open-ended
action prediction.

## Failure Modes

Replay target relocation is necessary because common replay identifiers fail
under realistic UI drift:

- coordinate replay fails under layout, density, or device changes;
- selector replay fails when resource ids, text, or hierarchy indices drift;
- text-only replay fails on icons, duplicate labels, localization, or renamed
  actions;
- raw UI-tree exact replay can be strong on one device but is brittle for
  privacy-safe sharing and cross-state reuse;
- forced argmax replay is unsafe when the target is absent or ambiguous.

## Relation To OmniFlow

In the OmniFlow paper, this problem appears inside source-to-current Function
transfer. OmniFlow can recall a cached Function and already knows the recorded
operation sequence, but each recorded operation must still be grounded on the
current GUI before execution. OmniTransfer isolates this component:

```text
Function recall says what to replay.
OmniTransfer says where each recorded operation should be replayed.
Checker / executor decides whether and how to execute it safely.
```

This makes OmniTransfer a replay-correctness primitive for cached GUI agents,
workflow reuse, mobile testing, and GUI RPA.
