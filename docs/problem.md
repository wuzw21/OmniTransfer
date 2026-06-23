# Problem

OmniTransfer studies history/context-aware UI grounding relocation for mobile
GUI agents.

## Task

Given:

```text
source UI graph G_s
source grounding p_s
target UI graph G_t
optional history H
```

Predict:

```text
target grounding p_t
```

The model outputs a ranked distribution over target candidates:

```text
s_i = f(G_s, p_s, G_t, c_i, H)
P(c_i | G_s, p_s, G_t, H) = softmax_i(s_i)
p_t = argmax_i P(c_i | ...)
```

## Non-Goal

OmniTransfer does not predict the action type. Click, input, and swipe are
provided by the cached function or planner and are only used as candidate
compatibility features.

## Why This Problem Matters

In cached GUI-agent execution, replay fails when source coordinates or source
selectors no longer point to the corresponding UI element on a target screen.
The necessary component is not a full VLA policy, but a fast relocation matcher:

```text
source grounding -> target grounding
```

This component should run on-device or near-device, with latency suitable for
replay hot paths.
