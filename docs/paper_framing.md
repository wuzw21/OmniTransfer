# Paper Framing

This note gives the paper-facing framing for OmniTransfer. It is intentionally
shorter than `docs/problem.md` and can be reused in an introduction,
motivation, or problem-formulation section.

## Core Claim

GUI record-and-replay systems do not mainly fail because they forget what to
do. They fail because the recorded target no longer has the same grounding at
replay time.

Traditional replay stores one or more identifiers for the operated target:

```text
coordinate, selector, resource id, text, XPath, hierarchy index
```

These identifiers are brittle under app updates, screen-size changes, language
changes, repeated list rows, dialogs, ads, personalized content, and state
changes. The replay system may still know the correct operation type, but it no
longer knows where the operation should be applied.

## Problem Statement

We formulate replay-time target relocation as source-conditioned GUI grounding.
Given a recorded source interaction and a target replay screen, predict the
target element corresponding to the recorded source element:

```text
input:
  recorded source screen G_s
  recorded source element e_s
  source local context C_s
  recorded operation a_s
  target replay screen G_t

output:
  target element e_t / bbox_t / point_t
```

The executor then reuses the recorded operation:

```text
a_s + e_t -> concrete replay action
```

The model does not perform action prediction. It performs target grounding for
a known recorded operation.

## Why This Is A Distinct GUI Grounding Problem

Most GUI grounding models solve:

```text
instruction + screenshot -> coordinate / region
```

Record-and-replay needs:

```text
recorded source element + source context + target screen
  -> corresponding target element
```

The source element and source local context are not auxiliary metadata. They are
the central supervision signal. The task is therefore closer to UI-to-UI
correspondence than to open-ended instruction grounding.

## Position In OmniFlow

In OmniFlow, Function recall and target relocation solve different problems:

```text
Function recall:
  decide what recorded Function or trace segment to reuse

OmniTransfer:
  decide where each recorded operation should be grounded now

checker / executor:
  decide whether execution is safe and apply the recorded operation
```

This makes OmniTransfer the replay-correctness layer between cache recall and
device execution.

## Method Framing

The method should be described as a two-layer source-conditioned grounder:

```text
Layer 1: target proposal grounding
  find plausible target elements or regions on the replay screen

Layer 2: source-conditioned correspondence
  select the candidate corresponding to the recorded source element
```

For a lightweight implementation, Layer 1 can use the accessibility tree or UI
graph as candidates, and Layer 2 can use a structured ranker. For a model-based
implementation, a small GUI VLM can be adapted with LoRA and a grounding head.
LoRA is an implementation mechanism, not the contribution.

## Paper-Ready Paragraph

Record-and-replay is a natural substrate for reducing the cost of GUI agents:
once a workflow has been executed successfully, later agents should reuse the
recorded trace instead of planning from scratch. However, replay correctness is
limited by target grounding. A recorded coordinate or selector often ceases to
identify the intended UI element after app updates, layout changes, device
changes, repeated list rows, or personalized content. We therefore formulate
replay-time target relocation as source-conditioned GUI grounding: given the
recorded source element and its local UI context, ground the corresponding
element on the current target screen. This separates action reuse from action
prediction: the recorded trace specifies what operation to perform, while the
grounder decides where that operation should be applied.

## Reviewer-Facing Boundary

Do not claim that this replaces general GUI agents. The point is narrower and
stronger: record-and-replay already has a recorded operation, so the missing
primitive is robust target grounding under UI drift.

Do not claim that raw UI-tree replay is always weak. Raw UI-tree replay can be
strong on one device or one app version. The contribution is robust relocation
under drift, absent/ambiguous target handling, and integration with cached GUI
agent replay.
