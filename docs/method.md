# Method

The baseline OmniTransfer method is a structured target-candidate ranker.

## Inputs

Each source/target UI element is represented by structured features:

- text hashes and normalized text features
- resource-id and class features
- action flags such as clickable/editable/scrollable
- normalized bounding box, area, aspect ratio, and screen region
- parent/sibling/list-item context
- local graph relations and relative layout

## Ranker

For each target candidate `c_i`, build pair features:

```text
phi_i = Phi(G_s, p_s, G_t, c_i, H)
score_i = Ranker(phi_i)
```

The first deployable version uses structured machine learning, especially GBDT
style rankers, because they are data efficient, fast, and interpretable for
mixed text/layout/graph features.

## Learning Target

Training uses one positive target candidate and hard negatives from the same
target screen:

```text
L = -log softmax(score_gold)
```

Optional margin:

```text
L_margin = max(0, m - score_gold + max score_negative)
```

## Design Principle

Do not learn a new UI foundation model in the hot path. Use existing structured
UI observations and learn how to combine evidence for this relocation task.
