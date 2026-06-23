# Method

OmniTransfer is a source-conditioned GUI grounder for record-and-replay. The
method should be understood as a two-layer grounding problem, not as action
prediction.

## Layer 1: Target Proposal Grounding

The first layer identifies plausible replay targets on the current target
screen:

```text
proposal(G_t) -> C_t = {c_1, ..., c_n}
```

In a structured implementation, candidates come from the target accessibility
tree or UI graph. In a model-based implementation, candidates can come from an
attention grounding head over target visual patches, similar in spirit to
coordinate-free GUI grounding: the model predicts a target-region heatmap rather
than directly generating a coordinate string.

## Layer 2: Source-Conditioned Correspondence

The second layer chooses which target candidate corresponds to the recorded
source element:

```text
score_i = f_theta(G_s, e_s, C_s, G_t, c_i, H)
e_t = argmax_i score_i
```

This is the part that makes the problem specific to record-and-replay. The
grounder is not asked to infer the user goal from scratch. It is given the
source element that was successfully operated on and uses its local context to
find the corresponding target element.

## Structured Baseline

The first deployable baseline is a structured candidate ranker. Each
source-target candidate pair is represented with:

- text, resource-id, and class similarity;
- action compatibility flags such as clickable, editable, and scrollable;
- normalized bounding box, area, aspect ratio, and screen region;
- parent, sibling, list-item, and row context;
- graph relations and relative layout;
- optional retrieved-history features.

For each target candidate:

```text
phi_i = Phi(G_s, e_s, C_s, G_t, c_i, H)
score_i = Ranker(phi_i)
```

The ranker can be a GBDT/LambdaMART-style model, logistic reranker, or shallow
MLP. This version is fast, inspectable, and compatible with on-device replay
constraints.

## Model-Based Version

The model-based version adapts a small GUI VLM with LoRA and a grounding head
instead of training a foundation model from scratch.

Input tokens:

```text
target screenshot patches
target UI candidate tokens
source element tokens
source local-context tokens
optional retrieved-history tokens
<GROUND> token
```

The grounding head uses the hidden state of `<GROUND>` to score target patches
or target candidates:

```text
h_g = hidden(<GROUND>)
score_patch = GroundHead(h_g, target_patch_tokens)
score_candidate_i = MatchHead(h_g, candidate_i)
```

Recommended trainable parameters:

```text
LoRA adapters on selected cross-modal / upper transformer layers
<GROUND> and source-context special token embeddings
grounding proposal head
candidate matching head
```

Frozen parameters:

```text
vision encoder
most language-model backbone weights
```

This keeps training feasible while directly modifying the grounding model for
the replay-time target-relocation task.

## Learning Target

Training uses a gold target element or NULL target for each recorded
source-target replay pair:

```text
L_candidate = -log softmax(score_gold)
```

If the model predicts a patch heatmap, target-bbox supervision can be added:

```text
L_patch = BCE(predicted_patch_mask, gold_bbox_patch_mask)
```

Optional hard-negative margin:

```text
L_margin = max(0, m - score_gold + max score_negative)
```

Full objective:

```text
L = L_candidate + lambda_patch L_patch + lambda_margin L_margin
```

## Design Principle

The contribution is not LoRA itself. LoRA is only the adaptation mechanism. The
core design is to convert GUI record-and-replay from coordinate or selector
replay into source-conditioned target grounding:

```text
recorded source target -> corresponding target replay element
```
