# OmniTransfer v9.3: Direct-text Spatial-XML alignment

OmniTransfer v9.3 keeps one four-stage mapping pipeline:

```text
direct lexical and multimodal scoring
  -> typed local attention
  -> page-level partial matching
  -> Spatial-XML alignment
```

Text and content description enter as zero-parameter exact, token-Dice,
trigram-Dice, containment, and presence evidence. Class identity uses a
deterministic signed hash. The learned token lookup has been removed. The v9
scorer retains visual, action-state, local relation, and page-assignment
evidence, plus one shared geometric-alignment residual head:

```text
g_ij     = RelativeGeometryConsensus(P_v9, R_source, R_target)
delta_ij = MLP([direct_ij, score_ij, relation_ij, g_ij])
S_ij     = PartialAssignment(S_v9 + delta_ij)
```

The head contains 7,978 of the release's 177,372 parameters. The direct-text
checkpoint is a deterministic no-training migration of official v9.2: shared
weights are copied, the token lookup and text-projection matrix are removed,
and text-present nodes retain the old projection bias. No historical
checkpoint is selected at runtime.

## Spatial-XML alignment

For source node `i` and every target candidate `j`, inference exposes three
scale-free coordinates:

```text
p = normalized center on the whole page
q = normalized center inside the largest branching XML ancestor
o = normalized direct-child order inside that XML ancestor
```

Within the model Top-5, the page coordinate proposes the nearest candidate.
The proposal is accepted only when at least one XML coordinate agrees:

```text
j_page = argmin_j distance(p_i, p_j)
j_box  = argmin_j distance(q_i, q_j)
j_tree = argmin_j abs(o_i - o_j)

accept j_page iff j_page = j_box or j_page = j_tree
```

The accepted candidate must remain within a bounded model-score gap. The gap
is at most `5.0` normally and `2.0` when the current and proposed target nodes
have the same observable text/description, class, and action role. This stricter
gate prevents high-confidence duplicate instances from being exchanged solely
by position. The rule never reads application identity, resource id, labels, or
source-device coordinates.

The final target is `argmax` after swapping the accepted candidate with the
model winner. Swapping preserves the learned score distribution and therefore
keeps the existing pair-confidence and rank-margin fail-closed gates intact.

## Frozen development result

On the reviewed-gold 702-row ASE Dev split:

```text
Top-1       633 / 702 = 90.17%
Recall@3    683 / 702 = 97.29%
Recall@5    689 / 702 = 98.15%
```

Relative to v9.2, direct text removes 395,520 parameters at a cost of 18 Top-1
rows, while adding four Recall@3 rows. Formal Test was not accessed during
development or promotion.

The canonical Dev report SHA-256 is
`320d0cee14d01cb16fd07be4de5f6712a9f3c55370bf9593cf8e38ffc78b0864` and
the 702-row prediction artifact SHA-256 is
`11ed7e541eb6a8459bebf566022d9b5c0ca10aa3cbf6e957ad497aedbc904023`.
Warm GPU model latency is 14.05 ms p50 and 16.17 ms p95.
