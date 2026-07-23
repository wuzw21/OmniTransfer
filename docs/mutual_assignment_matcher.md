# Pair-Evidence Mutual Assignment Matcher

This experiment preserves the relation-aware cross-attention matcher and the
original dot-product mutual matcher as frozen baselines. It does not change the
production transfer path.

## Explicit Feature Matrices

The matcher keeps the existing graph inputs but does not collapse them into a
dot-product score. It constructs one source-target matrix per evidence family:

```text
C_semantic[i,j]   = MLP_semantic(|s_i - t_j|, s_i * t_j)
C_visual[i,j]     = MLP_visual(|v_i - v_j|, v_i * v_j)
C_attributes[i,j] = MLP_attributes(|a_i - a_j|, a_i * a_j)
C_context[i,j]    = MLP_context(|c_i - c_j|, c_i * c_j)
C_geometry[i,j]   = MLP_geometry(r_ij)
```

`s` contains learned text, content-description, resource-id, class, and local
text-context token buckets. `v` is the jointly trained 32x32 crop CNN state.
`a` comes from the 18 node attributes, including action flags, normalized bbox,
size, depth, and degree. `c` is the relation-aware within-screen graph state.
`r_ij` contains the 18 explicit cross-screen geometry features.

Local-anchor support is a graph propagation matrix, not a coordinate rule. A
small learned seed fusion produces `P_seed`; learned local edge gates `L_s` and
`L_t` then propagate compatible neighboring correspondences:

```text
C_anchor = normalize(L_s @ P_seed @ transpose(L_t))
```

The seven observable matrices, including visual availability, are fused by one
shared pointwise MLP:

```text
M[i,j] = MLP_fuse(C_semantic[i,j], C_visual[i,j],
                  C_attributes[i,j], C_context[i,j],
                  C_geometry[i,j], C_anchor[i,j],
                  visual_available[i,j])
```

There is no `Q @ K.T` correspondence scorer. A NULL row and column are appended
to this one final matrix. Row normalization gives `P(target | source)` and
column normalization gives `P(source | target)`. The real-node assignment score
is their log-space geometric mean:

```text
score[i,j] = 0.5 * (log P(target[j] | source[i])
                    + log P(source[i] | target[j]))
```

Source and target nodes share all encoders and pair heads. There are no
source-to-target and target-to-source execution paths and no recovery path.

## Clean Relative-XML Result

The comparison uses the same immutable `clean_relative_xml_v1` bundle, split
seed 17, training seed 17, and three supervised epochs as the frozen baseline.

| Matcher | Test Top-1 | Recall@5 | Parameters | New-image P95 |
|---|---:|---:|---:|---:|
| Relation cross-attention | 70.73% | 91.39% | 821,015 | 50.21 ms |
| Dot-product mutual matrix | 66.46% | 92.21% | 611,902 | 46.92 ms |
| Explicit pair-evidence mutual matrix | 78.83% | 94.54% | 670,459 | 48.82 ms |

The dot-product mutual matcher is smaller and has slightly better Recall@5, but
loses 4.27 Top-1 points. It remains a frozen lightweight high-recall baseline;
the explicit pair-evidence version recovers pairwise ranking without restoring
two cross-attention passes. Its model-only P95 is 8.35 ms; the reported 48.82 ms
new-image P95 includes 34.54 ms input preparation.

## Visual Ablation

The visual evidence remains optional at the input mask, but it is useful when
trained jointly. With identical model initialization, split seed 17, training
seed 17, and three epochs, the dev comparison is:

| Training | Dev Top-1 | Dev Recall@5 |
|---|---:|---:|
| Visual enabled | 78.30% | 94.87% |
| Visual disabled from training | 76.23% | 93.59% |

The crop CNN therefore contributes 2.07 Top-1 points on dev without becoming a
mandatory execution branch. Missing screenshots set `visual_available` to zero;
semantic, attribute, context, geometry, and anchor matrices remain unchanged.
