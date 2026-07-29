# Pair-Evidence Mutual Matcher v3

This directory is the immutable replay-time release
`pair-evidence-mutual-matcher-v3.0.1`. Both files contain the same learned
weights for different inference backends. The runtime accepts no checkpoint
override and verifies the selected artifact before loading it.

This checkpoint removes the learned NULL row, column, head, and inference
class. Mutual assignment ranks concrete source-target pairs. The unnormalized
pair affinity is trained with balanced binary supervision and provides the
absolute confidence used by the fail-closed runtime gate.

- Release id: `pair-evidence-mutual-matcher-v3.0.1`
- Mapping mode: `mutual_graph_matcher_no_null_v3`
- Feature schema: `pemm-v3-node-context-v1`
- Feature schema SHA256: `171735252bbdaea3da8c2fd21967963698f89e45c085cc6801172dfff66d2e58`
- PyTorch schema: `omnitransfer_mutual_matcher_v3`
- NumPy schema: `omnitransfer_numpy_mutual_matcher_v2`
- PyTorch SHA256: `61beec6da26f7aab7c51fd778ea22b5cfc956ca0cb658f1e91f4e8debc6f95b8`
- NumPy SHA256: `6e5668343419da38776e1f32ad9da610abc323637d8f6c6df38fb72ddec062b8`
- Frozen test ranking Top-1: `0.7744974874`
- Frozen test Recall@5: `0.9246231156`
- Test coverage at the fixed gates: `0.8008793970`
- Test selective accuracy at the fixed gates: `0.8690196078`
- Parameters: `670362`

The fixed absolute pair-confidence gate is `0.5`; the fixed relative
rank-margin gate is `0.15`. A failed gate returns transfer failure so the caller
can use its normal VLM fallback; source coordinates are never replayed
directly.
