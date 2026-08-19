# OmniTransfer unified contextual association v1

This directory contains the frozen NumPy runtime checkpoint selected from the
relation-slot contextual matcher trained on the reviewed ASE correspondence
training split.

```text
Runtime file     relation_slots_l3_h64_seed17.npz
Runtime SHA-256  0494224f76c410f17d47b4aaaeacf99e2060c1174da628884c287a6922882ada
Torch source     relation_slots_l3_h64_seed17_final.pt
Torch SHA-256    9a930d6ef5fe2c73482b6bb1df568dc40416eeee78a4e3bcf81f547e0bf5ae9a
Architecture     omnitransfer_geometric_alignment_v9
Parameters       719,795
Hidden size      64
Refinement       3 local-graph + bidirectional cross-page layers
Training seed    17
```

Frozen reviewed-gold results are 75.10% Top-1 on Dev and 78.46% Top-1 on
Test. The textless/icon hard slice is 84.62% Top-1. The runtime is fail-closed:
missing or checksum-mismatched weights are a transfer failure and never cause
source-coordinate replay.
