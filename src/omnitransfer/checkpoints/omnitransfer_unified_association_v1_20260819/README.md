# OmniTransfer unified contextual association v1

This directory contains the frozen NumPy runtime checkpoint selected from the
relation-slot contextual matcher trained on the reviewed ASE correspondence
training split.

```text
Runtime file     relation_slots_l3_h64_seed17.npz
Runtime SHA-256  c262f03c32c4b88d2933323fe2b33007281224ef1a8aae1418a9844d354de232
Torch source     cleaned_retrain_v1/model.pt (audit artifact, not packaged)
Torch SHA-256    afcdcbab65aa8f43e3a879ccfb43991382777106c67251a28ba51e4c0975ece2
Architecture     omnitransfer_geometric_alignment_v9
Parameters       719,795
Hidden size      64
Refinement       3 local-graph + bidirectional cross-page layers
Training seed    17
```

The model resumes the prior unified checkpoint for one epoch on the cleaned
837-pair training set; 38 suspicious labels are quarantined rather than
relabelled. Frozen reviewed-gold results are 76.25% Top-1 on Dev. Test Top-1 is
79.94% in Torch and 79.86% in the packaged NumPy runtime; NumPy Top-5 is
94.01%. The runtime is fail-closed: missing or checksum-mismatched weights are
a transfer failure and never cause source-coordinate replay.
