# OmniTransfer direct-text alignment v9

This directory contains the immutable checkpoint for replay release
`omnitransfer-direct-text-alignment-v9.3.0`.

```text
File           v9_direct_text_alignment_seed29.pt
SHA-256        d1fcbfd3e4a79d74ca0bb5dc2a101360dc4e0942c31be3c83405363c58d212e4
Architecture   omnitransfer_geometric_alignment_v9
Text encoder   direct_text_evidence
Parameters     177,372 total; 7,978 alignment
Source SHA-256 9913bb389745ee6b70fe80197f0f3a270740be414344313229da9aa4b2c23875
Training       none; deterministic checkpoint migration
```

The migration deletes the 393,216-parameter learned token table and the
2,304-parameter text projection matrix. Text and content description remain
available as exact, token-Dice, trigram-Dice, containment, and presence
evidence; class identity uses a deterministic signed hash.

Frozen reviewed-gold Dev metrics are 633/702 Top-1 (90.17%), 683/702 Recall@3
(97.29%), and 689/702 Recall@5 (98.15%). The canonical evaluation report
SHA-256 is `320d0cee14d01cb16fd07be4de5f6712a9f3c55370bf9593cf8e38ffc78b0864`;
its 702-row prediction artifact SHA-256 is
`11ed7e541eb6a8459bebf566022d9b5c0ca10aa3cbf6e957ad497aedbc904023`.
Warm GPU model latency is 14.05 ms p50 and 16.17 ms p95. Formal Test was not
accessed.
