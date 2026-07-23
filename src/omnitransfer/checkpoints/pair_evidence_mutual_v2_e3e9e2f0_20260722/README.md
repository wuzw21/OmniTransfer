# Pair-Evidence Mutual Matcher v2

`seeded_visual_seed17.pt` is the frozen replay checkpoint for the explicit
pair-evidence mutual assignment matcher.

- Schema: `omnitransfer_mutual_matcher_v2`
- SHA256: `e2c879b5046c86e7493f3eb18c46a24bbe180ae5c483e779a53097384fdc4ad5`
- Test Top-1: `0.7883165829`
- Test Recall@5: `0.9453517588`

The replay runtime verifies the bundled checkpoint checksum before loading it.
`OMNITRANSFER_MATCHER_CHECKPOINT` may select another v2 checkpoint explicitly;
`OMNITRANSFER_MATCHER_DEVICE`, `OMNITRANSFER_MATCHER_MIN_PROBABILITY`, and
`OMNITRANSFER_MATCHER_MIN_MARGIN` control inference without enabling a legacy
or coordinate fallback.
