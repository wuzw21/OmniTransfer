# Spatial/XML alignment retirement note

This document records a retired experiment. Spatial/XML alignment is not part
of the current OmniTransfer runtime or training evaluation path.

The old experiment used a deterministic post-ranker to swap a learned winner
when page coordinates and XML branch order agreed. That made inference depend
on a rule that was absent from the training objective, so it was removed from
the unified pipeline. Local relative geometry remains available as a feature
inside the learnable matcher and is trained through correspondence labels.

The frozen checkpoint directory is retained as historical evidence only; it
must not be selected by the runtime or reported as the current matcher.
