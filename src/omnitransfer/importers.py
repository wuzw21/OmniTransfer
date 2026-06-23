"""Dataset importer outline.

Concrete importers will convert public datasets into `Query` rows without
changing the raw data layout.
"""

from __future__ import annotations

from pathlib import Path

from omnitransfer.schema import Query


def load_queries(path: str | Path) -> list[Query]:
    """Load canonical OmniTransfer queries.

    Args:
        path: Canonical JSONL path or future dataset registry id.

    Returns:
        Loaded relocation queries.
    """

    raise NotImplementedError("Importer implementation will be migrated next.")
