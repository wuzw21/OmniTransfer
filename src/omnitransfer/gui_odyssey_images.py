"""Pinned individual-file transports for GUIOdyssey review screenshots."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen


MIRRORS = (
    ("Voxel51/gui-odyssey-train", "920ab344102d68fc85c1a0fe37f38d9894b19e88"),
    ("Voxel51/gui-odyssey-test", "ccdb433074e1ae374621031fc011c906a0cfc1e9"),
)


def gui_odyssey_image_index(
    filenames: set[str] | None = None,
) -> dict[str, tuple[str, str, str]]:
    """Resolve screenshot basenames to immutable mirror paths."""

    index: dict[str, tuple[str, str, str]] = {}
    for repo_id, revision in MIRRORS:
        request = Request(
            f"https://huggingface.co/api/datasets/{repo_id}/revision/{revision}",
            headers={"User-Agent": "OmniTransfer/0.1 GUIOdyssey review"},
        )
        with urlopen(request, timeout=90) as response:
            metadata = json.load(response)
        for sibling in metadata.get("siblings") or ():
            remote_path = str(sibling.get("rfilename") or "")
            filename = remote_path.rsplit("/", 1)[-1]
            if filename.endswith(".png") and (filenames is None or filename in filenames):
                index.setdefault(filename, (repo_id, revision, remote_path))
        if filenames is not None and len(index) == len(filenames):
            break
    return index
