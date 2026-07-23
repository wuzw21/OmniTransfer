from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.download_hf_lfs_files import (
    load_file_manifest,
    load_lfs_pointers,
    select_lfs_pointers,
)


def test_lfs_pointer_parser_preserves_paths_with_spaces(monkeypatch) -> None:
    sha256 = "a" * 64

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            stdout=f"{sha256} - app name/screenshot_step0.png\ninvalid line\n"
        )

    monkeypatch.setattr("scripts.download_hf_lfs_files.subprocess.run", fake_run)

    assert load_lfs_pointers(Path("/tmp/repository")) == [
        (sha256, "app name/screenshot_step0.png")
    ]


def test_lfs_pointer_selection_requires_every_exact_path() -> None:
    pointers = [
        ("a" * 64, "mobile_domain/aw_mobile.json"),
        ("b" * 64, "mobile_domain/mobile_images.zip"),
        ("c" * 64, "web_domain/fineweb_3m.json"),
    ]

    assert select_lfs_pointers(
        pointers,
        paths=(
            "mobile_domain/aw_mobile.json",
            "mobile_domain/mobile_images.zip",
        ),
    ) == pointers[:2]
    with pytest.raises(ValueError, match="missing.zip"):
        select_lfs_pointers(pointers, paths=("mobile_domain/missing.zip",))


def test_lfs_file_manifest_supports_acquisition_manifest_schema(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
          "dataset": "owner/data",
          "source_revision": "0123456789012345678901234567890123456789",
          "selected_files": [
            {"path": "mobile/data.zip", "bytes": 10, "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
          ]
        }""",
        encoding="utf-8",
    )

    assert load_file_manifest(
        manifest,
        repo_id="owner/data",
        revision="0123456789012345678901234567890123456789",
    ) == [("a" * 64, "mobile/data.zip")]
