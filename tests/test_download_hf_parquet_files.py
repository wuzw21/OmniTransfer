import json

import pytest

from scripts.download_hf_parquet_files import fixed_parquet_url, load_file_manifest


def test_fixed_parquet_url_pins_convert_commit() -> None:
    assert fixed_parquet_url(
        "biglab/webui-70k-elements",
        revision="8b74f86a8418b4578ad85e73eaba3d6e7280ed9d",
        config="default",
        split="train",
        filename="0000.parquet",
    ) == (
        "https://huggingface.co/datasets/biglab/webui-70k-elements/resolve/"
        "8b74f86a8418b4578ad85e73eaba3d6e7280ed9d/default/train/0000.parquet"
    )


def test_file_manifest_rebuilds_urls_for_server_mirror(tmp_path) -> None:
    manifest = tmp_path / "files.json"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "biglab/webui-7k-elements",
                "revision": "c5974a3aed33fef3a5d2db5ac17c510ceb309573",
                "config": "default",
                "split": "train",
                "files": [{"filename": "0000.parquet", "bytes": 3, "sha256": "a" * 64}],
            }
        ),
        encoding="utf-8",
    )

    files = load_file_manifest(
        manifest,
        repo_id="biglab/webui-7k-elements",
        revision="c5974a3aed33fef3a5d2db5ac17c510ceb309573",
        config="default",
        split="train",
        endpoint="https://hf-mirror.com",
    )

    assert files[0]["url"].startswith("https://hf-mirror.com/datasets/")
    with pytest.raises(ValueError, match="revision"):
        load_file_manifest(
            manifest,
            repo_id="biglab/webui-7k-elements",
            revision="0" * 40,
            config="default",
            split="train",
            endpoint="https://hf-mirror.com",
        )
