from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts/build_compact_training_bundle.py"
    spec = importlib.util.spec_from_file_location("build_compact_training_bundle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pretrain_graphs_resolve_dynamic_bundle_directory(tmp_path: Path) -> None:
    module = _load_script_module()
    output = tmp_path / "custom_bundle"
    assets = output / "assets"
    assets.mkdir(parents=True)
    xml_path = assets / "screen.xml"
    xml_path.write_text("<hierarchy bounds=\"[0,0][1,1]\" />", encoding="utf-8")
    query = module.Query(
        query_id="q1",
        source={"metadata": {"platform": "ios"}},
        target_candidates=(),
        metadata={
            "app": "Example",
            "source_xml_path": "custom_bundle/assets/screen.xml",
        },
    )

    result = module._write_pretrain_graphs(
        [query],
        [query],
        output=output,
        split="train",
    )

    assert result["count"] == 1
    assert (output / result["path"]).is_file()
