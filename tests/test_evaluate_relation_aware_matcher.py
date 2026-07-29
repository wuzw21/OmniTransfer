import json

from omnitransfer.learned_matcher import MatcherConfig
from scripts.evaluate_relation_aware_matcher import load_evaluation_pairs
from tests.test_mapping_training import _record


def test_evaluation_uses_the_same_ui_correspondence_adapter_and_split(tmp_path) -> None:
    record = _record()
    record["split"] = "test"
    record["label_status"] = "gold"
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    pairs, adapter = load_evaluation_pairs(
        [path],
        split="test",
        config=MatcherConfig(),
    )

    assert len(pairs) == 1
    assert adapter["allowed_splits"] == ["test"]
    assert adapter["set_valued_match_rows_retained"] == 1
