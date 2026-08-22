import json

import torch

from omnitransfer.learned_matcher import MatcherConfig
from scripts.evaluate_geometric_v9_matcher import (
    configure_local_fusion_ablation,
    load_evaluation_pairs,
    write_predictions,
)
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


def test_prediction_writer_emits_each_row_once(tmp_path) -> None:
    output = tmp_path / "predictions.jsonl"
    predictions = [{"row": 1}, {"row": 2}, {"row": 3}]

    artifact = write_predictions(output, predictions)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == predictions
    assert artifact["rows"] == 3


def test_local_fusion_ablation_disconnects_only_requested_paths() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state_to_pair = torch.nn.Linear(4, 3)
            self.relation_encoder = torch.nn.Sequential(
                torch.nn.Linear(3, 3),
                torch.nn.ReLU(),
                torch.nn.Linear(3, 3),
                torch.nn.Linear(3, 3),
            )
            self.correspondence_evidence = torch.nn.Linear(1, 1, bias=False)
            self.pair_scorer = torch.nn.Sequential(
                torch.nn.Linear(3, 3),
                torch.nn.Linear(3, 1),
            )

    model = Model()
    neighbour_before = model.relation_encoder[3].weight.detach().clone()

    stats = configure_local_fusion_ablation(
        model,
        disable_state_context=True,
        disable_correspondence_feedback=True,
        disable_local_correction=True,
    )

    assert set(stats) == {
        "state_context",
        "correspondence_feedback",
        "local_correction",
    }
    assert torch.count_nonzero(model.state_to_pair.weight) == 0
    assert torch.count_nonzero(model.correspondence_evidence.weight) == 0
    assert torch.count_nonzero(model.pair_scorer[-1].weight) == 0
    assert torch.equal(model.relation_encoder[3].weight, neighbour_before)
