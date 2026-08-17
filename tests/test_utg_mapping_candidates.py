import importlib.util
from pathlib import Path
import sys


def _module():
    path = Path(__file__).parents[1] / "scripts" / "utg_mapping_candidates.py"
    spec = importlib.util.spec_from_file_location("utg_mapping_candidates", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _action(resource_id: str, text: str = "Open", bounds=None) -> dict:
    return {
        "type": "tap",
        "target": {
            "resource_id": resource_id,
            "text": text,
            "class_name": "android.widget.Button",
            "bounds": bounds or [10, 20, 110, 80],
        },
    }


def _edge(source_state: str, target_state: str, resource_id: str, event_index: int) -> dict:
    return {
        "source_state": source_state,
        "target_state": target_state,
        "target_package": "com.example",
        "in_app": True,
        "event_index": event_index,
        "action": _action(resource_id),
        "effect": {
            "state_changed": True,
            "xml_changed": True,
            "activity_changed": False,
        },
    }


def _state(root: Path, role: str, state_id: str) -> dict:
    states = root / role / "states"
    states.mkdir(parents=True, exist_ok=True)
    (states / f"{state_id}.xml").write_text(f"<hierarchy id='{state_id}'/>", encoding="utf-8")
    (states / f"{state_id}.png").write_bytes(b"png")
    (states / f"{state_id}.json").write_text("{}\n", encoding="utf-8")
    return {
        "state_id": state_id,
        "package": "com.example",
        "activity": "MainActivity",
        "xml": f"states/{state_id}.xml",
        "screenshot": f"states/{state_id}.png",
    }


def _member(role: str, state_id: str) -> dict:
    return {
        "page_key": f"{role}:{state_id}",
        "page_id": state_id,
        "package": "com.example",
        "activity": "MainActivity",
        "device_role": role,
        "device_serial": role,
        "width": 1080,
        "height": 1920,
        "node_count": 3,
        "screenshot_path": f"/{role}/{state_id}.png",
        "transitions": {"outgoing": [{"large": "catalog"}]},
    }


def _payload(tmp_path: Path, target_specs: list[tuple[str, str]]) -> dict:
    roles = ("source", "small", "fold")
    states_by_role = {
        role: [_state(tmp_path, role, state_id) for state_id in ("before", "after", "other")]
        for role in roles
    }
    clusters = [
        {
            "cluster_id": "before-cluster",
            "members": [_member(role, "before") for role in roles],
        },
        {
            "cluster_id": "after-cluster",
            "members": [_member(role, "after") for role in roles],
        },
        {
            "cluster_id": "other-cluster",
            "members": [_member(role, "other") for role in roles],
        },
    ]
    sources = {}
    for role in roles:
        role_root = tmp_path / role
        edges = [_edge("before", "after", "source_action", 1)] if role == "source" else []
        if role == "small":
            edges = [
                _edge("before", target_state, resource_id, index)
                for index, (resource_id, target_state) in enumerate(target_specs, start=1)
            ]
        sources[role] = {
            "path": str(role_root / "utg.json"),
            "states": states_by_role[role],
            "utg_edges": edges,
        }
    return {
        "summary": {
            "schema_version": "omnitransfer.utg_cluster_review.v1",
            "cluster_catalog": clusters,
            "utg_sources": sources,
        }
    }


def test_score_action_combines_resource_text_class_geometry_and_effect() -> None:
    module = _module()
    source_edge = _edge("before", "after", "toolbar_settings", 1)
    target_edge = _edge("before", "after", "toolbar_settings_button", 2)
    source_page = {"width": 1080, "height": 1920}
    target_page = {"width": 720, "height": 1280}
    target_edge["action"]["target"]["bounds"] = [7, 13, 73, 53]

    result = module.score_action(source_edge, source_page, target_edge, target_page)

    assert result is not None
    assert result["score"] > 0.80
    assert set(result["features"]) == {"resource", "text", "class", "geometry", "effect"}
    assert result["features"]["resource"] > 0.0
    assert result["features"]["text"] == 1.0
    assert result["features"]["class"] == 1.0
    assert result["features"]["geometry"] > 0.99
    assert result["features"]["effect"] == 1.0


def test_obvious_large_margin_same_cluster_match_is_excluded(tmp_path: Path) -> None:
    module = _module()
    payload = _payload(tmp_path, [("easy", "after")])
    module.score_action = lambda *args: {"score": 0.95, "features": {}}

    candidates, audit = module.build_candidates(payload)

    assert candidates == []
    assert audit["rejection_reasons"]["obvious_large_margin"] == 1


def test_small_margin_candidates_are_kept_and_limited_to_five(tmp_path: Path) -> None:
    module = _module()
    specs = [(f"candidate_{index}", "after") for index in range(7)]
    payload = _payload(tmp_path, specs)
    scores = {f"candidate_{index}": 0.72 - index * 0.02 for index in range(7)}

    def fake_score(_source_edge, _source_page, target_edge, _target_page):
        score = scores[module.resource_id(target_edge["action"])]
        return {"score": score, "features": {"resource": score}}

    module.score_action = fake_score
    candidates, _ = module.build_candidates(payload)

    assert len(candidates) == 1
    assert "small_margin" in candidates[0]["reasons"]
    assert len(candidates[0]["target_candidates"]) == 5
    assert "transitions" not in candidates[0]["source_transition"]["source_page"]


def test_target_cluster_divergence_is_kept_above_boundary_band(tmp_path: Path) -> None:
    module = _module()
    payload = _payload(tmp_path, [("divergent", "other")])
    module.score_action = lambda *args: {"score": 0.88, "features": {}}

    candidates, _ = module.build_candidates(payload)

    assert len(candidates) == 1
    assert candidates[0]["reasons"] == ["target_cluster_divergence"]
    assert candidates[0]["target_candidates"][0]["target_cluster_relation"] == "target_cluster_divergence"


def test_same_action_with_different_endpoint_cluster_is_flagged_as_conflict(tmp_path: Path) -> None:
    module = _module()
    payload = _payload(tmp_path, [("source_action", "other")])
    module.score_action = lambda *args: {"score": 0.95, "features": {}}

    candidates, audit = module.build_candidates(payload)

    assert len(candidates) == 1
    assert candidates[0]["action_group_status"] == "endpoint_cluster_conflict"
    assert "same_action_different_target_cluster" in candidates[0]["reasons"]
    assert candidates[0]["target_candidates"][0]["same_action_as_source"] is True
    assert candidates[0]["target_candidates"][0]["endpoint_cluster_conflict"] is True
    assert audit["reasons"]["same_action_different_target_cluster"] == 1
