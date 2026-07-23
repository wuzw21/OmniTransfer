from omnitransfer.generalization import split_gui_odyssey_pair_rows


def _row(
    pair_id: str,
    task: str,
    source_episode: str,
    target_episode: str,
    source_device: str,
    target_device: str,
) -> dict:
    return {
        "pair_id": pair_id,
        "episode_pair_id": f"trajectory-{pair_id}",
        "meta_task": task,
        "source": {
            "episode_id": source_episode,
            "device_name": source_device,
        },
        "target": {
            "episode_id": target_episode,
            "device_name": target_device,
        },
    }


def test_gui_odyssey_split_has_no_task_episode_or_trajectory_leakage() -> None:
    rows = [
        _row("a1", "task-a", "episode-a1", "episode-a2", "phone", "fold"),
        _row("a2", "task-a", "episode-a2", "episode-a3", "fold", "tablet"),
        _row("b1", "task-b", "episode-b1", "episode-b2", "phone", "tablet"),
        _row("c1", "task-c", "episode-c1", "episode-c2", "phone", "fold"),
        _row("d1", "task-d", "episode-d1", "episode-d2", "fold", "tablet"),
        _row("e1", "task-e", "episode-e1", "episode-e2", "phone", "tablet"),
        _row("f1", "task-f", "episode-f1", "episode-f2", "phone", "fold"),
        _row("g1", "task-g", "episode-g1", "episode-g2", "fold", "tablet"),
    ]

    result = split_gui_odyssey_pair_rows(rows, seed=13, train_percent=50, dev_percent=25)

    assert sum(len(values) for values in result.splits.values()) == len(rows)
    assert all(result.splits[name] for name in ("train", "dev", "test"))
    assert result.audit["overlap"]["meta_task"] == {}
    assert result.audit["overlap"]["episode_id"] == {}
    assert result.audit["overlap"]["trajectory_id"] == {}
    pair_splits = {
        row["pair_id"]: split
        for split, values in result.splits.items()
        for row in values
    }
    assert pair_splits["a1"] == pair_splits["a2"]
    assert result.audit["device_protocol"] == "stratified_and_reported_not_disjoint"
