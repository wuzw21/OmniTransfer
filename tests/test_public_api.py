def test_package_exports_only_candidate_ranking() -> None:
    import omnitransfer
    import omnitransfer.runtime as runtime

    assert omnitransfer.__all__ == ["rank_action_candidates"]
    assert callable(omnitransfer.rank_action_candidates)
    assert not hasattr(omnitransfer, "action_transfer")
    assert not hasattr(runtime, "action_transfer")
    assert not hasattr(omnitransfer, "describe_action_target")
    assert not hasattr(runtime, "describe_action_target")
    assert not hasattr(omnitransfer, "runtime_preflight")
    assert not hasattr(runtime, "runtime_preflight")
    assert not hasattr(runtime, "runtime_matcher_manifest")
