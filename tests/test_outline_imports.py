def test_outline_imports() -> None:
    import omnitransfer.eval
    import omnitransfer.features
    import omnitransfer.importers
    import omnitransfer.learned_matcher
    import omnitransfer.matchers
    import omnitransfer.reports
    import omnitransfer.self_supervised
    import omnitransfer.ui_graph
    import omnitransfer.unified_ui

    assert omnitransfer.reports.format_percent(0.767008) == "76.70%"
