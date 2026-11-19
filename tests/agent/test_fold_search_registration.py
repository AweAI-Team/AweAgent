from aweagent.scaffold.registry import agent_registry


def test_fold_search_is_available_without_internal_plugins():
    scaffold = agent_registry.get("fold_search")
    assert scaffold.__name__ == "FoldSearchAgent"
