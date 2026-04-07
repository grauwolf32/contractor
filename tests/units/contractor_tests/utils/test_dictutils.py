from contractor.utils.dictutils import deep_merge, dict_diff


def test_deep_merge_simple():
    base = {"a": 1, "b": {"x": 1}}
    diff = {"b": {"y": 2}, "c": 3}

    result = deep_merge(base, diff)

    assert result == {
        "a": 1,
        "b": {"x": 1, "y": 2},
        "c": 3,
    }


def test_deep_merge_replaces_lists():
    base = {"a": [1, 2]}
    diff = {"a": [3]}

    result = deep_merge(base, diff)
    assert result["a"] == [3]


def test_dict_diff_added_removed_changed():
    old = {"a": 1, "b": 2, "c": {"x": 1}}
    new = {"b": 2, "c": {"x": 2}, "d": 4}

    diff = dict_diff(old, new)

    assert diff.added == {"d": 4}
    assert diff.removed == {"a": 1}
    assert "c" in diff.changed
