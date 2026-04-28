from contractor.utils.fs import join_path


def test_join_path_simple():
    assert join_path("/a/b", "c.txt") == "/a/b/c.txt"


def test_join_path_strips_trailing_slash_from_directory():
    assert join_path("/a/b/", "c.txt") == "/a/b/c.txt"


def test_join_path_normalizes_backslashes():
    assert join_path(r"C:\proj", "file.py") == "C:/proj/file.py"


def test_join_path_root_directory():
    assert join_path("/", "x") == "/x"


def test_join_path_accepts_pathlike_via_str():
    from pathlib import PurePosixPath

    result = join_path(str(PurePosixPath("/a/b")), "c.txt")
    assert result == "/a/b/c.txt"
