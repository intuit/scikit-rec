import skrec


def test_import():
    assert skrec is not None


def test_version():
    assert isinstance(skrec.__version__, str)
    assert len(skrec.__version__) > 0
