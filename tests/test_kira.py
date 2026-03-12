from kira import Kira


def test_import_and_defaults():
    kira = Kira()
    assert kira is not None
    assert kira._config["backend"] == "ollama"
    assert kira._config["model"] == "llama3.1:8b"
    assert kira._config["memory_path"] == "~/.kira/memory.db"


def test_lazy_internals():
    kira = Kira()
    assert kira._memory is None
    assert kira._router is None
    assert kira._backend is None