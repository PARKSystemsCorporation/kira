__all__ = []

def __getattr__(name: str):
    raise AttributeError("_internals is private")