"""Document loader implementations."""


def load_from_path(path: str):
    """Placeholder loader to read files from disk."""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


