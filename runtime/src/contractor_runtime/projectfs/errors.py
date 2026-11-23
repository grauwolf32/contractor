"""Content-free errors shared by workspace providers and disk primitives."""


class WorkspaceStorageError(RuntimeError):
    """Stable storage/path failure without backend details."""
