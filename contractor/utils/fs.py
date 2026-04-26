def join_path(directory: str, filename: str) -> str:
    return f"{str(directory).rstrip('/')}/{filename}".replace("\\", "/")