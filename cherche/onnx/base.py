import os

__all__ = ["clean_folder"]


def clean_folder(path: str) -> None:
    """Delete files in a folder and then the folder."""
    if os.path.exists(path):
        for file in os.listdir(path):
            if os.path.isdir(os.path.join(path, file)):
                clean_folder(os.path.join(path, file))
            else:
                os.remove(os.path.join(path, file))
        os.rmdir(path)
