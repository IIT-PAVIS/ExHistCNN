import os
import shutil


def mkdir(path):
    """
    Recursively create a folder.

    Parameters
    ----------
    path : string
        Path to generate
    """
    os.makedirs(path, exist_ok=True)


def cp(src, dst, force=False):
    """
    Copy data from src to dst.

    Parameters
    ----------
    src : string
        Input filename
    dst : string
        Output path or filename where to copy src
    """
    if force and not os.path.isdir(dst):
        mkdir(os.path.dirname(dst))
    shutil.copyfile(src, dst)


def mv(src, dst):
    """
    Move data from src to dst.

    Parameters
    ----------
    src : string
        Input filename
    dst : string
        Output path or filename where to move src
    """
    if os.path.exists(dst):
        rm(dst)
    shutil.move(src, dst)


def rm(path):
    """
    Remove a folder or file.

    Parameters
    ----------
    path : string
        Path to remove; could be a folder or a file
    """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    elif os.path.isdir(path):
        # os.removedirs(path)
        shutil.rmtree(path)
