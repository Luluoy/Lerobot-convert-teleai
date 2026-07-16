"""Local, resumable conversion workbench for LeRobot datasets."""

import os


# PyArrow 25's mimalloc backend can crash after repeated short-lived HTTP threads.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

__version__ = "0.1.0"
