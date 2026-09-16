"""
Writing artifacts without leaking the machine they were produced on.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from . import paths

_WINDOWS_ABS = re.compile(r"^[A-Za-z]:[\\/]")


class AbsolutePathError(ValueError):
    """A value about to be written contains an absolute or home path."""


def _forbidden_substrings() -> set[str]:
    subs = {str(paths.REPO_ROOT)}
    home = os.path.expanduser("~")
    if home not in ("", "/", "~"):
        subs.add(home)
    return subs


def check_no_abs_paths(obj, where: str = "$") -> None:
    subs = _forbidden_substrings()

    def bad(s: str) -> bool:
        return (
            s.startswith("/")
            or bool(_WINDOWS_ABS.match(s))
            or any(x in s for x in subs)
        )

    def walk(x, loc):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and bad(k):
                    raise AbsolutePathError(f"absolute path in a key at {loc}")
                walk(v, f"{loc}.{k}")
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                walk(v, f"{loc}[{i}]")
        elif isinstance(x, str) and bad(x):
            raise AbsolutePathError(f"absolute path or home directory at {loc}")

    walk(obj, where)


def write_json(path, obj, indent: int = 2, allow_nan: bool = False) -> None:
    check_no_abs_paths(obj)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=indent, allow_nan=allow_nan)
            f.write("\n")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def checkpoint_sha256(path, n_hex: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n_hex]


def event_set_sha256(ids, n_hex: int = 16) -> str:
    """Order-independent fingerprint of a set of event ids (a split's ``global_match_id``)."""
    arr = np.sort(np.asarray(ids, dtype=np.int64))
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:n_hex]
