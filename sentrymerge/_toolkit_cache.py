"""Shared "last clip" cache for cross-tool integration.

Self-contained: no other sentrysearch imports. Designed to be copied
verbatim into sibling tools (e.g. sentryblur) so they share the cache
file format and read/write semantics.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_CACHE_DIR_NAME = ".sentrysearch"
_CACHE_FILENAME = "last_clip.json"
_LAST_SEARCH_FILENAME = "last_search.json"
_SCHEMA_VERSION = 1


def _cache_path() -> Path:
    return Path.home() / _CACHE_DIR_NAME / _CACHE_FILENAME


def _last_search_path() -> Path:
    return Path.home() / _CACHE_DIR_NAME / _LAST_SEARCH_FILENAME


def _atomic_write_json(payload: dict, target: Path) -> None:
    """Atomically write *payload* as JSON to *target*. Cleans up tmp on failure."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class LastClip:
    path: Path
    saved_at: datetime
    saved_by: str

    @property
    def age_seconds(self) -> int:
        now = datetime.now(timezone.utc)
        return int((now - self.saved_at).total_seconds())

    @property
    def file_exists(self) -> bool:
        return self.path.is_file()


def write_last_clip(path: Path, saved_by: str = "sentrysearch") -> None:
    """Atomically write the cache file."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError(f"path must be absolute: {path}")

    payload = {
        "version": _SCHEMA_VERSION,
        "path": str(path),
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "saved_by": saved_by,
    }
    _atomic_write_json(payload, _cache_path())


def read_last_clip() -> Optional[LastClip]:
    """Return the cached entry, or None if missing/corrupt/wrong-version."""
    cache_file = _cache_path()
    try:
        with open(cache_file) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    if not isinstance(data, dict) or data.get("version") != _SCHEMA_VERSION:
        return None

    try:
        path = Path(data["path"])
        saved_at_str = data["saved_at"]
        saved_by = data["saved_by"]
    except (KeyError, TypeError):
        return None

    try:
        # Accept the "Z" suffix that we write, plus any ISO-8601 form
        # fromisoformat understands.
        if saved_at_str.endswith("Z"):
            saved_at = datetime.fromisoformat(saved_at_str[:-1]).replace(
                tzinfo=timezone.utc,
            )
        else:
            saved_at = datetime.fromisoformat(saved_at_str)
            if saved_at.tzinfo is None:
                saved_at = saved_at.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, AttributeError):
        return None

    if not isinstance(saved_by, str):
        return None

    return LastClip(path=path, saved_at=saved_at, saved_by=saved_by)


# ----------------------------------------------------------------------------
# Last-search receipt: records the most recent SentrySearch query + results so
# downstream tools (e.g. sentrymerge) can pick up where the user left off
# without re-running the embedding/index lookup.
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class LastSearch:
    query: Optional[str]
    image_path: Optional[Path]
    results: list
    saved_at: datetime
    saved_by: str

    @property
    def age_seconds(self) -> int:
        now = datetime.now(timezone.utc)
        return int((now - self.saved_at).total_seconds())

    @property
    def is_image_query(self) -> bool:
        return self.image_path is not None


_RESULT_KEYS = ("source_file", "start_time", "end_time", "similarity_score")


def write_last_search(
    query: Optional[str],
    results: list,
    *,
    image_path: Optional[Path] = None,
    saved_by: str = "sentrysearch",
) -> None:
    """Atomically write the search-receipt file.

    Exactly one of *query* and *image_path* must be set. *results* is the list
    returned by ``sentrysearch.search.search_footage`` — each entry must carry
    the four ``_RESULT_KEYS``."""
    has_query = query is not None
    has_image = image_path is not None
    if has_query == has_image:
        raise ValueError("exactly one of query, image_path must be set")
    if image_path is not None:
        image_path = Path(image_path)
        if not image_path.is_absolute():
            raise ValueError(f"image_path must be absolute: {image_path}")

    serialized_results = []
    for r in results:
        try:
            serialized_results.append({k: r[k] for k in _RESULT_KEYS})
        except KeyError as e:
            raise ValueError(f"result missing required key: {e}") from None

    payload = {
        "version": _SCHEMA_VERSION,
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "saved_by": saved_by,
        "query": query,
        "image_path": str(image_path) if image_path is not None else None,
        "results": serialized_results,
    }
    _atomic_write_json(payload, _last_search_path())


def read_last_search() -> Optional[LastSearch]:
    """Return the cached search receipt, or None if missing/corrupt/wrong version."""
    cache_file = _last_search_path()
    try:
        with open(cache_file) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    if not isinstance(data, dict) or data.get("version") != _SCHEMA_VERSION:
        return None

    try:
        query = data["query"]
        image_path_str = data["image_path"]
        results = data["results"]
        saved_at_str = data["saved_at"]
        saved_by = data["saved_by"]
    except (KeyError, TypeError):
        return None

    if (query is None) == (image_path_str is None):
        return None  # exactly one must be present
    if query is not None and not isinstance(query, str):
        return None
    if not isinstance(results, list):
        return None
    if not isinstance(saved_by, str):
        return None
    for r in results:
        if not isinstance(r, dict) or not all(k in r for k in _RESULT_KEYS):
            return None

    image_path = Path(image_path_str) if image_path_str is not None else None

    try:
        if saved_at_str.endswith("Z"):
            saved_at = datetime.fromisoformat(saved_at_str[:-1]).replace(
                tzinfo=timezone.utc,
            )
        else:
            saved_at = datetime.fromisoformat(saved_at_str)
            if saved_at.tzinfo is None:
                saved_at = saved_at.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, AttributeError):
        return None

    return LastSearch(
        query=query,
        image_path=image_path,
        results=results,
        saved_at=saved_at,
        saved_by=saved_by,
    )
