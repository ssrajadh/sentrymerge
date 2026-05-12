"""Camera config loader. A config describes one dashcam system: its camera
names, the filename pattern that groups sister files, and the topology of
its front-back axis (used by build_ownership_timeline to infer travel
direction).

Built-in configs live under ``sentrymerge/cams/`` and are loaded by name
(``--cam tesla``, ``--cam blackvue``). External users can also pass a path
to a custom .toml file (``--cam /path/to/mycam.toml``).

The algorithm in core.py assumes one monotonic front-back axis. Two-camera
(BlackVue) and four-camera (Tesla) setups fit; 4-corner security rigs or
360° gimbals do not.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


_BUILTIN_DIR = Path(__file__).parent / "cams"


@dataclass(frozen=True)
class CamConfig:
    name: str
    description: str
    cameras: tuple[str, ...]                  # canonical camera ids
    axis_pos: dict[str, int]                  # canonical id → axis position
    filename_re: re.Pattern[str]              # must have named groups `timestamp`, `camera`
    camera_aliases: dict[str, str] = field(default_factory=dict)  # regex-captured → canonical

    def parse(self, path: str | Path) -> tuple[Optional[str], Optional[str]]:
        """Return ``(timestamp, camera_id)`` for a filename, else ``(None, None)``.

        The regex's `camera` capture is mapped through ``camera_aliases``
        when present, so the returned id always matches an entry in
        ``cameras`` / ``axis_pos``.
        """
        m = self.filename_re.match(Path(path).name)
        if m is None:
            return (None, None)
        raw = m.group("camera")
        canonical = self.camera_aliases.get(raw, raw)
        if canonical not in self.axis_pos:
            return (None, None)
        return (m.group("timestamp"), canonical)


def load(name_or_path: str) -> CamConfig:
    """Load a config by short name (``tesla``, ``blackvue``) or filesystem path.

    A name with ``/`` or a ``.toml`` extension is treated as a path; otherwise
    the built-in directory is searched.
    """
    if "/" in name_or_path or name_or_path.endswith(".toml"):
        path = Path(name_or_path).expanduser().resolve()
    else:
        path = _BUILTIN_DIR / f"{name_or_path}.toml"
    if not path.is_file():
        builtins = sorted(p.stem for p in _BUILTIN_DIR.glob("*.toml"))
        raise FileNotFoundError(
            f"cam config not found: {name_or_path!r}. "
            f"Built-ins: {', '.join(builtins)}. "
            f"Pass a path to a .toml file for custom configs."
        )
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return _from_dict(data, source=str(path))


def list_builtin() -> list[str]:
    """Return user-selectable built-in config names. Files whose names start
    with ``_`` (e.g. the template ``_example.toml``) are hidden."""
    return sorted(
        p.stem for p in _BUILTIN_DIR.glob("*.toml") if not p.stem.startswith("_")
    )


def detect_from_filenames(filenames: Iterable[str]) -> Optional[CamConfig]:
    """Probe each built-in config against *filenames* and return the one
    that recognizes the most. Returns ``None`` if no built-in matches any.

    Used by ``--cam auto`` so users with a known dashcam system don't have
    to specify the config explicitly.
    """
    names = list(filenames)
    best_cfg: Optional[CamConfig] = None
    best_count = 0
    for name in list_builtin():
        cfg = load(name)
        count = sum(1 for f in names if cfg.parse(f)[0] is not None)
        if count > best_count:
            best_count, best_cfg = count, cfg
    return best_cfg


def _from_dict(data: dict, *, source: str) -> CamConfig:
    try:
        name = data["name"]
        cameras = tuple(data["cameras"])
        axis_pos = dict(data["axis_positions"])
        pattern = data["filename_pattern"]
    except KeyError as e:
        raise ValueError(
            f"{source}: missing required field {e.args[0]!r} "
            "(need name, cameras, axis_positions, filename_pattern)"
        ) from None

    description = data.get("description", "")
    camera_aliases = dict(data.get("camera_aliases", {}))

    # Validate that every camera has an axis position and vice versa.
    cams_set = set(cameras)
    axis_set = set(axis_pos)
    if cams_set != axis_set:
        raise ValueError(
            f"{source}: `cameras` and `axis_positions` must list the same ids. "
            f"In cameras only: {sorted(cams_set - axis_set)}. "
            f"In axis_positions only: {sorted(axis_set - cams_set)}."
        )
    for raw, canonical in camera_aliases.items():
        if canonical not in cams_set:
            raise ValueError(
                f"{source}: camera_aliases maps {raw!r} → {canonical!r}, "
                f"but {canonical!r} is not in `cameras`."
            )

    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"{source}: invalid filename_pattern regex: {e}") from None
    if "timestamp" not in regex.groupindex or "camera" not in regex.groupindex:
        raise ValueError(
            f"{source}: filename_pattern must use named groups "
            "`(?P<timestamp>...)` and `(?P<camera>...)`."
        )

    return CamConfig(
        name=name,
        description=description,
        cameras=cameras,
        axis_pos=axis_pos,
        filename_re=regex,
        camera_aliases=camera_aliases,
    )


# The default config — kept as a module attribute so functions in core.py can
# default to it without callers needing to load it explicitly. Loaded lazily
# so importing this module is cheap.
_TESLA: Optional[CamConfig] = None


def tesla() -> CamConfig:
    global _TESLA
    if _TESLA is None:
        _TESLA = load("tesla")
    return _TESLA
