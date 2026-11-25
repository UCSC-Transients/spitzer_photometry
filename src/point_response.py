"""Native, detector-integrated point responses for Stage 2/3.

No scene pixel area, second pixel integral, crop normalization, or pixel-phase
correction belongs here. A coefficient is in the *declared* normalization
convention; none of the supplied warm tables is certified as absolute Jy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
from pathlib import Path
import re
from typing import Literal

import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates

Normalization = Literal["table", "legacy_integral", "zero_phase_aperture"]


def file_identity(path: str | Path) -> tuple:
    """Local invalidation key including replacements and restored-mtime writes.

    SHA-256 is computed when a snapshot is read. This stat key is not a content
    hash and assumes a local filesystem with meaningful nanosecond ctime.
    """
    p = Path(path).resolve()
    s = p.stat()
    return (str(p), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def sample_native_response(
    table: np.ndarray, shape: tuple[int, int], x: float, y: float,
    oversample: float, normalization: float, *, center_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    """Sample an already pixel-integrated table at native centers.

    ``x,y`` and ``center_xy`` use zero-based pixel centers. ``normalization`` is
    fixed for the complete calibration, never computed from the output crop.
    Values outside table support are zero; lost wings stay lost.
    """
    if not np.isfinite([x, y, oversample, normalization]).all() or oversample <= 0 or normalization <= 0:
        raise ValueError("Nonfinite position or nonpositive PRF scale/normalization")
    if table.ndim != 2 or len(shape) != 2 or min(shape) <= 0:
        raise ValueError("Expected a 2D PRF and positive native shape")
    cx, cy = center_xy if center_xy is not None else ((table.shape[1]-1)/2, (table.shape[0]-1)/2)
    yy, xx = np.indices(shape, dtype=float)
    return map_coordinates(table, [cy+(yy-y)*oversample, cx+(xx-x)*oversample],
                           order=1, mode="constant", cval=0., prefilter=False) / normalization


@dataclass(frozen=True)
class NativePRF:
    """Immutable calibration snapshot, including its amplitude/centroid gauge.

    ``synthetic`` is explicit and cannot count as real-calibration evidence.
    Aperture mode fixes a single zero-phase, radius-10 native-center sum; it
    does not certify warm-mission mean-phase, color or location corrections.
    """
    values: np.ndarray
    oversample: int
    convention: Normalization = "legacy_integral"
    synthetic: bool = False
    channel: str = "ch2"
    mission: str = "unverified"
    sha256: str = ""
    center_xy: tuple[float, float] | None = None
    normalization: float = field(init=False)

    def __post_init__(self) -> None:
        a = np.asarray(self.values, dtype=np.float64)
        if a.ndim != 2 or not np.isfinite(a).all() or np.any(a < 0) or a.sum() <= 0:
            raise ValueError("PRF must be finite, nonnegative and nonempty")
        if not isinstance(self.oversample, (int, np.integer)) or self.oversample <= 0:
            raise ValueError("PRF oversampling must be a positive integer")
        if self.convention not in ("table", "legacy_integral", "zero_phase_aperture"):
            raise ValueError("Unknown PRF amplitude convention")
        # Bytes-backed arrays cannot be made writeable by a caller.
        frozen = np.frombuffer(a.tobytes(), dtype=np.float64).reshape(a.shape)
        object.__setattr__(self, "values", frozen)
        center = self.center_xy or ((a.shape[1]-1)/2, (a.shape[0]-1)/2)
        if len(center) != 2 or not np.isfinite(center).all() or not (0 <= center[0] < a.shape[1] and 0 <= center[1] < a.shape[0]):
            raise ValueError("PRF center must lie within the finite calibration array")
        object.__setattr__(self, "center_xy", tuple(center))
        norm = 1.
        if self.convention == "legacy_integral":
            norm = float(a.sum()) / self.oversample**2
        elif self.convention == "zero_phase_aperture":
            if min(center[0], center[1], a.shape[1]-1-center[0], a.shape[0]-1-center[1]) < 10*self.oversample:
                raise ValueError("Calibration does not cover the radius-10 normalization aperture")
            zero = sample_native_response(a, (21, 21), 10., 10., self.oversample, 1., center_xy=center)
            yy, xx = np.indices(zero.shape)
            norm = float(zero[(xx-10)**2+(yy-10)**2 <= 100].sum())
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid fixed calibration normalization")
        object.__setattr__(self, "normalization", norm)
        if not self.sha256:
            object.__setattr__(self, "sha256", hashlib.sha256(a.tobytes()).hexdigest())

    def sample(self, shape: tuple[int, int], x: float, y: float) -> np.ndarray:
        """Return native response per declared amplitude, with no crop renormalization."""
        return sample_native_response(self.values, shape, x, y, self.oversample,
                                      self.normalization, center_xy=self.center_xy)

    def derivatives(self, shape: tuple[int, int], x: float, y: float) -> np.ndarray:
        """Exact bilinear derivatives per native source pixel, axis first.

        At a table knot this chooses the cell on the positive table-coordinate
        side. Hard support boundaries are not differentiable. Interior fits
        avoid finite differences spanning two different interpolation cells.
        """
        yy, xx = np.indices(shape, dtype=float)
        tx = self.center_xy[0]+(xx-x)*self.oversample
        ty = self.center_xy[1]+(yy-y)*self.oversample
        valid = (tx >= 0)&(ty >= 0)&(tx < self.values.shape[1]-1)&(ty < self.values.shape[0]-1)
        ix = np.clip(np.floor(tx).astype(int), 0, self.values.shape[1]-2)
        iy = np.clip(np.floor(ty).astype(int), 0, self.values.shape[0]-2)
        u, v = tx-ix, ty-iy
        a, b = self.values[iy, ix], self.values[iy, ix+1]
        c, d = self.values[iy+1, ix], self.values[iy+1, ix+1]
        dx = ((1-v)*(b-a)+v*(d-c))*valid
        dy = ((1-u)*(c-a)+u*(d-b))*valid
        return -self.oversample/self.normalization*np.array([dx, dy])


@lru_cache(maxsize=8)
def _read_snapshot(identity: tuple, channel: str, mission: str, convention: Normalization) -> NativePRF:
    path = Path(identity[0])
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    # Read the exact hashed bytes, not a second possibly changed file version.
    import io
    with fits.open(io.BytesIO(payload), memmap=False) as hdus:
        h = hdus[0].header
        if str(h.get("CHAN", "")).strip() != channel.removeprefix("ch"):
            raise ValueError("PRF channel header does not match requested channel")
        location = re.search(r"col(\d+)_row(\d+)", path.name)
        if location is None or tuple(map(int, location.groups())) != (int(h.get("PRFCOL", -1)), int(h.get("PRFROW", -1))):
            raise ValueError("PRF detector-position header does not match filename")
        sx, sy = int(h.get("PRFXRSMP", 0)), int(h.get("PRFYRSMP", 0))
        if sx != sy or sx != 100 or hdus[0].data.shape != (2559, 2559):
            raise ValueError("Only audited 2559-square x100 APEX tables are supported")
        if mission != "warm" or "postcryogenic" not in str(h.get("COMMENT", "")).lower():
            raise ValueError("Requested mission phase is not verified by this PRF header")
        if "1279" not in str(h.get("HISTORY", "")):
            raise ValueError("Missing audited warm-PRF centroid definition")
        result = NativePRF(hdus[0].data, sx, convention, False, channel, mission,
                           digest, (1279., 1279.))
    if file_identity(path) != identity:
        raise RuntimeError("Calibration changed during snapshot read")
    return result


@dataclass(frozen=True)
class PRFGrid:
    """Strict warm full-detector calibration provider with bounded snapshot IO.

    FITS PRFCOL/PRFROW are one-based detector positions. Input detector coordinates
    are zero-based. All nonzero-weight corners are required; missing tables never
    silently fall back. Positions outside the calibrated centers use the nearest
    edge, an explicitly unvalidated extrapolation regime.
    """
    directory: str
    channel: str
    mission: str = "warm"
    convention: Normalization = "legacy_integral"

    def sample(self, shape: tuple[int, int], x: float, y: float,
               detector_x: float, detector_y: float) -> np.ndarray:
        """Blend *responses* at full detector position while preserving each gauge."""
        if self.channel not in ("ch1", "ch2") or not np.isfinite([detector_x, detector_y]).all():
            raise ValueError("Invalid calibration channel or detector coordinates")
        centers = np.array([25, 77, 129, 181, 233])
        gx, gy = np.clip([detector_x+1, detector_y+1], centers[0], centers[-1])
        ix, iy = [int(np.clip(np.searchsorted(centers, v, side="right"), 1, 4)) for v in (gx, gy)]
        u, v = (gx-centers[ix-1])/52., (gy-centers[iy-1])/52.
        result = np.zeros(shape)
        for row, col, weight in ((iy-1, ix-1, (1-u)*(1-v)), (iy-1, ix, u*(1-v)),
                                  (iy, ix-1, (1-u)*v), (iy, ix, u*v)):
            if weight == 0:
                continue
            name = f"apex_sh_IRACPC{self.channel[-1]}_col{centers[col]:03d}_row{centers[row]:03d}_x100.fits"
            table = _read_snapshot(file_identity(Path(self.directory)/name), self.channel, self.mission, self.convention)
            result += weight * table.sample(shape, x, y)
        return result

    def identity(self) -> tuple:
        """Snapshot all channel calibration identities for fit/predict consistency."""
        paths = sorted(Path(self.directory).glob(f"apex_sh_IRACPC{self.channel[-1]}_*_x100.fits"))
        return (str(Path(self.directory).resolve()), self.channel, self.mission,
                self.convention, tuple(file_identity(p) for p in paths))
