"""
Science cutouts: pristine IRAC grids only (:func:`~preprocessing.extract_native_stamp_for_target`).
``extract_analysis_cutouts`` and ``extract_native_analysis_cutouts`` are aliases; supersampling widens the
square cutout linearly (`ANALYSIS_BOX_SIZE × SUPERSAMPLE_FACTOR`) — no interpolation or detector rotation.

``reproject_to_grid`` for mosaic/stack steps is unrelated to per-BCD stamps.
"""
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from src import config
from src import preprocessing


def _target_coord():
    return SkyCoord(float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC), unit="deg")


@pytest.mark.parametrize("supersample", [1, 2, 4])
def test_native_stamp_side_scales_with_supersample(monkeypatch, supersample):
    monkeypatch.setattr(config, "SUPERSAMPLE_FACTOR", supersample)
    captured = []

    def fake_extract(d_jy, s_jy, wcs_use, target_coord, size_xy, *, fill_value, fill_sigma):
        captured.append(tuple(int(x) for x in size_xy))
        h, w = int(size_xy[0]), int(size_xy[1])
        return np.zeros((h, w)), np.ones((h, w)), wcs_use.deepcopy()

    monkeypatch.setattr(preprocessing, "extract_native_stamp_for_target", fake_extract)

    def fake_load(_f_pair):
        from astropy.wcs import WCS

        w = WCS(naxis=2)
        w.wcs.crval = [float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)]
        w.wcs.crpix = [128.0, 128.0]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.cdelt = [-1.22 / 3600.0, 1.22 / 3600.0]
        w.wcs.pc = [[1.0, 0.0], [0.0, 1.0]]
        arr = np.zeros((256, 256), dtype=np.float64)
        return {
            "data": arr,
            "sigma": np.ones_like(arr),
            "wcs": w,
            "header": {},
            "filename": "stub_cbcd.fits",
            "mjd": 58000.0,
            "is_full_array": False,
        }

    monkeypatch.setattr(preprocessing, "load_data", fake_load)
    hits = []

    def catch_reproj(*_args, **_kwargs):
        hits.append(True)
        return np.zeros((1, 1)), None

    monkeypatch.setattr(preprocessing, "reproject_interp", catch_reproj)

    file_like = [{"image": "stub.fits", "unc": "stub_unc.fits", "epoch_id": 0, "is_template": False}]
    preprocessing.extract_native_analysis_cutouts(file_like, _target_coord())

    abox = int(config.ANALYSIS_BOX_SIZE)
    expect_side = abox * int(supersample)
    assert captured == [(expect_side, expect_side)]
    assert not hits, "pristine stamps must never call reproject_interp"


def test_analysis_and_native_paths_are_identical_geometry(monkeypatch):
    monkeypatch.setattr(config, "SUPERSAMPLE_FACTOR", 2)

    def fake_extract(d_jy, s_jy, wcs_use, target_coord, size_xy, *, fill_value, fill_sigma):
        h, w = int(size_xy[0]), int(size_xy[1])
        return np.zeros((h, w)), np.ones((h, w)), wcs_use.deepcopy()

    monkeypatch.setattr(preprocessing, "extract_native_stamp_for_target", fake_extract)

    def fake_load(_f_pair):
        from astropy.wcs import WCS

        w = WCS(naxis=2)
        w.wcs.crval = [float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)]
        w.wcs.crpix = [64.0, 64.0]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.cdelt = [-1.22 / 3600.0, 1.22 / 3600.0]
        w.wcs.pc = [[1.0, 0.0], [0.0, 1.0]]
        arr = np.zeros((128, 128), dtype=np.float64)
        return {
            "data": arr,
            "sigma": np.ones_like(arr),
            "wcs": w,
            "header": {},
            "filename": "stub.fits",
            "mjd": 58000.0,
            "is_full_array": False,
        }

    monkeypatch.setattr(preprocessing, "load_data", fake_load)
    fl = [{"image": "s.fits", "unc": "u.fits", "epoch_id": 0, "is_template": False}]
    t = _target_coord()
    ca, _ = preprocessing.extract_analysis_cutouts(fl, t)
    cb, _ = preprocessing.extract_native_analysis_cutouts(fl, t)
    ex = int(config.ANALYSIS_BOX_SIZE * 2)
    assert ca[0]["data"].shape == cb[0]["data"].shape == (ex, ex)
    assert ca[0]["stamp_frame"] == cb[0]["stamp_frame"] == "native"

