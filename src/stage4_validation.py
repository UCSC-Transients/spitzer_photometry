"""Independent analytic injections through the Stage 1/2 fixture contracts.

Truth is continuous optical emission integrated over native pixels. The fit uses
the unchanged established host F and native table point responses. Their finite
grid/footprint mismatch is retained and measured, not added to the error bars.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache

import numpy as np

from . import config, solver
from .gp_identifiability import StaticData, template_design, dense_host_operator
from .isolated_astrometry import StarFrame, _freeze
from .nucleus import TemplateFrame
from .stage2_validation import validation_wcs, synthetic_integrated_prf
from .stage3_validation import integrated_gaussian
from .validation_fixtures import DEVELOPMENT_SEEDS


@contextmanager
def synthetic_host_context():
    """Explicit synthetic fixture calibration; restore all global settings/caches."""
    names = dict(PRF_OVERSAMPLE=50, PRF_ORDER_PROJECT_THEN_CONVOLVE=True,
                 PRF_SPATIAL_ANCHORS_PER_AXIS=1, PRF_APPLY_PMAP_GAIN=False,
                 PRF_APPLY_APODIZATION=False, PRF_PREBLUR_IN_OVERSAMPLE_PIXELS=0.)
    old = {n: getattr(config, n, None) for n in names}
    loader = solver.load_prf
    table = synthetic_integrated_prf()
    caches = [getattr(solver, n) for n in ('_PRF_NATIVE_OPERATOR_BUNDLE_CACHE', '_PROJECTION_CACHE')]
    saved = [c.copy() for c in caches]
    try:
        for name, value in names.items():
            setattr(config, name, value)
        solver.load_prf = lambda *args, **kwargs: table.values
        for cache in caches:
            cache.clear()
        yield
    finally:
        solver.load_prf = loader
        for name, value in old.items():
            setattr(config, name, value)
        for cache, previous in zip(caches, saved):
            cache.clear()
            cache.update(previous)


@lru_cache(maxsize=12)
def _geometry(sr: int, phase: float, same_roll: bool) -> tuple:
    """Small fixed 9-native-pixel footprint, SR=1/2/4 via a coarse GP knot grid.

    Reference WCS pixel labels change with SR; host knots retain 1.22 arcsec
    spacing. This tests coordinate-unit invariance, NOT fine-grid convergence.
    """
    reference = validation_wcs(sr=sr)
    reference.wcs.crpix = (np.array([5., 5.])-.5)*sr+.5
    host_wcs = validation_wcs()
    host_wcs.wcs.crpix = [5., 5.]
    yy, xx = np.indices((9, 9))
    xy = np.column_stack([xx.ravel()-4, yy.ravel()-4])*1.22
    natives, matrices = [], []
    with synthetic_host_context():
        for i in range(5):
            native = validation_wcs(0 if same_roll else [0, 31, 79, 47, 137][i],
                                    (phase+.11*i, -.13*i))
            native.wcs.crpix = [5.+phase+.11*i, 5.-.13*i]
            matrix = dense_host_operator(lambda h: solver._apply_frame_forward_operator(
                h, host_wcs, native, (9, 9), (9, 9), 'ch2', is_full_array=True), (9, 9), (9, 9))
            natives.append(native)
            matrices.append(_freeze(matrix))
    return reference, xy, tuple(natives), tuple(matrices)


def gp_fixture(*, seed: int = 1729, flux: float = .002, separation: float = 2.,
               gradient: float = .15, phase: float = .17, masked: bool = False,
               templates: int = 3, contaminated: bool = False, sr: int = 1,
               same_roll: bool = False, nucleus_width: float = 0., position_offset: float = 0.) -> tuple:
    """Known-positive, non-GP truth with a gradient/asymmetric extended host.

    Sources share flux across templates. A deliberately contaminated exposure
    violates this assumption but is marked verified to test a *false* provenance
    assertion. Noise remains 15 microJy/native pixel in every case. No acceptance
    seed or reserved exposure ID is generated (47/137 here are development rolls).
    """
    if seed not in DEVELOPMENT_SEEDS or templates not in (1, 2, 3) or sr not in (1, 2, 4):
        raise ValueError('Declared development seeds/template count/SR required')
    reference, xy, natives, matrices = _geometry(sr, phase, same_roll)
    reference = reference.deepcopy()
    natives = tuple(w.deepcopy() for w in natives)
    to_ref = lambda p: (np.asarray(p)+.5)*sr-.5
    nucleus = to_ref([4., 4.])
    star = to_ref([4.+separation, 4.2])
    response = synthetic_integrated_prf()
    rng = np.random.default_rng(seed)
    frames = []
    truth_host, truth_star, truth_nucleus = [], [], []
    for i, native in enumerate(natives):
        def project(p):
            return np.asarray(native.world_to_pixel_values(*reference.pixel_to_world_values(*p)))
        center = project(nucleus)
        # Positive broad asymmetric mixture, outside the GP's fitted family.
        host = integrated_gaussian((9, 9), project(to_ref([3.6, 4.4])), np.eye(2)*5.**2, .025)
        host += integrated_gaussian((9, 9), project(to_ref([5.7, 2.5])), np.diag([1.8, 3.])**2, .003*gradient)
        nucleus_image = integrated_gaussian((9, 9), center, np.eye(2)*(.78**2+nucleus_width**2), .004)
        star_image = integrated_gaussian((9, 9), project(star+np.array([position_offset*sr, 0.])), np.eye(2)*.78**2,
                                         flux+(.002 if contaminated and i == 0 else 0))
        sigma = np.full((9, 9), 1.5e-5)
        background = (i-2)*1e-6
        data = host+nucleus_image+star_image+background+rng.normal(size=(9, 9))*sigma
        mask = np.ones((9, 9), dtype=bool)
        if masked:
            mask[3, 5] = False
            mask[6, 3:5] = False
        image = StarFrame(data, sigma, native, response, mask)
        split = 'train' if i < 3 else ('validation' if i == 3 else 'heldout')
        frames.append(TemplateFrame(image, f'stage4-development-{i}', split, True,
                                    background=background, provenance='independent analytic synthetic truth'))
        truth_host.append(host)
        truth_star.append(star_image)
        truth_nucleus.append(nucleus_image)
    keep = list(range(templates))+[3, 4]
    frames = tuple(frames[i] for i in keep)
    matrices = tuple(matrices[i] for i in keep)
    truth = dict(seed=seed, flux=flux, separation_native_pixels=float(np.hypot(separation, .2)),
                 source_x_offset_native_pixels=separation, gradient=gradient,
                 phase=phase, masked=masked, template_count=templates, contaminated=contaminated,
                 sr=sr, same_roll=same_roll, nucleus_width=nucleus_width, position_offset=position_offset, nucleus_flux=.004,
                 host=np.array(truth_host)[keep], star=np.array(truth_star)[keep],
                 nucleus=np.array(truth_nucleus)[keep], response_sha256=response.sha256)
    return frames, matrices, reference, xy, star, nucleus, truth


def fixture_design(fixture: tuple, family: str, partition: str, *, nucleus_profile: np.ndarray | None = None) -> StaticData:
    """Three nested static families; no science pixels accepted for selection."""
    frames, matrices, reference, xy, star, nucleus, _ = fixture
    if family not in ('gp', 'gp_stars', 'gp_stars_nucleus'):
        raise ValueError('Unknown static family')
    indices = [i for i, f in enumerate(frames) if f.split == partition]
    return template_design(tuple(frames[i] for i in indices), reference, xy,
                           tuple(matrices[i] for i in indices),
                           stars_xy=np.array([star]) if family != 'gp' else np.empty((0, 2)),
                           nucleus_xy=nucleus if family == 'gp_stars_nucleus' else None,
                           nucleus_profile=nucleus_profile,
                           operator_provenance='unchanged solver F; synthetic PRF; Jy/native-pixel host; 9x9 finite footprint')
