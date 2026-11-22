#!/usr/bin/env python3
"""Seeded synthetic validation; no external light-curve benchmark is assumed.

Truth is an analytic native-pixel response plus a static host, independently
evaluated from the solver's scene renderer. Products include the numerical
recovery/pull results and labelled plots. This is a small regression diagnostic,
not a substitute for a large coverage experiment on real-data noise models.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config, solver
from src.native_fit_campaign import _temporary_config


def run_validation(output_dir: str | Path, trials: int = 3, truth_flux_jy: float = .002) -> dict:
    """Run SR=1,2,4 recovery trials, return metrics, and save JSON/PNG products."""
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    native = WCS(naxis=2)
    native.wcs.crpix = [5.2, 4.8]
    native.wcs.crval = [config.TRANSIENT_RA, config.TRANSIENT_DEC]
    native.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    native.wcs.cdelt = [-1.22 / 3600, 1.22 / 3600]
    yy, xx = np.indices((9, 9), dtype=float)
    width = .9
    # Analytic detector response: the model samples a discretized table instead.
    response = np.exp(-((xx - 4.2)**2 + (yy - 3.8)**2) / (2 * width**2)) / (2 * np.pi * width**2)
    host = 2e-4 * np.exp(-((xx - 3.)**2 + (yy - 4.5)**2) / 12.)
    if not np.isfinite(truth_flux_jy) or truth_flux_jy < 0:
        raise ValueError('Injected flux must be finite and nonnegative')
    truth, noise = float(truth_flux_jy), 1e-5
    oversample = 40
    py, px = np.mgrid[-240:241, -240:241] / oversample
    calibration = np.exp(-(px**2 + py**2) / (2 * width**2))
    calibration /= calibration.sum()
    rows, example = [], None
    overrides = dict(USE_SCENE_GP_PRIOR=True, GP_KERNEL_TYPE='matern',
                     ENFORCE_GP_CENTRAL_MONOTONICITY=False, TRANSIENT_NONNEGATIVE=True,
                     FLOAT_TRANSIENT_POSITION=False, USE_HOST_GAUSSIAN_CORE=False,
                     USE_NUCLEAR_POINT_SOURCE=False, PRF_ORDER_PROJECT_THEN_CONVOLVE=True,
                     PRF_OPERATOR_MODE='anchor', PRF_SPATIAL_ANCHORS_PER_AXIS=1,
                     PRF_OVERSAMPLE=oversample, PRF_PIXEL_SCALE_ARCSEC=1.22 / oversample,
                     SIGMA_QUADRATURE_FLOOR_JY=0., SIGMA_RELATIVE_FLUX_EXPONENT=0.,
                     NUCLEAR_DATA_WEIGHT_RADIUS_NATIVE_PX=0., PRF_APPLY_PMAP_GAIN=False,
                     DIAGNOSTIC_DIR=str(dest))
    with _temporary_config(overrides), patch.object(solver, 'load_prf', lambda *a, **kw: calibration):
        for sr in (1, 2, 4):
            scene = native.deepcopy()
            scene.wcs.cdelt /= sr
            scene.wcs.crpix = (native.wcs.crpix - .5) * sr + .5
            shape = (9 * sr, 9 * sr)
            with _temporary_config({'SUPERSAMPLE_FACTOR': sr}):
                for seed in range(trials):
                    rng = np.random.default_rng(seed + 217)
                    cutouts = []
                    for i in range(4):
                        is_template = i >= 2
                        data = host + (1 + i / 5) * 1e-4
                        data = data + (0 if is_template else truth) * response + rng.normal(0, noise, (9, 9))
                        cutouts.append(dict(data=data, sigma=np.full((9, 9), noise),
                            wcs=native, raw_wcs=native, epoch_id=int(is_template),
                            is_template=is_template, filename=f'synthetic_I2_{i}_cbcd.fits',
                            mjd=(3 if is_template else 0) + (i % 2) * .2))
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = solver.run_gls_solve(cutouts, [], [], {'ell': 1.5, 'var': 1e-7},
                                                      (1.5, 1e-7), np.zeros(shape), scene, 2)
                        prediction = solver.predict_cutout_model(result, cutouts, [], [], 0)
                    flux, error = float(result['transient_fluxes'][0]), float(result['transient_errs'][0])
                    z = (cutouts[0]['data'] - prediction) / noise
                    rows.append(dict(sr=sr, seed=seed, flux_jy=flux, error_jy=error,
                                     truth_jy=truth, pull=(flux - truth) / error,
                                     residual_z_mean=float(z.mean()), residual_z_std=float(z.std())))
                    if sr == 2 and seed == 0:
                        example = (cutouts[0]['data'], prediction, z)
    pulls = np.array([r['pull'] for r in rows])
    report = dict(kind='synthetic_internal_consistency', trials_per_sr=trials,
                  truth_flux_jy=truth, rows=rows, max_abs_pull=float(np.max(np.abs(pulls))),
                  notes='Repeated seeds across SR are correlated; do not interpret them as independent coverage trials.')
    (dest / 'internal_consistency.json').write_text(json.dumps(report, indent=2) + '\n')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, values, label in zip(axes, example, ['Data (Jy/native pixel)', 'Model (Jy/native pixel)', 'Residual / sigma']):
        artist = ax.imshow(values, origin='lower', cmap='coolwarm' if 'sigma' in label else 'viridis')
        ax.set(title=label, xlabel='Detector x (pixel)', ylabel='Detector y (pixel)')
        fig.colorbar(artist, ax=ax, shrink=.8)
    fig.suptitle('Synthetic science BCD: GP host + shared nonnegative point source; SR=2')
    fig.tight_layout()
    fig.savefig(dest / 'residual_diagnostics.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for sr in (1, 2, 4):
        subset = [r for r in rows if r['sr'] == sr]
        axes[0].errorbar([sr + .05 * r['seed'] for r in subset],
                        [1e3 * r['flux_jy'] for r in subset],
                        yerr=[1e3 * r['error_jy'] for r in subset], fmt='o', label=f'SR={sr}')
    axes[0].axhline(1e3 * truth, color='black', linestyle='--', label='Injected truth')
    axes[0].set(xlabel='Scene supersampling factor', ylabel='Point-source flux (mJy)', title='Flux recovery versus GP resolution')
    axes[0].legend(fontsize=8)
    axes[1].hist(pulls, bins=np.linspace(-4, 4, 17))
    axes[1].set(xlabel='(Recovered − injected flux) / reported sigma', ylabel='Trials (correlated across SR)', title='Recovery pulls: regression diagnostic')
    fig.tight_layout()
    fig.savefig(dest / 'flux_recovery.png', dpi=160)
    plt.close(fig)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='output/review_validation')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--truth-flux-jy', type=float, default=.002)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    report = run_validation(args.output_dir, args.trials, args.truth_flux_jy)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
    if report['max_abs_pull'] > 5:
        raise SystemExit('Synthetic recovery failed the predeclared 5-sigma regression bound')
