#!/usr/bin/env python3
"""Reproduce Stage 3 template fits without using reserved acceptance seeds."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.nucleus import compare_models, fit_nucleus, residual_signatures
from src.stage3_validation import TRUTH_CASES, nucleus_fixture
from src.validation_fixtures import DEVELOPMENT_SEEDS
from src.validation_metrics import write_artifact_bundle


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--quick', action='store_true', help='One development noise seed; full truth cases retained')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, arrays, fitted = [], {}, {}
    seeds = (1729,) if args.quick else (None, *DEVELOPMENT_SEEDS)
    for case in TRUTH_CASES:
        for seed in seeds:
            begin = time.perf_counter()
            frames, scene, truth = nucleus_fixture(case, seed)
            comparison = compare_models(frames, scene)
            key = f'{case}_{seed}'
            metrics = comparison['metrics']
            for model, failure_arrays in comparison['failed_arrays'].items():
                for name, value in failure_arrays.items():
                    arrays[key+'_'+model+'_failed_'+name] = value
            for model, fit in comparison['fits'].items():
                predictions = [*fit.training_predictions(), *[fit.predict(f, propagate=False)[0] for f in frames[3:]]]
                residual = np.array([f.image.data-p for f, p in zip(frames, predictions)])
                prefix = key+'_'+model
                arrays[prefix+'_residual'] = residual
                arrays[prefix+'_parameters'] = fit.parameters
                arrays[prefix+'_covariance'] = fit.covariance
                arrays[prefix+'_objective'] = np.asarray(fit.diagnostics['objective_history'])
                factors = [None]*3+[fit.predict(f)[1] if fit.diagnostics['covariance_available'] else None for f in frames[3:]]
                signatures = [residual_signatures(f, scene, p, probe_xy=truth['probe_xy'], center_xy=fit.center_xy,
                                                   predictive_factor=factor,multiplicity=3*len(frames))
                              for f,p,factor in zip(frames,predictions,factors)]
                for i,factor in enumerate(factors):
                    if factor is not None: arrays[prefix+f'_predictive_factor_{i}']=factor
                metrics[model]['signatures'] = signatures
                metrics[model]['centroid_bias_reference_pixels'] = (fit.center_xy-truth['centroid']).tolist()
                metrics[model]['nuclear_flux_bias'] = float(sum(fit.flux[:fit.problem.nc])-.01)
                # Nonnegative nearby source, fixed template prediction; no transient tuning.
                levels = np.array([0., .0001, .0003, .001])
                biases = np.array([r['source_bias'] for r in signatures])
                recovered = np.maximum(0, levels[:, None]+biases[None])
                arrays[prefix+'_probe_injected'] = levels
                arrays[prefix+'_probe_recovered'] = recovered
                metrics[model]['probe_recovery_slope'] = np.polyfit(levels, recovered, 1)[0].tolist()
                if fit.diagnostics['covariance_available']:
                    variance = fit.covariance[:fit.problem.nc, :fit.problem.nc].sum()*scene.flux_scale**2
                    metrics[model]['total_nuclear_local_95_covers_truth'] = bool(abs(sum(fit.flux[:fit.problem.nc])-.01) <= 1.96*np.sqrt(variance))
            candidates = [arrays[key+'_'+m+'_residual'] for m in comparison['fits']]
            if candidates:
                arrays[key+'_model_residual_envelope_min'] = np.min(candidates,axis=0)
                arrays[key+'_model_residual_envelope_max'] = np.max(candidates,axis=0)
            arrays[key+'_data'] = np.array([f.image.data for f in frames])
            arrays[key+'_sigma'] = np.array([f.image.sigma for f in frames])
            arrays[key+'_valid'] = np.array([f.image.valid for f in frames])
            arrays[key+'_phase'] = truth['phases']
            arrays[key+'_roll'] = truth['rolls']
            records.append(dict(truth=truth, models=metrics, config=dict(quadrature_order=scene.quadrature_order,
                           host_order=scene.host_order,centroid_radius=scene.centroid_radius,maximum_width=scene.maximum_width,
                           flux_scale=scene.flux_scale,host_profile=scene.host_profile,
                           response_sha256=frames[0].image.response.sha256,
                           wcs_headers=[f.image.wcs.to_header(relax=True).tostring() for f in frames]),seconds=time.perf_counter()-begin))
            fitted[key] = comparison
            # Preserve failures/partial progress even if a later case fails.
            (args.output_dir/'progress.json').write_text(json.dumps(json_safe(records), indent=2, allow_nan=False))
            print(key, {m:r['status'] for m,r in metrics.items()}, f'{time.perf_counter()-begin:.2f}s', flush=True)
    progression = []
    if not args.quick:
        for geometry, sr in [('single', 1), ('dithers', 1), ('rolls', 2), ('rolls', 4)]:
            frames, scene, truth = nucleus_fixture('extent', geometry=geometry, sr=sr)
            for model in ('point', 'gaussian', 'core_envelope'):
                try:
                    fit = fit_nucleus(tuple(f for f in frames if f.split == 'train'), scene, model)
                    progression.append(dict(geometry=geometry, sr=sr, model=model, diagnostics=fit.diagnostics,
                                            center_native=(fit.center_xy+.5)/sr-.5, profile_native=fit.profile/np.array([sr,sr,1.])))
                except (ValueError, RuntimeError) as error:
                    progression.append(dict(geometry=geometry, sr=sr, model=model, error=str(error)))
    payload = dict(records=records, progression=progression, default='point',
                   science_promotion_approved=False, truth_label='independent analytic synthetic instrument',
                   real_templates='unavailable without verified-template/calibration manifest',
                   interval_interpretation='local interior curvature only; boundary fits unavailable',
                   limitations=['science-bias tolerance unset', 'no real-template acceptance', 'absolute warm calibration open',
                                'fixed catalog positions and controlled host shape', 'local covariance is not global model uncertainty',
                                'nearby source projection is conditional on fixed static fit', 'model alternatives remain explicit'])
    write_artifact_bundle(args.output_dir, metrics=json_safe(payload), arrays=arrays,
                          labels={'plot_title':'Stage 3 — synthetic residuals, not real calibration',
                                  'truth':'independent analytic optics integrated over native pixels'})
    # Four-case signed residual, phase/roll, centroid/profile, convergence summaries.
    fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    for row, case in enumerate(TRUTH_CASES):
        key = f'{case}_{seeds[0]}'
        for col, model in enumerate(('point', 'gaussian', 'core_envelope')):
            a = arrays.get(key+'_'+model+'_residual')
            ax = axes[row, col]
            if a is not None:
                lim = max(float(np.max(np.abs(a[-1]))), 1e-10)
                im = ax.imshow(a[-1], origin='lower', cmap='coolwarm', vmin=-lim, vmax=lim)
                fig.colorbar(im, ax=ax, label='synthetic flux / native pixel')
            else:
                ax.text(.1,.5,'Fit failed (see metrics)', transform=ax.transAxes)
            ax.set_title(case+' / '+model+' / held-out')
    fig.tight_layout(); fig.savefig(args.output_dir/'signed-residuals.png', dpi=130); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for r in records:
        case, seed = r['truth']['case'], r['truth']['seed']
        if seed != seeds[0]: continue
        for model, m in r['models'].items():
            if m['status'] != 'converged': continue
            label=case+'/'+model
            axes[0,0].plot(r['truth']['rolls'], [x['source_bias'] for x in m['signatures']], '.-', label=label)
            axes[0,1].plot(r['truth']['phases'][:,0], [x['derivative_coefficients'][0] for x in m['signatures']], '.-')
            axes[1,0].plot(*m['centroid_xy'], '.', label=label)
            h=m['objective_history']; axes[1,1].semilogy(np.arange(len(h)),h)
    axes[0,0].set(xlabel='roll (degrees)', ylabel='nearby-source signed bias (synthetic flux)')
    axes[0,1].set(xlabel='detector dither x (pixel)', ylabel='sky-x derivative coefficient')
    axes[1,0].set(xlabel='nucleus reference x (pixel)',ylabel='nucleus reference y (pixel)')
    axes[1,1].set(xlabel='residual evaluation',ylabel='objective including registration prior')
    axes[0,0].legend(fontsize=6, ncol=2)
    fig.tight_layout();fig.savefig(args.output_dir/'phase-roll-centroid-convergence.png', dpi=130);plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(10,4))
    for r in records:
        if r['truth']['seed'] != seeds[0]:continue
        for model,m in r['models'].items():
            if m['status']!='converged':continue
            axes[0].plot(m['profile'][0],m['profile'][1],'.',label=r['truth']['case']+'/'+model)
            pred=m.get('validation_predictive')
            if pred:axes[1].plot(m['parameter_count'],pred['nll'],'.')
    axes[0].set(xlabel='intrinsic sigma x (sky pixel)',ylabel='intrinsic sigma y (sky pixel)')
    axes[0].legend(fontsize=6)
    axes[1].set(xlabel='fitted parameter count',ylabel='validation predictive negative log likelihood')
    fig.tight_layout();fig.savefig(args.output_dir/'profile-comparison.png',dpi=130);plt.close(fig)
    manifest=args.output_dir/'manifest.json'
    m=json.loads(manifest.read_text())
    m['files'].update({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output_dir.glob('*.png')})
    manifest.write_text(json.dumps(m,indent=2))
    failures=sum(v['status']=='failed' for r in records for v in r['models'].values())
    print(f'{failures} fit failures retained; scientific promotion unapproved',flush=True)
    # A benchmark with any failed fit must not report an all-pass exit status.
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
