#!/usr/bin/env python3
"""Stage 1 worker protocol: paired dense/scalable posterior on identical pixels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--mode', choices=('cold', 'warm'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arrays-output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    import numpy as np
    from src import config, solver
    from src.gp_identifiability import dense_host_operator, template_design, fit_static
    from src.isolated_astrometry import StarFrame
    from src.nucleus import TemplateFrame
    from src.stage2_validation import validation_wcs, synthetic_integrated_prf
    from src.stage3_validation import integrated_gaussian
    from src.stage4_validation import synthetic_host_context

    spec = json.loads((args.output.parent/'stage5-config.json').read_text())
    head = (args.repo/'src/scalable_gp.py').is_file()
    if head:
        from src.scalable_operators import frozen_host_operator, native_template_data
        from src.scalable_gp import GridPrior, Budget, fit_positive
    n, sr, footprint = spec['native_size'], spec['sr'], spec['footprint_native']
    shape = (footprint*sr, footprint*sr)
    host_count = int(np.prod(shape))
    reference = validation_wcs(sr=sr)
    reference.wcs.crpix = (np.array([shape[1], shape[0]])+1)/2
    yy, xx = np.indices(shape)
    xy = np.column_stack((xx.ravel()-(shape[1]-1)/2, yy.ravel()-(shape[0]-1)/2))*1.22/sr
    nucleus = (np.array(shape[::-1])-1)/2
    star = nucleus+np.array([1.7, .2])*sr
    response = synthetic_integrated_prf()
    rng = np.random.default_rng(spec['seed'])
    frames = []
    for i in range(spec['bcd_count']):
        angle = [0, 31, 79, 47, 137][i % 5]
        native = validation_wcs(angle, (.17+.11*i, -.13*i))
        native.wcs.crpix = [(n+1)/2+.17+.11*i, (n+1)/2-.13*i]
        def project(point):
            return np.asarray(native.world_to_pixel_values(*reference.pixel_to_world_values(*point)))
        rotation = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
                             [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]])
        host = integrated_gaussian((n, n), project(nucleus+np.array([-.4, .4])*sr), np.eye(2)*25, .025)
        host += integrated_gaussian((n, n), project(nucleus+np.array([1.7, -1.5])*sr),
                                    rotation.T@np.diag([1.8**2, 3.**2])@rotation, .00045)
        point_star = integrated_gaussian((n, n), project(star), np.eye(2)*.78**2, .002)
        point_nucleus = integrated_gaussian((n, n), project(nucleus), np.eye(2)*.78**2, .004)
        sigma = np.full((n, n), 1.5e-5)
        background, background_sd = (i-2)*1e-6, 2e-6
        image = host+point_star+point_nucleus+background+rng.normal()*background_sd+rng.normal(size=(n, n))*sigma
        mask = np.ones((n, n), dtype=bool); mask[1, 1] = False
        frames.append(TemplateFrame(StarFrame(image, sigma, native, response, mask), f'stage5-development-{i}',
                                    'train', True, background=background, background_sigma=background_sd,
                                    provenance='independent continuous optical synthetic truth, fixed sky morphology'))
    frames = tuple(frames)
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    for frame in frames:
        for array in (frame.image.data, frame.image.sigma, frame.image.valid):
            digest.update(array.tobytes())
    input_digest = digest.hexdigest()
    record = dict(config=spec, input_sha256=input_digest, backend='scalable' if head else 'dense-reference',
                  head=subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip(),
                  mode=args.mode, threads=1, calibration='analytic synthetic; not real IRAC calibration',
                  native_pixels=sum(int(f.image.valid.sum()) for f in frames), parameters=host_count+2,
                  position_iterations='unsupported; conditional geometry inherited from Stage 4')
    arrays = {}
    start = time.perf_counter()

    def run(label):
        run_dir = args.output.parent/(args.output.stem+'-'+label)
        run_dir.mkdir()
        begin = time.perf_counter()
        with synthetic_host_context():
            config.PRF_SPATIAL_ANCHORS_PER_AXIS = spec['anchors']
            operators = []
            for frame in frames:
                native = frame.image.wcs
                if head:
                    operators.append(frozen_host_operator(reference, native, shape, (n, n), 'ch2', is_full_array=True))
                else:
                    operators.append(dense_host_operator(lambda x, w=native: solver._apply_frame_forward_operator(
                        x, reference, w, shape, (n, n), 'ch2', is_full_array=True), shape, (n, n)))
            kwargs = dict(stars_xy=np.array([star]), nucleus_xy=nucleus)
            if head:
                data = native_template_data(frames, reference, xy, tuple(operators), **kwargs)
            else:
                data = template_design(frames, reference, xy, tuple(operators),
                                       operator_provenance='established native host F, synthetic PRF', **kwargs)
            old_estimate = solver._estimate_solver_peak_memory_bytes(
                [dict(data=f.image.data) for f in frames], shape, n_scene_blocks=1, n_stars=1, n_science_epochs=0)
        setup = time.perf_counter()-begin
        selected = np.array([0, host_count//2, host_count-1, host_count, host_count+1])
        gp_kwargs = dict(scale_arcsec=3.66, amplitude=5e-5, host_mean=np.full(host_count, 1.2e-4), source_prior_sigma=.01)
        fit_start = time.perf_counter()
        if head:
            prior = GridPrior(xy, shape, source_count=2, budget=Budget(batch_size=spec['batch_size']), **gp_kwargs)
            fit = fit_positive(data, prior, output_dir=run_dir, seed=spec['seed'], draws=spec['draws'], minimum_accepted=512)
            gaussian = fit.gaussian
            gaussian_mean = gaussian.mean
            covariance_start = time.perf_counter()
            covariance = gaussian.selected_covariance(selected)
            covariance_seconds = time.perf_counter()-covariance_start
            logz = gaussian.log_evidence
            prediction_start = time.perf_counter()
            prediction = gaussian.data.design.forward(gaussian_mean)
            prediction_seconds = time.perf_counter()-prediction_start
            positive_mean, positive_cov = fit.moments(selected)
            # Only selected draws are retained in the compact public summary.
            source_samples = np.concatenate([x[selected].T for x in fit.batches()])
            details = fit.diagnostics
        else:
            fit = fit_static(data, seed=spec['seed'], draws=spec['draws'], minimum_accepted=512, **gp_kwargs)
            gaussian_mean, covariance = fit.gaussian_mean, fit.gaussian_covariance[np.ix_(selected, selected)]
            covariance_seconds = None  # included in the reference fit, not separately timed
            # Recover the already-computed Gaussian evidence; do not burden the
            # base timing with an extra update just for benchmark metadata.
            logz = fit.log_evidence-np.log(fit.diagnostics['accepted']/fit.diagnostics['prior_host_accepted'])-2*np.log(2)
            prediction_start = time.perf_counter()
            prediction = data.design@gaussian_mean
            prediction_seconds = time.perf_counter()-prediction_start
            positive_mean, positive_cov = fit.mean[selected], fit.covariance[np.ix_(selected, selected)]
            source_samples = fit.samples[:, selected]
            details = fit.diagnostics
        fit_seconds = time.perf_counter()-fit_start
        flux_error = fit.mean[-2:]-np.array([.002, .004])
        sample_centered = source_samples-positive_mean
        # MC SE of selected covariance from the IID central-product observations.
        products = sample_centered[:, :, None]*sample_centered[:, None, :]
        covariance_se = products.std(axis=0, ddof=1)/np.sqrt(len(source_samples))
        np.savez_compressed(run_dir/'positive-summary.npz', selected_samples=source_samples,
                            selected_mean=positive_mean, selected_covariance=positive_cov,
                            selected_covariance_mc_se=covariance_se, positive_mean=fit.mean,
                            mean_mc_se=details['mean_mc_se'], flux_bias_jy=flux_error)
        (run_dir/'diagnostics.json').write_text(json.dumps(details, indent=2))
        numerical = dict(gaussian_mean=gaussian_mean, selected_covariance=covariance,
                         gaussian_log_evidence=np.array([logz]), native_prediction=prediction)
        result = dict(wall_seconds=time.perf_counter()-begin, setup_seconds=setup, fit_seconds=fit_seconds,
                      prediction_seconds=prediction_seconds, selected_covariance_seconds=covariance_seconds,
                      gaussian_log_evidence=float(logz), positive_log_evidence=fit.log_evidence,
                      positive_log_evidence_95=details['log_evidence_95'], accepted=details['accepted'],
                      prior_host_accepted=details['prior_host_accepted'], flux_bias_jy=flux_error.tolist(),
                      legacy_dense_estimated_peak_bytes=old_estimate['peak_bytes'],
                      positive_arrays=str((run_dir/'positive-summary.npz').relative_to(args.output.parent)),
                      diagnostics=str((run_dir/'diagnostics.json').relative_to(args.output.parent)))
        return result, numerical

    try:
        if args.mode == 'warm':
            run('warmup')
        result, arrays = run('measured')
        record.update(result, status='completed')
    except Exception as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      failure_diagnostics=getattr(exc, 'diagnostics', None),
                      traceback=traceback.format_exc(), wall_seconds=time.perf_counter()-start)
    record['peak_memory_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
    record['rss_budget_bytes'] = 1024*2**20
    record['rss_within_budget'] = record['peak_memory_bytes'] <= record['rss_budget_bytes']
    np.savez_compressed(args.arrays_output, **arrays)
    record['numerical_arrays'] = dict(file=args.arrays_output.name,
                                     sha256=hashlib.sha256(args.arrays_output.read_bytes()).hexdigest(), keys=sorted(arrays))
    args.output.write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
