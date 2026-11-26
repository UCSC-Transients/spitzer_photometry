#!/usr/bin/env python3
"""Stage 1 artifact consumer: dense/scalable positive-posterior science comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gp_identifiability import fit_static, input_digest
from src.scalable_gp import Budget, GridPrior, fit_positive
from src.scalable_operators import from_dense
from src.stage4_validation import gp_fixture, fixture_design
from src.stage1_benchmark import compare_numerical_arrays
from src.validation_metrics import write_artifact_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--draws', type=int, default=8192)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, arrays = [], {}
    cases = [dict(flux=f, separation=s, masked=True, templates=t)
             for f in (0., .001, .003) for s, t in ((2., 3), (.15, 1))]
    seeds = (1729,) if args.quick else (1729, 2718, 31415)
    for seed in seeds:
        for case in cases:
            start = time.perf_counter(); index = len(records)
            entry = dict(seed=seed, config=case, truth='independent Stage 4 analytic sky; synthetic calibration',
                         science_promotion_approved=False)
            try:
                fixture = gp_fixture(seed=seed, **case)
                dense = fixture_design(fixture, 'gp_stars_nucleus', 'train')
                kwargs = dict(scale_arcsec=3.66, amplitude=5e-5, host_mean=np.full(81, 1.2e-4), source_prior_sigma=.01)
                oracle = fit_static(dense, seed=seed, draws=args.draws, **kwargs)
                data = from_dense(dense)
                prior = GridPrior(data.host_xy_arcsec, (9, 9), source_count=2, budget=Budget(batch_size=128), **kwargs)
                fit = fit_positive(data, prior, output_dir=args.output_dir, seed=seed, draws=args.draws)
                selected = np.array([40, 81, 82])
                gm = compare_numerical_arrays(oracle.gaussian_mean, fit.gaussian.mean)
                ref_cov = oracle.gaussian_covariance[np.ix_(selected, selected)]
                new_cov = fit.gaussian.selected_covariance(selected)
                gc = compare_numerical_arrays(ref_cov, new_cov)
                sd = np.sqrt(np.diag(ref_cov))
                gc['maximum_marginal_sd_scaled_error'] = float(np.max(np.abs(ref_cov-new_cov)/np.outer(sd, sd)))
                gc['passed'] = gc['passed'] and gc['maximum_marginal_sd_scaled_error'] <= 1e-8
                mean_se = np.hypot(oracle.diagnostics['mean_mc_se'], fit.diagnostics['mean_mc_se'])
                mean_z = float(np.max(np.abs(oracle.mean-fit.mean)/mean_se))
                evidence_overlap = max(oracle.diagnostics['log_evidence_95'][0], fit.diagnostics['log_evidence_95'][0]) <= min(oracle.diagnostics['log_evidence_95'][1], fit.diagnostics['log_evidence_95'][1])
                draws = np.concatenate([sample[selected].T for sample in fit.batches()])
                low, high = np.quantile(draws[:, 1], [.025, .975])
                upper = float(np.quantile(draws[:, 1], .95))
                bias = float(fit.mean[81]-case['flux'])
                heldout = from_dense(fixture_design(fixture, 'gp_stars_nucleus', 'heldout'))
                predictive = fit.predictive_evidence(heldout, seed=seed+1, draws=args.draws)
                entry.update(status='equivalent' if gm['passed'] and gc['passed'] and mean_z <= 6 and evidence_overlap else 'failed_equivalence',
                             input_sha256=input_digest(dense), gaussian_mean=gm, selected_gaussian_covariance=gc,
                             maximum_mean_mc_se=mean_z, positive_evidence_intervals_overlap=evidence_overlap,
                             dense_flux_jy=float(oracle.mean[81]), scalable_flux_jy=float(fit.mean[81]),
                             flux_bias_jy=bias, fractional_bias=bias/case['flux'] if case['flux'] else None,
                             positive_interval=[float(low), float(high)], upper_limit95_jy=upper,
                             interval_contains_positive_truth=bool(low <= case['flux'] <= high) if case['flux'] else None,
                             boundary_note='equal-tail continuous-positive interval excludes exact zero; upper-limit inclusion is not detection coverage',
                             heldout=predictive, dense_diagnostics=oracle.diagnostics, scalable_diagnostics=fit.diagnostics)
                arrays[f'case{index}_gaussian_means'] = np.array([oracle.gaussian_mean, fit.gaussian.mean])
                arrays[f'case{index}_positive_means'] = np.array([oracle.mean, fit.mean])
                arrays[f'case{index}_selected_draws'] = draws
                arrays[f'case{index}_selected_covariances'] = np.array([oracle.covariance[np.ix_(selected, selected)], fit.moments(selected)[1]])
            except Exception as exc:
                entry.update(status='failed', error_type=type(exc).__name__, error=str(exc), diagnostics=getattr(exc, 'diagnostics', {}))
            entry['wall_seconds'] = time.perf_counter()-start
            records.append(entry)
            (args.output_dir/'progress.json').write_text(json.dumps(records, indent=2))
    metrics = dict(records=records, science_promotion_approved=False,
                   uncertainty='positive-posterior credible intervals conditional on fixed geometry and hyperparameters',
                   coverage_caveat='at most three independent development noise seeds; no acceptance coverage claim',
                   failures=sum(r['status'] != 'equivalent' for r in records))
    write_artifact_bundle(args.output_dir/'comparison', metrics=metrics, arrays=arrays,
                          labels=dict(calibration='synthetic', truth='independent optical sky, different from fitted GP'))
    print(json.dumps(dict(cases=len(records), failures=metrics['failures'])))
    return int(metrics['failures'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
