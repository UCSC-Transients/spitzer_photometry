#!/usr/bin/env python3
"""Stage 4 template-only GP leakage experiment; no reserved acceptance data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from src.gp_identifiability import fit_static, model_weights, mixture_moments, input_digest
from src.stage4_validation import gp_fixture, fixture_design
from src.validation_fixtures import DEVELOPMENT_SEEDS
from src.validation_metrics import write_artifact_bundle


def ci(values: list[float]) -> list[float] | None:
    """Descriptive t interval across independent development seeds (small N)."""
    a = np.asarray(values)
    if len(a) < 2:
        return None
    radius = t.ppf(.975, len(a)-1)*np.std(a, ddof=1)/np.sqrt(len(a))
    return [float(a.mean()-radius), float(a.mean()+radius)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--quick', action='store_true', help='Required-CI subset; same physical/noise model')
    parser.add_argument('--draws', type=int, default=8192)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    seeds = DEVELOPMENT_SEEDS[:1] if args.quick else DEVELOPMENT_SEEDS
    scenarios = {'separated': {}, 'near_nucleus': {'separation': .2},
                 'masked_few_templates': {'masked': True, 'templates': 1, 'phase': .43},
                 'steep_host': {'gradient': 2.}, 'offset_star': {'position_offset': .15},
                 'contaminated': {'contaminated': True}, 'resolved_nucleus': {'nucleus_width': .4},
                 'single_roll': {'same_roll': True}}
    if args.quick:
        scenarios = {k: scenarios[k] for k in ('separated', 'near_nucleus', 'masked_few_templates')}
    fluxes = [.0, .0005, .002, .004] if not args.quick else [.0005, .004]
    scales, amplitudes = [1.22, 2.44, 4.88], [2.5e-5, 7.5e-5]
    families = [('gp', None), ('gp_stars', None), ('gp_stars_nucleus', None),
                ('gp_stars_nucleus', np.array([.4, .4, 0.]))]
    if args.quick:
        scales, amplitudes, families = [1.22, 4.88], [5e-5], families[:3]
    arrays, records, failures = {}, [], []
    example_fit = None
    begin = time.perf_counter()
    # This comparison uses templates only; injected source recovery never sets weights.
    for seed in seeds:
        fixture = gp_fixture(seed=seed)
        if not arrays:
            arrays['native_host_truth_example'] = fixture[-1]['host'][0]
        fits, labels = [], []
        for family, profile in families:
            for scale in scales:
                for amplitude in amplitudes:
                    key = f'sweep_{seed}_{family}_{profile is not None}_{scale}_{amplitude}'
                    data = fixture_design(fixture, family, 'train', nucleus_profile=profile)
                    fit = None
                    try:
                        fit = fit_static(data, scale_arcsec=scale, amplitude=amplitude,
                                         host_mean=np.full(81, 1.2e-4), source_prior_sigma=.01,
                                         seed=seed, draws=args.draws)
                        scores = {part: fit.predictive_evidence(fixture_design(fixture, family, part, nucleus_profile=profile),
                                                               seed=seed+1, draws=args.draws)
                                  for part in ('validation', 'heldout')}
                        fits.append(fit)
                        labels.append(key)
                        record = dict(key=key, kind='sweep', family=family, resolved=profile is not None,
                                      input_sha256=input_digest(data), provenance=data.provenance,
                                      calibration_gauge=data.calibration_gauge, scores=scores, diagnostics=fit.diagnostics,
                                      mean=fit.mean.tolist(), source_names=data.source_names)
                        arrays[key+'_mean'], arrays[key+'_covariance'] = fit.mean, fit.covariance
                        arrays[key+'_samples'] = (fit.samples[:, 81:] if len(data.source_names) else fit.samples[:128]).copy()
                        arrays[key+'_design'], arrays[key+'_y'], arrays[key+'_noise'] = data.design, data.y, data.noise
                        records.append(record)
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                        failures.append(dict(key=key, error=str(error), diagnostics=getattr(error, 'diagnostics', {}),
                                             input_sha256=input_digest(data), provenance=data.provenance,
                                             completed_fit_diagnostics=fit.diagnostics if fit is not None else None))
                        arrays[key+'_failed_y'], arrays[key+'_failed_design'], arrays[key+'_failed_noise'] = data.y, data.design, data.noise
                        if fit is not None:
                            arrays[key+'_failed_mean'], arrays[key+'_failed_covariance'] = fit.mean, fit.covariance
        # Do not conceal failed candidates by silently renormalizing the survivors.
        if len(fits) == len(families)*len(scales)*len(amplitudes):
            weights = model_weights(tuple(fits), np.ones(len(fits)))
            mix = mixture_moments(tuple(fits), weights['weights'])
            records.append(dict(kind='model_mixture', seed=seed, candidates=labels,
                                weights=weights['weights'].tolist(), status=weights['status'],
                                mc_bounds=weights['orthant_mc_weight_bounds'].tolist(),
                                between_variance=np.diag(mix['between_covariance']).tolist(),
                                source_names=mix['source_names']))
            arrays[f'mixture_{seed}_covariance'] = mix['covariance']
        print('template sweep', seed, len(fits), 'completed', flush=True)
    # Paired injection slopes at two predeclared kernel priors; no choice based on recovery.
    for scenario, options in scenarios.items():
        for seed in seeds:
            for flux in fluxes:
                fixture = gp_fixture(seed=seed, flux=flux, **options)
                for scale in (1.22, 4.88):
                    key = f'injection_{scenario}_{seed}_{flux}_{scale}'
                    data = fixture_design(fixture, 'gp_stars_nucleus', 'train')
                    fit = None
                    try:
                        fit = fit_static(data, scale_arcsec=scale, amplitude=5e-5,
                                         host_mean=np.full(81, 1.2e-4), source_prior_sigma=.01,
                                         seed=seed, draws=args.draws)
                        interval = np.quantile(fit.samples[:, -2], [.025, .975])
                        source_column = data.design[:, -2]
                        # Apparent point flux in the fitted host, in the fixed noise metric.
                        weights = np.linalg.solve(data.noise, source_column)
                        host_projection = float(weights@(data.design[:, :81]@fit.mean[:81])/(weights@source_column))
                        scores = {part: fit.predictive_evidence(fixture_design(fixture, 'gp_stars_nucleus', part),
                                                               seed=seed+1, draws=args.draws)
                                  for part in ('validation', 'heldout')}
                        records.append(dict(key=key, kind='injection', scenario=scenario, seed=seed, flux=flux,
                                            scale_arcsec=scale, recovered=float(fit.mean[-2]), nucleus=float(fit.mean[-1]),
                                            host_projection=host_projection, interval=interval.tolist(),
                                            covers=bool(interval[0] <= flux <= interval[1]),
                                            posterior_upper95=float(np.quantile(fit.samples[:, -2], .95)),
                                            bias=float(fit.mean[-2]-flux), input_sha256=input_digest(data),
                                            provenance=data.provenance, calibration_gauge=data.calibration_gauge,
                                            config=options, scores=scores, diagnostics=fit.diagnostics))
                        arrays[key+'_components'] = fit.samples[:, -2:].copy()
                        arrays[key+'_host_mean'] = fit.mean[:81]
                        arrays[key+'_covariance'] = fit.covariance
                        arrays[key+'_y'], arrays[key+'_design'], arrays[key+'_noise'] = data.y, data.design, data.noise
                        for component in ('host', 'star', 'nucleus'):
                            arrays[key+'_truth_'+component] = fixture[-1][component]
                        if scenario == 'near_nucleus':
                            example_fit = fit
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                        failures.append(dict(key=key, error=str(error), diagnostics=getattr(error, 'diagnostics', {}),
                                             input_sha256=input_digest(data), provenance=data.provenance,
                                             completed_fit_diagnostics=fit.diagnostics if fit is not None else None))
                        arrays[key+'_failed_y'], arrays[key+'_failed_design'], arrays[key+'_failed_noise'] = data.y, data.design, data.noise
                        if fit is not None:
                            arrays[key+'_failed_mean'], arrays[key+'_failed_covariance'] = fit.mean, fit.covariance
            print('injections', scenario, seed, flush=True)
        (out/'progress.json').write_text(json.dumps(dict(records=records, failures=failures), indent=2, allow_nan=False))
    summaries = []
    for scenario in scenarios:
        for scale in (1.22, 4.88):
            rows = [r for r in records if r['kind'] == 'injection' and r['scenario'] == scenario and r['scale_arcsec'] == scale]
            slopes = {name: [] for name in ('recovered', 'host_projection', 'nucleus')}
            for seed in seeds:
                selected = [r for r in rows if r['seed'] == seed]
                if len(selected) == len(fluxes):
                    for name in slopes:
                        slopes[name].append(float(np.polyfit([r['flux'] for r in selected], [r[name] for r in selected], 1)[0]))
            summaries.append(dict(scenario=scenario, scale_arcsec=scale, trials=len(rows),
                                  slopes={name: dict(per_seed=values, mean=float(np.mean(values)) if values else None,
                                                     descriptive_95=ci(values)) for name, values in slopes.items()},
                                  bias_mean=float(np.mean([r['bias'] for r in rows])) if rows else None,
                                  rmse=float(np.sqrt(np.mean([r['bias']**2 for r in rows]))) if rows else None,
                                  positive_coverage_count=sum(r['covers'] for r in rows if r['flux'] > 0),
                                  positive_trials=sum(r['flux'] > 0 for r in rows),
                                  zero_equal_tail_coverage_count=sum(r['covers'] for r in rows if r['flux'] == 0),
                                  zero_trials=sum(r['flux'] == 0 for r in rows)))
    payload = dict(records=records, summaries=summaries, failures=failures,
                   truth='independent analytic synthetic optics; not real IRAC validation',
                   config=dict(draws=args.draws, seeds=seeds, scales_arcsec=scales,
                               amplitudes_jy_per_native_pixel=amplitudes, source_prior_sigma_jy=.01,
                               host_mean_jy_per_native_pixel=1.2e-4, numerical_threads=1),
                   science_promotion_approved=False,
                   elapsed_seconds=time.perf_counter()-begin,
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                   limitations=['fixed geometry; no real-template provenance', 'finite 9x9 host footprint',
                                'only three independent noise seeds; small-N descriptive intervals',
                                'zero equal-tail intervals cannot cover zero for a continuous positive prior',
                                'source-prior and omitted geometry uncertainty remain explicit',
                                'no agreed science bias threshold; no negligible-leakage claim'])
    if not arrays:
        arrays['no_completed_fits'] = np.zeros((1, 1))
    write_artifact_bundle(out, metrics=payload, arrays=arrays,
                          labels={'plot_title': 'Stage 4 synthetic evidence; no science promotion',
                                  'truth': payload['truth']})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for scale in (1.22, 4.88):
        rows = [r for r in records if r['kind']=='injection' and r['scenario']=='separated' and r['scale_arcsec']==scale]
        axes[0].scatter([r['flux']*1e3 for r in rows], [r['recovered']*1e3 for r in rows], label=f'{scale} arcsec')
    axes[0].set(xlabel='Injected star (mJy)', ylabel='Posterior mean star (mJy)', title='Independent injections')
    axes[0].legend()
    for family, profile in families:
        rows = [r for r in records if r['kind']=='sweep' and r['family']==family and r['resolved']==(profile is not None)]
        axes[1].scatter([r['diagnostics']['scale_arcsec'] for r in rows],
                        [r['scores']['heldout']['log_score'] for r in rows], label=family+(' resolved' if profile is not None else ''))
    axes[1].set(xlabel='GP scale (arcsec)', ylabel='Held-out log predictive density', title='No science-based selection')
    axes[1].legend(fontsize=6)
    if example_fit is None:
        correlation = np.zeros((1, 1))
        axes[2].text(0, 0, 'No completed near-nucleus fit')
    else:
        sd = np.sqrt(np.diag(example_fit.covariance))
        correlation = example_fit.covariance/np.outer(sd, sd)
    artist = axes[2].imshow(correlation, vmin=-1, vmax=1, cmap='coolwarm', origin='lower')
    axes[2].set(title='Host/star/nucleus correlation', xlabel='Component index', ylabel='Component index')
    fig.colorbar(artist, ax=axes[2])
    fig.tight_layout()
    fig.savefig(out/'leakage-scale-correlation.png', dpi=150)
    fig.savefig(out/'leakage-scale-correlation.svg')
    plt.close(fig)
    public = dict(truth=payload['truth'], config=payload['config'], summaries=summaries,
                  science_promotion_approved=False, failures=failures,
                  mismatch_count=sum(r['diagnostics']['static_mismatch'] for r in records if 'diagnostics' in r),
                  completed_fits=sum('diagnostics' in r for r in records),
                  source_hashes={name: hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
                                 for name in ('src/gp_identifiability.py', 'src/stage4_validation.py', 'scripts/validate_stage4.py')},
                  injections=[{k: r[k] for k in ('key', 'scenario', 'seed', 'flux', 'scale_arcsec', 'recovered',
                              'nucleus', 'host_projection', 'interval', 'covers', 'posterior_upper95', 'bias', 'input_sha256')}
                              for r in records if r['kind']=='injection'],
                  limitations=payload['limitations'])
    (out/'public-evidence.json').write_text(json.dumps(public, indent=2, allow_nan=False))
    # Persist plot checksums alongside the Stage 1 artifact manifest.
    (out/'extra-checksums.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                      for p in out.iterdir() if p.suffix in ('.png', '.svg')}, indent=2))
    print(json.dumps(dict(failures=len(failures), records=len(records), seconds=payload['elapsed_seconds'])))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
