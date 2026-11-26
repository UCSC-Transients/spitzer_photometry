#!/usr/bin/env python3
"""Stage 5 consumers of the unchanged Stage 1 paired-process/budget interface."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_stage1_benchmarks import _collect, _numerical_check, _performance_input
from src.stage1_benchmark import compare_performance


CASES = {
    'small': dict(native_size=5, footprint_native=5, sr=1, bcd_count=1, anchors=1),
    'bcd3': dict(native_size=5, footprint_native=5, sr=1, bcd_count=3, anchors=1),
    'bcd10': dict(native_size=5, footprint_native=5, sr=1, bcd_count=10, anchors=1),
    'sr2': dict(native_size=5, footprint_native=5, sr=2, bcd_count=3, anchors=1),
    'sr4': dict(native_size=5, footprint_native=5, sr=4, bcd_count=3, anchors=1),
    'native9': dict(native_size=9, footprint_native=9, sr=1, bcd_count=3, anchors=1),
    'anchors3': dict(native_size=9, footprint_native=9, sr=1, bcd_count=3, anchors=3),
    'native12': dict(native_size=12, footprint_native=12, sr=1, bcd_count=3, anchors=1),
    'large': dict(native_size=9, footprint_native=32, sr=4, bcd_count=10, anchors=3),
}


def positive_comparison(records: dict, directory: Path) -> dict:
    """Predeclared six-MC-SE comparisons; no scientific bias tolerance is implied."""
    reference = records['base'][0]
    failures = []
    maximum_mean_z, maximum_covariance_z = 0., 0.
    maximum_gaussian_covariance_scaled_error = 0.
    with np.load(directory/reference['numerical_arrays']['file']) as values:
        reference_covariance = values['selected_covariance'].copy()
    covariance_scale = np.sqrt(np.diag(reference_covariance))
    with np.load(directory/reference['positive_arrays']) as ref:
        for side in ('base', 'head'):
            for record in records[side]:
                if record['input_sha256'] != reference['input_sha256']:
                    failures.append('input checksum mismatch')
                with np.load(directory/record['numerical_arrays']['file']) as values:
                    error = np.abs(values['selected_covariance']-reference_covariance)/np.outer(covariance_scale, covariance_scale)
                    maximum_gaussian_covariance_scaled_error = max(maximum_gaussian_covariance_scaled_error, float(error.max()))
                with np.load(directory/record['positive_arrays']) as candidate:
                    se = np.hypot(ref['mean_mc_se'], candidate['mean_mc_se'])
                    z = np.abs(ref['positive_mean']-candidate['positive_mean'])/se
                    maximum_mean_z = max(maximum_mean_z, float(np.max(z)))
                    cov_se = np.hypot(ref['selected_covariance_mc_se'], candidate['selected_covariance_mc_se'])
                    zcov = np.abs(ref['selected_covariance']-candidate['selected_covariance'])/np.maximum(cov_se, np.finfo(float).tiny)
                    maximum_covariance_z = max(maximum_covariance_z, float(np.max(zcov)))
                a, b = reference['positive_log_evidence_95'], record['positive_log_evidence_95']
                if max(a[0], b[0]) > min(a[1], b[1]):
                    failures.append('positive evidence Monte Carlo intervals do not overlap')
    if maximum_mean_z > 6 or maximum_covariance_z > 6:
        failures.append('positive posterior moments disagree beyond six combined Monte Carlo SE')
    if maximum_gaussian_covariance_scaled_error > 1e-8:
        failures.append('Gaussian covariance differs beyond 1e-8 in marginal-SD units')
    return dict(passed=not failures, maximum_mean_mc_se=maximum_mean_z,
                maximum_covariance_mc_se=maximum_covariance_z,
                maximum_gaussian_covariance_scaled_error=maximum_gaussian_covariance_scaled_error, failures=failures)


def assess(records: dict, directory: Path, confirmation: dict | None = None) -> dict:
    report = {}
    for mode in ('cold', 'warm'):
        sample = records[mode]
        complete = all(r['status'] == 'completed' for side in sample.values() for r in side)
        if complete:
            numerical = _numerical_check(sample, directory)
            positive = positive_comparison(sample, directory)
            kwargs = {} if confirmation is None else dict(
                confirmation_base=_performance_input(confirmation[mode]['base']),
                confirmation_head=_performance_input(confirmation[mode]['head']))
            performance = compare_performance(_performance_input(sample['base']), _performance_input(sample['head']), **kwargs)
            status = performance['status'] if numerical['passed'] and positive['passed'] else 'failed_equivalence'
            report[mode] = dict(status=status, numerical=numerical, positive=positive, performance=performance)
            if confirmation is not None:
                report[mode]['confirmation_numerical'] = _numerical_check(confirmation[mode], directory, reference_record=sample['base'][0])
                report[mode]['confirmation_positive'] = positive_comparison(confirmation[mode], directory)
                if not report[mode]['confirmation_numerical']['passed'] or not report[mode]['confirmation_positive']['passed']:
                    report[mode]['status'] = 'failed_equivalence'
        else:
            expected_guard = all(r['status'] == 'failed' and 'budget' in r['error'].lower() for r in sample['base'])
            head_success = all(r['status'] == 'completed' and r['rss_within_budget'] for r in sample['head'])
            report[mode] = dict(status='base_guarded_head_completed' if expected_guard and head_success else 'failed',
                                base=[{k:v for k,v in r.items() if k in ('status','error','wall_seconds','peak_memory_bytes')} for r in sample['base']],
                                head=[{k:v for k,v in r.items() if k in ('status','error','wall_seconds','peak_memory_bytes','rss_within_budget','accepted','prior_host_accepted','legacy_dense_estimated_peak_bytes')} for r in sample['head']],
                                note='No paired speed ratio when the reference cannot run; small-case equivalence remains required')
        if any(not r['rss_within_budget'] for r in sample['head']):
            report[mode]['status'] = 'failed_rss_budget'
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-repo', type=Path, required=True)
    parser.add_argument('--head-repo', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--draws', type=int, default=1024)
    parser.add_argument('--timeout', type=float, default=180.)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('At least three paired repeats required')
    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name('stage5_benchmark_worker.py').resolve()
    results, all_records = {}, {}
    for name in args.cases:
        directory = root/name; directory.mkdir()
        spec = dict(CASES[name], seed=1729, draws=max(args.draws, 8192) if name=='large' else args.draws,
                    minimum_accepted=512, batch_size=16 if name=='large' else 128)
        (directory/'stage5-config.json').write_text(json.dumps(spec, indent=2))
        kwargs = dict(base_repo=args.base_repo.resolve(), head_repo=args.head_repo.resolve(), worker=worker,
                      output_dir=directory, repeats=args.repeats, timeout_seconds=args.timeout)
        records = _collect(**kwargs, label='initial')
        report = assess(records, directory)
        if any(r['status'] == 'confirmation_required' for r in report.values()):
            confirmation = _collect(**kwargs, label='confirmation')
            report = assess(records, directory, confirmation)
        results[name], all_records[name] = report, records
        (root/'summary.json').write_text(json.dumps(dict(cases=results, science_promotion_approved=False), indent=2))
        print(json.dumps(dict(case=name, results={k:v['status'] for k,v in report.items()})), flush=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for side in ('base', 'head'):
        for mode in ('cold', 'warm'):
            positions, times, memory = [], [], []
            for i, name in enumerate(args.cases):
                values = all_records[name][mode][side]
                if all(r['status']=='completed' for r in values):
                    positions.append(i); times.append(np.median([r['wall_seconds'] for r in values]))
                    memory.append(np.median([r['peak_memory_bytes']/2**20 for r in values]))
            axes[0].plot(positions, times, 'o-', label=f'{side} {mode}')
            axes[1].plot(positions, memory, 'o-', label=f'{side} {mode}')
    for axis in axes:
        axis.set_xticks(range(len(args.cases)), args.cases, rotation=45); axis.legend(fontsize=7); axis.grid(alpha=.3)
    axes[0].set(ylabel='seconds', yscale='log'); axes[1].set(ylabel='process peak RSS (MiB)')
    figure.suptitle('Exact-objective synthetic Stage 5; missing dense points are explicit budget failures')
    figure.tight_layout(); figure.savefig(root/'scaling.svg'); figure.savefig(root/'scaling.png', dpi=150); plt.close(figure)
    hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (root/'manifest.json').write_text(json.dumps(dict(sha256=hashes, calibration='synthetic only', threads=1), indent=2))
    return 0 if all(r['status'] in ('passed','base_guarded_head_completed') for c in results.values() for r in c.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
