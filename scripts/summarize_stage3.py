#!/usr/bin/env python3
"""Summarize saved Stage 3 arrays; do not refit or consume acceptance data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
from astropy.io.fits import Header
from astropy.wcs import WCS
from src.stage2_validation import validation_wcs
from src.stage3_validation import nucleus_fixture


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path)
    args=parser.parse_args()
    metrics_path=args.bundle/'metrics.json'; array_path=args.bundle/'arrays.npz'
    payload=json.loads(metrics_path.read_text())['metrics']
    summary={'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (metrics_path,array_path)},
             'bias_units':'synthetic Jy-like declared amplitude (not absolute IRAC calibration)',
             'coverage_note':'three development noise seeds; local 95% curvature intervals; missing/failing fits recorded separately; models/truth cases reuse noise seeds',
             'models':{},'noiseless':{},'optimizer_failure_count':0,'total_fits':0}
    arrays=np.load(array_path)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    phase_arrays={}
    for case in ('centroid','extent','prf_mismatch','asymmetric'):
        for model in ('point','gaussian','core_envelope'):
            trials=[r['models'][model] for r in payload['records'] if r['truth']['case']==case and r['truth']['seed'] is not None]
            valid=[r for r in trials if r['status']=='converged']
            bias=np.array([r['nuclear_flux_bias'] for r in valid])
            intervals=[r['total_nuclear_local_95_covers_truth'] for r in valid if 'total_nuclear_local_95_covers_truth' in r]
            mean=float(bias.mean()) if len(bias) else None
            half=float(t.ppf(.975,len(bias)-1)*bias.std(ddof=1)/np.sqrt(len(bias))) if len(bias)>1 else None
            summary['models'][case+'/'+model]=dict(noise_trials=len(trials),converged=len(valid),
                failed=len(trials)-len(valid),mean_bias=mean,rmse=float(np.sqrt(np.mean(bias*bias))) if len(bias) else None,
                descriptive_mean_95_interval=[mean-half,mean+half] if half is not None else None,
                available_local_intervals=len(intervals),coverage_count=sum(intervals),
                maximum_absolute_nearby_source_bias=max((abs(v['source_bias']) for r in valid for v in r['signatures'][4:]),default=None))
    for r in payload['records']:
        truth=r['truth'];case=truth['case'];seed=truth['seed']
        summary['total_fits']+=len(r['models'])
        summary['optimizer_failure_count']+=sum(m['status']=='failed' for m in r['models'].values())
        if seed is None:
            summary['noiseless'][case]={m:dict(status=v['status'],bias=v.get('nuclear_flux_bias'),validation=v.get('validation_predictive'),
                                              profile=v.get('profile'),centroid=v.get('centroid_xy')) for m,v in r['models'].items()}
        if seed != 1729:continue
        reference=validation_wcs(sr=truth['sr'])
        frames,_,_=nucleus_fixture(case,seed,sr=truth['sr'])
        np.testing.assert_array_equal(np.array([f.image.data for f in frames]),arrays[f'{case}_{seed}_data'])
        for model,m in r['models'].items():
            if m['status']!='converged':continue
            sky=reference.pixel_to_world_values(*m['centroid_xy'])
            phases=[]
            for h in r['config']['wcs_headers']:
                w=WCS(Header.fromstring(h));phases.append(np.mod(w.world_to_pixel_values(*sky),1.))
            # Include the supplied detector translations. These are calibration
            # means for unseen frames and posterior translations for training.
            prefix=f'{case}_{seed}_{model}'
            parameters=arrays[prefix+'_parameters']
            nc=2 if model=='core_envelope' else 1;na=nc+3;ng=na+2+(3 if model!='point' else 0)
            for i in range(6):
                shift=frames[i].registration_xy.copy()
                if i in (1,2):shift+=parameters[ng+2*(i-1):ng+2*i]
                phases[i]=np.mod(phases[i]+shift,1.)
            phase_arrays[prefix+'_phase']=np.array(phases)
            coefficients=np.array([s['derivative_coefficients'][0] for s in m['signatures']])
            axes[0].scatter(np.array(phases)[:,0],coefficients,s=18,label=case+'/'+model)
            axes[1].plot(truth['rolls'],[s['source_bias'] for s in m['signatures']],'.-',label=case+'/'+model)
    summary['phase_note']='sky centroid through saved WCS plus calibration means and fitted training corrections; calibration regenerated from recorded seed only after exact equality of saved input arrays; no held-out fitting'
    (args.bundle/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    np.savez_compressed(args.bundle/'detector-phases.npz',**phase_arrays)
    axes[0].set(xlabel='native nuclear x phase (pixel fraction)',ylabel='signed sky-x derivative coefficient')
    axes[1].set(xlabel='roll (degrees)',ylabel='nearby-source signed bias (synthetic flux)')
    axes[0].legend(fontsize=6,ncol=2)
    fig.suptitle('Synthetic development seed 1729 — failures omitted from plot, retained in JSON')
    fig.tight_layout();fig.savefig(args.bundle/'detector-phase-roll.png',dpi=130);plt.close(fig)
    manifest_path=args.bundle/'manifest.json';manifest=json.loads(manifest_path.read_text())
    for name in ('summary.json','detector-phases.npz','detector-phase-roll.png'):
        manifest['files'][name]=hashlib.sha256((args.bundle/name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
