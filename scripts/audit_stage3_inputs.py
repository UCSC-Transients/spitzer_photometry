#!/usr/bin/env python3
"""Read-only real-input inventory and real-table algebra, not template science."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from astropy.io import fits
from src.point_response import _read_snapshot, file_identity
from src.isolated_astrometry import StarFrame, star_response
from src.nucleus import gaussian_response
from src.stage2_validation import validation_wcs
from src.validation_metrics import write_artifact_bundle


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,required=True)
    parser.add_argument('--prf-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    entries=[]
    for image in sorted(args.raw_dir.rglob('*I2*cbcd.fits')):
        uncertainty=Path(str(image).replace('cbcd.fits','cbunc.fits'))
        header=fits.getheader(image)
        entry=dict(image=str(image),sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                   metadata={k:header.get(k) for k in ('MJD_OBS','DATE_OBS','BUNIT','GAIN','EXPTIME','FLUXCONV','PXSCAL1','PXSCAL2')},
                   uncertainty=str(uncertainty),template_verified=False)
        if uncertainty.exists():
            uh=fits.getheader(uncertainty)
            entry['uncertainty_sha256']=hashlib.sha256(uncertainty.read_bytes()).hexdigest()
            entry['uncertainty_units']=uh.get('BUNIT')
            entry['unit_match']=uh.get('BUNIT')==header.get('BUNIT')=='MJy/sr'
        entries.append(entry)
    inventory=dict(status='available_but_template_purity_and_independent_registration_unverified',
                   count=len(entries),pipeline_date_candidate_count=sum(e['metadata']['MJD_OBS']>58750 for e in entries),
                   entries=entries,
                   noise_definition='archive cbunc is standard deviation in MJy/sr; includes pipeline Poisson/read and propagated calibration terms; no added sqrt(Jy) or floor',
                   unit_conversion='convert both CBCD and CBUNC by 1e6 times declared native pixel solid angle in steradians; keep gain/exposure/FLUXCONV provenance, do not reapply those factors',
                   limitations=['systematic detector and absolute flux errors not all represented by cbunc',
                                'date-only classification does not certify transient absence',
                                'no real template fit performed without independent verification'],
                   source='https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/50/')
    (args.output_dir/'bcd-inventory.json').write_text(json.dumps(inventory,indent=2))
    path=args.prf_dir/'apex_sh_IRACPC2_col129_row129_x100.fits'
    table=_read_snapshot(file_identity(path),'ch2','warm','legacy_integral')
    w=validation_wcs()
    frame=StarFrame(np.zeros((25,25)),np.full((25,25),1e-5),w,table)
    point=star_response(frame,w,np.array([12.13,11.81]),np.zeros(2))
    arrays={'real_prf_native_point':point}; comparisons=[]
    for width in (0.,.1,.4,.8):
        profile=np.array([width,.6*width,.2])
        low=gaussian_response(frame,w,np.array([12.13,11.81]),profile,np.zeros(2),order=11)
        medium=gaussian_response(frame,w,np.array([12.13,11.81]),profile,np.zeros(2),order=21)
        high=gaussian_response(frame,w,np.array([12.13,11.81]),profile,np.zeros(2),order=41)
        arrays[f'width_{width}_order11']=low;arrays[f'width_{width}_order21']=medium;arrays[f'width_{width}_order41']=high
        comparisons.append(dict(width_native_pixels=width,max_abs_11_21=float(np.max(abs(low-medium))),
                                max_abs_21_41=float(np.max(abs(medium-high))),
                                relative_l2_11_41=float(np.linalg.norm(low-high)/np.linalg.norm(high)),
                                finite_stamp_sum=float(high.sum())))
    np.testing.assert_array_equal(arrays['width_0.0_order11'],point)
    write_artifact_bundle(args.output_dir/'real-table-algebra',metrics=dict(calibration_sha256=table.sha256,
        calibration_path=str(path),comparisons=comparisons,point_limit='bitwise equal',
        interpretation='real calibration table on synthetic geometry; no real BCD fit or absolute Jy certification',
        science_promotion_approved=False),arrays=arrays,
        labels={'plot_title':'Real warm ch2 PRF — algebra only, synthetic geometry'})
    print(json.dumps({k:v for k,v in inventory.items() if k!='entries'},indent=2))


if __name__=='__main__':main()
