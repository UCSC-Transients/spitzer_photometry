"""src/postprocessing.py"""
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from . import config, preprocessing

def group_and_average(science_files, results):
    fluxes = results['transient_fluxes']
    errs = results['transient_errs']
    if len(fluxes) != len(science_files) or len(errs) != len(science_files):
        raise ValueError('science_files must align one-to-one with result arrays')
    data = []
    for i, f in enumerate(science_files):
        mjd = f.get('mjd', 0.0)
        chan = f.get('channel', 1)
        data.append({'mjd': mjd, 'flux': fluxes[i], 'err': errs[i], 'channel': chan, 'filename': f['filename'], 'bcd_index': i, 'epoch_id': f.get('epoch_id')})
    df = pd.DataFrame(data, columns=["mjd", "flux", "err", "channel", "filename", "bcd_index", "epoch_id"])
    df = df.sort_values('mjd').reset_index(drop=True)
    if not df.empty:
        print(f"[DEBUG] Post-processing {len(df)} records. MJD Range: {df['mjd'].min()} to {df['mjd'].max()}")
    if not df.empty and df['epoch_id'].isna().any():
        records = df.to_dict('records')
        preprocessing.assign_flux_epochs(records, config.EPOCH_WINDOW_DAYS)
        df['epoch_id'] = [r['epoch_id'] for r in records]
    os.makedirs(config.DIAGNOSTIC_DIR, exist_ok=True)
    df.to_csv(os.path.join(config.DIAGNOSTIC_DIR, 'step4_epoch_grouping.csv'), index=False)
    summary = []
    if not df.empty:
        for (epoch, chan), group in df.groupby(['epoch_id', 'channel']):
            safe_errs = group['err'].copy()
            safe_errs[safe_errs == 0] = np.inf
            w = 1.0 / (safe_errs**2)
            w_sum = np.sum(w)
            epoch_map = results.get('transient_epoch_index_by_id', {})
            if int(epoch) in epoch_map:
                j = epoch_map[int(epoch)]
                avg_flux = float(results['transient_epoch_fluxes'][j])
                avg_err = float(results['transient_epoch_errs'][j])
                avg_mjd = float(group['mjd'].mean())
            elif w_sum > 0:
                avg_flux = np.sum(group['flux'] * w) / w_sum
                avg_err = 1.0 / np.sqrt(w_sum)
                avg_mjd = np.average(group['mjd'], weights=w)
            else: avg_flux = float("nan"); avg_err = float("nan"); avg_mjd = group['mjd'].mean()
            summary.append({'epoch_id': epoch, 'channel': chan, 'mjd': avg_mjd, 'flux': avg_flux, 'err': avg_err, 'n_bcds': len(group)})
    df_summary = pd.DataFrame(summary)
    if not df_summary.empty: df_summary = df_summary.sort_values(['channel', 'mjd'])
    return df, df_summary

def save_outputs(df_bcds, df_summary, model_scene, output_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    df_bcds.to_csv(os.path.join(output_dir, 'flux_per_bcd.csv'), index=False)
    df_summary.to_csv(os.path.join(output_dir, 'flux_per_epoch.csv'), index=False)
    hdu = fits.PrimaryHDU(np.nan_to_num(model_scene))
    hdu.header['UNITS'] = 'Jy/pixel'
    hdu.writeto(os.path.join(output_dir, 'template_model_scene.fits'), overwrite=True)
    print(f"Saved results to {output_dir}")
