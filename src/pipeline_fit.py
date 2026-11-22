"""Pipeline through joint MAP solve (shared by main.py and standalone tools).

Native-fit path (`FIT_ON_NATIVE_PIXELS`): optional square crop (`NATIVE_CUTOUT_SIZE`), exposure-consensus CR mask,
then optional ``unmask_sigma_inf_in_radius`` before star selection and ``solver.run_gls_solve``.
GP hyperparameters are optimized only when ``USE_SCENE_GP_PRIOR`` and ``GP_OPTIMIZE_HYPERPARAMS``
are both enabled.
"""
import os
import warnings

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning
from reproject import reproject_interp

from . import config, diagnostics, gp_model, native_fit_campaign, preprocessing, run_manifest, solver

warnings.simplefilter('ignore', category=AstropyWarning)


def _native_renderer_aperture_fraction(
    source: SkyCoord,
    cutouts: list[dict],
    scene_wcs,
    scene_shape: tuple[int, int],
    channel: str,
    radius_arcsec: float,
) -> float:
    """Measure the catalog aperture fraction in the native point renderer.

    The median over template BCDs accounts for native pixel integration, phase,
    roll, detector position, and finite cutout support without renormalizing a
    cropped response.  This is an instrumental convention match, not an
    absolute IRAC aperture calibration.
    """
    fractions: list[float] = []
    for entry in cutouts:
        if not entry.get('is_template', False):
            continue
        native_shape = tuple(np.asarray(entry['data']).shape)
        response = solver.native_point_source_column(
            scene_wcs,
            entry['wcs'],
            scene_shape,
            native_shape,
            channel,
            float(source.ra.deg),
            float(source.dec.deg),
            is_full_array=bool(entry.get('is_full_array', False)),
        ).reshape(native_shape)
        yy, xx = np.indices(native_shape, dtype=float)
        ra, dec = entry['wcs'].pixel_to_world_values(xx, yy)
        separation = source.separation(SkyCoord(ra, dec, unit='deg')).arcsec
        fraction = float(np.sum(response[separation <= radius_arcsec]))
        if np.isfinite(fraction) and fraction > 0.0:
            fractions.append(fraction)
    if not fractions:
        raise ValueError("No template BCD provides a finite catalog-aperture response")
    fraction = float(np.median(fractions))
    if fraction <= np.finfo(float).eps:
        raise ValueError("Catalog aperture contains no native-renderer response")
    return fraction


def _catalog_prior_flux_jy(
    row,
    source: SkyCoord,
    cutouts: list[dict],
    scene_wcs,
    scene_shape: tuple[int, int],
    channel: str,
) -> tuple[float, float]:
    """Convert a catalog aperture measurement to renderer-integrated Jy."""
    aperture_flux = float(row['catalog_aperture_flux_jy']) if (
        'catalog_aperture_flux_jy' in row.colnames
    ) else float(row['flux'])
    radius_arcsec = float(row['catalog_aperture_radius_arcsec']) if (
        'catalog_aperture_radius_arcsec' in row.colnames
    ) else float(getattr(config, 'CATALOG_APERTURE_RADIUS_ARCSEC', 4.0 * config.PIXEL_SCALE))
    fraction = _native_renderer_aperture_fraction(
        source, cutouts, scene_wcs, scene_shape, channel, radius_arcsec,
    )
    return aperture_flux / fraction, fraction


def run_pipeline_fit_core(*, skip_pre_analysis_check: bool = False):
    """
    Run catalog → alignment → cutouts → CR flag → star detect → joint solve.
    Does not reconstruct full model or write diagnostics PDFs.

    Returns None on failure, else dict with keys:
      cutouts, cutout_wcs, results, all_stars, full_flux_list, median_stamp,
      stretch_mask, chan_str, target, n_epochs
    """
    print("=== Pipeline fit stage ===")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.DIAGNOSTIC_DIR, exist_ok=True)

    print("\n[Init] Finding Files...")
    all_files = preprocessing.find_spitzer_files(config.DATA_DIR)

    print("\n[Step 0] Initializing Source Catalog...")
    preprocessing.get_or_create_source_catalog(all_files)

    sci_files, tpl_files = preprocessing.categorize_observations(all_files, config.SPLIT_DATE_MJD)
    if not tpl_files:
        print("No template files; abort.")
        return None

    # Effective band comes from `config.CHANNEL`; filenames still match via
    # `find_spitzer_files` / `is_correct_channel` (no infer-from-path ambiguity).
    chan_str = str(config.CHANNEL).lower()

    full_list = []
    for f in sci_files:
        f['is_template'] = False
        full_list.append(f)
    for f in tpl_files:
        f['is_template'] = True
        full_list.append(f)
    full_list.sort(key=lambda x: x.get('mjd', 0))

    n_epochs = preprocessing.assign_flux_epochs(full_list, config.EPOCH_WINDOW_DAYS)

    target = SkyCoord(config.TRANSIENT_RA, config.TRANSIENT_DEC, unit='deg')

    print("\n[Step 1] Initial Full-Field Reprojection...")
    mosaic_wcs, mosaic_shape = preprocessing.define_mosaic_wcs(full_list, target)
    tpl_list_sub = [f for f in full_list if f['is_template']]
    processed_tpl = preprocessing.reproject_to_grid(tpl_list_sub, mosaic_wcs, mosaic_shape)

    print("\n[Step 2] Creating Median Template Stack...")
    tpl_cube = np.array([p['data'] for p in processed_tpl])
    med_stack, mad_stack = preprocessing.create_median_stack(tpl_cube)

    print("\n[Step 2b] Detecting Sources on Deep Template...")
    template_stars_table = preprocessing.update_catalog_with_template(
        med_stack, mosaic_wcs, 'deep_template',
    )

    print("\n[Step 3] Loading Full Source Catalog for Alignment...")
    source_cat = preprocessing.get_or_create_source_catalog(all_files)

    tpl_stars = []
    tpl_fluxes = []
    if template_stars_table is not None:
        tpl_stars = [SkyCoord(r['ra'], r['dec'], unit='deg') for r in template_stars_table]
        tpl_fluxes = template_stars_table['flux']

    if not skip_pre_analysis_check:
        deep_stretch = np.isfinite(np.asanyarray(med_stack)) & (np.asanyarray(med_stack) != 0)
        diagnostics.plot_pre_analysis_check(
            med_stack, mosaic_wcs, tpl_stars, tpl_fluxes, target, stretch_mask=deep_stretch,
        )

    fits.writeto(
        os.path.join(config.OUTPUT_DIR, f'deep_template_{chan_str}.fits'),
        med_stack, mosaic_wcs.to_header(), overwrite=True,
    )
    fits.writeto(
        os.path.join(config.OUTPUT_DIR, f'deep_variance_{chan_str}.fits'),
        mad_stack**2, mosaic_wcs.to_header(), overwrite=True,
    )

    print("\n[Step 5-6] Astrometric Alignment (KDTree)...")
    preprocessing.align_frames_to_template(full_list, med_stack, mosaic_wcs, source_cat)

    print("\n[Step 9] Extracting Analysis Cutouts...")
    fit_on_native = bool(getattr(config, 'FIT_ON_NATIVE_PIXELS', False))
    if fit_on_native:
        cutouts, cutout_wcs = preprocessing.extract_native_analysis_cutouts(full_list, target)
        if not cutouts:
            raise RuntimeError('No usable native cutouts were extracted')
        native_cutout_size = int(getattr(config, 'NATIVE_CUTOUT_SIZE', 0))
        # Tighter ROI around the transient for cost / conditioning; WCS slices stay consistent.
        if native_cutout_size > 0:
            cutouts = [
                native_fit_campaign.crop_cutout_to_size(
                    c, native_cutout_size, float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)
                )
                for c in cutouts
            ]
            cutout_wcs = cutouts[0]['wcs']
    else:
        cutouts, cutout_wcs = preprocessing.extract_analysis_cutouts(full_list, target)

    if not cutouts:
        raise RuntimeError('No usable cutouts were extracted')
    # Native stamp WCS is not the latent sky grid. Keep the requested SR scale.
    scene_wcs, scene_shape = native_fit_campaign._scene_wcs_from_bcd_footprints(
        [c['wcs'] for c in cutouts], cutouts[0]['data'].shape,
        config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR,
        [(target.ra.deg, target.dec.deg)], int(config.NATIVE_SCENE_PAD_PX))

    print("\n[Step 8] Final CR Pass on Cutouts...")
    tpl_cutouts = [reproject_interp((c['data'], c['wcs']), scene_wcs, shape_out=scene_shape)[0]
                   for c in cutouts if c['is_template']]
    if tpl_cutouts:
        median_stamp = np.nanmedian(np.array(tpl_cutouts), axis=0)
    else:
        median_stamp = np.zeros(scene_shape)

    if fit_on_native:
        native_fit_campaign.apply_consensus_cr_masks(cutouts)
        for c in cutouts:
            # In-place: recover pixels that were already inf-sigma in the FITS (not from our CR pass).
            native_fit_campaign.unmask_sigma_inf_in_radius(
                c,
                float(getattr(config, 'UNMASK_SIGMA_INF_RADIUS_PX', 0.0)),
                str(getattr(config, 'UNMASK_SIGMA_INF_CENTER', 'nuclear')),
            )
        print("   [Native fit] Exposure-consensus CR masks applied; using template star list.")
        raw_stars_table = template_stars_table
    else:
        native_fit_campaign.apply_consensus_cr_masks(cutouts)
        print("   Detecting stars in analysis cutout (Robust Method)...")
        raw_stars_table = preprocessing.detect_sources(
            median_stamp,
            scene_wcs,
            sigma_map=None,
            is_template=True,
            mask_nucleus=False,
            integrated_jy_catalog=True,
        )

    solver_stars = []
    solver_init_fluxes = []
    solver_prior_metadata = []

    print(f"\n   [Star Detection Report]")
    print(f"   {'Index':<5} {'Dist(as)':<10} {'Init Flux':<12} {'Category'}")
    print(f"   {'-'*40}")

    if raw_stars_table is not None:
        for i, row in enumerate(raw_stars_table):
            sc = SkyCoord(row['ra'], row['dec'], unit='deg')
            flux_val, aperture_fraction = _catalog_prior_flux_jy(
                row, sc, cutouts, scene_wcs, scene_shape, chan_str,
            )
            dist = sc.separation(target).arcsec
            if fit_on_native and dist > 40.0:
                # Keep only stars near the analysis window for native stamps.
                continue
            print(f"   {i:<5} {dist:<10.3f} {flux_val:<12.4e} Solver (all in joint fit)")
            solver_stars.append(sc)
            solver_init_fluxes.append(flux_val)
            solver_prior_metadata.append({
                'aperture_fraction': aperture_fraction,
                'aperture_radius_arcsec': float(
                    row['catalog_aperture_radius_arcsec']
                    if 'catalog_aperture_radius_arcsec' in row.colnames
                    else getattr(config, 'CATALOG_APERTURE_RADIUS_ARCSEC', 4.0 * config.PIXEL_SCALE)
                ),
            })

    print(f"   Total Detections: {len(raw_stars_table) if raw_stars_table else 0}")
    print(f"   Solver Stars:     {len(solver_stars)}")

    print(f"\n[Step 10] Solving Joint System ({len(cutouts)} frames)...")
    vals = [c['data'][c['data'] != 0] for c in cutouts]
    if vals:
        flat = np.concatenate(vals)
        low, high = np.percentile(flat, [16, 84])
        var_est = ((high - low) / 2.0) ** 2
    else:
        var_est = 1e-7

    tpl_cutout_entries = [c for c in cutouts if c.get('is_template')]
    gp_ell, gp_var = config.INIT_LENGTH_SCALE, max(var_est, config.INIT_VARIANCE)
    if (
        bool(getattr(config, 'USE_SCENE_GP_PRIOR', True))
        and getattr(config, 'GP_OPTIMIZE_HYPERPARAMS', False)
        and tpl_cutout_entries
    ):
        gp_ell, gp_var = gp_model.optimize_hyperparameters(tpl_cutout_entries)
    gp_params = {'ell': gp_ell, 'var': gp_var}

    results = solver.run_gls_solve(
        cutouts,
        solver_stars,
        solver_init_fluxes,
        gp_params,
        (gp_ell, gp_var),
        median_stamp,
        scene_wcs,
        n_epochs,
    )
    results['star_catalog_prior_metadata'] = solver_prior_metadata

    if results is None:
        print("CRITICAL ERROR: Solver returned None.")
        return None

    full_flux_list = np.asarray(results['star_fluxes'], dtype=float)

    print("\n   [Final Fluxes for Reconstruction]")
    for i, flux in enumerate(full_flux_list):
        print(f"   Star {i}: {flux:.4e} Jy")

    stretch_mask = diagnostics.diagnostic_stretch_mask(cutouts, cutouts[0]['data'].shape)

    run_manifest.write_run_manifest(
        os.path.join(config.OUTPUT_DIR, "run_manifest.json"),
        [f["image"] for f in full_list],
        extra={
            "chan_str": chan_str,
            "n_epochs": n_epochs,
            "n_cutouts": len(cutouts),
        },
    )

    return {
        'cutouts': cutouts,
        'cutout_wcs': cutout_wcs,
        'results': results,
        'all_stars': solver_stars,
        'full_flux_list': full_flux_list,
        'median_stamp': median_stamp,
        'stretch_mask': stretch_mask,
        'chan_str': chan_str,
        'target': target,
        'n_epochs': n_epochs,
    }
