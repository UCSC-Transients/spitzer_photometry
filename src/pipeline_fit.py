"""Pipeline through joint MAP solve (shared by main.py and standalone tools).

Native-fit path (`FIT_ON_NATIVE_PIXELS`): optional square crop (`NATIVE_CUTOUT_SIZE`), exposure-consensus CR mask,
then optional ``unmask_sigma_inf_in_radius`` before star selection and ``solver.run_gls_solve``.
GP hyperparameters are optimized only when ``USE_SCENE_GP_PRIOR`` and ``GP_OPTIMIZE_HYPERPARAMS``
are both enabled.
"""
import os
import warnings

import numpy as np
import sep
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning
from scipy.ndimage import binary_dilation

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


def _source_has_native_fit_support(
    source: SkyCoord,
    cutouts: list[dict],
    channel: str,
) -> bool:
    """Return whether a catalog source contributes to any valid fit pixel.

    The test uses the same detector-integrated point renderer and fit masks as
    the solve. This avoids angular-distance heuristics for distorted or cropped
    detector footprints.
    """
    for entry in cutouts:
        native_shape = tuple(np.asarray(entry['data']).shape)
        response = solver.native_point_source_column(
            entry['wcs'],
            entry['wcs'],
            native_shape,
            native_shape,
            channel,
            float(source.ra.deg),
            float(source.dec.deg),
            is_full_array=bool(entry.get('is_full_array', False)),
        ).reshape(native_shape)
        valid = solver.valid_fit_mask(entry)
        supported = np.asarray(response, dtype=float)[valid]
        if (
            supported.size
            and np.isfinite(supported).all()
            and float(np.sum(np.abs(supported))) > np.finfo(float).eps
        ):
            return True
    return False


def _catalog_frame_positive_outliers(
    data: np.ndarray,
    response: np.ndarray,
    sigma: np.ndarray,
    support: np.ndarray,
    aperture_pixels: np.ndarray,
    annulus_pixels: np.ndarray,
    x_offset: np.ndarray,
    y_offset: np.ndarray,
    threshold: float = 8.0,
) -> np.ndarray:
    """Identify detector artifacts after fitting a source plus affine sky.

    The fit is used only to classify positive outliers; neither the catalog
    data nor its uncertainties are replaced.  A bounded IRLS fit prevents one
    bright pixel from controlling the source amplitude while retaining a real
    coherent gradient.  The exact rendered response is included in the same
    design as the affine background, so detector-position PRF changes are not
    mistaken for sky structure.
    """
    fit_support = support & (aperture_pixels | annulus_pixels)
    if np.count_nonzero(fit_support) < 8:
        return np.zeros(data.shape, dtype=bool)
    design = np.column_stack((
        response[fit_support],
        np.ones(np.count_nonzero(fit_support), dtype=float),
        x_offset[fit_support],
        y_offset[fit_support],
    ))
    values = data[fit_support]
    errors = sigma[fit_support]
    annulus_fit = annulus_pixels[fit_support]
    coefficients = np.zeros(4, dtype=float)
    if np.count_nonzero(annulus_fit) >= 3:
        background_design = design[annulus_fit, 1:]
        coefficients[1:] = np.linalg.lstsq(
            background_design / errors[annulus_fit, None],
            values[annulus_fit] / errors[annulus_fit],
            rcond=None,
        )[0]
    source_response = design[:, 0]
    source_dominated = (
        aperture_pixels[fit_support]
        & (source_response > 1.0e-3 * np.max(source_response, initial=0.0))
    )
    if np.any(source_dominated):
        background = design[:, 1:] @ coefficients[1:]
        ratios = (
            values[source_dominated] - background[source_dominated]
        ) / source_response[source_dominated]
        finite_ratios = ratios[np.isfinite(ratios)]
        if finite_ratios.size:
            coefficients[0] = float(np.median(finite_ratios))
    clip = float(max(threshold, np.finfo(float).eps))
    for _ in range(8):
        normalized = (values - design @ coefficients) / errors
        robust_weight = np.ones(normalized.shape, dtype=float)
        large = np.abs(normalized) > clip
        robust_weight[large] = clip / np.abs(normalized[large])
        weighted = np.sqrt(robust_weight) / errors
        updated = np.linalg.lstsq(
            design * weighted[:, None], values * weighted, rcond=None,
        )[0]
        if np.allclose(updated, coefficients, rtol=1.0e-10, atol=1.0e-15):
            coefficients = updated
            break
        coefficients = updated
    model = (
        coefficients[0] * response
        + coefficients[1]
        + coefficients[2] * x_offset
        + coefficients[3] * y_offset
    )
    # The same retained support feeds the subsequent affine nuisance
    # projection. Positive detector artifacts in either the source aperture or
    # its background annulus must therefore be removed before that projection;
    # otherwise an annulus CR is spread coherently across the frame by the
    # fitted plane and can no longer be rejected by the pixelwise median.
    measurement_pixels = aperture_pixels | annulus_pixels
    positive = support & measurement_pixels & (
        (data - model) > threshold * sigma
    )
    # Catalog products are reprojected before this classifier. A single native
    # detector impulse therefore occupies a compact interpolation footprint,
    # including low-amplitude negative lobes that cannot independently cross a
    # positive-only threshold. Exclude the adjacent footprint from both data
    # and response so those lobes cannot bias the subsequent affine nuisance.
    return support & measurement_pixels & binary_dilation(
        positive,
        iterations=2,
    )


def _catalog_remove_affine_nuisance(
    data: np.ndarray,
    response: np.ndarray,
    sigma: np.ndarray,
    support: np.ndarray,
    annulus_pixels: np.ndarray,
    x_offset: np.ndarray,
    y_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project data and unit response through one affine-annulus operator.

    The weighted annulus fit is a linear nuisance projection.  Applying the
    identical projection to the observed image and its rendered unit source
    preserves ``data = flux * response`` after any supported affine sky is
    removed, including asymmetric masks.  Artifact selection must precede this
    operation so the projection itself remains data-independent and linear.

    Parameters
    ----------
    data, response, sigma
        Reprojected catalog-grid image, unit-source response, and unchanged
        measurement uncertainty for one exposure.
    support
        Pixels retained after native/reprojection validity and CR selection.
    annulus_pixels
        Catalog measurement's recorded background-annulus footprint.
    x_offset, y_offset
        Catalog-grid pixel offsets from the measured source position.

    Returns
    -------
    tuple of numpy.ndarray
        Affine-residualized data and response on the original grid.

    Raises
    ------
    ValueError
        If the retained annulus cannot identify a constant and two gradients.
    """
    fit_support = support & annulus_pixels
    if np.count_nonzero(fit_support) < 3:
        raise ValueError('Catalog frame has insufficient affine-background support')
    design = np.column_stack((
        np.ones(np.count_nonzero(fit_support), dtype=float),
        x_offset[fit_support],
        y_offset[fit_support],
    ))
    weighted_design = design / sigma[fit_support, None]
    if np.linalg.matrix_rank(weighted_design) < 3:
        raise ValueError('Catalog frame affine background is not identifiable')
    # Solve the weighted rectangular problems directly.  Forming normal
    # equations squares the condition number and left a measurable constant-
    # sky residual on fine (SR=4) catalog grids after asymmetric CR masks.
    # The two right-hand sides still use the identical retained design.
    data_coefficients = np.linalg.lstsq(
        weighted_design,
        data[fit_support] / sigma[fit_support],
        rcond=None,
    )[0]
    response_coefficients = np.linalg.lstsq(
        weighted_design,
        response[fit_support] / sigma[fit_support],
        rcond=None,
    )[0]
    full_design = np.stack((
        np.ones(data.shape, dtype=float), x_offset, y_offset,
    ), axis=-1)
    return (
        data - full_design @ data_coefficients,
        response - full_design @ response_coefficients,
    )


def _catalog_measurement_aperture_fraction(
    row,
    source: SkyCoord,
    catalog_frames: list[dict],
    catalog_wcs,
    catalog_shape: tuple[int, int],
    channel: str,
) -> tuple[float | None, float]:
    """Apply the catalog's measurement operator to a unit rendered source.

    Unit sources are rendered on the original native template BCDs, masked by
    their measurement support, reprojected and median-combined like the deep
    template, then measured with the recorded aperture, annulus and subpixel
    settings. The result is independent of subsequent fit-stamp cropping.
    """
    if not catalog_frames:
        raise ValueError('No template frames are available for catalog calibration')
    frame_records: list[dict] = []
    for entry in catalog_frames:
        native_shape = tuple(entry.get('native_shape', ()))
        if len(native_shape) != 2 or min(native_shape) <= 0:
            raise ValueError('Catalog calibration frame lacks its native shape')
        native_wcs = entry.get('corrected_wcs', entry.get('raw_wcs'))
        if native_wcs is None:
            raise ValueError('Catalog calibration frame lacks its native WCS')
        response = solver.native_point_source_column(
            catalog_wcs,
            native_wcs,
            catalog_shape,
            native_shape,
            channel,
            float(source.ra.deg),
            float(source.dec.deg),
            is_full_array=bool(entry.get('is_full_array', False)),
        ).reshape(native_shape)
        native_valid = np.asarray(
            entry.get('native_valid_mask', np.ones(native_shape, dtype=bool)),
            dtype=bool,
        )
        if native_valid.shape != native_shape:
            raise ValueError('Catalog calibration mask shape does not match native data')
        projected, support = preprocessing.reproject_valid_native_samples(
            response,
            native_valid,
            native_wcs,
            catalog_wcs,
            tuple(catalog_shape),
            return_support=True,
        )
        data = None
        projected_sigma = None
        if 'data' in entry:
            data = np.asarray(entry['data'], dtype=float)
            if data.shape != tuple(catalog_shape):
                raise ValueError('Catalog calibration data shape does not match its grid')
            projected_support = np.asarray(
                entry.get('projected_valid_mask', support), dtype=bool,
            )
            if projected_support.shape != tuple(catalog_shape):
                raise ValueError('Catalog projected mask shape does not match its grid')
            support = support & projected_support & np.isfinite(data)
            if 'sigma' in entry:
                projected_sigma = np.asarray(entry['sigma'], dtype=float)
                if projected_sigma.shape != tuple(catalog_shape):
                    raise ValueError('Catalog projected error shape does not match its grid')
                support &= np.isfinite(projected_sigma) & (projected_sigma > 0.0)
        frame_records.append({
            'response': projected,
            'support': support,
            'data': data,
            'sigma': projected_sigma,
        })

    pixel_scale_arcsec = (
        float(np.sqrt(abs(np.linalg.det(catalog_wcs.pixel_scale_matrix)))) * 3600.0
    )
    if not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0.0:
        raise ValueError('Catalog WCS has no finite positive pixel scale')
    area_factor = preprocessing._scene_sample_to_integrated_jy_factor(catalog_wcs)
    radius = float(row['catalog_aperture_radius_arcsec']) / pixel_scale_arcsec
    annulus = (
        float(row['catalog_aperture_annulus_inner_arcsec']) / pixel_scale_arcsec,
        float(row['catalog_aperture_annulus_outer_arcsec']) / pixel_scale_arcsec,
    )
    subpix = int(row['catalog_aperture_subpix'])
    x = np.asarray([float(row['x'])])
    y = np.asarray([float(row['y'])])
    yy, xx = np.indices(tuple(catalog_shape), dtype=float)
    x_offset = xx - float(x[0])
    y_offset = yy - float(y[0])
    distance = np.hypot(x_offset, y_offset)
    aperture_pixels = distance <= radius
    annulus_pixels = (distance >= annulus[0]) & (distance <= annulus[1])
    usable_records = []
    for record in frame_records:
        response = np.asarray(record['response'], dtype=float)
        support = np.asarray(record['support'], dtype=bool)
        source_support = support & aperture_pixels & np.isfinite(response)
        response_scale = float(np.sum(np.abs(response[source_support])))
        annulus_support = support & annulus_pixels
        annulus_design = np.column_stack((
            np.ones(np.count_nonzero(annulus_support), dtype=float),
            x_offset[annulus_support],
            y_offset[annulus_support],
        ))
        if (
            np.count_nonzero(annulus_support) >= 3
            and np.linalg.matrix_rank(annulus_design) == 3
            and np.any(source_support)
            and response_scale > np.finfo(float).eps
        ):
            usable_records.append(record)
    if not usable_records:
        raise ValueError('No catalog exposure has supported source and background pixels')
    responses = [np.asarray(record['response'], dtype=float) for record in usable_records]
    supports = [np.asarray(record['support'], dtype=bool) for record in usable_records]
    projected_data = [record['data'] for record in usable_records]
    projected_sigmas = [record['sigma'] for record in usable_records]
    calibrated_flux = None
    if all(data is not None for data in projected_data):
        data_cube = np.asarray(projected_data, dtype=float).copy()
        response_cube = np.asarray(responses, dtype=float).copy()
        support_cube = np.asarray(supports, dtype=bool)
        # Classify detector artifacts relative to each exposure's exact source
        # response and a fitted affine sky. Only support changes in this step.
        if all(sigma is not None for sigma in projected_sigmas):
            sigma_cube = np.asarray(projected_sigmas, dtype=float)
            for frame_index in range(data_cube.shape[0]):
                support_cube[frame_index] &= ~_catalog_frame_positive_outliers(
                    data_cube[frame_index],
                    response_cube[frame_index],
                    sigma_cube[frame_index],
                    support_cube[frame_index],
                    aperture_pixels,
                    annulus_pixels,
                    x_offset,
                    y_offset,
                )
            # Remove the retained affine nuisance with one exactly matched
            # linear operator for observed data and rendered unit response.
            for frame_index in range(data_cube.shape[0]):
                data_cube[frame_index], response_cube[frame_index] = (
                    _catalog_remove_affine_nuisance(
                        data_cube[frame_index],
                        response_cube[frame_index],
                        sigma_cube[frame_index],
                        support_cube[frame_index],
                        annulus_pixels,
                        x_offset,
                        y_offset,
                    )
                )
        else:
            # Without projected uncertainties, use equal weights while still
            # applying the identical affine nuisance operator to both arrays.
            unit_sigma = np.ones(tuple(catalog_shape), dtype=float)
            for frame_index in range(data_cube.shape[0]):
                data_cube[frame_index], response_cube[frame_index] = (
                    _catalog_remove_affine_nuisance(
                        data_cube[frame_index],
                        response_cube[frame_index],
                        unit_sigma,
                        support_cube[frame_index],
                        annulus_pixels,
                        x_offset,
                        y_offset,
                    )
                )
        sortable_data = np.where(support_cube, data_cube, np.inf)
        order = np.argsort(sortable_data, axis=0, kind='stable')
        valid_count = np.sum(support_cube, axis=0)
        low_rank = np.maximum((valid_count - 1) // 2, 0)[None, ...]
        high_rank = np.maximum(valid_count // 2, 0)[None, ...]
        low_index = np.take_along_axis(order, low_rank, axis=0)
        high_index = np.take_along_axis(order, high_rank, axis=0)
        selected_data = 0.5 * (
            np.take_along_axis(data_cube, low_index, axis=0)[0]
            + np.take_along_axis(data_cube, high_index, axis=0)[0]
        )
        selected_response = 0.5 * (
            np.take_along_axis(response_cube, low_index, axis=0)[0]
            + np.take_along_axis(response_cube, high_index, axis=0)[0]
        )
        catalog_support = valid_count > 0
        selected_data = np.where(catalog_support, selected_data, 0.0)
        selected_response = np.where(catalog_support, selected_response, 0.0)
        data_measurement = sep.sum_circle(
            np.ascontiguousarray(selected_data, dtype=np.float64),
            x, y, radius, mask=~catalog_support, bkgann=annulus, subpix=subpix,
        )[0][0] * area_factor
        response_measurement = sep.sum_circle(
            np.ascontiguousarray(selected_response, dtype=np.float64),
            x, y, radius, mask=~catalog_support, bkgann=annulus, subpix=subpix,
        )[0][0] * area_factor
        fraction = float(response_measurement)
        if (
            not np.isfinite(data_measurement)
            or not np.isfinite(fraction)
            or fraction <= np.finfo(float).eps
        ):
            raise ValueError('Catalog frames provide no finite calibrated source flux')
        calibrated_flux = float(data_measurement / fraction)
        if not np.isfinite(calibrated_flux) or calibrated_flux <= 0.0:
            raise ValueError('Catalog frames provide no positive calibrated source flux')
    else:
        catalog_response, _ = preprocessing.create_median_stack(
            np.asarray(responses),
        )
        catalog_support = np.any(np.asarray(supports, dtype=bool), axis=0)
        measured, _, _ = sep.sum_circle(
            np.ascontiguousarray(catalog_response, dtype=np.float64),
            x, y, radius, mask=~catalog_support, bkgann=annulus, subpix=subpix,
        )
        fraction = float(measured[0]) * area_factor
        if not np.isfinite(fraction) or fraction <= np.finfo(float).eps:
            raise ValueError('Catalog measurement contains no finite renderer response')
    return calibrated_flux, fraction


def _catalog_prior_flux_jy(
    row,
    source: SkyCoord,
    cutouts: list[dict],
    scene_wcs,
    scene_shape: tuple[int, int],
    channel: str,
    *,
    catalog_frames: list[dict] | None = None,
    catalog_wcs=None,
    catalog_shape: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """Convert a catalog aperture measurement to renderer-integrated Jy.

    Production callers supply the original catalog measurement geometry. The
    native-stamp fallback supports older programmatic catalogs without complete
    aperture provenance, but cannot claim crop-independent calibration.
    """
    aperture_flux = float(row['catalog_aperture_flux_jy']) if (
        'catalog_aperture_flux_jy' in row.colnames
    ) else float(row['flux'])
    radius_arcsec = float(row['catalog_aperture_radius_arcsec']) if (
        'catalog_aperture_radius_arcsec' in row.colnames
    ) else float(getattr(config, 'CATALOG_APERTURE_RADIUS_ARCSEC', 4.0 * config.PIXEL_SCALE))
    provenance = {
        'x',
        'y',
        'catalog_aperture_annulus_inner_arcsec',
        'catalog_aperture_annulus_outer_arcsec',
        'catalog_aperture_subpix',
    }
    if (
        catalog_frames is not None
        and catalog_wcs is not None
        and catalog_shape is not None
        and provenance.issubset(set(row.colnames))
    ):
        calibrated_flux, fraction = _catalog_measurement_aperture_fraction(
            row,
            source,
            catalog_frames,
            catalog_wcs,
            tuple(catalog_shape),
            channel,
        )
        if calibrated_flux is not None:
            aperture_flux = calibrated_flux * fraction
    else:
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
    template_measurement_mask = np.any(np.asarray([
        p['projected_valid_mask']
        if 'projected_valid_mask' in p
        else np.isfinite(p.get('sigma', p['data']))
        for p in processed_tpl
    ], dtype=bool), axis=0)
    template_stars_table = preprocessing.update_catalog_with_template(
        med_stack,
        mosaic_wcs,
        'deep_template',
        measurement_mask=template_measurement_mask,
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
    catalog_cutout_frames = []
    tpl_cutouts = []
    for c in cutouts:
        if not c['is_template']:
            continue
        valid = solver.valid_fit_mask(c)
        projected, support = preprocessing.reproject_valid_native_samples(
            c['data'],
            valid,
            c['wcs'],
            scene_wcs,
            scene_shape,
            return_support=True,
        )
        tpl_cutouts.append(projected)
        catalog_cutout_frames.append({
            'data': projected,
            'raw_wcs': c['wcs'],
            'corrected_wcs': c['wcs'],
            'native_shape': tuple(np.asarray(c['data']).shape),
            'native_valid_mask': valid.copy(),
            'projected_valid_mask': support,
            'is_full_array': bool(c.get('is_full_array', False)),
        })
    if tpl_cutouts:
        median_stamp, _ = preprocessing.create_median_stack(np.asarray(tpl_cutouts))
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
            measurement_mask=np.any(np.asarray([
                entry['projected_valid_mask'] for entry in catalog_cutout_frames
            ], dtype=bool), axis=0),
        )

    if fit_on_native:
        catalog_frames = processed_tpl
        catalog_wcs = mosaic_wcs
        catalog_shape = mosaic_shape
    else:
        catalog_frames = catalog_cutout_frames
        catalog_wcs = scene_wcs
        catalog_shape = scene_shape

    solver_stars = []
    solver_init_fluxes = []
    solver_prior_metadata = []

    print(f"\n   [Star Detection Report]")
    print(f"   {'Index':<5} {'Dist(as)':<10} {'Init Flux':<12} {'Category'}")
    print(f"   {'-'*40}")

    if raw_stars_table is not None:
        for i, row in enumerate(raw_stars_table):
            sc = SkyCoord(row['ra'], row['dec'], unit='deg')
            if fit_on_native and not _source_has_native_fit_support(sc, cutouts, chan_str):
                continue
            flux_val, aperture_fraction = _catalog_prior_flux_jy(
                row, sc, cutouts, scene_wcs, scene_shape, chan_str,
                catalog_frames=catalog_frames,
                catalog_wcs=catalog_wcs,
                catalog_shape=catalog_shape,
            )
            dist = sc.separation(target).arcsec
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
