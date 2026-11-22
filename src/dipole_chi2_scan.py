"""Dipole-directed χ² scan for shifting host/nucleus position (template BCDs)."""
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates
from reproject import reproject_interp

from . import config, solver
from . import residual_metrics


def _template_indices(cutouts):
    return [i for i, c in enumerate(cutouts) if c.get('is_template')]


def _bcd_valid_mask(c, stretch_mask):
    return solver.valid_fit_mask(c)


def _template_intersection_mask(cutouts, tpl_indices, stretch_mask, scene_shape, scene_wcs):
    from .diagnostics import diagnostic_stretch_mask
    return diagnostic_stretch_mask([cutouts[i] for i in tpl_indices], scene_shape, scene_wcs)


def _to_stack_grid(arr, arr_wcs, stack_wcs, stack_shape):
    return reproject_interp((np.asarray(arr, dtype=float), arr_wcs), stack_wcs,
                            shape_out=stack_shape)[0]


def shift_model_in_scene_pixels(img, native_wcs, scene_wcs, dx, dy):
    """Shift a native model by a displacement defined on the sky scene grid."""
    yy, xx = np.indices(np.shape(img), dtype=float)
    ra, dec = native_wcs.pixel_to_world_values(xx, yy)
    sx, sy = scene_wcs.world_to_pixel_values(ra, dec)
    ra, dec = scene_wcs.pixel_to_world_values(sx - dx, sy - dy)
    nx, ny = native_wcs.world_to_pixel_values(ra, dec)
    return map_coordinates(np.asarray(img, dtype=float), [ny, nx],
                           order=1, mode='nearest', prefilter=False)


def compute_dipole_chi2_scan(
    cutouts,
    results,
    stars,
    star_fluxes,
    stretch_mask=None,
    *,
    coarse_step: float = 0.1,
    coarse_max: float = 3.0,
    fine_half_width: float = 0.2,
    fine_step: float = 0.01,
    poly_degree: int = 2,
) -> Dict[str, Any]:
    """
    Template BCDs only: median residual → dipole unit vector; χ²(s) along +u from anchor pixel.

    Coarse: s in [-coarse_max, +coarse_max] step coarse_step
    (pixels in scene coordinates along dipole unit vector u).
    Fit polynomial to coarse χ²; vertex seeds fine grid [vertex - fine_half_width, vertex + fine_half_width].
    """
    out: Dict[str, Any] = {'dipole_chi2_refinement': {}}
    sec = out['dipole_chi2_refinement']
    tpl = _template_indices(cutouts)
    if len(tpl) < 1:
        sec['skipped'] = True
        sec['reason'] = 'need ≥1 template BCD'
        return out

    scene_shape = results['scene_shape']
    vm = _template_intersection_mask(cutouts, tpl, stretch_mask, scene_shape, results['scene_wcs'])
    stack_wcs = results.get('scene_wcs')
    stack_shape = tuple(scene_shape)

    # Build the fitted structured model we want to shift: host+nucleus+BG on each template BCD.
    # In this codebase, this maps to GP + host Gaussian + epoch background.
    model_cube = []
    resid0_cube = []
    for i in tpl:
        pred_host_nuc_bg = solver.predict_cutout_model(
            results, cutouts, stars, star_fluxes, i,
            include_transient=False, include_stars=False, include_gp=True, include_host=True,
        )
        model_cube.append(np.asarray(pred_host_nuc_bg, dtype=float))
        resid0_cube.append(
            _to_stack_grid(
                np.asarray(cutouts[i]['data'], dtype=float) - np.asarray(pred_host_nuc_bg, dtype=float),
                cutouts[i]['wcs'],
                stack_wcs,
                stack_shape,
            )
        )
    med_res = np.nanmedian(np.stack(resid0_cube, axis=0), axis=0)
    vm = np.isfinite(med_res) if vm is None else (vm & np.isfinite(med_res))
    mx, my, dmag = residual_metrics.dipole_moment_xy(med_res, vm)
    norm = float(np.hypot(mx, my))
    if norm < 1e-8:
        sec['skipped'] = True
        sec['reason'] = 'dipole direction norm too small'
        return out
    ux, uy = mx / norm, my / norm

    # Pre-cache data/sigma/mask for each template frame.
    tpl_data = [np.asarray(cutouts[i]['data'], dtype=float) for i in tpl]
    tpl_sig = [np.asarray(results['effective_sigmas'][i] if 'effective_sigmas' in results else cutouts[i]['sigma'], dtype=float) for i in tpl]
    tpl_mask = [_bcd_valid_mask(cutouts[i], stretch_mask) for i in tpl]

    def chi2_at_shift(s_pix: float) -> float:
        tot = 0.0
        for index, (data_i, sig_i, m_i, model_i) in enumerate(zip(tpl_data, tpl_sig, tpl_mask, model_cube)):
            pred = shift_model_in_scene_pixels(model_i, cutouts[tpl[index]]['wcs'],
                                               stack_wcs, s_pix * ux, s_pix * uy)
            r = data_i - pred
            tot += float(np.sum((r[m_i] / sig_i[m_i]) ** 2))
        return tot

    coarse_s = np.arange(-float(coarse_max), float(coarse_max) + 1e-9, float(coarse_step), dtype=float)
    coarse_chi2 = [chi2_at_shift(float(s)) for s in coarse_s]

    deg = int(max(1, min(poly_degree, len(coarse_s) - 1)))
    coef = np.polyfit(coarse_s, np.asarray(coarse_chi2, dtype=float), deg)
    if deg >= 2:
        a, b, cc = float(coef[0]), float(coef[1]), float(coef[2])
        s_vertex = float(-b / (2.0 * a)) if abs(a) > 1e-20 else float(coarse_s[int(np.argmin(coarse_chi2))])
    else:
        a, b, cc = float(coef[0]), float(coef[1]), 0.0
        s_vertex = float(coarse_s[int(np.argmin(coarse_chi2))])

    s_vertex = float(np.clip(s_vertex, float(coarse_s[0]), float(coarse_s[-1])))

    fine = np.arange(
        s_vertex - float(fine_half_width),
        s_vertex + float(fine_half_width) + 1e-9,
        float(fine_step),
        dtype=float,
    )
    fine_chi2 = [chi2_at_shift(float(s)) for s in fine]
    i_min = int(np.argmin(fine_chi2))
    s_best = float(fine[i_min])
    chi2_best = float(fine_chi2[i_min])

    n_f = len(fine)
    deg_f = min(2, max(1, n_f - 1))
    coef2 = np.polyfit(fine, np.asarray(fine_chi2, dtype=float), deg_f)
    if len(coef2) >= 3:
        a2, b2, c2 = float(coef2[0]), float(coef2[1]), float(coef2[2])
        s_parab = float(-b2 / (2.0 * a2)) if abs(a2) > 1e-20 else s_best
    else:
        a2 = float(coef2[0])
        b2 = float(coef2[1]) if len(coef2) > 1 else 0.0
        c2 = 0.0
        s_parab = s_best

    sec.update({
        'skipped': False,
        'n_template_bcds': len(tpl),
        'shifted_submodel': 'host+nucleus+BG (implemented as GP+host+BG)',
        'dipole_mx_pix': float(mx),
        'dipole_my_pix': float(my),
        'dipole_mag_pix': float(dmag),
        'unit_vector_scene_xy': [float(ux), float(uy)],
        'coarse_grid_s_pix': coarse_s.tolist(),
        'coarse_chi2': [float(x) for x in coarse_chi2],
        'coarse_poly_degree': deg,
        'coarse_poly_coeffs_high_to_low': [float(c) for c in coef],
        'parabola_vertex_coarse_s_pix': s_vertex,
        'fine_half_width_pix': float(fine_half_width),
        'fine_step_pix': float(fine_step),
        'fine_scan_n': int(len(fine)),
        'fine_grid_s_pix': fine.tolist(),
        'fine_chi2': [float(x) for x in fine_chi2],
        'best_s_pix': s_best,
        'chi2_at_best': chi2_best,
        'parabola_vertex_fine_s_pix': s_parab,
        'parabola_coeffs_fine': [a2, b2, c2] if len(coef2) >= 3 else list(coef2.astype(float)),
    })
    return out


def plot_chi2_scan(
    scan_result: Dict[str, Any],
    out_path: str,
    title: Optional[str] = None,
) -> None:
    """Save χ² vs shift: coarse scan + poly fit; fine scan around vertex; best shift marked."""
    import matplotlib.pyplot as plt

    sec = scan_result.get('dipole_chi2_refinement', {})
    if sec.get('skipped'):
        return
    c_s = np.asarray(sec['coarse_grid_s_pix'], dtype=float)
    c_x = np.asarray(sec['coarse_chi2'], dtype=float)
    s_best = sec['best_s_pix']
    chi2_best = sec['chi2_at_best']

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), height_ratios=[1.1, 1.0])
    ax0.plot(c_s, c_x, 'ko-', label='coarse (0–3 px, step 0.1)', markersize=4)

    deg = int(sec.get('coarse_poly_degree', 2))
    coef = sec.get('coarse_poly_coeffs_high_to_low')
    if coef is not None and len(coef) > 1 and deg >= 2:
        coef = np.asarray(coef, dtype=float)
        s_plot = np.linspace(float(c_s[0]), float(c_s[-1]), 200)
        y_fit = np.polyval(coef, s_plot)
        ax0.plot(s_plot, y_fit, 'b--', alpha=0.7, label=f'poly fit (deg {deg})')

    v_coarse = sec.get('parabola_vertex_coarse_s_pix')
    if v_coarse is not None:
        ax0.axvline(float(v_coarse), color='g', ls=':', lw=1, alpha=0.8, label='coarse vertex')

    ax0.set_ylabel('χ²')
    ax0.set_title(title or 'Template dipole χ² scan')
    ax0.legend(loc='best', fontsize=7)
    ax0.grid(True, alpha=0.3)

    f_s = sec.get('fine_grid_s_pix')
    f_x = sec.get('fine_chi2')
    if f_s and f_x:
        f_s = np.asarray(f_s, dtype=float)
        f_x = np.asarray(f_x, dtype=float)
        ax1.plot(f_s, f_x, 's-', color='C1', ms=3, label='fine (±0.2 px, step 0.01)')
    ax1.axvline(s_best, color='r', ls='--', lw=1.5, label=f'best s = {s_best:.4f} px')
    ax1.scatter([s_best], [chi2_best], c='r', s=90, zorder=5, marker='*')
    ax1.set_xlabel('shift along dipole (scene pixels)')
    ax1.set_ylabel('χ²')
    ax1.legend(loc='best', fontsize=7)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_shifted_residual_gallery(
    scan_result: Dict[str, Any],
    cutouts,
    results,
    stars,
    star_fluxes,
    stretch_mask,
    out_path: str,
) -> None:
    """
    Plot median template residual images for shifts:
    -0.1, -0.05, 0, 0.05, 0.1, and best shift.
    """
    import matplotlib.pyplot as plt

    sec = scan_result.get('dipole_chi2_refinement', {})
    if sec.get('skipped'):
        return

    tpl = _template_indices(cutouts)
    if not tpl:
        return

    ux, uy = sec['unit_vector_scene_xy']
    s_best = float(sec['best_s_pix'])
    shifts = [-0.1, -0.05, 0.0, 0.05, 0.1, s_best]
    labels = ['-0.10', '-0.05', '0.00', '0.05', '0.10', f'best={s_best:.3f}']

    model_cube = []
    for i in tpl:
        pred = solver.predict_cutout_model(
            results, cutouts, stars, star_fluxes, i,
            include_transient=False, include_stars=False, include_gp=True, include_host=True,
        )
        model_cube.append(np.asarray(pred, dtype=float))
    tpl_data = [np.asarray(cutouts[i]['data'], dtype=float) for i in tpl]
    vm = _template_intersection_mask(cutouts, tpl, stretch_mask, results['scene_shape'], results['scene_wcs'])
    stack_wcs = results.get('scene_wcs')
    stack_shape = tuple(results['scene_shape'])

    resid_maps = []
    for s in shifts:
        cube = []
        for i, (d, m) in enumerate(zip(tpl_data, model_cube)):
            cube.append(
                _to_stack_grid(d - shift_model_in_scene_pixels(m, cutouts[tpl[i]]['wcs'], stack_wcs, float(s) * ux, float(s) * uy), cutouts[tpl[i]]['wcs'], stack_wcs, stack_shape)
            )
        resid_maps.append(np.nanmedian(np.stack(cube, axis=0), axis=0))
    if resid_maps:
        vm = vm & np.isfinite(resid_maps[0])

    # Symmetric residual scaling around zero via p1/p99 of |resid| over all panels.
    vals = []
    for r in resid_maps:
        vv = r[vm & np.isfinite(r)]
        if vv.size:
            vals.append(np.abs(vv))
    if vals:
        merged = np.concatenate(vals)
        lim = float(np.nanpercentile(merged, 99))
        lim = max(lim, 1e-20)
    else:
        lim = 1.0

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=False)
    for ax, rmap, lab in zip(axes.ravel(), resid_maps, labels):
        im = ax.imshow(rmap, origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.set_title(f'shift {lab} px')
        ax.axis('off')
    # Keep colorbar in a dedicated side axis to avoid overlap/warnings with tight layout.
    fig.subplots_adjust(left=0.04, right=0.90, top=0.90, bottom=0.06, wspace=0.10, hspace=0.10)
    cax = fig.add_axes([0.92, 0.20, 0.02, 0.60])
    fig.colorbar(im, cax=cax, label='Median template residual (Jy)')
    plt.suptitle('Template residuals after shifting fitted host+nucleus+BG along dipole')
    plt.savefig(out_path, dpi=150)
    plt.close()


def json_sanitize(obj):
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.floating):
        x = float(obj)
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(obj, np.integer):
        return int(obj)
    return obj
