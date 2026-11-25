#!/usr/bin/env python3
"""Emit Stage 2 numerical/plot evidence through the Stage 1 artifact interface.

Real files are audited locally; only derived arrays/hashes are written. The
synthetic recovery and real-table diagnostics are separate labeled bundles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.isolated_astrometry import fit_isolated_stars
from src.point_response import NativePRF
from src.stage2_validation import isolated_star_fixture
from src.validation_fixtures import DEVELOPMENT_SEEDS
from src.validation_metrics import write_artifact_bundle


def synthetic_recovery(output: Path) -> dict:
    """Measure independent injection recovery, local coverage and convergence."""
    records, arrays = [], {}
    for sr in (1, 2, 4):
        for seed in (None, *DEVELOPMENT_SEEDS):
            frames, reference, initial, truth, flux, shifts = isolated_star_fixture(seed, sr)
            fit = fit_isolated_stars(frames, initial, .8*flux, reference_wcs=reference)
            t = time.perf_counter()
            predictions = fit.predict(frames)
            predict_seconds = time.perf_counter()-t
            sigma = np.sqrt(np.diag(fit.covariance)[:len(flux)])
            name = f"sr{sr}_seed{seed}"
            arrays.update({f"{name}_data": np.array([f.data for f in frames]),
                           f"{name}_prediction": np.array(predictions),
                           f"{name}_flux": fit.flux, f"{name}_flux_truth": flux,
                           f"{name}_centroid": fit.positions_xy, f"{name}_centroid_truth": truth,
                           f"{name}_registration": fit.translations_xy, f"{name}_registration_truth": shifts,
                           f"{name}_covariance": fit.covariance,
                           f"{name}_residual": np.array([p-f.data for p, f in zip(predictions, frames)])})
            records.append({"sr": sr, "seed": seed, "flux_bias_jy": (fit.flux-flux).tolist(),
                            "fractional_bias": (fit.flux/flux-1).tolist(), "flux_sigma_jy": sigma.tolist(),
                            "covered_local_95": (np.abs(fit.flux-flux) <= 1.96*sigma).tolist(),
                            "centroid_error_native_px": ((fit.positions_xy-truth)/sr).tolist(),
                            "registration_error_native_px": (fit.translations_xy-shifts).tolist(),
                            "predict_seconds": predict_seconds, "diagnostics": fit.diagnostics})
    metrics = {"trials": records, "failures": 0, "roll_deg": [0, 27, 73],
               "sigma_jy_per_native_pixel": 1e-5, "derivative_step_reference_pixels": .001,
               "science_tolerance": None, "promotion": "unapproved",
               "coverage_limit": "3 independent noise seeds, 9 star estimates; SR repeats are correlated; local positive-flux curvature only"}
    write_artifact_bundle(output, metrics=metrics, arrays=arrays,
                          labels={"kind": "synthetic independent analytic integration", "plot_title": "Synthetic native star data (Jy/pixel)"})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for sr in (1, 2, 4):
        rr = [r for r in records if r["sr"] == sr]
        axes[0].plot([r["seed"] or 0 for r in rr], [max(np.abs(r["fractional_bias"])) for r in rr], "o", label=f"SR={sr}")
    axes[0].set(xlabel="Development seed (0 = noiseless)", ylabel="Max |fractional flux bias|", title="Synthetic recovery")
    axes[0].legend()
    first = records[0]
    axes[1].plot([0, 27, 73], np.asarray(first["registration_error_native_px"]), "o-")
    axes[1].set(xlabel="Detector roll (deg)", ylabel="Registration error (native pixel)", title="Independent noiseless offsets")
    axes[2].semilogy(first["diagnostics"]["objective_history"], "o-")
    axes[2].set(xlabel="Residual evaluation", ylabel="Weighted squared residual", title="Joint fit convergence")
    fig.tight_layout(); fig.savefig(output/"recovery_roll_convergence.png", dpi=140); plt.close(fig)
    return metrics


def audit_calibration(directory: Path, output: Path) -> dict:
    """Audit all local warm tables without silently certifying amplitude units."""
    paths = sorted(directory.glob("apex_sh_IRACPC[12]_col*_row*_x100.fits"))
    if not paths:
        raise FileNotFoundError(f"No warm PRFs found in {directory}")
    rows, profiles, energy, phase_sums, centroid_errors, stamps = [], [], [], [], [], []
    radii = np.arange(0, 13, dtype=float)
    phases = np.arange(4)/4
    for path in paths:
        data, header = fits.getdata(path, header=True)
        table = NativePRF(data, int(header["PRFXRSMP"]), "table", False,
                          f"ch{str(header['CHAN']).strip()}", "warm",
                          hashlib.sha256(path.read_bytes()).hexdigest(), (1279., 1279.))
        native = table.sample((25, 25), 12, 12)
        yy, xx = np.indices(native.shape)
        radius = np.hypot(xx-12, yy-12)
        ap10 = float(native[radius <= 10].sum())
        integral = float(data.sum(dtype=np.float64))/table.oversample**2
        rows.append({"file": path.name, "sha256": table.sha256, "shape": list(data.shape),
                     "channel": table.channel, "mission": "warm (postcryogenic header)",
                     "header": {key: str(header.get(key, "")) for key in
                                ("CHAN", "PRFXRSMP", "PRFYRSMP", "PRFROW", "PRFCOL", "PRFDATE", "COMMENT", "HISTORY")},
                     "zero_based_detector_xy": [int(header["PRFCOL"])-1, int(header["PRFROW"])-1],
                     "zero_phase_radius10_sum": ap10, "array_integral": integral,
                     "legacy_zero_phase_radius10_sum": ap10/integral,
                     "legacy_amplitude_to_zero_phase_aperture_amplitude": ap10/integral,
                     "header_unit_aperture_discrepancy": ap10-1.,
                     "full_native_sum": float(native.sum()), "absolute_flux_gate": "blocked"})
        profiles.append([float(np.mean(native[(radius >= r)&(radius < r+1)])) for r in radii])
        energy.append([float(native[radius <= r].sum()) for r in radii])
        sums, centroids = [], []
        for phase in phases:
            response = table.sample((25, 25), 12+phase, 12+.37)
            local_ap = np.hypot(xx-12-phase, yy-12-.37) <= 10
            weights = response*local_ap
            sums.append(float(weights.sum()/integral))
            centroids.append([float((weights*xx).sum()/weights.sum()-12-phase),
                              float((weights*yy).sum()/weights.sum()-12-.37)])
        phase_sums.append(sums); centroid_errors.append(centroids); stamps.append(native)
    arrays = {"zero_phase_native_response": np.asarray(stamps), "radial_profile": np.asarray(profiles),
              "encircled_energy": np.asarray(energy), "radius_native_pixel": radii, "phase_native_pixel": phases,
              "legacy_phase_aperture_sum": np.asarray(phase_sums), "first_moment_centroid_error": np.asarray(centroid_errors)}
    metrics = {"files": rows, "n_files": len(rows), "absolute_calibration": "blocked",
               "units": "raw table response or legacy-integral response per amplitude; scene SR absent",
               "normalization": "fixed full-table integral; never a per-phase/crop sum",
               "centroid_definition": "array [1279,1279]; diagnostic first moment in radius10, not optical truth",
               "missing": ["external standard-star flux validation", "warm mean-phase/color/location convention", "held-out real BCDs", "absolute science-bias tolerance"]}
    write_artifact_bundle(output, metrics=metrics, arrays=arrays,
                          labels={"kind": "real warm PRF calibration audit, not absolute calibration", "plot_title": "Real warm zero-phase PRF (table units)"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for channel in ("ch1", "ch2"):
        indices = [i for i, row in enumerate(rows) if row["channel"] == channel]
        i = next((i for i in indices if rows[i]["zero_based_detector_xy"] == [128,128]), indices[0])
        axes[0,0].semilogy(radii, profiles[i], "o-", label=channel)
        axes[0,1].plot(radii, energy[i], "o-", label=channel)
        axes[1,0].plot(phases, phase_sums[i], "o-", label=channel)
        axes[1,1].plot([rows[j]["zero_based_detector_xy"][0]+256*rows[j]["zero_based_detector_xy"][1] for j in indices],
                       [rows[j]["legacy_zero_phase_radius10_sum"]-1 for j in indices], "o", label=channel)
    axes[0,0].set(xlabel="Radius (native pixel)", ylabel="Mean response (table units)", title="Real PRF radial profile")
    axes[0,1].set(xlabel="Radius (native pixel)", ylabel="Encircled response (table units)", title="Real PRF encircled energy")
    axes[1,0].set(xlabel="x phase (native pixel), y phase=0.37", ylabel="Legacy radius10 response", title="Phase response is preserved")
    axes[1,1].set(xlabel="Full-detector location x + 256y (zero based)", ylabel="Legacy aperture response − 1", title="Position-dependent convention discrepancy")
    for ax in axes.ravel(): ax.legend()
    fig.tight_layout(); fig.savefig(output/"radial_energy_phase_position.png", dpi=140); plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prf-dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = {"synthetic": "pending", "real_calibration": "missing", "promotion": "unapproved"}
    try:
        synthetic_recovery(args.output_dir/"synthetic")
        status["synthetic"] = "completed"
        if args.prf_dir:
            audit_calibration(args.prf_dir, args.output_dir/"real-prf")
            status["real_calibration"] = "audited; absolute flux blocked"
    except Exception as exc:
        status["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output_dir/"status.json").write_text(json.dumps(status, indent=2)+"\n")
        hashes = {str(p.relative_to(args.output_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(args.output_dir.rglob("*")) if p.is_file() and p.name != "checksums.json"}
        (args.output_dir/"checksums.json").write_text(json.dumps(hashes, indent=2)+"\n")


if __name__ == "__main__":
    main()
