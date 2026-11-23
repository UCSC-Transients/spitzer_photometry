import numpy as np
import pytest

from src.validation_fixtures import (
    ACCEPTANCE_SEEDS,
    DEVELOPMENT_SEEDS,
    RESERVED_ACCEPTANCE_EXPOSURES,
    make_validation_fixture,
)


def test_development_fixture_is_deterministic_and_independent_truth():
    first = make_validation_fixture(seed=DEVELOPMENT_SEEDS[0])
    second = make_validation_fixture(seed=DEVELOPMENT_SEEDS[0])
    assert first.checksum() == second.checksum()
    assert np.array_equal(first.data, second.data, equal_nan=True)
    assert first.metadata["truth_renderer"].startswith("independent_analytic")
    assert first.metadata["calibration_status"] == "synthetic_analytic_response_not_real_prf_calibration"


def test_fixture_contains_required_scene_and_detector_failures():
    fixture = make_validation_fixture(seed=DEVELOPMENT_SEEDS[1])
    assert set(fixture.truth_components) == {
        "host",
        "nucleus",
        "isolated_star",
        "blended_stars",
        "transient",
        "background",
        "total",
    }
    assert len({exposure.roll_deg for exposure in fixture.exposures}) > 3
    assert len({(exposure.phase_x, exposure.phase_y) for exposure in fixture.exposures}) == len(fixture.exposures)
    assert np.count_nonzero(fixture.cosmic_ray_mask) == len(fixture.exposures)
    assert np.count_nonzero(~fixture.valid_mask) == 2 * len(fixture.exposures)
    assert np.isnan(fixture.data[~fixture.valid_mask]).all()
    assert np.isinf(fixture.uncertainty[~fixture.valid_mask]).all()
    assert np.all(fixture.truth_components["host"] >= 0.0)
    assert np.all(fixture.truth_components["nucleus"] >= 0.0)


@pytest.mark.parametrize("truth_flux", [0.0, 2.0e-3])
def test_transient_is_nonnegative_shared_and_exactly_zero_in_templates(truth_flux: float):
    fixture = make_validation_fixture(seed=DEVELOPMENT_SEEDS[2], transient_flux_jy=truth_flux)
    integrated = np.sum(fixture.truth_components["transient"], axis=(1, 2))
    for exposure, flux in zip(fixture.exposures, integrated, strict=True):
        if exposure.is_template:
            assert flux == 0.0
        else:
            assert flux >= 0.0
            # Finite fixture support loses a tiny analytic tail; the shared input
            # amplitude remains identical and is recorded in metadata.
            assert flux == pytest.approx(truth_flux, rel=2e-10, abs=1e-15)
    assert fixture.metadata["science_transient_flux_jy"] == truth_flux


def test_epoch_edges_do_not_single_linkage_chain():
    fixture = make_validation_fixture()
    science = [exposure for exposure in fixture.exposures if not exposure.is_template]
    by_epoch = {
        epoch: [exposure.mjd for exposure in science if exposure.epoch_id == epoch]
        for epoch in {exposure.epoch_id for exposure in science}
    }
    assert max(by_epoch[0]) - min(by_epoch[0]) == pytest.approx(1.0)
    assert max(by_epoch[1]) - min(by_epoch[1]) == pytest.approx(1.0)
    assert min(by_epoch[1]) - max(by_epoch[0]) > 1.0


def test_acceptance_partition_is_reserved_from_required_pr_seeds():
    with pytest.raises(ValueError, match="registered development seed"):
        make_validation_fixture(seed=ACCEPTANCE_SEEDS[0])
    acceptance = make_validation_fixture(seed=ACCEPTANCE_SEEDS[0], partition="acceptance")
    assert acceptance.metadata["partition"] == "acceptance"
    assert all(exposure.partition == "acceptance" for exposure in acceptance.exposures)
    assert all(
        any(exposure.exposure_id.startswith(prefix) for prefix in RESERVED_ACCEPTANCE_EXPOSURES)
        for exposure in acceptance.exposures
    )
