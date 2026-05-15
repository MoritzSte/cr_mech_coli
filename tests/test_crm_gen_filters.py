"""
Tests for crm_gen background, filters and metrics.
"""

import numpy as np
import pytest

from cr_mech_coli.crm_gen.background import generate_phase_contrast_background
from cr_mech_coli.crm_gen.filters import (
    create_gaussian_psf,
    apply_psf_blur,
    add_poisson_noise,
    add_gaussian_noise,
    apply_halo_effect,
    apply_edge_diffraction_fringe,
    create_halo_gradient,
    _mask_distance_fields,
)
from cr_mech_coli.crm_gen.metrics import compute_all_metrics


H, W = 64, 64


def _gray_image(value: int = 128) -> np.ndarray:
    """Uniform uint8 RGB image."""
    return np.full((H, W, 3), value, dtype=np.uint8)


def _random_image(seed: int = 0) -> np.ndarray:
    """Random uint8 RGB image."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


def _circular_mask(radius: int = 15) -> np.ndarray:
    """Boolean mask with a filled circle in the centre."""
    y, x = np.ogrid[:H, :W]
    return (x - W // 2) ** 2 + (y - H // 2) ** 2 <= radius**2


# Background

def test_background_has_spatial_variation():
    """Background should exhibit non-trivial spatial variation."""
    bg = generate_phase_contrast_background(shape=(H, W), seed=0)
    assert bg.std() > 15, f"background is nearly uniform: std={bg.std():.1f}"


def test_background_different_seeds_differ():
    bg0 = generate_phase_contrast_background(shape=(H, W), seed=0)
    bg1 = generate_phase_contrast_background(shape=(H, W), seed=99)
    assert not np.array_equal(bg0, bg1), "different seeds produced identical backgrounds"


def test_background_brightness_affects_mean():
    dark = generate_phase_contrast_background(shape=(H, W), seed=0, base_brightness=0.1)
    bright = generate_phase_contrast_background(shape=(H, W), seed=0, base_brightness=0.9)
    assert bright.mean() > dark.mean() + 120, (
        f"base_brightness had no effect: dark={dark.mean():.1f}, bright={bright.mean():.1f}"
    )


# Filters

def test_psf_kernel_sums_to_one():
    psf = create_gaussian_psf(size=7, sigma=1.0)
    assert abs(psf.sum() - 1.0) < 1e-10, f"PSF does not sum to 1: {psf.sum()}"


def test_psf_blur_smooths_image():
    """PSF blur should reduce high-frequency variation (std decreases)."""
    img = _random_image()
    result = apply_psf_blur(img, psf_sigma=2.0)
    assert result.std() < img.std(), "PSF blur did not reduce image variation"


def test_poisson_noise_statistical_properties():
    """Poisson residual: mean ≈ 0 and std ≈ sqrt(signal / peak_signal)."""
    gray, peak_signal = 0.5, 500.0
    img = np.full((256, 256), gray, dtype=np.float64)
    result = add_poisson_noise(img, peak_signal=peak_signal, seed=0)
    noise = result - img
    expected_std = np.sqrt(gray / peak_signal)
    assert abs(noise.mean()) < 0.001, f"mean={noise.mean():.5f}"
    assert abs(noise.std() - expected_std) / expected_std < 0.02, (
        f"expected std≈{expected_std:.5f}, got {noise.std():.5f}"
    )


def test_gaussian_noise_statistical_properties():
    """Gaussian residual: mean ≈ 0 and std ≈ sigma."""
    sigma = 0.05
    img = np.full((256, 256), 0.5, dtype=np.float64)
    result = add_gaussian_noise(img, sigma=sigma, seed=0)
    noise = result - img
    assert abs(noise.mean()) < 0.001, f"mean={noise.mean():.5f}"
    assert abs(noise.std() - sigma) / sigma < 0.02, (
        f"expected std≈{sigma}, got {noise.std():.5f}"
    )


def test_halo_effect_modifies_boundary_pixels():
    """Pixels at the mask boundary should be brighter after a 'bright' halo."""
    img = _gray_image(100)
    mask = _circular_mask()

    result = apply_halo_effect(img, mask, halo_intensity=0.4)

    # A thin ring just outside the mask boundary
    from scipy.ndimage import distance_transform_edt
    dist_outside = distance_transform_edt(~mask)
    boundary_ring = (dist_outside > 0) & (dist_outside <= 3)

    assert result[boundary_ring].mean() > img[boundary_ring].mean(), (
        "halo effect did not brighten boundary pixels"
    )


# Boundary-aware distance fields (shared by halo + edge fringe)

def _two_touching_labels() -> np.ndarray:
    """2-D integer label mask: two cells sharing a contact line at column 32."""
    lab = np.zeros((H, W), dtype=np.int32)
    lab[16:48, 12:32] = 1
    lab[16:48, 32:52] = 2
    return lab


def _legacy_distance_inside(binary: np.ndarray) -> np.ndarray:
    """The pre-refactor inner-distance field: distance to nearest background."""
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(binary)


def test_mask_distance_fields_binary_matches_legacy():
    """For binary masks (bool, {0,1}, {0,255}) the helper's distance_inside
    must equal the legacy distance-to-background field."""
    from scipy.ndimage import distance_transform_edt

    circ = _circular_mask()
    for m in (circ, circ.astype(np.uint8), (circ * 255).astype(np.uint8)):
        binary, dist_in, dist_out = _mask_distance_fields(m)
        assert np.array_equal(binary, circ)
        assert np.allclose(dist_in, _legacy_distance_inside(circ)), (
            "binary mask should use the legacy distance-to-background field"
        )
        assert np.allclose(dist_out, distance_transform_edt(~circ))


def test_mask_distance_fields_labeled_uses_contacts():
    """A labelled mask must yield a distance_inside that is ~0 on the
    cell-cell contact line (the legacy field would be large there)."""
    lab = _two_touching_labels()
    binary, dist_in, _ = _mask_distance_fields(lab)
    # Column 31/32 straddle the contact; a mid-height row sits deep in both cells.
    assert dist_in[32, 31] < 1.5 and dist_in[32, 32] < 1.5, (
        "labelled mask should put a boundary at the cell-cell contact"
    )
    assert _legacy_distance_inside(binary)[32, 31] > 3.0, (
        "sanity: the legacy field is large at the contact (the bug being fixed)"
    )


def test_create_halo_gradient_unchanged_for_binary():
    """The helper refactor must not change create_halo_gradient output for a
    plain binary mask — recompute the legacy formula by hand and compare."""
    from scipy.ndimage import distance_transform_edt

    circ = _circular_mask()
    inner_width, outer_width = 2.0, 8.0
    d_in = distance_transform_edt(circ)
    d_out = distance_transform_edt(~circ)
    ref = np.zeros((H, W), dtype=np.float64)
    ref[circ] = np.clip(1.0 - d_in[circ] / inner_width, 0, 1)
    dn = d_out[~circ] / outer_width
    ref[~circ] = np.exp(-6 * dn) * (dn <= 1.0)

    got = create_halo_gradient(circ, inner_width=inner_width, outer_width=outer_width)
    assert np.array_equal(got, ref), "create_halo_gradient binary output changed"


# Edge diffraction fringe

def test_edge_fringe_ripples_at_cell_contacts():
    """The fringe must ripple along a cell-cell contact when given a labelled
    mask; a merged binary mask leaves that contact line essentially flat."""
    img = _gray_image(120)
    lab = _two_touching_labels()
    binary = lab > 0

    out_labeled = apply_edge_diffraction_fringe(img, lab, edge_fringe_intensity=0.05)
    out_merged = apply_edge_diffraction_fringe(img, binary, edge_fringe_intensity=0.05)

    # Sample the shared contact column, away from the outer cell edges.
    contact = (slice(20, 44), slice(31, 33))
    labeled_delta = np.abs(
        out_labeled[contact].astype(np.int16) - img[contact].astype(np.int16)
    ).mean()
    merged_delta = np.abs(
        out_merged[contact].astype(np.int16) - img[contact].astype(np.int16)
    ).mean()

    assert labeled_delta > 1.0, (
        f"labelled-mask fringe did not ripple at the contact (delta={labeled_delta:.2f})"
    )
    assert merged_delta < labeled_delta / 2, (
        f"merged-mask fringe unexpectedly active at contact "
        f"(merged={merged_delta:.2f}, labelled={labeled_delta:.2f})"
    )


def test_edge_fringe_empty_mask_is_noop():
    """An all-zero mask must leave the image untouched (clean no-op)."""
    img = _gray_image(120)
    empty = np.zeros((H, W), dtype=np.int32)
    out = apply_edge_diffraction_fringe(img, empty, edge_fringe_intensity=0.05)
    assert np.array_equal(out, img)


def test_edge_fringe_binary_backward_compat():
    """For a 2-D bool mask the fringe must match the pre-refactor formula."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    img = _gray_image(120)
    circ = _circular_mask()
    intensity, width = 0.05, 1.5

    d_out = distance_transform_edt(~circ)
    d_in = distance_transform_edt(circ)
    dist = np.where(circ, -d_in, d_out)
    w = max(width, 1e-6)
    fringe = intensity * np.cos(2.0 * np.pi * dist / w) * np.exp(-np.abs(dist) / (w * 2.0))
    fringe = gaussian_filter(fringe, sigma=0.5)
    ref = np.clip(img.astype(np.float64) / 255.0 + fringe[..., np.newaxis], 0.0, 1.0)
    ref = (ref * 255).astype(np.uint8)

    got = apply_edge_diffraction_fringe(img, circ, edge_fringe_intensity=intensity,
                                        edge_fringe_width=width)
    assert np.array_equal(got, ref), "edge fringe binary output changed"


# Metrics

@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_metrics_identical_images():
    """Identical images must give SSIM=1 and histogram_distance=0."""
    img = _random_image().astype(np.float32) / 255.0
    metrics = compute_all_metrics(img, img)

    assert abs(metrics["summary"]["ssim_score"] - 1.0) < 1e-6, (
        f"SSIM for identical images should be 1.0, got {metrics['summary']['ssim_score']}"
    )
    assert metrics["summary"]["histogram_distance"] == 0.0, (
        f"Histogram distance for identical images should be 0, "
        f"got {metrics['summary']['histogram_distance']}"
    )


def test_metrics_different_images():
    """Very different images must give SSIM < 1 and histogram_distance > 0."""
    black = np.zeros((H, W), dtype=np.float32)
    white = np.ones((H, W), dtype=np.float32)
    metrics = compute_all_metrics(black, white)

    assert metrics["summary"]["ssim_score"] < 1.0
    assert metrics["summary"]["histogram_distance"] > 0.0


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_metrics_output_keys():
    img = _random_image().astype(np.float32) / 255.0
    metrics = compute_all_metrics(img, img)

    assert "summary" in metrics
    for key in ("ssim_score", "psnr_db", "histogram_distance"):
        assert key in metrics["summary"], f"metrics summary missing key: {key}"
