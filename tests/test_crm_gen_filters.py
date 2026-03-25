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
