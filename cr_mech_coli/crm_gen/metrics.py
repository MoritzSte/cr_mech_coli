#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics for comparing synthetic images with real microscope images.

This module provides three key metrics:

- Color/Intensity Distribution (histogram comparison)
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)

These metrics are used to evaluate and optimize synthetic image generation
to match real microscope images.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional
import tifffile
from skimage import img_as_float
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr,
)
import matplotlib.pyplot as plt
import json


def load_image(path: Path) -> np.ndarray:
    """
    Load a TIFF image and convert to float [0,1].

    Args:
        path (Path): Path to the image file.

    Returns:
        np.ndarray: Image as float array with values in [0,1].
    """
    img = tifffile.imread(path)
    img = img_as_float(img)
    if img.min() < 0.0 or img.max() > 1.0:
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
    return img


def compute_color_distribution(
    image1: np.ndarray, image2: np.ndarray, bins: int = 256
) -> Dict[str, np.ndarray]:
    """
    Compute and compare color/intensity distributions of two images.

    Args:
        image1 (np.ndarray): First image (e.g., original).
        image2 (np.ndarray): Second image (e.g., synthetic).
        bins (int): Number of histogram bins.

    Returns:
        dict: Dictionary containing 'hist1', 'hist2', 'bin_edges',
            'histogram_diff', and 'histogram_distance' (L1 norm).
    """
    # Flatten images to 1D
    flat1 = image1.flatten()
    flat2 = image2.flatten()

    # Compute histograms with same bins
    hist1, bin_edges = np.histogram(flat1, bins=bins, range=(0, 1), density=True)
    hist2, _ = np.histogram(flat2, bins=bins, range=(0, 1), density=True)

    # Compute difference and distance
    hist_diff = np.abs(hist1 - hist2)
    hist_distance = np.sum(hist_diff)

    return {
        "hist1": hist1,
        "hist2": hist2,
        "bin_edges": bin_edges,
        "histogram_diff": hist_diff,
        "histogram_distance": float(hist_distance),
    }


def compute_ssim(
    image1: np.ndarray,
    image2: np.ndarray,
    data_range: float = 1.0,
    return_full: bool = False,
) -> Dict[str, float]:
    """
    Compute Structural Similarity Index (SSIM) between two images.

    SSIM measures the structural similarity between images, considering
    luminance, contrast, and structure. Values range from -1 to 1, where
    1 indicates perfect similarity.

    Args:
        image1 (np.ndarray): First image (e.g., original).
        image2 (np.ndarray): Second image (e.g., synthetic).
        data_range (float): Data range of the images (1.0 for float images).
        return_full: If True, also return the per-pixel SSIM map under key
            ``ssim_map``. Required for region-weighted aggregation.

    Returns:
        dict: Dictionary containing 'ssim' score (higher is better, max=1.0),
            and optionally 'ssim_map' (per-pixel array of the same spatial
            shape as the inputs).
    """
    is_rgb = len(image1.shape) == 3 and image1.shape[2] == 3
    kwargs = dict(data_range=data_range)
    if is_rgb:
        kwargs["channel_axis"] = 2

    if return_full:
        ssim_score, ssim_map = ssim(image1, image2, full=True, **kwargs)
        # For RGB, skimage returns an (H, W, C) map — collapse to (H, W).
        if ssim_map.ndim == 3:
            ssim_map = ssim_map.mean(axis=-1)
        # Float32 keeps the per-eval map small (~5 MB vs ~10 MB for ~1.3 MP).
        # Pairwise summation in numpy keeps reduction error well below the
        # noise floor of any downstream metric.
        ssim_map = ssim_map.astype(np.float32, copy=False)
        return {"ssim": float(ssim_score), "ssim_map": ssim_map}
    else:
        ssim_score = ssim(image1, image2, **kwargs)
        return {"ssim": float(ssim_score)}


def compute_psnr(
    image1: np.ndarray, image2: np.ndarray, data_range: float = 1.0
) -> Dict[str, float]:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR measures the ratio between the maximum possible signal power
    and the power of corrupting noise. Higher values indicate better quality.
    Typical range: 20-50 dB (higher is better).

    Args:
        image1 (np.ndarray): First image (e.g., original).
        image2 (np.ndarray): Second image (e.g., synthetic).
        data_range (float): Data range of the images (1.0 for float images).

    Returns:
        dict: Dictionary containing 'psnr' value in dB (higher is better).
    """
    psnr_value = psnr(image1, image2, data_range=data_range)

    return {"psnr": float(psnr_value)}


def compute_ms_ssim(
    image1: np.ndarray,
    image2: np.ndarray,
    data_range: float = 1.0,
    weights: np.ndarray = None,
    return_full: bool = False,
) -> Dict[str, float]:
    """
    Compute Multi-Scale SSIM between two images.

    Evaluates structural similarity at multiple resolutions by successively
    downsampling. More robust than single-scale SSIM for microscopy where
    features exist at multiple scales.

    Args:
        image1: First image (float [0,1]).
        image2: Second image (float [0,1]).
        data_range: Data range (1.0 for float images).
        weights: Per-scale weights. Default uses 5 scales with the
            standard weights from Wang et al. (2003).
        return_full: If True, also return the finest-scale per-pixel SSIM
            map under key ``ms_ssim_map`` (same spatial shape as inputs).
            Used for region-weighted aggregation — coarser scales don't
            have per-input-pixel values, so we expose the level-0 map as a
            proxy; this is consistent with how MS-SSIM weights the first
            level most heavily anyway.

    Returns:
        dict with 'ms_ssim' score (higher is better, max=1.0), and
        optionally 'ms_ssim_map'.
    """
    from scipy.ndimage import gaussian_filter

    if weights is None:
        weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    levels = len(weights)
    css = []  # contrast-structure per scale
    im1, im2 = image1.astype(np.float64), image2.astype(np.float64)
    finest_map = None

    for i in range(levels):
        s = ssim(im1, im2, data_range=data_range, full=True)
        if isinstance(s, tuple):
            ssim_val, ssim_map = s
        else:
            ssim_val = float(s)
            ssim_map = None

        if i == 0 and return_full and ssim_map is not None:
            finest_map = ssim_map

        if i < levels - 1:
            # Compute contrast * structure (SSIM without luminance)
            # Approximate: use SSIM itself at intermediate scales
            css.append(max(ssim_val, 0.0))
            # Downsample by 2x with Gaussian anti-aliasing
            im1 = gaussian_filter(im1, sigma=1.0)[::2, ::2]
            im2 = gaussian_filter(im2, sigma=1.0)[::2, ::2]
            if min(im1.shape) < 11:
                # Image too small for further downsampling
                # Pad remaining scales with current value
                for j in range(i + 1, levels):
                    css.append(max(ssim_val, 0.0))
                break
        else:
            css.append(max(ssim_val, 0.0))

    # Weighted geometric mean
    ms_ssim_val = np.prod(np.array(css[: len(weights)]) ** weights)
    result = {"ms_ssim": float(ms_ssim_val)}
    if return_full and finest_map is not None:
        result["ms_ssim_map"] = finest_map.astype(np.float32, copy=False)
    return result


def compute_ms_ssim_with_region_weights(
    image1: np.ndarray,
    image2: np.ndarray,
    w_fg: np.ndarray,
    w_bg: np.ndarray,
    data_range: float = 1.0,
    weights: np.ndarray = None,
) -> Dict[str, float]:
    """Compute true per-region MS-SSIM with the same weight maps as the loss.

    Runs the standard MS-SSIM pyramid on the full image AND downsamples
    the foreground / background weight maps with the same Gaussian +
    decimation chain so they stay aligned with the per-pixel SSIM map at
    every scale.  At each scale we compute three scalars (full / fg / bg)
    via :func:`compute_weighted_scalar`, clamp them to ``[0, 1]`` (mirrors
    the existing ``max(s, 0)`` clamp in :func:`compute_ms_ssim` so a 0
    score doesn't NaN the geometric mean), then combine across scales
    with the Wang-2003 weighted geometric mean.

    The per-scale "score" approximation matches the existing
    :func:`compute_ms_ssim`: rather than splitting SSIM into separate
    contrast/structure/luminance terms, we use the full SSIM at each
    scale.  This is the same approximation, just region-aware.

    If the image becomes too small for the SSIM window
    (``min(im.shape) < 11``) at some level, the loop breaks early and
    re-uses the most recent per-region scalars for the remaining scales —
    consistent with :func:`compute_ms_ssim`'s behaviour.

    Args:
        image1: Reference image.  RGB inputs are collapsed to grayscale.
        image2: Candidate image, same shape as ``image1``.
        w_fg, w_bg: Soft region weight maps from
            :func:`crm_gen.weight_maps.build_region_weight_maps`.
        data_range: SSIM data range (1.0 for ``[0, 1]`` images).
        weights: Per-scale weights; default Wang-2003 5-scale weights.

    Returns:
        ``{"full_ms_ssim": float, "fg_ms_ssim": float, "bg_ms_ssim": float}``
        — each in ``[0, 1]``, higher is better.
    """
    from scipy.ndimage import gaussian_filter

    if weights is None:
        weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    levels = len(weights)
    im1 = _ensure_grayscale(image1).astype(np.float64)
    im2 = _ensure_grayscale(image2).astype(np.float64)
    wfg = w_fg.astype(np.float64)
    wbg = w_bg.astype(np.float64)

    full_scores = []
    fg_scores = []
    bg_scores = []

    for i in range(levels):
        s = ssim(im1, im2, data_range=data_range, full=True)
        if isinstance(s, tuple):
            s_val, s_map = s
        else:
            s_val = float(s)
            s_map = np.full(im1.shape, s_val, dtype=np.float64)

        full_scores.append(max(float(s_val), 0.0))
        fg_scores.append(max(compute_weighted_scalar(s_map, wfg), 0.0))
        bg_scores.append(max(compute_weighted_scalar(s_map, wbg), 0.0))

        if i < levels - 1:
            im1 = gaussian_filter(im1, sigma=1.0)[::2, ::2]
            im2 = gaussian_filter(im2, sigma=1.0)[::2, ::2]
            wfg = gaussian_filter(wfg, sigma=1.0)[::2, ::2]
            wbg = gaussian_filter(wbg, sigma=1.0)[::2, ::2]
            if min(im1.shape) < 11:
                # Pad remaining scales with the most recent per-region scalars.
                for _ in range(i + 1, levels):
                    full_scores.append(full_scores[-1])
                    fg_scores.append(fg_scores[-1])
                    bg_scores.append(bg_scores[-1])
                break

    w = np.asarray(weights, dtype=np.float64)
    full_ms = float(np.prod(np.array(full_scores[: levels]) ** w))
    fg_ms = float(np.prod(np.array(fg_scores[: levels]) ** w))
    bg_ms = float(np.prod(np.array(bg_scores[: levels]) ** w))

    return {
        "full_ms_ssim": full_ms,
        "fg_ms_ssim": fg_ms,
        "bg_ms_ssim": bg_ms,
    }


def compute_gradient_ssim(
    image1: np.ndarray,
    image2: np.ndarray,
    data_range: float = 1.0,
    return_full: bool = False,
) -> Dict[str, float]:
    """Compute SSIM on Sobel-magnitude edge maps of both images.

    Puts direct optimisation pressure on edge features (fringes, halo
    boundaries, cell outlines) that plain intensity-SSIM under-weights.

    Both gradient magnitude maps are normalised by the **maximum across
    both images** (``max(real_edges.max(), synth_edges.max())``).  This
    keeps both maps in ``[0, 1]`` without clipping and preserves the
    relative magnitude between them, so a synthetic with stronger edges
    than the real reference still produces a score below 1 (over-sharp
    halos / PSF stay penalised).

    Typical scores run slightly higher than intensity-SSIM (gradient maps
    are sparse, so most windows compare near-zero to near-zero), hence a
    default weight matched to SSIM's is appropriate.

    Args:
        image1: Reference (real) image, float [0, 1].
        image2: Candidate (synthetic) image, float [0, 1].
        data_range: Data range after normalisation; SSIM internal use.
        return_full: If True, also return per-pixel gradient-SSIM map under
            'ssim_map' (delegated from compute_ssim).

    Returns:
        dict with 'ssim' in [-1, 1] (higher = better) and optionally
        'ssim_map' (shape matches the input images).
    """
    from scipy.ndimage import sobel

    def grad_mag(img):
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.mean(axis=-1)
        gx = sobel(img, axis=1)
        gy = sobel(img, axis=0)
        return np.hypot(gx, gy)

    real_edges = grad_mag(image1)
    synth_edges = grad_mag(image2)
    scale = max(float(real_edges.max()), float(synth_edges.max())) + 1e-8
    real_n = real_edges / scale
    synth_n = synth_edges / scale
    return compute_ssim(real_n, synth_n, data_range=data_range, return_full=return_full)


# ---------------------------------------------------------------------------
# LPIPS perceptual distance (opt-in via weight; lazy-loaded per worker)
# ---------------------------------------------------------------------------

_LPIPS_MODEL = None
_LPIPS_IMPORT_ERROR = None


def _get_lpips_model():
    """Return a cached LPIPS model, loading it on first call per process.

    The ``_LPIPS_MODEL`` global is module-local: under spawn-based
    multiprocessing each worker has its own cache (which is what we want
    — fork-inherited torch state can be unsafe).  In interactive setups
    using ``%autoreload`` or `importlib.reload(metrics)`, the cache is
    discarded along with the module and the AlexNet weights are reloaded
    on the next call.  Both behaviours are intentional.

    Respects the ``CRM_GEN_DISABLE_LPIPS=1`` environment variable as an
    opt-out for environments where torch / lpips aren't installed and the
    user wants to bypass the check.  Raises ImportError with a clear hint
    if the packages are missing but the weight is non-zero.
    """
    global _LPIPS_MODEL, _LPIPS_IMPORT_ERROR
    import os

    if os.environ.get("CRM_GEN_DISABLE_LPIPS") == "1":
        raise RuntimeError(
            "LPIPS is disabled via CRM_GEN_DISABLE_LPIPS=1; set the 'lpips' "
            "metric weight to 0 to skip it."
        )
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL
    if _LPIPS_IMPORT_ERROR is not None:
        raise _LPIPS_IMPORT_ERROR
    try:
        import torch  # noqa: F401
        import lpips

        model = lpips.LPIPS(net="alex", verbose=False).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _LPIPS_MODEL = model
        return _LPIPS_MODEL
    except ImportError as e:  # pragma: no cover — import-time environment
        _LPIPS_IMPORT_ERROR = ImportError(
            "LPIPS metric requires the 'lpips' and 'torch' packages. "
            "Install them with `pip install lpips torch`, or set "
            "CRM_GEN_DISABLE_LPIPS=1 and the 'lpips' metric weight to 0."
        )
        raise _LPIPS_IMPORT_ERROR from e


def compute_lpips_distance(
    image1: np.ndarray, image2: np.ndarray
) -> Dict[str, float]:
    """Perceptual distance via a pretrained AlexNet feature network.

    Lazy-loads the LPIPS model once per process (safe under spawn-based
    multiprocessing — each worker builds its own cache).  Always runs on
    CPU because GPU contexts don't survive a fork and aren't guaranteed to
    exist under SLURM worker allocations.

    LPIPS is a *distance* (lower = better), not a similarity.  Typical
    values during fits sit in the 0.15–0.30 range.

    Args:
        image1: Reference image, float [0, 1], grayscale or RGB.
        image2: Candidate image, same shape as ``image1``.

    Returns:
        dict with 'lpips' distance (>= 0, lower is better).
    """
    import torch

    model = _get_lpips_model()

    def to_tensor(x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=0)
        elif x.ndim == 3 and x.shape[-1] == 3:
            x = x.transpose(2, 0, 1)
        elif x.ndim == 3 and x.shape[0] == 3:
            pass
        else:
            raise ValueError(f"Unsupported image shape for LPIPS: {x.shape}")
        t = torch.from_numpy(x).unsqueeze(0)
        # LPIPS expects [-1, 1]
        return t * 2.0 - 1.0

    with torch.no_grad():
        d = model(to_tensor(image1), to_tensor(image2))
    return {"lpips": float(d.item())}


def compute_weighted_scalar(score_map: np.ndarray, weight_map: np.ndarray) -> float:
    """Return the weighted mean of a per-pixel score map.

    Used to reduce a full-image SSIM / gradient-SSIM map to a single
    region-specific score: ``Σ(w · score) / Σ(w)``.

    Args:
        score_map: Per-pixel scores (any shape matching ``weight_map``).
        weight_map: Non-negative per-pixel weights.

    Returns:
        Weighted mean as a float.  Returns 0.0 if the weight map sums to
        zero.
    """
    w_sum = float(weight_map.sum())
    if w_sum <= 0.0:
        return 0.0
    return float((score_map * weight_map).sum() / w_sum)


def compute_power_spectrum_distance(
    image1: np.ndarray, image2: np.ndarray
) -> Dict[str, float]:
    """
    Compute power spectrum distance between two images.

    Compares frequency-domain content via the magnitude of 2D FFT.
    Captures noise characteristics and texture patterns that spatial
    metrics miss.

    Args:
        image1: First image (float [0,1]).
        image2: Second image (float [0,1]).

    Returns:
        dict with 'power_spectrum_distance' (lower is better).
    """
    # Compute 2D FFT and shift zero-frequency to center
    fft1 = np.fft.fftshift(np.fft.fft2(image1))
    fft2 = np.fft.fftshift(np.fft.fft2(image2))

    # Power spectra (log scale to compress dynamic range)
    ps1 = np.log1p(np.abs(fft1))
    ps2 = np.log1p(np.abs(fft2))

    # Normalize each spectrum
    ps1 = ps1 / (ps1.sum() + 1e-10)
    ps2 = ps2 / (ps2.sum() + 1e-10)

    # L1 distance between normalized power spectra
    distance = float(np.sum(np.abs(ps1 - ps2)))

    return {"power_spectrum_distance": distance}


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Collapse a 3-channel RGB image to grayscale; pass 2-D through.

    Uses an unweighted channel mean (not Y/luminance-weighted).  This is
    appropriate for microscopy, where channels are typically identical
    (single-channel sensor replicated to RGB by the loader) or near-
    identical; for natural-scene RGB the luminance weights would be
    preferred.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        return image.mean(axis=2)
    return image


def compute_all_metrics(
    original: np.ndarray,
    synthetic: np.ndarray,
    bins: int = 256,
    include_ms_ssim: bool = False,
    include_gradient_ssim: bool = False,
    include_power_spectrum: bool = False,
    include_lpips: bool = False,
) -> Dict[str, any]:
    """
    Compute all metrics comparing original and synthetic images.

    This is the main method that computes the full image metrics in one
    call, with optional additional metrics behind flags.  Used by the
    reporting / plotting path and by tests.  The DE inner loop uses the
    region-weighted variant :func:`compute_all_metrics_with_weights`
    instead, which avoids recomputing SSIM three times per image pair.

    Args:
        original (np.ndarray): Original microscope image (float [0,1]).
        synthetic (np.ndarray): Synthetic image (float [0,1]).
        bins (int): Number of bins for histogram.
        include_ms_ssim: If True, also compute Multi-Scale SSIM.
        include_gradient_ssim: If True, compute SSIM on Sobel-edge maps.
        include_power_spectrum: If True, compute power spectrum distance.
        include_lpips: If True, compute LPIPS perceptual distance.
            Requires the 'lpips' and 'torch' packages; see
            :func:`compute_lpips_distance` for details.

    Returns:
        dict: Dictionary containing 'color_distribution', 'ssim', 'psnr'
            (always — PSNR remains available for reporting),
            'summary' statistics, and any requested optional metrics.
    """
    original = _ensure_grayscale(original)
    synthetic = _ensure_grayscale(synthetic)

    if original.shape != synthetic.shape:
        raise ValueError(
            f"Image shapes don't match after conversion: {original.shape} vs {synthetic.shape}"
        )

    color_dist = compute_color_distribution(original, synthetic, bins=bins)
    ssim_result = compute_ssim(original, synthetic)
    psnr_result = compute_psnr(original, synthetic)

    results = {
        "color_distribution": color_dist,
        "ssim": ssim_result,
        "psnr": psnr_result,
        "summary": {
            "histogram_distance": color_dist["histogram_distance"],
            "ssim_score": ssim_result["ssim"],
            "psnr_db": psnr_result["psnr"],
        },
    }

    if include_ms_ssim:
        ms_ssim_result = compute_ms_ssim(original, synthetic)
        results["ms_ssim"] = ms_ssim_result
        results["summary"]["ms_ssim_score"] = ms_ssim_result["ms_ssim"]

    if include_gradient_ssim:
        g_result = compute_gradient_ssim(original, synthetic)
        results["gradient_ssim"] = g_result
        results["summary"]["gradient_ssim_score"] = g_result["ssim"]

    if include_power_spectrum:
        ps_result = compute_power_spectrum_distance(original, synthetic)
        results["power_spectrum"] = ps_result
        results["summary"]["power_spectrum_distance"] = ps_result[
            "power_spectrum_distance"
        ]

    if include_lpips:
        lpips_result = compute_lpips_distance(original, synthetic)
        results["lpips"] = lpips_result
        results["summary"]["lpips_distance"] = lpips_result["lpips"]

    return results


def compute_all_metrics_with_weights(
    original: np.ndarray,
    synthetic: np.ndarray,
    w_fg: np.ndarray,
    w_bg: np.ndarray,
    bins: int = 256,
    include_ms_ssim: bool = False,
    include_gradient_ssim: bool = False,
    include_power_spectrum: bool = False,
    include_lpips: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics once on the full image and aggregate by region.

    Runs SSIM / MS-SSIM / gradient-SSIM **once** on the unmodified full
    image (no zero-masking, so no fake cliffs at cell boundaries), then
    derives foreground and background scores from the per-pixel maps
    weighted by ``w_fg`` and ``w_bg``.

    Histogram distance, PSNR, LPIPS, and power spectrum are computed
    once on the full image and placed in the ``full`` summary only:

    - Histogram has no spatially-resolved counterpart and would otherwise
      get region-weight-scaled twice via the loss mixer.
    - PSNR is a single scalar over the whole image.
    - LPIPS's deep features need full image context; masking would defeat
      the point of using a perceptual network.
    - Power spectrum is translation-invariant and inherently global.

    SSIM, MS-SSIM, and gradient-SSIM ARE region-partitioned: each is
    computed once with ``return_full=True`` to produce a per-pixel map,
    then aggregated against ``w_fg`` and ``w_bg`` to produce the fg / bg
    scores.  MS-SSIM uses :func:`compute_ms_ssim_with_region_weights`
    which downsamples the weight maps in lockstep with the image pyramid.

    Args:
        original: Real image, float [0, 1], 2-D or (H, W, 3).
        synthetic: Synthetic image, same shape as ``original``.
        w_fg, w_bg: Soft region weight maps from
            :func:`crm_gen.weight_maps.build_region_weight_maps`.
        bins: Histogram bin count.
        include_ms_ssim, include_gradient_ssim, include_power_spectrum,
        include_lpips: Opt-in flags for the corresponding metrics.

    Returns:
        ``{"full": {...}, "fg": {...}, "bg": {...}}`` — each inner dict is
        a ``summary``-style flat mapping of metric-name → score, directly
        consumable by ``compute_weighted_loss``.
    """
    original = _ensure_grayscale(original)
    synthetic = _ensure_grayscale(synthetic)

    if original.shape != synthetic.shape:
        raise ValueError(
            f"Image shapes don't match: {original.shape} vs {synthetic.shape}"
        )
    if original.shape != w_fg.shape or original.shape != w_bg.shape:
        raise ValueError(
            f"Weight-map shape {w_fg.shape} doesn't match image {original.shape}"
        )

    color_dist = compute_color_distribution(original, synthetic, bins=bins)
    hist_distance = color_dist["histogram_distance"]

    ssim_res = compute_ssim(original, synthetic, return_full=True)
    ssim_map = ssim_res["ssim_map"]

    # Histogram + PSNR live in `full` only; they are not region-partitioned.
    full = {
        "histogram_distance": hist_distance,
        "ssim_score": ssim_res["ssim"],
        "psnr_db": compute_psnr(original, synthetic)["psnr"],
    }
    fg = {"ssim_score": compute_weighted_scalar(ssim_map, w_fg)}
    bg = {"ssim_score": compute_weighted_scalar(ssim_map, w_bg)}

    if include_ms_ssim:
        ms = compute_ms_ssim_with_region_weights(original, synthetic, w_fg, w_bg)
        full["ms_ssim_score"] = ms["full_ms_ssim"]
        fg["ms_ssim_score"] = ms["fg_ms_ssim"]
        bg["ms_ssim_score"] = ms["bg_ms_ssim"]

    if include_gradient_ssim:
        g = compute_gradient_ssim(original, synthetic, return_full=True)
        full["gradient_ssim_score"] = g["ssim"]
        if "ssim_map" in g:
            fg["gradient_ssim_score"] = compute_weighted_scalar(g["ssim_map"], w_fg)
            bg["gradient_ssim_score"] = compute_weighted_scalar(g["ssim_map"], w_bg)
        else:
            fg["gradient_ssim_score"] = g["ssim"]
            bg["gradient_ssim_score"] = g["ssim"]

    if include_power_spectrum:
        # Power spectrum is translation-invariant and inherently global; place
        # in `full` only so the global-terms branch of `compute_weighted_loss`
        # adds it once instead of double-counting via the region mixer.
        ps = compute_power_spectrum_distance(original, synthetic)["power_spectrum_distance"]
        full["power_spectrum_distance"] = ps

    if include_lpips:
        lp = compute_lpips_distance(original, synthetic)["lpips"]
        # LPIPS needs the full image context; don't region-split.  Include it
        # only in the "full" bucket so it appears as a global pressure term
        # that isn't double-counted via the fg/bg region weights.
        full["lpips_distance"] = lp

    return {"full": full, "fg": fg, "bg": bg}


def save_metrics_json(metrics: Dict, output_path: Path) -> None:
    """
    Save metrics to a JSON file.

    Args:
        metrics (dict): Metrics dictionary from compute_all_metrics().
        output_path (Path): Path where to save the JSON file.
    """
    # Dynamically convert all metrics for JSON serialization
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            serialized = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    serialized[k] = v.tolist()
                elif isinstance(v, (np.floating, np.integer)):
                    serialized[k] = float(v)
                else:
                    serialized[k] = v
            metrics_serializable[key] = serialized
        elif isinstance(value, np.ndarray):
            metrics_serializable[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            metrics_serializable[key] = float(value)
        else:
            metrics_serializable[key] = value

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    print(f"Metrics saved to: {output_path}")


def plot_metrics(
    original: np.ndarray,
    synthetic: np.ndarray,
    metrics: Dict,
    output_path: Optional[Path] = None,
    title: str = "Image Comparison Metrics",
) -> None:
    """
    Create a visualization of all metrics.

    Args:
        original (np.ndarray): Original image.
        synthetic (np.ndarray): Synthetic image.
        metrics (dict): Metrics dictionary from compute_all_metrics().
        output_path (Path): If provided, save plot to this path. If None, plot is
            only displayed.
        title (str): Title for the plot.
    """
    # Handle shape mismatches (grayscale vs RGB) for visualization
    # Convert RGB to grayscale if needed
    if len(original.shape) == 3 and original.shape[2] == 3:
        original = np.mean(original, axis=2)

    if len(synthetic.shape) == 3 and synthetic.shape[2] == 3:
        synthetic = np.mean(synthetic, axis=2)

    has_ms_ssim = "ms_ssim_score" in metrics.get("summary", {})
    has_power_spectrum = "power_spectrum_distance" in metrics.get("summary", {})
    extra_panels = int(has_ms_ssim) + int(has_power_spectrum)
    nrows = 2 + (1 if extra_panels > 0 else 0)

    fig, axes = plt.subplots(nrows, 2, figsize=(14, 5 * nrows))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Color/Intensity Distribution (top-left)
    ax = axes[0, 0]
    bin_centers = (
        metrics["color_distribution"]["bin_edges"][:-1]
        + metrics["color_distribution"]["bin_edges"][1:]
    ) / 2
    ax.plot(
        bin_centers,
        metrics["color_distribution"]["hist1"],
        label="Original",
        color="#2E86AB",
        linewidth=2,
        alpha=0.7,
    )
    ax.plot(
        bin_centers,
        metrics["color_distribution"]["hist2"],
        label="Synthetic",
        color="#A23B72",
        linewidth=2,
        alpha=0.7,
    )
    ax.set_xlabel("Intensity", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        f"Color Distribution\nL1 Distance: {metrics['summary']['histogram_distance']:.4f}",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 2. SSIM Score (top-right)
    ax = axes[0, 1]
    ssim_score = metrics["summary"]["ssim_score"]
    color = (
        "#2ECC71" if ssim_score > 0.9 else "#F39C12" if ssim_score > 0.7 else "#E74C3C"
    )
    ax.text(
        0.5,
        0.5,
        f"{ssim_score:.4f}",
        ha="center",
        va="center",
        fontsize=48,
        fontweight="bold",
        color=color,
        transform=ax.transAxes,
    )
    ax.set_title("SSIM Score\n(Structural Similarity)", fontsize=11, fontweight="bold")
    ax.text(
        0.5,
        0.25,
        "Range: -1 to 1 (higher is better)",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
        transform=ax.transAxes,
        color="gray",
    )
    ax.axis("off")

    # 3. PSNR Value (bottom-left)
    ax = axes[1, 0]
    psnr_value = metrics["summary"]["psnr_db"]
    color = (
        "#2ECC71" if psnr_value > 30 else "#F39C12" if psnr_value > 20 else "#E74C3C"
    )
    ax.text(
        0.5,
        0.5,
        f"{psnr_value:.2f} dB",
        ha="center",
        va="center",
        fontsize=40,
        fontweight="bold",
        color=color,
        transform=ax.transAxes,
    )
    ax.set_title("PSNR\n(Peak Signal-to-Noise Ratio)", fontsize=11, fontweight="bold")
    ax.text(
        0.5,
        0.25,
        "Typical range: 20-50 dB (higher is better)",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
        transform=ax.transAxes,
        color="gray",
    )
    ax.axis("off")

    # 4. Side-by-side image comparison (bottom-right)
    ax = axes[1, 1]
    # Both images are now grayscale after conversion at function start
    combined = np.hstack([original, synthetic])
    ax.imshow(combined, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Visual Comparison", fontsize=11, fontweight="bold")
    ax.axis("off")

    # Add labels
    h, w = original.shape[:2]
    ax.text(w // 2, h + 10, "Original", ha="center", fontsize=9, fontweight="bold")
    ax.text(w + w // 2, h + 10, "Synthetic", ha="center", fontsize=9, fontweight="bold")

    # 5. Optional MS-SSIM panel
    if has_ms_ssim:
        ax = axes[2, 0]
        ms_ssim_val = metrics["summary"]["ms_ssim_score"]
        color = (
            "#2ECC71" if ms_ssim_val > 0.9 else "#F39C12" if ms_ssim_val > 0.7 else "#E74C3C"
        )
        ax.text(
            0.5, 0.5, f"{ms_ssim_val:.4f}", ha="center", va="center",
            fontsize=48, fontweight="bold", color=color, transform=ax.transAxes,
        )
        ax.set_title("MS-SSIM Score\n(Multi-Scale Structural Similarity)",
                      fontsize=11, fontweight="bold")
        ax.text(
            0.5, 0.25, "Range: 0 to 1 (higher is better)", ha="center",
            va="center", fontsize=9, style="italic", transform=ax.transAxes, color="gray",
        )
        ax.axis("off")

    # 6. Optional Power Spectrum panel
    if has_power_spectrum:
        col = 1 if has_ms_ssim else 0
        ax = axes[2, col]
        ps_val = metrics["summary"]["power_spectrum_distance"]
        color = (
            "#2ECC71" if ps_val < 0.05 else "#F39C12" if ps_val < 0.2 else "#E74C3C"
        )
        ax.text(
            0.5, 0.5, f"{ps_val:.4f}", ha="center", va="center",
            fontsize=40, fontweight="bold", color=color, transform=ax.transAxes,
        )
        ax.set_title("Power Spectrum Distance\n(Frequency Domain Similarity)",
                      fontsize=11, fontweight="bold")
        ax.text(
            0.5, 0.25, "Lower is better", ha="center", va="center",
            fontsize=9, style="italic", transform=ax.transAxes, color="gray",
        )
        ax.axis("off")

    # Hide unused extra axes
    if extra_panels == 1 and nrows == 3:
        axes[2, 1].set_visible(False)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.close()
