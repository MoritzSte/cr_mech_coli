#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic plotting for synthetic image optimization.

Generates side-by-side comparisons of real and synthetic microscope images
with intensity histograms and region-specific quality metrics (SSIM, PSNR,
histogram distance).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Tuple, Dict
import random
import tifffile as tiff
from tqdm import tqdm

from .scene import create_synthetic_scene
from .metrics import (
    load_image,
    compute_all_metrics,
    compute_color_distribution,
    compute_psnr,
    compute_ssim,
)
from .parameter_registry import PARAMETER_REGISTRY


def _extract_masked_region(image, mask, region):
    """Zero out pixels outside ``region`` for histogram-style display only.

    The fake-cliff problem that motivated removing this helper from the
    DE inner loop does NOT apply to plotting — the histograms here are
    pixel-value distributions (no sliding window), so a hard mask is
    fine.  Kept private to this module; the loss path uses soft weight
    maps via ``crm_gen.weight_maps``.
    """
    if mask.ndim == 3:
        mask = np.max(mask, axis=2)
    out = image.copy()
    if region == "foreground":
        out[mask == 0] = 0.0
    elif region == "background":
        out[mask > 0] = 0.0
    else:
        raise ValueError(f"Invalid region: {region!r}")
    return out


def _create_histogram_subplot(
    ax: plt.Axes,
    real_img: np.ndarray,
    synth_img: np.ndarray,
    title: str,
    histogram_distance: float,
    ssim_score: float,
    psnr_db: float,
    exclude_zeros: bool = False,
    ms_ssim: float = None,
    power_spectrum: float = None,
) -> None:
    """
    Create a histogram subplot comparing intensity distributions with metrics overlay.

    Args:
        ax (plt.Axes): Matplotlib axes to plot on.
        real_img (np.ndarray): Real microscope image (float [0,1]).
        synth_img (np.ndarray): Synthetic image (float [0,1]).
        title (str): Title for the subplot.
        histogram_distance (float): L1 histogram distance between images.
        ssim_score (float): SSIM score between images.
        psnr_db (float): PSNR value in dB.
        exclude_zeros (bool): If True, exclude zero-valued pixels from histograms.
        ms_ssim (float): Optional MS-SSIM score. Displayed when not None.
        power_spectrum (float): Optional power spectrum distance. Displayed when not None.
    """
    real_values = real_img.flatten()
    synth_values = synth_img.flatten()

    if exclude_zeros:
        real_values = real_values[real_values > 0]
        synth_values = synth_values[synth_values > 0]

    hist_real, bins_real = np.histogram(
        real_values, bins=256, range=(0, 1), density=True
    )
    hist_synth, _ = np.histogram(synth_values, bins=256, range=(0, 1), density=True)
    bin_centers = (bins_real[:-1] + bins_real[1:]) / 2

    ax.plot(
        bin_centers, hist_real, label="Real", color="#2E86AB", linewidth=2, alpha=0.8
    )
    ax.plot(
        bin_centers,
        hist_synth,
        label="Synthetic",
        color="#A23B72",
        linewidth=2,
        alpha=0.8,
    )

    ax.set_xlabel("Intensity", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_xlim(0, 1)

    metrics_text = f"Hist: {histogram_distance:.2f}\nSSIM: {ssim_score:.3f}\nPSNR: {psnr_db:.1f} dB"
    if ms_ssim is not None:
        metrics_text += f"\nMS-SSIM: {ms_ssim:.3f}"
    if power_spectrum is not None:
        metrics_text += f"\nPSD: {power_spectrum:.4f}"
    ax.text(
        0.02,
        0.97,
        metrics_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def _format_param_subtitle(
    params: Dict,
    active_param_names: List[str] = None,
    prefix: str = "",
    max_per_line: int = 4,
) -> str:
    """Format parameter values into a multi-line subtitle string.

    Args:
        params: Full parameter dictionary.
        active_param_names: If provided, only these params are shown.
        prefix: Optional first line (e.g. "Optimized Parameters:").
        max_per_line: Max number of params per line before wrapping.

    Returns:
        Multi-line string suitable for use as a figure suptitle.
    """
    display_names = active_param_names if active_param_names is not None else list(params.keys())
    param_parts = []
    for name in display_names:
        val = params[name]
        if name in PARAMETER_REGISTRY and PARAMETER_REGISTRY[name].dtype in ("int", "int_odd"):
            param_parts.append(f"{name}={int(val)}")
        else:
            param_parts.append(f"{name}={val:.4g}")

    lines = []
    if prefix:
        lines.append(prefix)
    for i in range(0, len(param_parts), max_per_line):
        lines.append(", ".join(param_parts[i:i + max_per_line]))
    return "\n".join(lines)


def generate_detailed_plots(
    image_pairs: List[Tuple[Path, Path]],
    params: Dict,
    per_image_metrics: List[Dict],
    output_dir: Path,
    n_vertices: int,
    synth_cache: Dict = None,
    active_param_names: List[str] = None,
) -> None:
    """
    Generate detailed per-image plots with region-specific analysis.

    Creates one plot per image pair showing full, background, and foreground
    histogram comparisons along with image and mask visualizations.

    Args:
        image_pairs (List[Tuple[Path, Path]]): List of (image_path, mask_path) tuples.
        params (Dict): Optimized synthetic parameters (bg_base_brightness, etc.).
        per_image_metrics (List[Dict]): Per-image metrics from compute_final_metrics().
        output_dir (Path): Output directory for saving plots.
        n_vertices (int): Number of vertices for cell shape extraction.
        synth_cache (Dict): Optional cache mapping image names to
            (synthetic_img_float, synthetic_mask) tuples from compute_final_metrics().
    """
    print(f"\nGenerating detailed plots for {len(image_pairs)} images...")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    metrics_lookup = {m["image_name"]: m for m in per_image_metrics}

    for real_img_path, mask_path in tqdm(
        image_pairs, desc="Creating detailed plots"
    ):
        if synth_cache and real_img_path.name in synth_cache:
            synthetic_img, synthetic_mask = synth_cache[real_img_path.name]
        else:
            synthetic_img, synthetic_mask = create_synthetic_scene(
                microscope_image_path=real_img_path,
                segmentation_mask_path=mask_path,
                output_dir=None,
                n_vertices=n_vertices,
                params=params,
                save=False,
            )
            if synthetic_img.dtype == np.uint8:
                synthetic_img = synthetic_img.astype(np.float64) / 255.0

        real_img = load_image(real_img_path)
        original_mask = tiff.imread(mask_path)

        real_fg = _extract_masked_region(real_img, original_mask, "foreground")
        real_bg = _extract_masked_region(real_img, original_mask, "background")
        synth_fg = _extract_masked_region(
            synthetic_img, synthetic_mask, "foreground"
        )
        synth_bg = _extract_masked_region(
            synthetic_img, synthetic_mask, "background"
        )

        img_metrics = metrics_lookup[real_img_path.name]

        fig = plt.figure(figsize=(14, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

        ax_full = fig.add_subplot(gs[0, 0])
        _create_histogram_subplot(
            ax_full,
            real_img,
            synthetic_img,
            "Full Image",
            img_metrics["full_histogram_distance"],
            img_metrics["full_ssim_score"],
            img_metrics["full_psnr_db"],
            ms_ssim=img_metrics.get("full_ms_ssim"),
            power_spectrum=img_metrics.get("full_power_spectrum"),
        )

        # The fit's CSV only stores SSIM / MS-SSIM per region (histogram,
        # PSNR, power-spectrum are global under the new metrics layout).
        # For these diagnostic plots we recompute hist + PSNR locally on
        # the masked arrays — these are display-only numbers, not the
        # values DE optimised against, so they're labelled with "(local)".
        bg_hist = compute_color_distribution(real_bg, synth_bg)["histogram_distance"]
        bg_psnr = compute_psnr(real_bg, synth_bg)["psnr"]
        ax_bg = fig.add_subplot(gs[0, 1])
        _create_histogram_subplot(
            ax_bg,
            real_bg,
            synth_bg,
            "Background Region",
            bg_hist,
            img_metrics["bg_ssim_score"],
            bg_psnr,
            exclude_zeros=True,
            ms_ssim=img_metrics.get("bg_ms_ssim"),
            power_spectrum=None,  # global metric; not recomputed per region
        )

        fg_hist = compute_color_distribution(real_fg, synth_fg)["histogram_distance"]
        fg_psnr = compute_psnr(real_fg, synth_fg)["psnr"]
        ax_fg = fig.add_subplot(gs[1, 0])
        _create_histogram_subplot(
            ax_fg,
            real_fg,
            synth_fg,
            "Foreground Region (Bacteria)",
            fg_hist,
            img_metrics["fg_ssim_score"],
            fg_psnr,
            exclude_zeros=True,
            ms_ssim=img_metrics.get("fg_ms_ssim"),
            power_spectrum=None,
        )

        ax_images = fig.add_subplot(gs[1, 1])
        real_display = (
            real_img if len(real_img.shape) == 2 else np.mean(real_img, axis=2)
        )
        synth_display = (
            synthetic_img
            if len(synthetic_img.shape) == 2
            else np.mean(synthetic_img, axis=2)
        )
        combined = np.hstack([real_display, synth_display])
        ax_images.imshow(combined, cmap="gray", vmin=0, vmax=1)
        ax_images.set_title("Image Comparison", fontsize=10, fontweight="bold")
        ax_images.axis("off")
        h, w = real_display.shape
        ax_images.text(
            w // 2, h + 15, "Real", ha="center", fontsize=9, fontweight="bold"
        )
        ax_images.text(
            w + w // 2,
            h + 15,
            "Synthetic",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

        ax_real_mask = fig.add_subplot(gs[2, 0])
        mask_display = (
            original_mask
            if len(original_mask.shape) == 2
            else np.max(original_mask, axis=2)
        )
        ax_real_mask.imshow(mask_display, cmap="tab20", interpolation="nearest")
        ax_real_mask.set_title(
            "Real Segmentation Mask", fontsize=10, fontweight="bold"
        )
        ax_real_mask.axis("off")

        ax_synth_mask = fig.add_subplot(gs[2, 1])
        synth_mask_display = (
            synthetic_mask
            if len(synthetic_mask.shape) == 2
            else np.max(synthetic_mask, axis=2)
        )
        ax_synth_mask.imshow(
            synth_mask_display, cmap="tab20", interpolation="nearest"
        )
        ax_synth_mask.set_title(
            "Synthetic Segmentation Mask", fontsize=10, fontweight="bold"
        )
        ax_synth_mask.axis("off")

        param_sub = _format_param_subtitle(params, active_param_names)
        n_lines = param_sub.count("\n") + 1
        fig.suptitle(
            f"{real_img_path.stem}\n{param_sub}",
            fontsize=12 if n_lines <= 2 else 10,
            fontweight="bold",
        )

        plot_path = plots_dir / f"plot_{real_img_path.stem}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved detailed plots to: {plots_dir}")


def generate_comparison_plot(
    image_pairs: List[Tuple[Path, Path]],
    params: Dict,
    output_dir: Path,
    n_vertices: int,
    synth_cache: Dict = None,
    active_param_names: List[str] = None,
    weights: Dict = None,
):
    """
    Generate comparison plot showing original vs synthetic for example images.

    Creates a single figure with histogram comparisons and side-by-side image
    views for a sample of image pairs (up to 3).

    Args:
        image_pairs (List[Tuple[Path, Path]]): List of (image_path, mask_path) tuples.
        params (Dict): Optimized synthetic parameters (bg_base_brightness, etc.).
        output_dir (Path): Output directory for saving the plot.
        n_vertices (int): Number of vertices for cell shape extraction.
        synth_cache (Dict): Optional cache mapping image names to
            (synthetic_img_float, synthetic_mask) tuples from compute_final_metrics().
    """
    print("\nGenerating comparison plot...")

    if len(image_pairs) <= 3:
        example_pairs = list(image_pairs)
    else:
        random.seed(42)
        example_pairs = random.sample(image_pairs, 3)

    num_examples = len(example_pairs)
    print(f"  Showing {num_examples} example(s)...")

    include_ms_ssim = weights.get("ms_ssim", 0.0) > 0 if weights else False
    include_power_spectrum = weights.get("power_spectrum", 0.0) > 0 if weights else False

    examples = []
    for real_img_path, mask_path in tqdm(example_pairs, desc="Generating examples"):
        if synth_cache and real_img_path.name in synth_cache:
            synthetic_img, _ = synth_cache[real_img_path.name]
        else:
            synthetic_img, _ = create_synthetic_scene(
                microscope_image_path=real_img_path,
                segmentation_mask_path=mask_path,
                output_dir=None,
                n_vertices=n_vertices,
                params=params,
                save=False,
            )
            if synthetic_img.dtype == np.uint8:
                synthetic_img = synthetic_img.astype(np.float64) / 255.0

        real_img = load_image(real_img_path)
        metrics = compute_all_metrics(
            real_img, synthetic_img,
            include_ms_ssim=include_ms_ssim,
            include_power_spectrum=include_power_spectrum,
        )

        examples.append(
            {
                "name": real_img_path.stem,
                "real": real_img,
                "synthetic": synthetic_img,
                "metrics": metrics["summary"],
            }
        )

    nrows = 2
    ncols = num_examples
    fig = plt.figure(figsize=(6 * num_examples, 8))

    gs = GridSpec(
        nrows, ncols, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.2
    )

    for idx, example in enumerate(examples):
        real_display = example["real"]
        synth_display = example["synthetic"]

        if len(real_display.shape) == 3:
            real_display = np.mean(real_display, axis=2)
        if len(synth_display.shape) == 3:
            synth_display = np.mean(synth_display, axis=2)

        ax_hist = fig.add_subplot(gs[0, idx])
        _create_histogram_subplot(
            ax_hist,
            real_display,
            synth_display,
            example["name"],
            example["metrics"]["histogram_distance"],
            example["metrics"]["ssim_score"],
            example["metrics"]["psnr_db"],
            ms_ssim=example["metrics"].get("ms_ssim_score"),
            power_spectrum=example["metrics"].get("power_spectrum_distance"),
        )

        ax_img = fig.add_subplot(gs[1, idx])

        combined = np.hstack([real_display, synth_display])

        ax_img.imshow(combined, cmap="gray", vmin=0, vmax=1)
        ax_img.set_title("Image Comparison", fontsize=10, fontweight="bold")
        ax_img.axis("off")

        h, w = real_display.shape
        ax_img.text(
            w // 2, h + 20, "Original", ha="center", fontsize=10, fontweight="bold"
        )
        ax_img.text(
            w + w // 2,
            h + 20,
            "Synthetic",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    title_str = _format_param_subtitle(
        params, active_param_names, prefix="Optimized Parameters:"
    )
    n_lines = title_str.count("\n") + 1
    fig.suptitle(title_str, fontsize=14 if n_lines <= 2 else 11, fontweight="bold", y=0.995)

    plt.tight_layout()

    plot_path = output_dir / "comparison_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {plot_path.name}")
