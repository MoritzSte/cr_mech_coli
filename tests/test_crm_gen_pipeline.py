import json
import numpy as np
import tifffile
from glob import glob
from pathlib import Path
from cr_mech_coli.crm_gen.pipeline import run_pipeline


# Minimal synthetic parameters for a fast, non-trivial test run of the full pipeline.
_SYNTHETIC_CONFIG = {
    "bg_base_brightness": 0.5,
    "bg_gradient_strength": 0.02,
    "bac_halo_intensity": 0.3,
    "bg_noise_scale": 20,
    "psf_sigma": 1.0,
    "peak_signal": 1000.0,
    "gaussian_sigma": 0.01,
}


def test_run_pipeline_smoke(tmp_path):
    """
    Run a single-frame, small-image pipeline and verify all outputs exist.

    Parameters chosen to be as fast as possible while still exercising every
    step.
    """
    run_pipeline(
        output_dir=str(tmp_path),
        n_frames=1,
        image_size=(128, 128),
        n_bacteria_range=(2, 2),
        n_simulations=1,
        simulation_seed=42,
        n_workers=1,
        synthetic_config=_SYNTHETIC_CONFIG,
        brightness_config={"mode": "age"},
    )

    generated_dir = tmp_path / "generated"
    synthetic_dir = tmp_path / "synthetic"

    # --- output directories exist ---
    assert generated_dir.is_dir(), "generated/ directory was not created"
    assert synthetic_dir.is_dir(), "synthetic/ directory was not created"

    # --- metadata file is written and is valid JSON ---
    metadata_path = tmp_path / "metadata.json"
    assert metadata_path.is_file(), "metadata.json was not written"
    with open(metadata_path) as f:
        metadata = json.load(f)
    assert "iterations" in metadata
    assert "n_frames" in metadata
    assert metadata["n_frames"] == 1

    # --- generated images (raw renders) ---
    raw_images = sorted(generated_dir.glob("*.tif"))
    raw_masks = sorted(generated_dir.glob("*_masks.tif"))
    assert len(raw_images) >= 1, "no generated .tif files found"
    assert len(raw_masks) >= 1, "no generated mask .tif files found"

    raw_img = tifffile.imread(raw_images[0])
    assert raw_img.ndim == 3 and raw_img.shape[2] == 3, (
        f"expected RGB image (H, W, 3), got shape {raw_img.shape}"
    )
    assert raw_img.dtype == np.uint8

    # --- synthetic images (post-processed) ---
    syn_images = sorted(synthetic_dir.glob("syn_*.tif"))
    syn_masks = sorted(synthetic_dir.glob("syn_*_masks.tif"))
    assert len(syn_images) >= 1, "no synthetic .tif files found"
    assert len(syn_masks) >= 1, "no synthetic mask .tif files found"

    syn_img = tifffile.imread(syn_images[0])
    assert syn_img.ndim == 2, (
        f"expected grayscale uint16 synthetic image (H, W), got shape {syn_img.shape}"
    )
    assert syn_img.dtype == np.uint16

    syn_mask = tifffile.imread(syn_masks[0])
    assert syn_mask.ndim == 2, (
        f"expected integer label uint16 synthetic mask (H, W), got shape {syn_mask.shape}"
    )
    assert syn_mask.dtype == np.uint16

    # --- synthetic image has the same spatial dimensions as the raw image ---
    assert raw_img.shape[:2] == syn_img.shape, (
        f"generated and synthetic images have different spatial shapes: "
        f"{raw_img.shape[:2]} vs {syn_img.shape}"
    )

    # --- synthetic image is not trivially empty (all-black) ---
    assert syn_img.max() > 0, "synthetic image is completely black"
