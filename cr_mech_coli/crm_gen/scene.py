"""
Single-frame compositing of synthetic microscope images.

Takes a rendered cell image and segmentation mask, then applies a sequence of
optical effects to produce a realistic microscope frame:

    1. Generate a phase-contrast background with gradient and texture
    2. Composite foreground cells onto the background
    3. Adjust per-cell brightness (age-based or matched to a real image)
    4. Add halo artifacts around cell boundaries
    5. Apply Point Spread Function (PSF) blur, Poisson shot noise, and Gaussian readout noise

"""


import cr_mech_coli as crm
import numpy as np
import tifffile as tiff
from pathlib import Path

from . import filters
from . import bacteria
from .background import generate_phase_contrast_background
from .parameter_registry import _get_cached_defaults


def apply_synthetic_effects(
    raw_image: np.ndarray,
    mask: np.ndarray,
    bg_base_brightness: float = 0.56,
    bg_gradient_strength: float = 0.027,
    bac_halo_intensity: float = 0.30,
    bg_noise_scale: int = 20,
    psf_sigma: float = 1.0,
    peak_signal: float = 1000.0,
    gaussian_sigma: float = 0.01,
    seed: int = None,
    brightness_mode: str = "original",
    cell_ages: dict = None,
    cell_colors_map: dict = None,
    brightness_range: tuple = (0.8, 0.3),
    max_age: int = None,
    num_dark_spots_range: tuple = (0, 5),
    original_image: np.ndarray = None,
    original_mask: np.ndarray = None,
    original_colors: list = None,
    # Effect toggles (new params with backwards-compatible defaults)
    apply_psf: bool = True,
    apply_poisson: bool = True,
    apply_gaussian: bool = True,
    # Background params (previously hardcoded)
    dark_spot_size_range: tuple = (2, 5),
    num_light_spots_range: tuple = (0, 0),
    texture_strength: float = 0.02,
    texture_scale: float = 1.5,
    bg_blur_sigma: float = 0.8,
    # Halo params (previously hardcoded)
    halo_inner_width: float = 2.0,
    halo_outer_width: float = 50.0,
    halo_blur_sigma: float = 0.5,
    halo_fade_type: str = "exponential",
    # Brightness noise strength
    brightness_noise_strength: float = 0.0,
    # Separate background seed for consistent backgrounds across frames
    bg_seed: int = None,
    # Dict-based parameter override — keys override individual kwargs above
    params: dict = None,
) -> np.ndarray:
    """
    Apply synthetic microscope effects to a raw rendered image.

    This function handles the post-processing pipeline:

    1. Generate phase contrast background
    2. Combine foreground with background
    3. Apply brightness adjustment (original or age-based)
    4. Apply halo effect
    5. Apply microscope effects (PSF, noise)

    Args:
        raw_image (np.ndarray): Raw rendered image from cr_mech_coli (H x W x 3), uint8.
        mask (np.ndarray): RGB segmentation mask (H x W x 3).
        bg_base_brightness (float): Base brightness for background generation.
        bg_gradient_strength (float): Gradient strength for background generation.
        bac_halo_intensity (float): Intensity of the halo effect around bacteria.
        bg_noise_scale (int): Scale factor for background illumination noise (1-25).
        psf_sigma (float): Sigma for Gaussian PSF blur (optical diffraction).
        peak_signal (float): Peak photon count for Poisson noise (higher = less noise).
        gaussian_sigma (float): Sigma for Gaussian readout noise.
        seed (int): Random seed for reproducibility. If None, a random seed is generated.
        brightness_mode (str): "original" to match brightness from original image,
            "age" to compute brightness based on cell age.
        cell_ages (dict): Dict mapping cell identifier to age (required if
            brightness_mode="age").
        cell_colors_map (dict): Dict mapping cell identifier to RGB color (required if
            brightness_mode="age").
        brightness_range (tuple): (young_brightness, old_brightness) for age-based mode.
        max_age (int): Maximum age for normalization in age-based mode.
        num_dark_spots_range (tuple): (min, max) range for number of dark spots in
            background.
        original_image (np.ndarray): Original microscope image (required if
            brightness_mode="original").
        original_mask (np.ndarray): Original segmentation mask (required if
            brightness_mode="original").
        original_colors (list): List of colors from crm.extract_positions (required if
            brightness_mode="original").
        apply_psf (bool): Whether to apply PSF blur.
        apply_poisson (bool): Whether to apply Poisson shot noise.
        apply_gaussian (bool): Whether to apply Gaussian readout noise.
        dark_spot_size_range (tuple): Range of dark spot sizes (sigma values).
        num_light_spots_range (tuple): Range for number of light spots in background.
        texture_strength (float): Fine texture strength (0.0-1.0).
        texture_scale (float): Texture smoothness (higher = smoother).
        bg_blur_sigma (float): Gaussian blur sigma for background optical effects.
        halo_inner_width (float): Width of inner halo in pixels.
        halo_outer_width (float): Total halo width in pixels.
        halo_blur_sigma (float): Gaussian blur sigma for halo transition.
        halo_fade_type (str): Fade type: "linear", "exponential", or "gaussian".
        brightness_noise_strength (float): Perlin-like noise strength for brightness
            variation (0-1).
        bg_seed (int): Separate seed for background generation. When provided, the
            background is generated with this seed (enabling consistent backgrounds
            across frames) while ``seed`` is still used for noise and brightness.
            If None, ``seed`` is used for everything (default).

    Returns:
        np.ndarray: Processed synthetic image (H x W x 3), uint8.
    """
    # When a params dict is provided, merge with registry defaults and use
    # those values.  Individual kwargs are overridden by the dict.
    if params is not None:
        _p = {**_get_cached_defaults(), **params}
        bg_base_brightness = _p["bg_base_brightness"]
        bg_gradient_strength = _p["bg_gradient_strength"]
        bac_halo_intensity = _p["bac_halo_intensity"]
        bg_noise_scale = int(_p["bg_noise_scale"])
        psf_sigma = _p["psf_sigma"]
        peak_signal = _p["peak_signal"]
        gaussian_sigma = _p["gaussian_sigma"]
        texture_strength = _p["texture_strength"]
        texture_scale = _p["texture_scale"]
        bg_blur_sigma = _p["bg_blur_sigma"]
        dark_spot_intensity = _p["dark_spot_intensity"]
        num_dark_spots_max = int(_p["num_dark_spots_max"])
        num_dark_spots_range = (0, num_dark_spots_max)
        halo_inner_width = _p["halo_inner_width"]
        halo_outer_width = _p["halo_outer_width"]
        halo_blur_sigma = _p["halo_blur_sigma"]
        brightness_noise_strength = _p["brightness_noise_strength"]
        psf_size = int(_p["psf_size"])
        absorption_coeff = _p["absorption_coeff"]
        cell_optical_thickness = _p["cell_optical_thickness"]
        defocus_strength = _p["defocus_strength"]
        defocus_scale = int(_p["defocus_scale"])
        vignette_strength = _p["vignette_strength"]
        edge_fringe_intensity = _p["edge_fringe_intensity"]
        edge_fringe_width = _p["edge_fringe_width"]
    else:
        dark_spot_intensity = 0.15  # legacy default
        psf_size = 7
        absorption_coeff = 0.0
        cell_optical_thickness = 3.0
        defocus_strength = 0.0
        defocus_scale = 10
        vignette_strength = 0.0
        edge_fringe_intensity = 0.0
        edge_fringe_width = 1.5

    # Generate random seed if not provided
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    # Extract foreground using mask
    binary_mask = mask.sum(axis=2) > 0
    synthetic_image = raw_image.copy().astype(np.float32)
    synthetic_image *= binary_mask[..., np.newaxis]

    # Generate phase contrast background
    # Use bg_seed if provided (for consistent backgrounds across video frames),
    # otherwise fall back to the general seed.
    bg = generate_phase_contrast_background(
        shape=synthetic_image.shape[:2],
        seed=bg_seed if bg_seed is not None else seed,
        noise_scale=bg_noise_scale,
        base_brightness=bg_base_brightness,
        gradient_strength=bg_gradient_strength,
        num_dark_spots_range=num_dark_spots_range,
        dark_spot_intensity=dark_spot_intensity,
        dark_spot_size_range=dark_spot_size_range,
        num_light_spots_range=num_light_spots_range,
        texture_strength=texture_strength,
        texture_scale=texture_scale,
        blur_sigma=bg_blur_sigma,
    )

    # Combine synthetic image with background
    synthetic_image = np.where(
        binary_mask[..., np.newaxis], synthetic_image, bg[..., np.newaxis]
    )

    # Apply brightness based on selected mode
    if brightness_mode == "age":
        if cell_ages is None or cell_colors_map is None:
            raise ValueError(
                "cell_ages and cell_colors_map are required when brightness_mode='age'"
            )
        brightness_adjustment = bacteria.apply_age_based_brightness(
            synthetic_image,
            synthetic_mask=mask,
            cell_ages=cell_ages,
            cell_colors=cell_colors_map,
            brightness_range=brightness_range,
            max_age=max_age,
            noise_strength=brightness_noise_strength,
            seed=seed,
        )
    else:
        if original_image is None or original_mask is None or original_colors is None:
            raise ValueError(
                "original_image, original_mask, and original_colors are required when brightness_mode='original'"
            )
        brightness_adjustment = bacteria.apply_original_brightness(
            synthetic_image,
            synthetic_mask=mask,
            original_image=original_image,
            original_mask=original_mask,
            colors=original_colors,
            noise_strength=brightness_noise_strength,
            seed=seed,
        )

    # Apply adjustment (additive)
    synthetic_image = synthetic_image + brightness_adjustment

    # Clip to valid range
    synthetic_image = np.clip(synthetic_image, 0, 255).astype(np.uint8)

    # 4. Beer-Lambert absorption (brightfield: darkens cells)
    if absorption_coeff > 0:
        synthetic_image = filters.apply_beer_lambert_absorption(
            synthetic_image,
            binary_mask,
            absorption_coeff=absorption_coeff,
            cell_optical_thickness=cell_optical_thickness,
        )

    # 5. Phase contrast halo (no-op when bac_halo_intensity <= 0)
    if bac_halo_intensity > 0:
        synthetic_image = filters.apply_halo_effect(
            synthetic_image,
            binary_mask,
            halo_intensity=bac_halo_intensity,
            inner_width=halo_inner_width,
            outer_width=halo_outer_width,
            blur_sigma=halo_blur_sigma,
            fade_type=halo_fade_type,
        )

    # 6. Edge diffraction fringe (brightfield: thin boundary effect)
    if edge_fringe_intensity > 0:
        synthetic_image = filters.apply_edge_diffraction_fringe(
            synthetic_image,
            binary_mask,
            edge_fringe_intensity=edge_fringe_intensity,
            edge_fringe_width=edge_fringe_width,
        )

    # 7. Defocus blur (spatially varying depth-of-field)
    if defocus_strength > 0:
        synthetic_image = filters.apply_defocus_blur(
            synthetic_image,
            defocus_strength=defocus_strength,
            defocus_scale=defocus_scale,
            base_psf_sigma=psf_sigma,
            seed=seed,
        )

    # 8. PSF blur (optical diffraction)
    if apply_psf:
        synthetic_image = filters.apply_psf_blur(
            synthetic_image,
            psf_sigma=psf_sigma,
            psf_size=psf_size,
        )

    # 9. Vignetting (after optics, before sensor noise)
    if vignette_strength > 0:
        synthetic_image = filters.apply_vignetting(
            synthetic_image,
            vignette_strength=vignette_strength,
        )

    # 10. Sensor noise — Poisson (shot) then Gaussian (readout)
    if apply_poisson:
        synthetic_image = filters.add_poisson_noise(
            synthetic_image,
            peak_signal=peak_signal,
            seed=seed,
        )
    if apply_gaussian:
        gaussian_seed = (seed + 1) if seed is not None else None
        synthetic_image = filters.add_gaussian_noise(
            synthetic_image,
            sigma=gaussian_sigma,
            seed=gaussian_seed,
        )

    return synthetic_image


def create_synthetic_scene(
    microscope_image_path,
    segmentation_mask_path,
    output_dir=None,
    n_vertices: int = 8,
    bg_base_brightness: float = 0.56,
    bg_gradient_strength: float = 0.027,
    bac_halo_intensity: float = 0.30,
    bg_noise_scale: int = 20,
    psf_sigma: float = 1.0,
    peak_signal: float = 1000.0,
    gaussian_sigma: float = 0.01,
    seed: int = None,
    # Brightness mode parameters
    brightness_mode: str = "original",
    cell_ages: dict = None,
    cell_colors_map: dict = None,
    brightness_range: tuple = (0.8, 0.3),
    max_age: int = None,
    # Dark spots range
    num_dark_spots_range: tuple = (0, 5),
    # Output naming
    output_prefix: str = "syn_",
    # Brightness noise
    brightness_noise_strength: float = 0.0,
    # Effect toggles
    apply_psf: bool = True,
    apply_poisson: bool = True,
    apply_gaussian: bool = True,
    # Halo params
    halo_inner_width: float = 2.0,
    halo_outer_width: float = 50.0,
    halo_blur_sigma: float = 0.5,
    halo_fade_type: str = "exponential",
    # Remaining background params
    dark_spot_size_range: tuple = (2, 5),
    num_light_spots_range: tuple = (0, 0),
    texture_strength: float = 0.02,
    texture_scale: float = 1.5,
    bg_blur_sigma: float = 0.8,
    # Dict-based parameter override — keys override individual kwargs above
    params: dict = None,
    # Whether to write output files to disk (set False for optimization/screening)
    save: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a synthetic microscope image from a real one using cr_mech_coli.

    Args:
        microscope_image_path: Path to the real microscope image (TIF).
        segmentation_mask_path: Path to the segmentation mask (TIF).
        output_dir: Directory to save synthetic images.
        n_vertices (int): Number of vertices to extract per cell.
        bg_base_brightness (float): Base brightness for background generation.
        bg_gradient_strength (float): Gradient strength for background generation.
        bac_halo_intensity (float): Intensity of the halo effect around bacteria.
        bg_noise_scale (int): Scale factor for background illumination noise (1-25).
        psf_sigma (float): Sigma for Gaussian PSF blur (optical diffraction).
        peak_signal (float): Peak photon count for Poisson noise (higher = less noise).
        gaussian_sigma (float): Sigma for Gaussian readout noise.
        seed (int): Random seed for reproducibility. If None, a random seed is generated.
        brightness_mode (str): "original" to match brightness from original image,
            "age" to compute brightness based on cell age.
        cell_ages (dict): Dict mapping cell identifier to age (required if
            brightness_mode="age").
        cell_colors_map (dict): Dict mapping cell identifier to RGB color (required if
            brightness_mode="age").
        brightness_range (tuple): (young_brightness, old_brightness) for age-based mode.
        max_age (int): Maximum age for normalization in age-based mode.
        num_dark_spots_range (tuple): (min, max) range for number of dark spots in
            background.
        output_prefix (str): Prefix for output filenames.

    Returns:
        tuple[np.ndarray, np.ndarray]: (synthetic_image, synthetic_mask).
    """
    # Generate random seed if not provided
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)
    # Create output directory (only when saving)
    if save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load real microscope image and mask
    microscope_image = tiff.imread(microscope_image_path)
    segmentation_mask = tiff.imread(segmentation_mask_path)

    # Use image dimensions as domain size (working in pixel units)
    domain_size = (float(segmentation_mask.shape[1]), float(segmentation_mask.shape[0]))

    # Extract cell positions from the segmentation mask
    positions, lengths, radii, colors = crm.extract_positions(
        segmentation_mask, n_vertices=n_vertices, domain_size=domain_size
    )

    n_cells = len(positions)

    if n_cells == 0:
        print("Warning: No cells found in mask. Returning blank image.")
        h, w = segmentation_mask.shape[:2]
        blank_img = np.ones((h, w, 3), dtype=np.uint8) * 128
        blank_mask = np.zeros((h, w, 3), dtype=np.uint8)
        if save:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            image_name = Path(microscope_image_path).stem
            mask_name = Path(segmentation_mask_path).stem
            tiff.imwrite(
                output_dir / f"{output_prefix}{image_name}.tif",
                blank_img, compression="zlib",
            )
            tiff.imwrite(
                output_dir / f"{output_prefix}{mask_name}.tif",
                blank_mask, compression="zlib",
            )
        return blank_img, blank_mask

    # Create cells dictionary and assign colors
    from cr_mech_coli import RodAgent, CellIdentifier

    # Create default agent settings to get all parameters with default values
    agent_settings = crm.AgentSettings()
    rod_args = agent_settings.to_rod_agent_dict()

    # Create cells dictionary
    cells = {}
    colors_dict = {}
    for i, (pos, radius) in enumerate(zip(positions, radii)):
        cell_id = CellIdentifier.new_initial(i)

        # Convert 2D positions to 3D by adding z=0
        # pos has shape (n_vertices, 2), we need (n_vertices, 3)
        pos_3d = np.zeros((pos.shape[0], 3), dtype=np.float32)
        pos_3d[:, :2] = pos  # Copy x, y coordinates
        pos_3d[:, 2] = 0.0  # Set z coordinate to 0

        # Ensure position array is contiguous
        pos_array = np.ascontiguousarray(pos_3d, dtype=np.float32)

        # Create velocity array (all zeros for static snapshot)
        vel = np.zeros_like(pos_array, dtype=np.float32)

        # Create RodAgent with default settings
        rod_agent = RodAgent(pos=pos_array, vel=vel, **rod_args)

        # Override radius with extracted value
        rod_agent.radius = float(radius) * 0.95

        cells[cell_id] = (rod_agent, None)  # (agent, parent_id)

        # Assign a unique color to this cell for the mask
        color = crm.counter_to_color(i + 1)  # Start from 1 to avoid black (0,0,0)
        colors_dict[cell_id] = color

    # Create render settings
    # Since we're working in pixel units, set pixel_per_micron = 1.0
    render_settings = crm.RenderSettings()
    render_settings.pixel_per_micron = 1.0

    # Adjust settings for better image quality
    render_settings.kernel_size = 2
    render_settings.noise = 0
    render_settings.bg_brightness = 0
    render_settings.cell_brightness = (
        0  # Start with black bacteria, brightness added from original
    )

    # Additional settings for a more microscope-like appearance
    render_settings.ambient = 0.3
    render_settings.diffuse = 0.7
    render_settings.specular = 0.0
    render_settings.specular_power = 0.0
    render_settings.metallic = 0.0
    render_settings.pbr = False
    render_settings.ssao_radius = 0.0

    # Render synthetic microscope image
    synthetic_image = crm.render_image(
        cells,
        domain_size=domain_size,
        render_settings=render_settings,
    )

    # Render synthetic mask
    synthetic_mask = crm.render_mask(
        cells,
        colors=colors_dict,
        domain_size=domain_size,
        render_settings=render_settings,
    )

    # Apply synthetic effects using shared function
    if params is not None:
        # Dict-based path: only pass non-imaging kwargs + params dict
        synthetic_image = apply_synthetic_effects(
            raw_image=synthetic_image,
            mask=synthetic_mask,
            seed=seed,
            brightness_mode=brightness_mode,
            cell_ages=cell_ages,
            cell_colors_map=cell_colors_map,
            brightness_range=brightness_range,
            max_age=max_age,
            original_image=microscope_image,
            original_mask=segmentation_mask,
            original_colors=colors,
            params=params,
        )
    else:
        # Legacy kwargs path (used by clone subcommand)
        synthetic_image = apply_synthetic_effects(
            raw_image=synthetic_image,
            mask=synthetic_mask,
            bg_base_brightness=bg_base_brightness,
            bg_gradient_strength=bg_gradient_strength,
            bac_halo_intensity=bac_halo_intensity,
            bg_noise_scale=bg_noise_scale,
            psf_sigma=psf_sigma,
            peak_signal=peak_signal,
            gaussian_sigma=gaussian_sigma,
            seed=seed,
            brightness_mode=brightness_mode,
            cell_ages=cell_ages,
            cell_colors_map=cell_colors_map,
            brightness_range=brightness_range,
            max_age=max_age,
            num_dark_spots_range=num_dark_spots_range,
            original_image=microscope_image,
            original_mask=segmentation_mask,
            original_colors=colors,
            brightness_noise_strength=brightness_noise_strength,
            apply_psf=apply_psf,
            apply_poisson=apply_poisson,
            apply_gaussian=apply_gaussian,
            halo_inner_width=halo_inner_width,
            halo_outer_width=halo_outer_width,
            halo_blur_sigma=halo_blur_sigma,
            halo_fade_type=halo_fade_type,
            dark_spot_size_range=dark_spot_size_range,
            num_light_spots_range=num_light_spots_range,
            texture_strength=texture_strength,
            texture_scale=texture_scale,
            bg_blur_sigma=bg_blur_sigma,
        )

    # Save results with output_prefix + original filename
    if save:
        image_name = Path(microscope_image_path).stem
        mask_name = Path(segmentation_mask_path).stem

        output_image_path = output_dir / f"{output_prefix}{image_name}.tif"
        output_mask_path = output_dir / f"{output_prefix}{mask_name}.tif"

        tiff.imwrite(output_image_path, synthetic_image, compression="zlib")
        tiff.imwrite(output_mask_path, synthetic_mask, compression="zlib")

    return synthetic_image, synthetic_mask
