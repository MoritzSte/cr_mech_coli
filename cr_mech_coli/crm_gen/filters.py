"""
Filters for adding realistic microscope effects to synthetic images.

This module provides methods to simulate:

- Phase contrast halo effects (bright/dark halos around bacteria edges)
- Point Spread Function (PSF) blur from optical systems
- Poisson noise (shot noise from photon counting)
- Gaussian noise (camera readout noise)

"""

import numpy as np
from scipy.ndimage import gaussian_filter, convolve, distance_transform_edt, zoom
from typing import Tuple, Optional


def create_gaussian_psf(size: int = 7, sigma: float = 1.0) -> np.ndarray:
    """
    Create a 2D Gaussian Point Spread Function kernel.

    The PSF models the optical blur caused by the microscope's optical system,
    including diffraction and aberrations.

    Args:
        size (int): Size of the PSF kernel (size x size). Should be odd.
        sigma (float): Standard deviation of the Gaussian PSF in pixels.

    Returns:
        np.ndarray: 2D normalized PSF kernel that sums to 1.
    """
    # Ensure odd size
    if size % 2 == 0:
        size += 1

    # Guard against zero or very small sigma (return identity-like kernel)
    if sigma <= 1e-10:
        psf = np.zeros((size, size), dtype=np.float64)
        psf[size // 2, size // 2] = 1.0
        return psf

    # Create coordinate grid centered at 0
    center = size // 2
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)

    # 2D Gaussian function
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalize so sum equals 1
    psf = psf / psf.sum()

    return psf


def create_airy_psf(size: int = 15, radius: float = 3.0) -> np.ndarray:
    """
    Create a 2D Airy disk Point Spread Function (more physically accurate).

    The Airy pattern is the diffraction pattern from a circular aperture,
    which is more accurate for microscope optics than a Gaussian.

    Args:
        size (int): Size of the PSF kernel (size x size). Should be odd.
        radius (float): Radius parameter for the Airy disk (first zero crossing).

    Returns:
        np.ndarray: 2D normalized PSF kernel that sums to 1.
    """
    # Ensure odd size
    if size % 2 == 0:
        size += 1

    # Create coordinate grid centered at 0
    center = size // 2
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
    r = np.sqrt(x**2 + y**2)

    # Avoid division by zero at center
    r[r == 0] = 1e-10

    # Airy pattern: (2 * J1(x) / x)^2 where J1 is Bessel function of first kind
    from scipy.special import j1

    # Normalize radius
    x_normalized = 2 * np.pi * r / radius

    # Airy disk pattern
    psf = (2 * j1(x_normalized) / x_normalized) ** 2

    # Normalize so sum equals 1
    psf = psf / psf.sum()

    return psf


def apply_psf_blur(
    image: np.ndarray,
    psf_type: str = "gaussian",
    psf_sigma: float = 1.0,
    psf_size: int = 7,
    airy_radius: float = 3.0,
) -> np.ndarray:
    """
    Apply Point Spread Function blur to simulate optical blur from microscope.

    This is crucial for smoothing sharp edges around bacteria that result from
    compositing rendered cells onto the background. The PSF models diffraction
    and optical aberrations in the microscope system.

    Args:
        image (np.ndarray): Input image (grayscale or RGB). Can be float [0,1] or
            uint8 [0,255].
        psf_type (str): Type of PSF: 'gaussian' (fast, good approximation) or 'airy'
            (more accurate).
        psf_sigma (float): Sigma for Gaussian PSF (in pixels). Typical range: 0.5-2.0.
            Higher values = more blur.
        psf_size (int): Size of PSF kernel. Should be large enough to capture PSF extent.
            Typical range: 5-15.
        airy_radius (float): Radius parameter for Airy disk PSF (only used if
            psf_type='airy').

    Returns:
        np.ndarray: Blurred image with same dtype as input.

    Notes:
        PSF blur should be applied BEFORE adding noise (it's an optical effect).
        For phase contrast microscopy, sigma=0.8-1.5 is typical at 60-100x magnification.
        Bacteria edges will be softened, making compositing look more natural.
    """
    # Remember input dtype
    input_dtype = image.dtype
    is_uint8 = input_dtype == np.uint8

    # Convert to float for processing
    if is_uint8:
        img_float = image.astype(np.float64) / 255.0
    else:
        img_float = image.astype(np.float64)

    # Create PSF kernel
    if psf_type.lower() == "gaussian":
        psf_kernel = create_gaussian_psf(size=psf_size, sigma=psf_sigma)
    elif psf_type.lower() == "airy":
        psf_kernel = create_airy_psf(size=psf_size, radius=airy_radius)
    else:
        raise ValueError(f"Unknown PSF type: {psf_type}. Use 'gaussian' or 'airy'")

    # Apply convolution
    if len(img_float.shape) == 2:
        # Grayscale image
        result = convolve(img_float, psf_kernel, mode="reflect")
    else:
        # RGB/multi-channel image - convolve each channel
        result = np.zeros_like(img_float)
        for i in range(img_float.shape[2]):
            result[:, :, i] = convolve(img_float[:, :, i], psf_kernel, mode="reflect")

    # Clip to valid range
    result = np.clip(result, 0.0, 1.0)

    # Convert back to original dtype
    if is_uint8:
        result = (result * 255).astype(np.uint8)

    return result


def add_poisson_noise(
    image: np.ndarray, peak_signal: float = 1000.0, seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Poisson (shot) noise to simulate photon counting noise.

    Poisson noise is signal-dependent: brighter regions have more noise.
    This is the dominant noise source in well-lit microscopy images.
    The noise model: noisy_value = Poisson(clean_value * peak_signal) / peak_signal.

    Args:
        image (np.ndarray): Input image (grayscale or RGB). Can be float [0,1] or
            uint8 [0,255].
        peak_signal (float): Peak photon count at maximum intensity. Controls noise
            strength. Higher values = less noise (more photons collected). Typical
            range: 100-500 (high noise), 500-2000 (moderate), 2000-10000 (low noise).
        seed (int): Random seed for reproducibility. If None, uses random state.

    Returns:
        np.ndarray: Noisy image with same dtype as input.

    Notes:
        Apply AFTER PSF blur (noise happens during photon detection) and BEFORE
        Gaussian noise. Variance = mean for Poisson, so brighter = noisier in
        absolute terms.
    """
    rng = np.random.default_rng(seed)

    # Remember input dtype
    input_dtype = image.dtype
    is_uint8 = input_dtype == np.uint8

    # Convert to float [0,1]
    if is_uint8:
        img_float = image.astype(np.float64) / 255.0
    else:
        img_float = image.astype(np.float64)
        img_float = np.clip(img_float, 0.0, 1.0)

    # Scale up to photon counts
    photon_counts = img_float * peak_signal

    # Guard against zero or very small peak_signal (return original image)
    if peak_signal <= 1e-10:
        return image.copy()

    # Apply Poisson noise
    noisy_photons = rng.poisson(photon_counts)

    # Scale back to [0,1]
    result = noisy_photons / peak_signal

    # Handle any NaN/Inf values that might have occurred
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)

    # Clip to valid range
    result = np.clip(result, 0.0, 1.0)

    # Convert back to original dtype
    if is_uint8:
        result = (result * 255).astype(np.uint8)

    return result


def add_gaussian_noise(
    image: np.ndarray, sigma: float = 0.01, seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Gaussian (readout) noise to simulate camera electronics noise.

    Gaussian noise is signal-independent: same noise level across all intensities.
    This represents noise from camera electronics (readout, thermal, etc.).

    Args:
        image (np.ndarray): Input image (grayscale or RGB). Can be float [0,1] or
            uint8 [0,255].
        sigma (float): Standard deviation of Gaussian noise (in range [0,1] for float
            images). Typical range: 0.001-0.005 (low), 0.01-0.02 (moderate),
            0.03-0.05 (high).
        seed (int): Random seed for reproducibility. If None, uses random state.

    Returns:
        np.ndarray: Noisy image with same dtype as input.

    Notes:
        Apply AFTER Poisson noise (it's the last noise source in the imaging chain).
        Also called "additive white Gaussian noise" (AWGN).
    """
    rng = np.random.default_rng(seed)

    # Remember input dtype
    input_dtype = image.dtype
    is_uint8 = input_dtype == np.uint8

    # Convert to float [0,1]
    if is_uint8:
        img_float = image.astype(np.float64) / 255.0
    else:
        img_float = image.astype(np.float64)

    # Generate Gaussian noise
    noise = rng.normal(0, sigma, img_float.shape)

    # Add noise
    result = img_float + noise

    # Clip to valid range
    result = np.clip(result, 0.0, 1.0)

    # Convert back to original dtype
    if is_uint8:
        result = (result * 255).astype(np.uint8)

    return result


def apply_gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur to an image.

    Simple wrapper around scipy's gaussian_filter that handles both
    grayscale and RGB images, and preserves the input dtype.

    Args:
        image (np.ndarray): Input image (grayscale or RGB). Can be float [0,1] or
            uint8 [0,255].
        sigma (float): Standard deviation of the Gaussian kernel (blur strength).
            Higher values = more blur. Typical range: 0.5-1.0 (slight), 1.0-2.0
            (moderate), 2.0-5.0 (strong).

    Returns:
        np.ndarray: Blurred image with same dtype as input.

    Notes:
        This is a simple spatial domain blur. For PSF blur (optical effects), use
        apply_psf_blur() instead.
    """
    # Remember input dtype
    input_dtype = image.dtype
    is_uint8 = input_dtype == np.uint8

    # Convert to float for processing
    if is_uint8:
        img_float = image.astype(np.float64) / 255.0
    else:
        img_float = image.astype(np.float64)

    # Apply Gaussian blur
    if len(img_float.shape) == 2:
        # Grayscale image
        result = gaussian_filter(img_float, sigma=sigma)
    else:
        # RGB/multi-channel image - blur each channel
        result = np.zeros_like(img_float)
        for i in range(img_float.shape[2]):
            result[:, :, i] = gaussian_filter(img_float[:, :, i], sigma=sigma)

    # Clip to valid range
    result = np.clip(result, 0.0, 1.0)

    # Convert back to original dtype
    if is_uint8:
        result = (result * 255).astype(np.uint8)

    return result


# ============================================================================
# Phase Contrast Halo Effects
# ============================================================================


def create_halo_gradient(
    mask: np.ndarray,
    inner_width: float = 2.0,
    outer_width: float = 8.0,
    fade_type: str = "exponential",
) -> np.ndarray:
    """
    Create a smooth gradient field for halo intensity that fades from edge outward.

    Args:
        mask (np.ndarray): Binary mask where True/1 = bacteria (foreground).
        inner_width (float): Width of inner bright halo (in pixels).
        outer_width (float): Total width of halo region (in pixels).
        fade_type (str): Type of fade: 'linear', 'exponential', or 'gaussian'.

    Returns:
        np.ndarray: Gradient field with values [0,1] where 1 is at the edge,
            fading to 0.
    """
    from scipy.ndimage import distance_transform_edt

    # Convert to boolean if needed
    if mask.dtype != bool:
        mask_bool = mask > 0
    else:
        mask_bool = mask.copy()

    # Distance from foreground edge (outside)
    distance_outside = distance_transform_edt(~mask_bool)

    # Distance from background edge (inside)
    distance_inside = distance_transform_edt(mask_bool)

    # Initialize gradient
    gradient = np.zeros_like(mask, dtype=np.float64)

    # Create gradient based on distance
    # Inside bacteria: gradient increases towards edge
    inside_region = mask_bool
    gradient[inside_region] = np.clip(
        1.0 - distance_inside[inside_region] / inner_width, 0, 1
    )

    # Outside bacteria: gradient decreases from edge
    outside_region = ~mask_bool
    dist_norm = distance_outside[outside_region] / outer_width

    if fade_type == "linear":
        gradient[outside_region] = np.clip(1.0 - dist_norm, 0, 1)
    elif fade_type == "exponential":
        # Exponential decay for more realistic falloff
        gradient[outside_region] = np.exp(-6 * dist_norm) * (dist_norm <= 1.0)
    elif fade_type == "gaussian":
        # Gaussian falloff
        gradient[outside_region] = np.exp(-0.5 * (dist_norm * 3) ** 2) * (
            dist_norm <= 1.0
        )
    else:
        raise ValueError(f"Unknown fade_type: {fade_type}")

    return gradient


def apply_halo_effect(
    image: np.ndarray,
    mask: np.ndarray,
    halo_intensity: float = 0.15,
    inner_width: float = 2.0,
    outer_width: float = 8.0,
    fade_type: str = "exponential",
    blur_sigma: float = 1.5,
) -> np.ndarray:
    """
    Apply bright phase contrast halo effect around bacteria edges.

    Args:
        image (np.ndarray): Input image (grayscale or RGB). Can be float [0,1] or
            uint8 [0,255].
        mask (np.ndarray): Binary mask where True/1 = bacteria (foreground).
        halo_intensity (float): Strength of halo effect (0.0 to 1.0). Higher values
            create more pronounced halos. Typical range: 0.1-0.3.
        inner_width (float): Width of inner halo in pixels (typically bright).
        outer_width (float): Total width of halo region in pixels.
        fade_type (str): How halo fades: 'linear', 'exponential', or 'gaussian'.
        blur_sigma (float): Gaussian blur sigma to smooth the halo transition.

    Returns:
        np.ndarray: Image with halo effect applied (same dtype as input).

    Notes:
        Apply BEFORE final PSF blur and noise for best results.
    """
    # Remember input dtype
    input_dtype = image.dtype
    is_uint8 = input_dtype == np.uint8

    # Convert to float [0,1]
    if is_uint8:
        img_float = image.astype(np.float64) / 255.0
    else:
        img_float = image.astype(np.float64)

    # Convert to boolean if needed
    if mask.dtype != bool:
        mask_bool = mask > 0
    else:
        mask_bool = mask.copy()

    # Create halo gradient
    gradient = create_halo_gradient(
        mask_bool, inner_width=inner_width, outer_width=outer_width, fade_type=fade_type
    )

    # Apply Gaussian blur to smooth the gradient
    gradient_smooth = gaussian_filter(gradient, sigma=blur_sigma)

    # Bright halo (positive phase shift)
    halo_field = gradient_smooth * halo_intensity

    # Apply halo to image
    if len(img_float.shape) == 2:
        # Grayscale
        result = img_float + halo_field
    else:
        # RGB - apply to all channels
        result = img_float + halo_field[..., np.newaxis]

    # Clip to valid range
    result = np.clip(result, 0.0, 1.0)

    # Convert back to original dtype
    if is_uint8:
        result = (result * 255).astype(np.uint8)

    return result


def apply_phase_contrast_pipeline(
    image: np.ndarray,
    mask: np.ndarray,
    # Halo parameters
    apply_halo: bool = True,
    halo_intensity: float = 0.15,
    halo_inner_width: float = 2.0,
    halo_outer_width: float = 8.0,
    halo_fade_type: str = "exponential",
    halo_blur_sigma: float = 1.5,
    # PSF parameters
    apply_psf: bool = True,
    psf_type: str = "gaussian",
    psf_sigma: float = 1.0,
    psf_size: int = 7,
    # Noise parameters
    apply_poisson: bool = True,
    peak_signal: float = 1000.0,
    apply_gaussian: bool = True,
    gaussian_sigma: float = 0.01,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Complete phase contrast microscopy pipeline with halo effects.

    This pipeline applies effects in the correct order for realistic phase contrast:

    1. Halo effects (optical phase shift at edges)
    2. PSF blur (optical diffraction)
    3. Poisson noise (photon shot noise)
    4. Gaussian noise (camera readout noise)

    This should be applied AFTER combining foreground and background.

    Args:
        image (np.ndarray): Input synthetic image (bacteria on background).
        mask (np.ndarray): Binary mask where True/1 = bacteria (foreground).
        apply_halo (bool): Whether to apply phase contrast halo effect.
        halo_intensity (float): Strength of halo effect (0.1-0.3 typical).
        halo_inner_width (float): Width of inner bright halo (pixels).
        halo_outer_width (float): Total halo width (pixels).
        halo_fade_type (str): Fade type: 'linear', 'exponential', or 'gaussian'.
        halo_blur_sigma (float): Blur sigma for smoothing halo.
        apply_psf (bool): Whether to apply PSF blur.
        psf_type (str): PSF type: 'gaussian' or 'airy'.
        psf_sigma (float): PSF blur strength.
        psf_size (int): PSF kernel size.
        apply_poisson (bool): Whether to add Poisson noise.
        peak_signal (float): Peak photon count for Poisson noise.
        apply_gaussian (bool): Whether to add Gaussian noise.
        gaussian_sigma (float): Gaussian noise sigma.
        seed (int): Random seed for reproducibility. If None, uses random state.

    Returns:
        np.ndarray: Processed image with all effects applied (same dtype as input).
    """
    result = image.copy()

    # Step 1: Apply phase contrast halo effect
    if apply_halo:
        result = apply_halo_effect(
            result,
            mask=mask,
            halo_intensity=halo_intensity,
            inner_width=halo_inner_width,
            outer_width=halo_outer_width,
            fade_type=halo_fade_type,
            blur_sigma=halo_blur_sigma,
        )

    # Step 2: Apply PSF blur (optical diffraction)
    if apply_psf:
        result = apply_psf_blur(
            result, psf_type=psf_type, psf_sigma=psf_sigma, psf_size=psf_size
        )

    # Step 3: Apply Poisson noise (photon shot noise)
    if apply_poisson:
        result = add_poisson_noise(result, peak_signal=peak_signal, seed=seed)

    # Step 4: Apply Gaussian noise (camera readout noise)
    if apply_gaussian:
        gaussian_seed = (seed + 1) if seed is not None else None
        result = add_gaussian_noise(result, sigma=gaussian_sigma, seed=gaussian_seed)

    return result


def apply_microscope_effects(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    # PSF parameters
    apply_psf: bool = True,
    psf_type: str = "gaussian",
    psf_sigma: float = 1.0,
    psf_size: int = 7,
    airy_radius: float = 3.0,
    blur_bacteria_more: bool = False,
    bacteria_blur_factor: float = 1.5,
    # Noise parameters
    apply_poisson: bool = True,
    peak_signal: float = 1000.0,
    apply_gaussian: bool = True,
    gaussian_sigma: float = 0.01,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply complete microscope imaging effects pipeline to synthetic image.

    This is the main method that applies all realistic microscope effects in the
    correct order: PSF blur -> Poisson noise -> Gaussian noise.

    Optionally applies stronger blur to bacteria (foreground) to better integrate
    them with the slightly blurred background.

    Args:
        image (np.ndarray): Input synthetic image.
        mask (np.ndarray): Binary mask where True/1 = bacteria (foreground),
            False/0 = background. Used for differential blurring if
            blur_bacteria_more=True. If None, uniform blur is applied.
        apply_psf (bool): Whether to apply PSF blur.
        psf_type (str): Type of PSF: 'gaussian' or 'airy'.
        psf_sigma (float): Sigma for Gaussian PSF blur (in pixels).
        psf_size (int): Size of PSF kernel.
        airy_radius (float): Radius for Airy disk PSF.
        blur_bacteria_more (bool): Apply stronger blur to bacteria to smooth
            sharp edges.
        bacteria_blur_factor (float): Multiply PSF sigma by this factor for bacteria
            regions. Only used if blur_bacteria_more=True and mask is provided.
        apply_poisson (bool): Whether to add Poisson noise.
        peak_signal (float): Peak photon count for Poisson noise.
        apply_gaussian (bool): Whether to add Gaussian noise.
        gaussian_sigma (float): Sigma for Gaussian noise.
        seed (int): Random seed for reproducibility. If None, uses random state.

    Returns:
        np.ndarray: Processed image with all effects applied (same dtype as input).
    """
    result = image.copy()

    # Step 1: Apply PSF blur (optical effect)
    if apply_psf:
        if blur_bacteria_more and mask is not None:
            # Apply differential blurring: stronger blur on bacteria
            # This helps smooth the sharp edges from compositing

            # Convert mask to boolean if needed
            if mask.dtype != bool:
                mask_bool = mask > 0
            else:
                mask_bool = mask

            # Blur background with standard PSF
            bg_blurred = apply_psf_blur(
                result,
                psf_type=psf_type,
                psf_sigma=psf_sigma,
                psf_size=psf_size,
                airy_radius=airy_radius,
            )

            # Blur bacteria with stronger PSF
            bacteria_blurred = apply_psf_blur(
                result,
                psf_type=psf_type,
                psf_sigma=psf_sigma * bacteria_blur_factor,
                psf_size=max(psf_size, int(psf_size * bacteria_blur_factor)),
                airy_radius=airy_radius * bacteria_blur_factor,
            )

            # Combine: use bacteria blur where mask is True, background blur elsewhere
            if len(result.shape) == 2:
                # Grayscale
                result = np.where(mask_bool, bacteria_blurred, bg_blurred)
            else:
                # RGB
                result = np.where(
                    mask_bool[..., np.newaxis], bacteria_blurred, bg_blurred
                )
        else:
            # Apply uniform blur across entire image
            result = apply_psf_blur(
                result,
                psf_type=psf_type,
                psf_sigma=psf_sigma,
                psf_size=psf_size,
                airy_radius=airy_radius,
            )

    # Step 2: Apply Poisson noise (shot noise from photon detection)
    if apply_poisson:
        result = add_poisson_noise(result, peak_signal=peak_signal, seed=seed)

    # Step 3: Apply Gaussian noise (camera readout noise)
    if apply_gaussian:
        # Use different seed if seed was provided
        gaussian_seed = (seed + 1) if seed is not None else None
        result = add_gaussian_noise(result, sigma=gaussian_sigma, seed=gaussian_seed)

    return result


# ============================================================================
# Brightfield-style Filters
# ============================================================================


def apply_beer_lambert_absorption(
    image: np.ndarray,
    mask: np.ndarray,
    absorption_coeff: float = 0.5,
    cell_optical_thickness: float = 3.0,
) -> np.ndarray:
    """
    Apply Beer-Lambert absorption inside cell regions.

    Models the core brightfield contrast mechanism: cells appear darker than
    the background because they absorb part of the transmitted light. The
    optical thickness is derived from the distance-transform of the mask,
    so cell centers absorb more than the edges.

    Args:
        image (np.ndarray): Input image (grayscale or RGB). Can be float [0,1]
            or uint8 [0,255].
        mask (np.ndarray): Binary mask where True/1 = cell (foreground).
        absorption_coeff (float): Beer-Lambert absorption coefficient. 0.0 = no
            absorption (no-op). Typical range: 0.2-1.5.
        cell_optical_thickness (float): Maximum optical thickness at the cell
            center, in the normalized distance-transform units (pixels).

    Returns:
        np.ndarray: Image with absorption applied inside mask region.
    """
    if absorption_coeff <= 0:
        return image.copy()

    input_dtype = image.dtype
    is_uint8 = input_dtype == np.uint8

    if is_uint8:
        img_float = image.astype(np.float64) / 255.0
    else:
        img_float = image.astype(np.float64)

    mask_bool = mask > 0 if mask.dtype != bool else mask

    # Distance from edge inside the cell — proxy for optical thickness
    distance_inside = distance_transform_edt(mask_bool)
    max_dist = distance_inside.max()
    if max_dist <= 0:
        # No foreground pixels — nothing to absorb
        return image.copy()

    thickness_map = (distance_inside / max_dist) * cell_optical_thickness
    transmission = np.exp(-absorption_coeff * thickness_map)

    # Only apply where mask is True; background unchanged
    if img_float.ndim == 2:
        result = np.where(mask_bool, img_float * transmission, img_float)
    else:
        trans_3d = transmission[..., np.newaxis]
        mask_3d = mask_bool[..., np.newaxis]
        result = np.where(mask_3d, img_float * trans_3d, img_float)

    result = np.clip(result, 0.0, 1.0)

    if is_uint8:
        result = (result * 255).astype(np.uint8)

    return result


def apply_defocus_blur(
    image: np.ndarray,
    defocus_strength: float = 1.0,
    defocus_scale: int = 10,
    base_psf_sigma: float = 0.5,
    num_bins: int = 8,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply spatially varying Gaussian blur to simulate depth-of-field defocus.

    Generates a smooth z-offset field (Perlin-like multi-octave noise) and
    blurs the image with a sigma that depends on the local z-offset. For
    efficiency, the sigma range is discretized into a small number of bins,
    each bin is blurred separately, and the pre-blurred images are blended
    per-pixel based on bin membership.

    Args:
        image (np.ndarray): Input image (grayscale or RGB).
        defocus_strength (float): Maximum additional sigma added on top of
            *base_psf_sigma*. 0.0 = no-op. Typical range: 0.5-2.5.
        defocus_scale (int): Spatial scale of the z-variation field. Larger =
            smoother / lower-frequency variation.
        base_psf_sigma (float): In-focus PSF sigma — minimum blur applied to
            every pixel.
        num_bins (int): Number of discrete sigma levels used for the
            spatially-varying blur approximation.
        seed (int): Random seed for the z-field.

    Returns:
        np.ndarray: Defocus-blurred image.
    """
    if defocus_strength <= 0:
        return image.copy()

    input_dtype = image.dtype
    is_uint8 = input_dtype == np.uint8

    if is_uint8:
        img_float = image.astype(np.float64) / 255.0
    else:
        img_float = image.astype(np.float64)

    h, w = img_float.shape[:2]
    rng = np.random.default_rng(seed)

    # Build a smooth [0, 1] z-field via low-resolution noise + upsample + smooth
    low_h = max(h // max(defocus_scale, 1), 4)
    low_w = max(w // max(defocus_scale, 1), 4)
    low_noise = rng.standard_normal((low_h, low_w))
    low_noise = gaussian_filter(low_noise, sigma=2.0)
    z_field = zoom(low_noise, (h / low_h, w / low_w), order=3)
    # Use absolute value so blur is symmetric around the focal plane
    z_field = np.abs(z_field)
    z_max = z_field.max()
    if z_max > 0:
        z_field = z_field / z_max

    # Sigma per pixel
    sigmas = base_psf_sigma + defocus_strength * z_field

    # Discretize sigmas into num_bins values between min and max
    s_min, s_max = sigmas.min(), sigmas.max()
    if s_max - s_min < 1e-6:
        # Uniform blur — just one gaussian_filter call
        if img_float.ndim == 2:
            result = gaussian_filter(img_float, sigma=float(s_min))
        else:
            result = np.stack(
                [gaussian_filter(img_float[..., c], sigma=float(s_min))
                 for c in range(img_float.shape[2])],
                axis=-1,
            )
    else:
        bin_sigmas = np.linspace(s_min, s_max, num_bins)

        # Pre-blur image at each sigma level
        blurred_stack = []
        for bs in bin_sigmas:
            if img_float.ndim == 2:
                blurred_stack.append(gaussian_filter(img_float, sigma=float(bs)))
            else:
                blurred_stack.append(
                    np.stack(
                        [gaussian_filter(img_float[..., c], sigma=float(bs))
                         for c in range(img_float.shape[2])],
                        axis=-1,
                    )
                )

        # For each pixel, linearly blend between the two nearest bins
        # bin index (continuous) in [0, num_bins-1]
        bin_idx = (sigmas - s_min) / (s_max - s_min) * (num_bins - 1)
        lower = np.floor(bin_idx).astype(int)
        upper = np.clip(lower + 1, 0, num_bins - 1)
        frac = bin_idx - lower

        if img_float.ndim == 2:
            result = np.zeros_like(img_float)
            for b in range(num_bins):
                w_lower = (lower == b) * (1.0 - frac)
                w_upper = (upper == b) * frac
                # Also account for the case where lower == upper (at s_max)
                if b == num_bins - 1:
                    w_upper = w_upper + (lower == b) * (1.0 - frac) * (upper == lower)
                result += (w_lower + w_upper) * blurred_stack[b]
        else:
            result = np.zeros_like(img_float)
            for b in range(num_bins):
                w_lower = (lower == b) * (1.0 - frac)
                w_upper = (upper == b) * frac
                if b == num_bins - 1:
                    w_upper = w_upper + (lower == b) * (1.0 - frac) * (upper == lower)
                weight = (w_lower + w_upper)[..., np.newaxis]
                result += weight * blurred_stack[b]

    result = np.clip(result, 0.0, 1.0)

    if is_uint8:
        result = (result * 255).astype(np.uint8)

    return result


def apply_vignetting(
    image: np.ndarray,
    vignette_strength: float = 0.2,
) -> np.ndarray:
    """
    Apply radial intensity falloff (vignetting) from the image center.

    Models uneven illumination often seen in brightfield microscopy due to
    imperfect Köhler alignment or condenser misalignment.

    Formula::

        V(x, y) = 1 - strength * ((x - cx)^2 + (y - cy)^2) / r_max^2

    Args:
        image (np.ndarray): Input image (grayscale or RGB).
        vignette_strength (float): Falloff strength. 0.0 = no vignetting.
            At strength=0.5, corners are at 50% of center intensity.

    Returns:
        np.ndarray: Vignetted image.
    """
    if vignette_strength <= 0:
        return image.copy()

    input_dtype = image.dtype
    is_uint8 = input_dtype == np.uint8

    if is_uint8:
        img_float = image.astype(np.float64) / 255.0
    else:
        img_float = image.astype(np.float64)

    h, w = img_float.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y_grid, x_grid = np.ogrid[:h, :w]
    r2 = (y_grid - cy) ** 2 + (x_grid - cx) ** 2
    r_max2 = cx ** 2 + cy ** 2
    vignette = 1.0 - vignette_strength * (r2 / r_max2)
    vignette = np.clip(vignette, 0.0, 1.0)

    if img_float.ndim == 2:
        result = img_float * vignette
    else:
        result = img_float * vignette[..., np.newaxis]

    result = np.clip(result, 0.0, 1.0)

    if is_uint8:
        result = (result * 255).astype(np.uint8)

    return result


def apply_edge_diffraction_fringe(
    image: np.ndarray,
    mask: np.ndarray,
    edge_fringe_intensity: float = 0.03,
    edge_fringe_width: float = 1.5,
) -> np.ndarray:
    """
    Apply a thin bright/dark diffraction fringe at cell boundaries.

    Models the subtle diffraction ripple sometimes visible at cell edges in
    brightfield microscopy. Much thinner and weaker than the phase contrast
    halo — the pattern is a dampened cosine in the distance-from-edge.

    Args:
        image (np.ndarray): Input image (grayscale or RGB).
        mask (np.ndarray): Binary mask where True/1 = cell.
        edge_fringe_intensity (float): Peak amplitude of the fringe.
            0.0 = no-op. Typical range: 0.01-0.05.
        edge_fringe_width (float): Width parameter of the fringe (pixels).

    Returns:
        np.ndarray: Image with fringe added.
    """
    if edge_fringe_intensity <= 0:
        return image.copy()

    input_dtype = image.dtype
    is_uint8 = input_dtype == np.uint8

    if is_uint8:
        img_float = image.astype(np.float64) / 255.0
    else:
        img_float = image.astype(np.float64)

    mask_bool = mask > 0 if mask.dtype != bool else mask

    dist_outside = distance_transform_edt(~mask_bool)
    dist_inside = distance_transform_edt(mask_bool)
    # Signed distance: positive outside, negative inside (but dampening uses |d|)
    dist = np.where(mask_bool, -dist_inside, dist_outside)

    w = max(edge_fringe_width, 1e-6)
    fringe = (
        edge_fringe_intensity
        * np.cos(2.0 * np.pi * dist / w)
        * np.exp(-np.abs(dist) / (w * 2.0))
    )
    # Smooth slightly for physical realism
    fringe = gaussian_filter(fringe, sigma=0.5)

    if img_float.ndim == 2:
        result = img_float + fringe
    else:
        result = img_float + fringe[..., np.newaxis]

    result = np.clip(result, 0.0, 1.0)

    if is_uint8:
        result = (result * 255).astype(np.uint8)

    return result
