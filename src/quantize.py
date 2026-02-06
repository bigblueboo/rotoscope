"""
Color quantization for Another World style palettes.
"""
import numpy as np
from typing import Tuple, List


def median_cut_quantize(image: np.ndarray, num_colors: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize an image to a limited color palette using median cut algorithm.

    Args:
        image: RGB image array (H, W, 3)
        num_colors: Number of colors in palette (default 16 for Another World style)

    Returns:
        indexed: Indexed image (H, W) with values 0 to num_colors-1
        palette: Color palette (num_colors, 3) RGB values
    """
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)

    # Remove duplicates for efficiency
    unique_pixels = np.unique(pixels, axis=0)

    # Recursive median cut
    def median_cut(pixels: np.ndarray, depth: int) -> List[np.ndarray]:
        if depth == 0 or len(pixels) == 0:
            return [pixels]

        # Find channel with greatest range
        ranges = pixels.max(axis=0) - pixels.min(axis=0)
        channel = np.argmax(ranges)

        # Sort by that channel and split at median
        sorted_idx = np.argsort(pixels[:, channel])
        sorted_pixels = pixels[sorted_idx]
        median = len(sorted_pixels) // 2

        return (median_cut(sorted_pixels[:median], depth - 1) +
                median_cut(sorted_pixels[median:], depth - 1))

    # Calculate depth needed for num_colors buckets
    depth = int(np.ceil(np.log2(num_colors)))
    buckets = median_cut(unique_pixels, depth)

    # Take mean of each bucket as palette color
    palette = []
    for bucket in buckets:
        if len(bucket) > 0:
            palette.append(bucket.mean(axis=0))
        if len(palette) >= num_colors:
            break

    # Pad palette if needed
    while len(palette) < num_colors:
        palette.append(np.array([0, 0, 0]))

    palette = np.array(palette[:num_colors], dtype=np.uint8)

    # Map each pixel to nearest palette color
    indexed = np.zeros((h, w), dtype=np.uint8)
    pixels = image.reshape(-1, 3)

    for i, pixel in enumerate(pixels):
        distances = np.sum((palette.astype(np.float32) - pixel) ** 2, axis=1)
        indexed.flat[i] = np.argmin(distances)

    return indexed, palette


def kmeans_quantize(image: np.ndarray, num_colors: int = 16,
                    max_iterations: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize using k-means clustering (often better quality than median cut).

    Args:
        image: RGB image array (H, W, 3)
        num_colors: Number of colors in palette
        max_iterations: Maximum k-means iterations

    Returns:
        indexed: Indexed image (H, W)
        palette: Color palette (num_colors, 3)
    """
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)

    # Initialize centroids randomly from pixels
    rng = np.random.default_rng(42)
    indices = rng.choice(len(pixels), size=num_colors, replace=False)
    centroids = pixels[indices].copy()

    for _ in range(max_iterations):
        # Assign each pixel to nearest centroid
        distances = np.zeros((len(pixels), num_colors))
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.sum((pixels - centroid) ** 2, axis=1)
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(num_colors):
            mask = labels == i
            if mask.any():
                new_centroids[i] = pixels[mask].mean(axis=0)
            else:
                new_centroids[i] = centroids[i]

        # Check for convergence
        if np.allclose(centroids, new_centroids, atol=1.0):
            break
        centroids = new_centroids

    palette = centroids.astype(np.uint8)
    indexed = labels.reshape(h, w).astype(np.uint8)

    return indexed, palette


def create_another_world_palette(base_colors: np.ndarray) -> np.ndarray:
    """
    Create a 16-color palette in Another World style:
    - 8 base colors
    - 8 brightened variants

    Args:
        base_colors: Array of 8 RGB colors (8, 3)

    Returns:
        Full 16-color palette (16, 3)
    """
    if len(base_colors) != 8:
        raise ValueError("Need exactly 8 base colors")

    palette = np.zeros((16, 3), dtype=np.uint8)

    # Base colors (0x0-0x7)
    palette[:8] = base_colors

    # Brightened variants (0x8-0xF)
    brightened = (base_colors.astype(np.float32) * 1.3 + 30).clip(0, 255)
    palette[8:] = brightened.astype(np.uint8)

    return palette


def posterize(image: np.ndarray, levels: int = 4) -> np.ndarray:
    """
    Reduce color depth by posterizing (useful preprocessing for quantization).

    Args:
        image: RGB image
        levels: Number of levels per channel

    Returns:
        Posterized image
    """
    factor = 256 // levels
    return (image // factor * factor).astype(np.uint8)


def apply_palette(indexed: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Convert indexed image back to RGB using palette.

    Args:
        indexed: Indexed image (H, W) with values 0 to len(palette)-1
        palette: Color palette (N, 3)

    Returns:
        RGB image (H, W, 3)
    """
    return palette[indexed]
