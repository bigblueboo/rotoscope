"""
Polygon simplification using Douglas-Peucker algorithm.
"""
import numpy as np
from typing import List, Tuple, Optional

# Try to use OpenCV if available, otherwise use pure Python implementation
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# Style presets for epsilon ratio
STYLE_PRESETS = {
    'detailed': 0.005,      # Many vertices, smooth curves
    'moderate': 0.015,      # Balance
    'stylized': 0.03,       # Visible simplification
    'another_world': 0.05,  # Heavy simplification, angular
    'extreme': 0.08,        # Very few vertices
}


def douglas_peucker(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Pure Python implementation of Douglas-Peucker line simplification.

    Args:
        points: Array of points (N, 2)
        epsilon: Distance threshold for simplification

    Returns:
        Simplified array of points
    """
    if len(points) <= 2:
        return points

    # Find point with maximum distance from line between first and last
    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        return np.array([start])

    # Calculate perpendicular distances
    line_unit = line_vec / line_len
    point_vecs = points - start
    projections = np.dot(point_vecs, line_unit)
    closest_points = start + np.outer(projections, line_unit)
    distances = np.linalg.norm(points - closest_points, axis=1)

    max_idx = np.argmax(distances)
    max_dist = distances[max_idx]

    if max_dist > epsilon:
        # Recursively simplify
        left = douglas_peucker(points[:max_idx + 1], epsilon)
        right = douglas_peucker(points[max_idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])


def simplify_contour(contour: np.ndarray, epsilon_ratio: float = 0.03,
                     min_vertices: int = 3, max_vertices: int = 30,
                     _recursion_depth: int = 0) -> np.ndarray:
    """
    Simplify a contour using Douglas-Peucker algorithm.

    Args:
        contour: Array of contour points (N, 2)
        epsilon_ratio: Ratio of perimeter to use as epsilon (0.01-0.1)
        min_vertices: Minimum number of vertices to keep
        max_vertices: Maximum number of vertices to allow

    Returns:
        Simplified contour
    """
    if len(contour) < 3:
        return contour

    # Calculate perimeter for epsilon
    if HAS_CV2:
        # Ensure contour is in OpenCV format (N, 1, 2)
        if contour.ndim == 2:
            contour_cv = contour.reshape(-1, 1, 2).astype(np.float32)
        else:
            contour_cv = contour.astype(np.float32)
        perimeter = cv2.arcLength(contour_cv, closed=True)
        epsilon = epsilon_ratio * perimeter
        simplified = cv2.approxPolyDP(contour_cv, epsilon, closed=True)
        result = simplified.reshape(-1, 2)
    else:
        # Pure Python fallback
        perimeter = np.sum(np.linalg.norm(np.diff(contour, axis=0), axis=1))
        perimeter += np.linalg.norm(contour[-1] - contour[0])  # Close the loop
        epsilon = epsilon_ratio * perimeter
        result = douglas_peucker(contour, epsilon)

    # Force max_vertices limit first (this is the priority for Another World style)
    if len(result) > max_vertices:
        if _recursion_depth < 10 and epsilon_ratio < 0.3:
            # Try with more aggressive epsilon
            return simplify_contour(contour, epsilon_ratio * 2.0, min_vertices, max_vertices,
                                   _recursion_depth + 1)
        else:
            # Force reduce by uniform sampling to ensure max_vertices is respected
            indices = np.linspace(0, len(result) - 1, max_vertices, dtype=int)
            result = result[indices]

    # Enforce minimum vertices (but don't return original if it would exceed max)
    if len(result) < min_vertices:
        # If original also exceeds max_vertices, sample it
        if len(contour) > max_vertices:
            indices = np.linspace(0, len(contour) - 1, max_vertices, dtype=int)
            return contour[indices].astype(np.int16)
        return contour.astype(np.int16)

    return result.astype(np.int16)


def simplify_contour_adaptive(contour: np.ndarray,
                               target_vertices: int = 10,
                               tolerance: int = 2) -> np.ndarray:
    """
    Simplify contour to approximately target number of vertices.

    Uses binary search on epsilon to achieve target vertex count.

    Args:
        contour: Input contour
        target_vertices: Desired number of vertices
        tolerance: Acceptable deviation from target

    Returns:
        Simplified contour
    """
    if len(contour) <= target_vertices:
        return contour

    low, high = 0.001, 0.3
    best_result = contour

    for _ in range(20):  # Binary search iterations
        mid = (low + high) / 2
        result = simplify_contour(contour, mid, min_vertices=3, max_vertices=1000, _recursion_depth=0)
        vertex_count = len(result)

        if abs(vertex_count - target_vertices) <= tolerance:
            return result

        if vertex_count > target_vertices:
            low = mid
            best_result = result
        else:
            high = mid

    return best_result


def mask_to_polygons(mask: np.ndarray, epsilon_ratio: float = 0.03,
                     min_area: int = 100, max_vertices: int = 30) -> List[np.ndarray]:
    """
    Convert a binary mask to simplified polygons.

    Args:
        mask: Binary mask (H, W) with values 0 or non-zero
        epsilon_ratio: Simplification ratio (see STYLE_PRESETS)
        min_area: Minimum polygon area to keep (filters noise)

    Returns:
        List of simplified polygon contours
    """
    # Ensure mask is proper format
    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    if HAS_CV2:
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,  # Only outer contours
            cv2.CHAIN_APPROX_SIMPLE
        )
    else:
        # Very basic contour extraction fallback (not recommended)
        contours = _extract_contours_basic(mask_uint8)

    polygons = []
    for contour in contours:
        # Flatten OpenCV contour format
        if contour.ndim == 3:
            contour = contour.reshape(-1, 2)

        # Filter by area
        if HAS_CV2:
            area = cv2.contourArea(contour.reshape(-1, 1, 2))
        else:
            area = _polygon_area(contour)

        if area < min_area:
            continue

        # Simplify
        simplified = simplify_contour(contour, epsilon_ratio, max_vertices=max_vertices)
        if len(simplified) >= 3:
            polygons.append(simplified)

    return polygons


def _polygon_area(vertices: np.ndarray) -> float:
    """Calculate polygon area using shoelace formula."""
    n = len(vertices)
    if n < 3:
        return 0.0
    x = vertices[:, 0].astype(np.float64)
    y = vertices[:, 1].astype(np.float64)
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _extract_contours_basic(mask: np.ndarray) -> List[np.ndarray]:
    """
    Very basic contour extraction without OpenCV.
    Not efficient, use only as fallback.
    """
    # This is a placeholder - proper implementation would use
    # marching squares or similar algorithm
    from scipy import ndimage

    labeled, num_features = ndimage.label(mask > 0)
    contours = []

    for label_id in range(1, num_features + 1):
        region = labeled == label_id
        # Find boundary pixels (very naive)
        boundary = region & ~ndimage.binary_erosion(region)
        points = np.argwhere(boundary)[:, ::-1]  # Convert to (x, y)
        if len(points) >= 3:
            contours.append(points)

    return contours


def simplify_by_style(mask: np.ndarray, style: str = 'another_world',
                      min_area: int = 100) -> List[np.ndarray]:
    """
    Convenience function to simplify mask using a named style preset.

    Args:
        mask: Binary mask
        style: One of 'detailed', 'moderate', 'stylized', 'another_world', 'extreme'
        min_area: Minimum polygon area

    Returns:
        List of simplified polygons
    """
    epsilon_ratio = STYLE_PRESETS.get(style, STYLE_PRESETS['another_world'])
    return mask_to_polygons(mask, epsilon_ratio, min_area)
