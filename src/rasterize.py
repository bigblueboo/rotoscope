"""
Polygon rasterizer - renders frames to images.
"""
import numpy as np
from typing import List, Optional

from .polygon import Polygon, Frame

# Try to use OpenCV for fast polygon filling
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def render_frame(frame: Frame, width: int = 320, height: int = 200,
                 background_color: tuple = (0, 0, 0)) -> np.ndarray:
    """
    Render a frame to an RGB image.

    Uses painter's algorithm - polygons are drawn in order,
    later polygons cover earlier ones.

    Args:
        frame: Frame containing polygons and palette
        width: Output image width
        height: Output image height
        background_color: RGB tuple for background

    Returns:
        RGB image as numpy array (H, W, 3)
    """
    # Create blank image
    image = np.full((height, width, 3), background_color, dtype=np.uint8)

    # Get palette (default to grayscale if not set)
    if frame.palette is not None:
        palette = frame.palette
    else:
        palette = np.array([[i * 16, i * 16, i * 16] for i in range(16)], dtype=np.uint8)

    # Draw polygons in order (painter's algorithm)
    for polygon in frame.polygons:
        color = tuple(int(c) for c in palette[polygon.color_index])
        _fill_polygon(image, polygon.vertices, color)

    return image


def _fill_polygon(image: np.ndarray, vertices: np.ndarray, color: tuple):
    """Fill a polygon on the image."""
    if len(vertices) < 3:
        return

    if HAS_CV2:
        pts = vertices.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(image, [pts], color)
    else:
        _fill_polygon_scanline(image, vertices, color)


def _fill_polygon_scanline(image: np.ndarray, vertices: np.ndarray, color: tuple):
    """
    Pure Python scanline polygon fill algorithm.
    Fallback when OpenCV is not available.
    """
    height, width = image.shape[:2]
    n = len(vertices)

    if n < 3:
        return

    # Find bounding box
    min_y = max(0, int(vertices[:, 1].min()))
    max_y = min(height - 1, int(vertices[:, 1].max()))

    # For each scanline
    for y in range(min_y, max_y + 1):
        # Find intersections with all edges
        intersections = []

        for i in range(n):
            j = (i + 1) % n
            y1, y2 = vertices[i, 1], vertices[j, 1]
            x1, x2 = vertices[i, 0], vertices[j, 0]

            # Check if edge crosses this scanline
            if (y1 <= y < y2) or (y2 <= y < y1):
                # Linear interpolation to find x intersection
                if y2 != y1:
                    x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                    intersections.append(x)

        # Sort intersections and fill between pairs
        intersections.sort()

        for i in range(0, len(intersections) - 1, 2):
            x_start = max(0, int(intersections[i]))
            x_end = min(width - 1, int(intersections[i + 1]))
            if x_start <= x_end:
                image[y, x_start:x_end + 1] = color


def render_frame_with_outlines(frame: Frame, width: int = 320, height: int = 200,
                                background_color: tuple = (0, 0, 0),
                                outline_color: tuple = (255, 255, 255),
                                outline_thickness: int = 1) -> np.ndarray:
    """
    Render frame with polygon outlines visible.
    Useful for debugging and style visualization.
    """
    image = render_frame(frame, width, height, background_color)

    if not HAS_CV2:
        # Can't draw outlines without OpenCV
        return image

    for polygon in frame.polygons:
        pts = polygon.vertices.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(image, [pts], isClosed=True,
                     color=outline_color, thickness=outline_thickness)

    return image


def render_animation(frames: List[Frame], width: int = 320, height: int = 200,
                     background_color: tuple = (0, 0, 0)) -> List[np.ndarray]:
    """
    Render all frames to a list of images.

    Args:
        frames: List of Frame objects
        width: Output image width
        height: Output image height
        background_color: RGB background color

    Returns:
        List of RGB images
    """
    return [render_frame(f, width, height, background_color) for f in frames]


def save_frame(frame: Frame, filepath: str, width: int = 320, height: int = 200):
    """Save a single frame to an image file."""
    image = render_frame(frame, width, height)

    if HAS_CV2:
        # OpenCV uses BGR
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        from PIL import Image
        Image.fromarray(image).save(filepath)


def save_animation_gif(frames: List[Frame], filepath: str,
                       width: int = 320, height: int = 200,
                       fps: float = 24.0, loop: int = 0):
    """
    Save animation as GIF.

    Args:
        frames: List of Frame objects
        filepath: Output GIF path
        width: Image width
        height: Image height
        fps: Frames per second
        loop: Number of loops (0 = infinite)
    """
    from PIL import Image

    images = render_animation(frames, width, height)
    pil_images = [Image.fromarray(img) for img in images]

    duration = int(1000 / fps)  # milliseconds per frame

    pil_images[0].save(
        filepath,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=loop
    )


def save_animation_mp4(frames: List[Frame], filepath: str,
                       width: int = 320, height: int = 200,
                       fps: float = 24.0):
    """
    Save animation as MP4 video.
    Requires OpenCV.
    """
    if not HAS_CV2:
        raise ImportError("OpenCV required for MP4 export")

    images = render_animation(frames, width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

    for img in images:
        # OpenCV uses BGR
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    out.release()
