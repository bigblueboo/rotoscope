"""
Video to polygon pipeline.

This module provides the main pipeline for converting video to polygon animation.
It can work with or without SAM 2 - falling back to color-based segmentation.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Iterator
from pathlib import Path

from .polygon import Polygon, Frame
from .quantize import kmeans_quantize, median_cut_quantize
from .simplify import mask_to_polygons, STYLE_PRESETS

# Try to import video libraries
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class VideoToPolygonPipeline:
    """
    Pipeline for converting video frames to polygon animation.

    Supports multiple segmentation backends:
    - 'color': Simple color-based segmentation (no ML required)
    - 'sam2': Segment Anything Model 2 (requires SAM 2 installation)
    """

    def __init__(self,
                 width: int = 320,
                 height: int = 200,
                 num_colors: int = 16,
                 style: str = 'another_world',
                 min_polygon_area: int = 50,
                 segmentation_backend: str = 'color'):
        """
        Initialize the pipeline.

        Args:
            width: Output width (default 320 for Another World style)
            height: Output height (default 200)
            num_colors: Number of colors in palette (default 16)
            style: Simplification style preset
            min_polygon_area: Minimum polygon area to keep
            segmentation_backend: 'color' or 'sam2'
        """
        self.width = width
        self.height = height
        self.num_colors = num_colors
        self.epsilon_ratio = STYLE_PRESETS.get(style, STYLE_PRESETS['another_world'])
        self.min_polygon_area = min_polygon_area
        self.backend = segmentation_backend

        self._polygon_id_counter = 0

    def _next_polygon_id(self) -> int:
        """Get next unique polygon ID."""
        self._polygon_id_counter += 1
        return self._polygon_id_counter

    def process_image(self, image: np.ndarray) -> Tuple[Frame, np.ndarray]:
        """
        Process a single image to polygons.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            Tuple of (Frame, palette)
        """
        # Resize to target dimensions
        resized = self._resize_image(image)

        # ---- Pre-smooth to create coherent colour regions ----
        # Bilateral filter preserves edges while smoothing colour noise
        if HAS_CV2:
            smoothed = cv2.bilateralFilter(resized, d=9, sigmaColor=75, sigmaSpace=75)
            # Second pass for stronger effect
            smoothed = cv2.bilateralFilter(smoothed, d=9, sigmaColor=50, sigmaSpace=50)
        else:
            smoothed = resized

        # Quantize colors
        indexed, palette = kmeans_quantize(smoothed, self.num_colors)

        # ---- Morphological cleaning on each colour mask ----
        # This merges nearby fragments and fills small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) if HAS_CV2 else None

        # Find the most common colour to use as the background layer
        color_areas = []
        for ci in range(self.num_colors):
            color_areas.append(((indexed == ci).sum(), ci))
        color_areas.sort(reverse=True)
        bg_color = color_areas[0][1]

        # Build polygon list: background first, then smaller regions on top
        polygons = []

        # Full-frame background rectangle in the dominant colour
        polygons.append(Polygon(
            id=self._next_polygon_id(),
            vertices=np.array([[0, 0], [self.width - 1, 0],
                               [self.width - 1, self.height - 1],
                               [0, self.height - 1]], dtype=np.int16),
            color_index=bg_color,
        ))

        # Process colours from largest area to smallest (painter's algorithm)
        for _, color_idx in color_areas:
            # Skip the background colour - already covered by the rectangle
            if color_idx == bg_color:
                continue

            mask = (indexed == color_idx).astype(np.uint8) * 255

            # Morphological close: fill small gaps and merge nearby fragments
            if kernel is not None:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                # Small open to remove remaining specks
                small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)

            if mask.sum() == 0:
                continue

            # Convert mask to simplified polygons
            color_polygons = mask_to_polygons(
                mask,
                epsilon_ratio=max(self.epsilon_ratio, 0.04),
                min_area=self.min_polygon_area,
                max_vertices=20,
            )

            for vertices in color_polygons:
                if len(vertices) >= 3:
                    polygons.append(Polygon(
                        id=self._next_polygon_id(),
                        vertices=vertices,
                        color_index=color_idx,
                    ))

        frame = Frame(index=0, polygons=polygons, palette=palette)
        return frame, palette

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions."""
        if HAS_CV2:
            return cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        elif HAS_PIL:
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize((self.width, self.height), Image.Resampling.LANCZOS)
            return np.array(pil_img)
        else:
            # Very basic nearest neighbor resize
            h, w = image.shape[:2]
            y_indices = (np.arange(self.height) * h // self.height).astype(int)
            x_indices = (np.arange(self.width) * w // self.width).astype(int)
            return image[y_indices][:, x_indices]

    def process_video(self, video_path: str,
                      max_frames: Optional[int] = None,
                      frame_skip: int = 1) -> Iterator[Frame]:
        """
        Process a video file to polygon frames.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None for all)
            frame_skip: Process every Nth frame

        Yields:
            Frame objects
        """
        if not HAS_CV2:
            raise ImportError("OpenCV required for video processing")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        frame_idx = 0
        processed = 0
        global_palette = None

        try:
            while True:
                ret, bgr_frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

                # Process frame
                frame, palette = self.process_image(rgb_frame)
                frame.index = processed

                # Use first frame's palette as global palette for consistency
                if global_palette is None:
                    global_palette = palette
                else:
                    frame.palette = global_palette

                yield frame

                processed += 1
                if max_frames and processed >= max_frames:
                    break

                frame_idx += 1

        finally:
            cap.release()

    def process_video_to_list(self, video_path: str,
                              max_frames: Optional[int] = None,
                              frame_skip: int = 1) -> List[Frame]:
        """Process video and return list of all frames."""
        return list(self.process_video(video_path, max_frames, frame_skip))


class SAM2Pipeline:
    """
    Hybrid pipeline: SAM 2 for foreground objects + colour quantization for background.

    For video:
      1. Auto-segment first frame with SAM 2
      2. Track object masks through subsequent frames
      3. Fill uncovered background with colour-quantized polygons
      4. Simplify all contours to low-vertex polygons

    Requires SAM 2:  cd sam2_repo && pip3 install -e .
    """

    # Palette layout: 0-7 background colours, 8-15 SAM object colours
    N_BG_COLORS = 8
    N_FG_COLORS = 8

    def __init__(
        self,
        sam_checkpoint: str = "sam2_repo/checkpoints/sam2.1_hiera_tiny.pt",
        sam_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        width: int = 320,
        height: int = 200,
        style: str = "another_world",
        min_polygon_area: int = 50,
        max_objects: int = 8,
        points_per_side: int = 24,
        device: Optional[str] = None,
    ):
        self.sam_checkpoint = sam_checkpoint
        self.sam_config = sam_config
        self.width = width
        self.height = height
        self.epsilon_ratio = STYLE_PRESETS.get(style, STYLE_PRESETS["another_world"])
        self.min_polygon_area = min_polygon_area
        self.max_objects = min(max_objects, self.N_FG_COLORS)
        self.points_per_side = points_per_side
        self.device = device

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_hybrid_frame(
        self,
        frame_idx: int,
        image_rgb: np.ndarray,
        sam_masks: Dict[int, np.ndarray],
        palette: np.ndarray,
    ) -> Frame:
        """
        Build one Frame by:
          – full-frame background in dominant colour
          – colour-quantized background regions (smoothed + cleaned)
          – SAM-segmented foreground polygons on top
        """
        h, w = self.height, self.width

        # Resize image
        if image_rgb.shape[:2] != (h, w):
            img = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)
        else:
            img = image_rgb

        # Pre-smooth for cleaner background regions
        smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        smoothed = cv2.bilateralFilter(smoothed, d=9, sigmaColor=50, sigmaSpace=50)

        # Resize SAM masks to output dims
        resized_masks: Dict[int, np.ndarray] = {}
        sam_union = np.zeros((h, w), dtype=bool)
        for oid, mask in sam_masks.items():
            if mask.shape[:2] != (h, w):
                m = cv2.resize(mask.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST)
            else:
                m = mask.astype(np.uint8)
            resized_masks[oid] = m
            sam_union |= m > 0

        polygons: List[Polygon] = []
        pid = [0]  # mutable counter

        def next_pid():
            pid[0] += 1
            return pid[0]

        # ---- background: colour-quantize uncovered areas ----
        bg_indexed, bg_palette = kmeans_quantize(smoothed, self.N_BG_COLORS)
        palette[:self.N_BG_COLORS] = bg_palette

        # Find most common bg colour for the base rectangle
        bg_areas = []
        for ci in range(self.N_BG_COLORS):
            bg_areas.append(((bg_indexed == ci).sum(), ci))
        bg_areas.sort(reverse=True)
        base_color = bg_areas[0][1]

        # Full-frame base rectangle (eliminates black gaps)
        polygons.append(Polygon(
            id=next_pid(),
            vertices=np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                              dtype=np.int16),
            color_index=base_color,
        ))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for _, ci in bg_areas:
            if ci == base_color:
                continue
            mask = ((bg_indexed == ci) & ~sam_union).astype(np.uint8) * 255
            # Morphological cleanup
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)
            if mask.sum() == 0:
                continue
            polys = mask_to_polygons(
                mask, epsilon_ratio=0.05,
                min_area=self.min_polygon_area, max_vertices=15,
            )
            for v in polys:
                if len(v) >= 3:
                    polygons.append(Polygon(id=next_pid(), vertices=v, color_index=ci))

        # ---- foreground: SAM objects ----
        for i, (oid, mask) in enumerate(sorted(resized_masks.items())):
            if i >= self.N_FG_COLORS:
                break
            ci = self.N_BG_COLORS + i

            # average colour from source image
            if mask.sum() > 0:
                palette[ci] = img[mask > 0].mean(axis=0).astype(np.uint8)

            polys = mask_to_polygons(
                mask, epsilon_ratio=max(self.epsilon_ratio, 0.04),
                min_area=self.min_polygon_area, max_vertices=20,
            )
            for v in polys:
                if len(v) >= 3:
                    polygons.append(Polygon(
                        id=oid * 1000 + next_pid(),
                        vertices=v, color_index=ci,
                    ))

        return Frame(index=frame_idx, polygons=polygons, palette=palette.copy())

    # ------------------------------------------------------------------
    # single image
    # ------------------------------------------------------------------

    def process_image(self, image: np.ndarray) -> Tuple[Frame, np.ndarray]:
        """Process one image (SAM auto-seg + colour quantization background)."""
        from .segmentation import SAM2ImageSegmenter

        segmenter = SAM2ImageSegmenter(
            checkpoint=self.sam_checkpoint,
            config=self.sam_config,
            device=self.device,
            points_per_side=self.points_per_side,
            pred_iou_thresh=0.65,
            stability_score_thresh=0.80,
            min_mask_region_area=self.min_polygon_area,
        )

        results = segmenter.segment(image)
        results.sort(key=lambda r: r["area"], reverse=True)
        mask_dict = {i + 1: r["segmentation"].astype(np.uint8)
                     for i, r in enumerate(results[:self.max_objects])}

        palette = np.zeros((16, 3), dtype=np.uint8)
        frame = self._build_hybrid_frame(0, image, mask_dict, palette)
        return frame, palette

    # ------------------------------------------------------------------
    # video with tracking
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        frame_skip: int = 1,
    ) -> List[Frame]:
        """
        Full pipeline: video → SAM 2 tracking → hybrid polygon frames.
        """
        from .segmentation import SAM2VideoTracker

        tracker = SAM2VideoTracker(
            checkpoint=self.sam_checkpoint,
            config=self.sam_config,
            device=self.device,
            max_frames=max_frames,
            frame_skip=frame_skip,
        )

        tracker.auto_segment_and_track(
            video_path,
            resize=(self.width, self.height),
            points_per_side=self.points_per_side,
            min_mask_region_area=self.min_polygon_area,
            max_objects=self.max_objects,
        )

        # Read original frames for colour sampling
        cap = cv2.VideoCapture(video_path)
        raw_frames: List[np.ndarray] = []
        raw_idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if raw_idx % frame_skip != 0:
                raw_idx += 1
                continue
            raw_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if max_frames and len(raw_frames) >= max_frames:
                break
            raw_idx += 1
        cap.release()

        # Build global palette from first frame
        palette = np.zeros((16, 3), dtype=np.uint8)

        frames: List[Frame] = []
        for fidx in range(tracker.num_frames):
            masks = tracker.get_masks(fidx)
            img = raw_frames[fidx] if fidx < len(raw_frames) else raw_frames[-1]
            frame = self._build_hybrid_frame(fidx, img, masks, palette)
            frames.append(frame)
            if fidx == 0:
                palette = frame.palette.copy()  # lock palette from frame 0

        tracker.cleanup()
        total_p = sum(len(f.polygons) for f in frames)
        total_v = sum(sum(p.vertex_count for p in f.polygons) for f in frames)
        print(f"Generated {len(frames)} frames, {total_p} polygons, {total_v} vertices")
        return frames


def quick_convert(image_path: str, output_path: str,
                  style: str = 'another_world',
                  width: int = 320, height: int = 200):
    """
    Quick conversion of a single image to polygon visualization.

    Args:
        image_path: Input image path
        output_path: Output image path
        style: Simplification style
        width: Output width
        height: Output height
    """
    from .rasterize import render_frame, save_frame

    # Load image
    if HAS_CV2:
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    else:
        img = np.array(Image.open(image_path).convert('RGB'))

    # Process
    pipeline = VideoToPolygonPipeline(width=width, height=height, style=style)
    frame, _ = pipeline.process_image(img)

    # Render and save
    save_frame(frame, output_path, width, height)

    return frame
