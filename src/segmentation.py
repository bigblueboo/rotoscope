"""
SAM 2 segmentation integration.

Provides automatic and prompted segmentation for the rotoscope pipeline.
Supports both per-image and video tracking modes.
"""
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Lazy-loaded SAM 2 imports
_sam2_available = None


def _check_sam2():
    global _sam2_available
    if _sam2_available is None:
        try:
            import sam2  # noqa: F401
            _sam2_available = True
        except ImportError:
            _sam2_available = False
    return _sam2_available


def _get_device():
    """Pick the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Image-level segmentation (one frame at a time, no temporal tracking)
# ---------------------------------------------------------------------------

class SAM2ImageSegmenter:
    """Segment a single image using SAM 2 automatic mask generator."""

    def __init__(
        self,
        checkpoint: str = "sam2/checkpoints/sam2.1_hiera_tiny.pt",
        config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        device: Optional[str] = None,
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0.90,
        min_mask_region_area: int = 100,
    ):
        if not _check_sam2():
            raise ImportError("SAM 2 not installed. See sam2/INSTALL.md")

        self.device = torch.device(device) if device else _get_device()
        self.checkpoint = checkpoint
        self.config = config
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        self._generator = None

    def _load(self):
        if self._generator is not None:
            return
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        model = build_sam2(self.config, self.checkpoint, device=self.device)
        self._generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
            output_mode="binary_mask",
        )

    def segment(self, image: np.ndarray) -> List[Dict]:
        """
        Automatically segment an image into masks.

        Args:
            image: RGB uint8 array (H, W, 3)

        Returns:
            List of dicts, each with keys:
                'segmentation': bool mask (H, W)
                'area': int pixel count
                'bbox': [x, y, w, h]
                'predicted_iou': float
                'stability_score': float
        """
        self._load()
        with torch.inference_mode():
            return self._generator.generate(image)

    def segment_to_masks(self, image: np.ndarray) -> List[np.ndarray]:
        """Return just the binary masks, sorted largest-first."""
        results = self.segment(image)
        results.sort(key=lambda r: r["area"], reverse=True)
        return [r["segmentation"].astype(np.uint8) for r in results]


# ---------------------------------------------------------------------------
# Prompted image segmentation (clicks / boxes)
# ---------------------------------------------------------------------------

class SAM2PromptedSegmenter:
    """Segment objects in a single image with explicit point/box prompts."""

    def __init__(
        self,
        checkpoint: str = "sam2/checkpoints/sam2.1_hiera_tiny.pt",
        config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        device: Optional[str] = None,
    ):
        if not _check_sam2():
            raise ImportError("SAM 2 not installed.")
        self.device = torch.device(device) if device else _get_device()
        self.checkpoint = checkpoint
        self.config = config
        self._predictor = None

    def _load(self):
        if self._predictor is not None:
            return
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(self.config, self.checkpoint, device=self.device)
        self._predictor = SAM2ImagePredictor(model)

    def segment(
        self,
        image: np.ndarray,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Segment with point or box prompts.

        Returns:
            Binary mask (H, W), dtype bool
        """
        self._load()
        self._predictor.set_image(image)
        masks, scores, _ = self._predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            multimask_output=True,
        )
        return masks[np.argmax(scores)]


# ---------------------------------------------------------------------------
# Video segmentation with temporal tracking
# ---------------------------------------------------------------------------

def extract_frames_to_dir(
    video_path: str,
    output_dir: str,
    max_frames: Optional[int] = None,
    frame_skip: int = 1,
    resize: Optional[Tuple[int, int]] = None,
) -> int:
    """
    Extract video frames as numbered JPEGs (required by SAM 2 video predictor).

    Args:
        video_path: Input video
        output_dir: Directory to write %05d.jpg files
        max_frames: Cap on number of frames to extract
        frame_skip: Keep every Nth frame
        resize: Optional (width, height) to resize

    Returns:
        Number of frames written
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    written = 0
    raw_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if raw_idx % frame_skip != 0:
                raw_idx += 1
                continue
            if resize:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            path = os.path.join(output_dir, f"{written:05d}.jpg")
            cv2.imwrite(path, frame)
            written += 1
            if max_frames and written >= max_frames:
                break
            raw_idx += 1
    finally:
        cap.release()
    return written


class SAM2VideoTracker:
    """
    Track objects through a video using SAM 2 video predictor.

    Workflow:
        1. init(video_path)               – extract frames & init state
        2. add_objects(frame_idx, ...)     – provide prompts on reference frame
        3. propagate()                     – track through all frames
        4. get_masks()                     – retrieve per-frame masks
    """

    def __init__(
        self,
        checkpoint: str = "sam2/checkpoints/sam2.1_hiera_tiny.pt",
        config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        device: Optional[str] = None,
        max_frames: Optional[int] = None,
        frame_skip: int = 1,
    ):
        if not _check_sam2():
            raise ImportError("SAM 2 not installed.")
        self.device = torch.device(device) if device else _get_device()
        self.checkpoint = checkpoint
        self.config = config
        self.max_frames = max_frames
        self.frame_skip = frame_skip

        self._predictor = None
        self._inference_state = None
        self._frames_dir = None
        self._temp_dir = None
        self._num_frames = 0

        # Collected results: {frame_idx: {obj_id: mask}}
        self.video_segments: Dict[int, Dict[int, np.ndarray]] = {}

    def _load(self):
        if self._predictor is not None:
            return
        from sam2.build_sam import build_sam2_video_predictor

        self._predictor = build_sam2_video_predictor(
            self.config, self.checkpoint, device=self.device
        )

    def init(
        self,
        video_path: str,
        frames_dir: Optional[str] = None,
        resize: Optional[Tuple[int, int]] = None,
    ):
        """
        Extract frames and initialise the video predictor state.

        Args:
            video_path: Path to input video
            frames_dir: Optional directory for frames (uses tempdir otherwise)
            resize: Optional (width, height) to resize frames
        """
        self._load()

        if frames_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="roto_frames_")
            self._frames_dir = self._temp_dir
        else:
            os.makedirs(frames_dir, exist_ok=True)
            self._frames_dir = frames_dir

        self._num_frames = extract_frames_to_dir(
            video_path,
            self._frames_dir,
            max_frames=self.max_frames,
            frame_skip=self.frame_skip,
            resize=resize,
        )
        print(f"Extracted {self._num_frames} frames to {self._frames_dir}")

        with torch.inference_mode():
            self._inference_state = self._predictor.init_state(
                video_path=self._frames_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )

    # ---- prompt methods ----

    def add_points(
        self,
        frame_idx: int,
        obj_id: int,
        points: np.ndarray,
        labels: np.ndarray,
    ):
        """Add click prompts for an object on a given frame."""
        with torch.inference_mode():
            _, _, logits = self._predictor.add_new_points_or_box(
                inference_state=self._inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )
        return (logits[0] > 0.0).cpu().numpy()

    def add_box(self, frame_idx: int, obj_id: int, box: np.ndarray):
        """Add a bounding-box prompt for an object on a given frame."""
        with torch.inference_mode():
            _, _, logits = self._predictor.add_new_points_or_box(
                inference_state=self._inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box,
            )
        return (logits[0] > 0.0).cpu().numpy()

    def add_mask(self, frame_idx: int, obj_id: int, mask: np.ndarray):
        """Add a pre-existing mask as a prompt (e.g. from auto-segmentation)."""
        with torch.inference_mode():
            _, _, logits = self._predictor.add_new_mask(
                inference_state=self._inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask,
            )
        return (logits[0] > 0.0).cpu().numpy()

    # ---- propagation ----

    def propagate(self, reverse: bool = False):
        """
        Propagate all prompted objects through the video.

        Populates self.video_segments with per-frame mask dictionaries.
        """
        with torch.inference_mode():
            for frame_idx, obj_ids, logits in self._predictor.propagate_in_video(
                self._inference_state, reverse=reverse
            ):
                masks = {}
                for i, oid in enumerate(obj_ids):
                    masks[oid] = (logits[i] > 0.0).squeeze().cpu().numpy()
                self.video_segments[frame_idx] = masks
        print(f"Propagated to {len(self.video_segments)} frames, "
              f"{len(self.video_segments.get(0, {}))} objects")

    def get_masks(self, frame_idx: int) -> Dict[int, np.ndarray]:
        """Retrieve tracked masks for a frame."""
        return self.video_segments.get(frame_idx, {})

    def get_all_masks(self) -> Dict[int, Dict[int, np.ndarray]]:
        return self.video_segments

    # ---- convenience: auto-segment first frame, then track ----

    def auto_segment_and_track(
        self,
        video_path: str,
        resize: Optional[Tuple[int, int]] = None,
        points_per_side: int = 8,
        min_mask_region_area: int = 200,
        max_objects: int = 32,
    ):
        """
        Fully automatic pipeline:
          1. Extract frames
          2. Auto-segment first frame with SAM 2 mask generator
          3. Register best masks as tracking prompts
          4. Propagate through video

        Args:
            video_path: Input video path
            resize: Optional (w, h) to resize frames
            points_per_side: Grid density for auto-segmentation
            min_mask_region_area: Minimum mask pixel area
            max_objects: Cap on number of tracked objects

        Returns:
            self (for chaining)
        """
        self.init(video_path, resize=resize)

        # Auto-segment first frame
        first_frame_path = os.path.join(self._frames_dir, "00000.jpg")
        first_frame = cv2.cvtColor(cv2.imread(first_frame_path), cv2.COLOR_BGR2RGB)

        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        # Build a separate image model for auto-seg (reuses weights)
        image_model = build_sam2(self.config, self.checkpoint, device=self.device)
        generator = SAM2AutomaticMaskGenerator(
            image_model,
            points_per_side=points_per_side,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
            min_mask_region_area=min_mask_region_area,
            output_mode="binary_mask",
        )

        print("Auto-segmenting first frame...")
        with torch.inference_mode():
            auto_masks = generator.generate(first_frame)

        # Sort by area descending, take top N
        auto_masks.sort(key=lambda m: m["area"], reverse=True)
        auto_masks = auto_masks[:max_objects]
        print(f"Found {len(auto_masks)} masks")

        # Register each mask as a tracking prompt on frame 0
        for obj_id, mask_data in enumerate(auto_masks, start=1):
            mask = mask_data["segmentation"]
            self.add_mask(frame_idx=0, obj_id=obj_id, mask=mask)

        # Free the image model
        del generator, image_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Propagate
        print("Propagating masks through video...")
        self.propagate()

        return self

    @property
    def num_frames(self) -> int:
        return self._num_frames

    def cleanup(self):
        """Remove temporary frame directory."""
        if self._temp_dir and os.path.isdir(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
