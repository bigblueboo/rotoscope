# Rotoscope Research: Initial Findings

## Project Goal

Create a radically compressed video format by converting arbitrary video into polygon-based rotoscoped animations, inspired by Another World and Flashback. Target: compress entire movies into megabytes of polygon data.

---

## Historical Context: How Delphine Did It

### Eric Chahi's Process (Another World, 1991)

**Equipment:**
- Amiga 500 (1MB RAM, 20MB HDD)
- Camcorder to film live actors
- GenLock device for real-time video overlay on Amiga
- VHS player with digital frame memory for frame-by-frame pauses
- Custom editor written in GFA Basic

**Process:**
1. Film live actors performing movements
2. Overlay video on Amiga screen via GenLock
3. Manually trace each frame as polygons in custom editor
4. Store in bytecode format for custom virtual machine

### Technical Specifications

**Virtual Machine:**
- 256 variables for state management
- 64 concurrent execution threads
- 29 distinct opcodes
- Target: 20 FPS minimum
- Original DOS executable: only 20 KiB

**Graphics System:**
- Fixed 320x200 coordinate system
- 16-color palette (0x0-0x7 base, 0x8-0xF brightened)
- 4 framebuffers (double buffer + 2 background caches)
- Hierarchical polygon structures for animation reuse

**Why Polygons?**
> "It was born from a technical choice to create 2D polygons, as the approach overcame technical memory limitations. It only needs a few points to create shapes which cover the entire screen, which is much fewer than traditional bitmaps." — Eric Chahi

**Advantages over sprites:**
- Huge animations without memory constraints
- Movie-style cuts without load latency
- Smooth, scalable shapes
- Impressionistic, cinematic quality from abstraction

---

## Modern Approaches

### Video Vectorization Research

| Paper/Project | Key Contribution |
|---------------|------------------|
| Video Vectorization via Tetrahedral Remeshing (Wang et al.) | Spatial-temporal mesh over video volume |
| Real-time Image Vectorization on GPU (Xiong et al.) | GPU-parallel boundary detection and vectorization |
| Towards Layer-wise Image Vectorization (Ma et al., 2022) | Layered vector graphics preserving structure |
| SuperSVG (2024) | Superpixel segmentation as vectorization foundation |
| Tutrace (Vital, 2025) | VFX rotoscoping with vertex correspondence tracking |

### Tools

| Tool | Description |
|------|-------------|
| **Potrace** | Classic bitmap-to-vector, GPL, polygon-based tracing |
| **VTracer** | Modern Rust vectorizer, handles color images, WebAssembly available |
| **FrameSVG** | GIF-to-animated-SVG using VTracer |

### Temporal Coherence

The key challenge: maintaining polygon identity across frames to avoid jitter.

**Solutions:**
- Optical flow guidance
- Deep Video Prior (learns from single video pair)
- Segment tracking (DeAOT, XMem++)
- Vertex-based shape encoding with temporal prediction

### Simplification Algorithms

| Algorithm | Approach |
|-----------|----------|
| **Douglas-Peucker** | Recursive subdivision, eliminate points within tolerance |
| **Visvalingam-Whyatt** | Remove points by triangular area importance |
| **Quadric Error Metrics** | Surface simplification via vertex pair contraction |

---

## The Lester Project (2024)

**Author:** Ruben Tous
**Paper:** https://arxiv.org/abs/2402.09883
**Code:** https://github.com/rtous/lester

**Pipeline:**
```
Video → SAM Segmentation → DeAOT Tracking → Douglas-Peucker → Polygon Animation
```

**Key Innovations:**
1. Uses Segment Anything Model (SAM) for semantic segmentation
2. DeAOT for hierarchical temporal mask propagation
3. Douglas-Peucker for contour simplification
4. Handles diverse poses, appearances, dynamic shots

**Why it works:**
- Segmentation provides semantic regions (not just edges)
- Tracking maintains polygon identity across frames
- Simplification reduces vertex count while preserving shape
- More deterministic than diffusion-based approaches

---

## Compression Potential

### Traditional Video Compression
- H.264/H.265: ~1-5 Mbps for 1080p
- 2-hour movie at 3 Mbps ≈ 2.7 GB

### Polygon-Based Compression Estimate

**Per-frame polygon data:**
- Vertex: 2 bytes x-coord + 2 bytes y-coord = 4 bytes
- Average polygon: ~8 vertices = 32 bytes
- Color index: 1 byte
- Total per polygon: ~33 bytes

**Assumptions for aggressive compression:**
- 50-100 polygons per frame (aggressive simplification)
- 24 FPS, keyframes every 2 seconds
- Delta encoding between frames (only changed vertices)
- Palette: 16 colors × 3 bytes = 48 bytes per scene

**Rough estimate:**
- Keyframe: 100 polygons × 33 bytes = 3.3 KB
- Delta frame: ~500 bytes average (only vertex deltas)
- Per second: 3.3 KB + (23 × 500 bytes) = ~15 KB/sec
- 2-hour movie: 15 KB × 7200 sec = **108 MB**

With further optimization (hierarchical polygons, better delta encoding):
- Target: **10-50 MB for a full movie**

---

## Existing Implementations

### Engine Recreations
- **Fabother World** - Bytecode interpreter by Fabien Sanglard
- **REminiscence** - Flashback engine recreation
- **a5k** - Another World on FPGA
- **another_js** - HTML5/JavaScript port

### References
- Fabien Sanglard's analysis: https://fabiensanglard.net/another_world_polygons/
- Official site: https://anotherworld.fr/anotherworld_uk/page_realisation.htm
- GitHub interpreter: https://github.com/fabiensanglard/Another-World-Bytecode-Interpreter

---

## Proposed Pipeline

```
┌─────────────┐
│ Input Video │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Preprocessing    │  ← Downscale, temporal smoothing
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ SAM 2 Segment    │  ← Extract semantic regions
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ DeAOT Track      │  ← Maintain region identity across frames
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Palette Quantize │  ← Reduce to 16 colors
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Douglas-Peucker  │  ← Simplify contours to polygons
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Delta Encode     │  ← Store only vertex changes
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Keyframe Detect  │  ← Scene changes reset polygon set
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│ .roto Format    │  ← Custom bytecode output
└─────────────────┘
```

---

## Implementation Stack (Recommended)

### Segmentation: SAM 2
- **Repo**: https://github.com/facebookresearch/segment-anything-2
- **Model**: `sam2_hiera_base_plus` (80.8M params, ~35 FPS, 8-10GB VRAM)
- **Key feature**: Built-in video propagation (no separate tracker needed)

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 && pip install -e .
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
```

### Alternative Tracker: Cutie (if needed)
- **Repo**: https://github.com/hkchengrex/Cutie
- 3x faster than DeAOT, better robustness to distractors
- CVPR 2024 Highlight

### Polygon Simplification: OpenCV Douglas-Peucker
```python
import cv2
import numpy as np

def mask_to_polygon(mask, epsilon_ratio=0.04):
    """Another World style: epsilon_ratio 0.04-0.06"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_ratio * perimeter
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        polygons.append(simplified.squeeze())
    return polygons
```

### Epsilon Values for Different Styles
| Style | Epsilon Ratio | Vertices/Shape |
|-------|---------------|----------------|
| Detailed | 0.005 | 50+ |
| Moderate | 0.015 | 20-30 |
| Stylized | 0.03 | 10-20 |
| Another World | 0.05 | 5-12 |
| Extreme | 0.08 | 3-6 |

---

## Lester Results Analysis

From the Lester repository samples:

| Video | Input Size | Output (H.264) | Compression |
|-------|-----------|----------------|-------------|
| youtube1 | 5.9 MB | 62 KB | **95x** |
| youtube3 | 1.8 MB | 101 KB | **18x** |

**Note**: These are still H.264 video outputs. Raw polygon data would be even smaller.

---

## Next Steps

1. [x] Clone and analyze Lester implementation (closed source, but algorithm understood)
2. [x] Understand SAM 2 + tracking integration
3. [ ] Design custom .roto bytecode format (inspired by Another World VM)
4. [ ] Prototype pipeline with sample video
5. [ ] Measure compression ratios vs quality
6. [ ] Build lightweight player (WebAssembly target)
