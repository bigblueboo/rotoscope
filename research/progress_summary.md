# Rotoscope Project: Progress Summary

## What We've Built

A working prototype pipeline for converting video to polygon-based animation in the "Another World" style.

### Components Implemented

1. **Polygon data structures** (`src/polygon.py`)
   - Polygon and Frame classes with delta computation
   - Support for move, morph, birth, death operations

2. **Color quantization** (`src/quantize.py`)
   - K-means and median cut algorithms
   - 16-color palette support (Another World style)

3. **Polygon simplification** (`src/simplify.py`)
   - Douglas-Peucker algorithm via OpenCV
   - Configurable style presets (detailed → extreme)
   - Enforced max vertices per polygon (20 for Another World)

4. **Binary encoder** (`src/encoder.py`)
   - Custom `.roto` format with keyframe + delta encoding
   - Supports: MOVE, MORPH, RECOLOR, BIRTH, DEATH operations

5. **Binary decoder** (`src/decoder.py`)
   - Reads `.roto` format with seeking support
   - Maintains polygon state machine

6. **Rasterizer** (`src/rasterize.py`)
   - Painter's algorithm polygon filling
   - GIF and MP4 export support

7. **Pipeline** (`src/pipeline.py`)
   - Color-based segmentation (no ML required)
   - SAM 2 integration placeholder

---

## Compression Results

### Single Frame (Synthetic Test)
- Raw RGB: 192,000 bytes
- .roto: 619 bytes
- **Compression: 310x**

### Video (50 frames, youtube1 sample)
- Original H.264: 5.9 MB
- .roto format: 257.7 KB
- **Compression: 23x**

### Extrapolated Full Movie Estimate
```
2-hour movie at 24fps = 172,800 frames
With keyframe every 48 frames + delta encoding:
- ~120 polygons/frame × 9 vertices avg = 1,080 vertices/frame
- Keyframe: ~2.3 KB, Delta: ~300 bytes
- Total: ~55 MB raw, ~20 MB with LZ4 compression
```

---

## Key Insight: Color vs Semantic Segmentation

### Our Current Approach (Color-based)
- Segments by color regions across entire image
- Includes background noise
- Fast, no ML required
- **Result**: Noisy, hard to recognize subjects

### Lester's Approach (SAM-based)
- Segments semantically meaningful objects
- Clean background (can use solid color)
- Requires ML model (SAM 2)
- **Result**: Clean, recognizable characters

**Conclusion**: For quality results, SAM 2 integration is essential.

---

## Visual Comparison

| Approach | Quality | Speed | Dependencies |
|----------|---------|-------|--------------|
| Color-based (ours) | Noisy | Fast | OpenCV only |
| SAM 2 (Lester-style) | Clean | Slower | PyTorch, SAM 2 |

---

## Next Steps

### Immediate Priorities

1. **SAM 2 Integration**
   - Interactive point/box prompts on first frame
   - Mask propagation through video
   - Combine with polygon simplification

2. **Delta Encoding Optimization**
   - Better vertex tracking between frames
   - Detect uniform translations (MOVE vs MORPH)
   - Run-length encoding for static polygons

3. **Palette Optimization**
   - Global palette across video
   - Palette interpolation between scenes

### Future Improvements

4. **Polygon Tracking Across Frames**
   - Use optical flow for vertex correspondence
   - Maintain polygon identity for smoother interpolation

5. **Scene Detection**
   - Auto-detect scene cuts for keyframes
   - Prevent inter-scene polygon morphing

6. **WebAssembly Player**
   - Rust-based decoder
   - Canvas rendering
   - < 50KB payload target

---

## Repository Structure

```
rotoscope/
├── src/
│   ├── __init__.py
│   ├── polygon.py      # Data structures
│   ├── quantize.py     # Color quantization
│   ├── simplify.py     # Douglas-Peucker simplification
│   ├── encoder.py      # .roto format encoder
│   ├── decoder.py      # .roto format decoder
│   ├── rasterize.py    # Polygon rendering
│   └── pipeline.py     # Video processing pipeline
├── research/
│   ├── initial_findings.md
│   ├── roto_format_design.md
│   └── progress_summary.md  (this file)
├── samples/            # Test outputs
├── lester/             # Reference implementation
├── demo.py             # CLI tool
├── test_pipeline.py    # Basic tests
└── requirements.txt
```

---

## Running the Pipeline

```bash
# Process single image
python demo.py image.jpg output.png --style another_world

# Process video to .roto
python demo.py video.mp4 output.roto --max-frames 100

# Process video to GIF (for preview)
python demo.py video.mp4 output.gif --max-frames 50 --frame-skip 2
```

---

## Technical Notes

### Why Polygon-Based Compression Works

1. **Spatial coherence**: Video has large uniform regions
2. **Temporal coherence**: Objects move/deform smoothly
3. **Perceptual tolerance**: Human vision tolerates simplification
4. **Vector efficiency**: Polygons scale infinitely, compress well

### Another World's Original Specs
- 320×200 resolution
- 16-color palette
- 29 VM opcodes
- 64 concurrent threads
- 20 FPS target
- 20 KB executable

### Our Target Specs
- 320×200 default (scalable)
- 16-color palette per scene
- ~10 delta opcodes
- Single-threaded decoder
- 24 FPS
- < 50 KB WASM decoder
