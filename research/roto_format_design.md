# .roto Format Design

A radically compressed polygon animation format inspired by Another World's bytecode VM.

## Design Goals

1. **Extreme compression**: Full movie in 10-50 MB
2. **Streaming playback**: No need to load entire file
3. **Lightweight decoder**: WebAssembly target, <50KB
4. **Temporal coherence**: Smooth polygon interpolation
5. **Scene awareness**: Keyframes at cuts, delta encoding between

---

## Compression Strategies

### 1. Coordinate Quantization

Instead of float32 (4 bytes per coord), use relative coordinates:

```
Resolution: 320x200 (Another World native)
            or 640x400 (2x for modern displays)

Coordinate storage:
- Absolute: 10 bits x + 9 bits y = 19 bits (~2.5 bytes)
- Delta: signed 6-bit (-32 to +31) = 12 bits per vertex

For 320x200:
- x: 0-319 = 9 bits
- y: 0-199 = 8 bits
- Total: 17 bits = ~2 bytes per vertex (absolute)
```

### 2. Delta Encoding (Key Innovation)

Between frames, polygons usually move/deform slightly. Store only changes:

```
Frame N:   polygon vertices at [100,50], [120,50], [120,80], [100,80]
Frame N+1: polygon vertices at [102,51], [122,51], [122,81], [102,81]

Delta:     [+2,+1], [+2,+1], [+2,+1], [+2,+1]

Storage: 4 vertices × 2 bytes = 8 bytes (absolute)
         4 vertices × 1 byte  = 4 bytes (delta, if all deltas fit in signed byte)
```

### 3. Polygon Identity Tracking

Each polygon has a persistent ID across frames:

```
Polygon lifecycle:
- BIRTH: New polygon appears (full vertex data)
- MORPH: Polygon changes shape (delta vertices)
- MOVE:  Polygon translates uniformly (single delta)
- DEATH: Polygon disappears (just the ID)
```

### 4. Palette-Based Colors

```
16-color palette per scene (Another World style):
- 8 base colors
- 8 highlight variants

Color index: 4 bits per polygon
Palette definition: 16 × 3 bytes = 48 bytes per scene
```

### 5. Keyframe Strategy

```
Keyframe (I-frame):
- Full polygon definitions
- Full palette
- Triggered by: scene change, every N seconds

Delta frame (P-frame):
- Polygon deltas only
- Reference previous frame

Bidirectional (B-frame): [optional, increases complexity]
- Can reference both previous and next keyframe
```

---

## Binary Format Specification

### File Header

```
Offset  Size  Description
------  ----  -----------
0x00    4     Magic: "ROTO"
0x04    2     Version: 0x0001
0x06    2     Width (pixels)
0x08    2     Height (pixels)
0x0A    2     Frame rate (fps × 100, e.g., 2400 = 24fps)
0x0C    4     Total frames
0x10    4     Keyframe count
0x14    4     Offset to keyframe index
0x18    4     Offset to frame data
0x1C    4     Offset to audio (0 if none)
```

### Keyframe Index (for seeking)

```
Array of:
  4 bytes: frame number
  4 bytes: offset in frame data
```

### Frame Data

#### Keyframe (I-frame)

```
1 byte:  Frame type (0x00 = keyframe)
2 bytes: Polygon count
48 bytes: Palette (16 colors × RGB)

For each polygon:
  2 bytes: Polygon ID
  1 byte:  Vertex count
  1 byte:  Color index (4 bits) + flags (4 bits)
  N × 3 bytes: Vertices (10-bit x, 9-bit y, packed)
```

#### Delta Frame (P-frame)

```
1 byte:  Frame type (0x01 = delta)
1 byte:  Operation count

For each operation:
  1 byte:  Opcode

  Opcodes:
    0x00 NOOP      - No change to polygon
    0x01 MOVE      - Uniform translation
                     2 bytes: polygon ID
                     2 bytes: dx (signed 10-bit), dy (signed 9-bit)

    0x02 MORPH     - Vertex-by-vertex delta
                     2 bytes: polygon ID
                     1 byte:  vertex count
                     N bytes: deltas (signed 6-bit pairs, packed)

    0x03 RECOLOR   - Change polygon color
                     2 bytes: polygon ID
                     1 byte:  new color index

    0x04 BIRTH     - New polygon appears
                     [same as keyframe polygon data]

    0x05 DEATH     - Polygon disappears
                     2 bytes: polygon ID

    0x06 VERTEX_ADD    - Add vertex to polygon
    0x07 VERTEX_REMOVE - Remove vertex from polygon

    0x10 PALETTE_SHIFT - Shift palette colors
    0x11 PALETTE_SET   - Set specific palette entry
```

---

## Compression Estimate

### Assumptions
- 1080p source → 320x200 output
- 24 fps, 2-hour movie = 172,800 frames
- Average 80 polygons per frame
- Average 8 vertices per polygon
- Keyframe every 2 seconds (48 frames)

### Per-Frame Costs

**Keyframe:**
```
Header:        3 bytes
Palette:      48 bytes
80 polygons × (2 + 1 + 1 + 8×3) bytes = 80 × 28 = 2,240 bytes
Total: ~2.3 KB per keyframe
```

**Delta frame (typical):**
```
Header:        2 bytes
~40 operations average:
  - 20 MOVE ops:  20 × 4 = 80 bytes
  - 15 MORPH ops: 15 × (3 + 8) = 165 bytes
  - 5 other:      ~20 bytes
Total: ~270 bytes per delta frame
```

### Movie Total

```
Keyframes: 3,600 × 2.3 KB = 8.3 MB
Delta frames: 169,200 × 270 bytes = 45.7 MB
Raw total: ~54 MB

With general compression (LZ4/zstd):
Estimated: 15-25 MB for a 2-hour movie
```

---

## Optimizations

### 1. Run-Length Encoding for Static Scenes

If a polygon doesn't change for N frames:
```
0x08 REPEAT polygon_id, frame_count
```

### 2. Polygon Grouping

Characters/objects as groups that move together:
```
0x20 GROUP_MOVE group_id, dx, dy
```

### 3. Bezier Curves (Optional)

For smoother motion with fewer keyframes:
```
0x30 BEZIER_PATH polygon_id, control_points, duration
```

### 4. Hierarchical Polygons

Like Another World, define polygon hierarchies:
```
0x40 ATTACH child_polygon_id, parent_polygon_id, offset
```

---

## Decoder Architecture

```
┌─────────────────────────────────────────────────────┐
│                    .roto File                        │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              Stream Demuxer                          │
│  - Parse headers                                     │
│  - Seek to keyframes                                 │
│  - Yield frame operations                            │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              Polygon State Machine                   │
│  - Maintain current polygon set                      │
│  - Apply operations (MOVE, MORPH, BIRTH, DEATH)     │
│  - Track polygon IDs                                 │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              Rasterizer                              │
│  - Fill polygons (painter's algorithm)               │
│  - Apply palette                                     │
│  - Output to canvas/framebuffer                      │
└─────────────────────────────────────────────────────┘
```

### WebAssembly Target

```rust
// Core decoder in Rust, compile to WASM
pub struct RotoDecoder {
    polygons: HashMap<u16, Polygon>,
    palette: [Color; 16],
    width: u16,
    height: u16,
}

impl RotoDecoder {
    pub fn decode_frame(&mut self, data: &[u8]) -> Vec<Polygon> {
        // Apply operations to polygon state
    }

    pub fn rasterize(&self, buffer: &mut [u8]) {
        // Fill polygons to RGBA buffer
    }
}
```

---

## Comparison to Other Formats

| Format | 2hr Movie | Decode Complexity | Seeking |
|--------|-----------|-------------------|---------|
| H.264 (1080p) | 2-4 GB | High (hardware) | Good |
| H.264 (240p) | 200-400 MB | High | Good |
| GIF | N/A | Low | Poor |
| SVG Animation | ~500 MB | Medium | Poor |
| **.roto** | **15-25 MB** | **Low** | **Good** |

---

## Implementation Phases

### Phase 1: Proof of Concept
- [ ] Basic encoder: video → polygons → .roto
- [ ] Basic decoder: .roto → canvas
- [ ] Test with short clips

### Phase 2: Compression Tuning
- [ ] Optimize delta encoding
- [ ] Add polygon tracking heuristics
- [ ] Implement keyframe detection

### Phase 3: Player
- [ ] WebAssembly decoder
- [ ] HTML5 canvas renderer
- [ ] Playback controls, seeking

### Phase 4: Production
- [ ] Audio track support
- [ ] Batch encoding pipeline
- [ ] Quality presets
