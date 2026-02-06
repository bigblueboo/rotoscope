#!/usr/bin/env python3
"""Quick test of the pipeline with a synthetic image."""
import numpy as np
import sys
sys.path.insert(0, '/home/cdeck/dev/rotoscope')

from src.pipeline import VideoToPolygonPipeline
from src.rasterize import render_frame
from src.encoder import encode_to_file
from src.decoder import RotoDecoder

def create_test_image(width=640, height=480):
    """Create a simple test image with geometric shapes."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Background gradient (sky blue to darker blue)
    for y in range(height):
        t = y / height
        img[y, :] = [int(135 + 50*t), int(206 - 80*t), int(235 - 50*t)]

    # Green ground
    img[height//2:, :] = [34, 139, 34]

    # Red rectangle (building)
    img[100:300, 200:350] = [180, 50, 50]

    # Yellow rectangle (window)
    img[150:200, 230:280] = [255, 220, 100]

    # Blue circle (sun) - approximated as filled region
    cy, cx, r = 80, 500, 50
    Y, X = np.ogrid[:height, :width]
    mask = (X - cx)**2 + (Y - cy)**2 <= r**2
    img[mask] = [255, 200, 50]

    # Dark triangle (tree)
    for y in range(250, 400):
        half_width = (400 - y) // 3
        cx = 100
        x_start = max(0, cx - half_width)
        x_end = min(width, cx + half_width)
        img[y, x_start:x_end] = [20, 80, 20]

    return img


def main():
    print("=== Rotoscope Pipeline Test ===\n")

    # Create test image
    print("Creating test image...")
    img = create_test_image(640, 480)

    # Process with pipeline
    print("Processing with pipeline...")
    pipeline = VideoToPolygonPipeline(
        width=320,
        height=200,
        num_colors=16,
        style='another_world',
        min_polygon_area=30
    )

    frame, palette = pipeline.process_image(img)

    print(f"  Polygons: {len(frame.polygons)}")
    total_vertices = sum(p.vertex_count for p in frame.polygons)
    print(f"  Total vertices: {total_vertices}")
    print(f"  Palette colors: {len(palette)}")

    # Show polygon details
    print("\n  Polygon breakdown by color:")
    color_counts = {}
    for p in frame.polygons:
        color_counts[p.color_index] = color_counts.get(p.color_index, 0) + 1
    for idx, count in sorted(color_counts.items()):
        color = palette[idx]
        print(f"    Color {idx:2d} (RGB {color[0]:3d},{color[1]:3d},{color[2]:3d}): {count} polygons")

    # Render
    print("\nRendering frame...")
    rendered = render_frame(frame, 320, 200)

    # Save rendered image
    import cv2
    cv2.imwrite('test_output.png', cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
    print("Saved rendered output: test_output.png")

    # Save original resized for comparison
    resized = cv2.resize(img, (320, 200))
    cv2.imwrite('test_original.png', cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    print("Saved original (resized): test_original.png")

    # Test encoder
    print("\nTesting .roto encoder...")
    frames = [frame]  # Just one frame for test
    encode_to_file(frames, 'test.roto', 320, 200, 24.0)

    import os
    roto_size = os.path.getsize('test.roto')
    print(f"  .roto file size: {roto_size} bytes")

    # Estimate equivalent raw size
    raw_size = 320 * 200 * 3  # RGB image
    print(f"  Raw frame size: {raw_size} bytes")
    print(f"  Compression ratio: {raw_size / roto_size:.1f}x")

    # Test decoder
    print("\nTesting .roto decoder...")
    decoder = RotoDecoder()
    decoder.open('test.roto')
    print(f"  Width: {decoder.width}")
    print(f"  Height: {decoder.height}")
    print(f"  FPS: {decoder.fps}")
    print(f"  Total frames: {decoder.total_frames}")

    # Read back frame
    decoded_frame = decoder.get_frame(0)
    print(f"  Decoded polygons: {len(decoded_frame.polygons)}")
    decoder.close()

    # Render decoded frame
    decoded_rendered = render_frame(decoded_frame, 320, 200)
    cv2.imwrite('test_decoded.png', cv2.cvtColor(decoded_rendered, cv2.COLOR_RGB2BGR))
    print("Saved decoded output: test_decoded.png")

    # Verify roundtrip
    if np.array_equal(rendered, decoded_rendered):
        print("\n✓ Encode/decode roundtrip successful!")
    else:
        diff = np.abs(rendered.astype(int) - decoded_rendered.astype(int)).sum()
        print(f"\n⚠ Minor differences in roundtrip (total diff: {diff})")

    print("\n=== Test Complete ===")


if __name__ == '__main__':
    main()
