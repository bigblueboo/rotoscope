#!/usr/bin/env python3
"""Debug script to check simplification."""
import numpy as np
import cv2
import sys
sys.path.insert(0, '/home/cdeck/dev/rotoscope')

from src.simplify import mask_to_polygons, simplify_contour, STYLE_PRESETS

# Load a frame from the video
cap = cv2.VideoCapture('lester/results/youtube1/footage.mp4')
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read frame")
    exit(1)

# Resize
frame = cv2.resize(frame, (320, 200))
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Quantize colors
from src.quantize import kmeans_quantize
indexed, palette = kmeans_quantize(rgb, 16)

# Process each color
print("Color region analysis:")
print("-" * 60)

total_vertices = 0
total_polygons = 0

for color_idx in range(16):
    mask = (indexed == color_idx).astype(np.uint8) * 255

    if mask.sum() == 0:
        continue

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if contour.ndim == 3:
            contour = contour.reshape(-1, 2)

        area = cv2.contourArea(contour.reshape(-1, 1, 2))
        if area < 30:
            continue

        original_vertices = len(contour)
        perimeter = cv2.arcLength(contour.reshape(-1, 1, 2), True)
        epsilon_ratio = 0.05
        epsilon = epsilon_ratio * perimeter

        # Simplify
        simplified = cv2.approxPolyDP(contour.reshape(-1, 1, 2), epsilon, True)
        simplified_vertices = len(simplified)

        total_polygons += 1
        total_vertices += simplified_vertices

        if original_vertices > 100 or simplified_vertices > 20:
            print(f"Color {color_idx:2d}: area={area:6.0f} perim={perimeter:6.0f} "
                  f"verts: {original_vertices:4d} -> {simplified_vertices:3d} (eps={epsilon:.1f})")

print("-" * 60)
print(f"Total: {total_polygons} polygons, {total_vertices} vertices")
print(f"Average: {total_vertices/total_polygons:.1f} vertices per polygon")
