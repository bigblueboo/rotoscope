#!/usr/bin/env python3
"""Fresh test without any caching issues."""
import sys
import importlib

# Remove any cached modules
for mod in list(sys.modules.keys()):
    if mod.startswith('src'):
        del sys.modules[mod]

import numpy as np
import cv2

# Now import fresh
sys.path.insert(0, '/home/cdeck/dev/rotoscope')
from src.pipeline import VideoToPolygonPipeline
from src.rasterize import render_frame

# Load a frame from the video
cap = cv2.VideoCapture('lester/results/youtube1/footage.mp4')
ret, frame = cap.read()
cap.release()

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process with pipeline
pipeline = VideoToPolygonPipeline(
    width=320,
    height=200,
    num_colors=16,
    style='another_world',
    min_polygon_area=30
)

result_frame, palette = pipeline.process_image(rgb)

print(f"Polygons: {len(result_frame.polygons)}")

total_vertices = 0
vertex_counts = []
for p in result_frame.polygons:
    total_vertices += p.vertex_count
    vertex_counts.append(p.vertex_count)

print(f"Total vertices: {total_vertices}")
print(f"Average vertices per polygon: {total_vertices / len(result_frame.polygons):.1f}")
print(f"Max vertices in a polygon: {max(vertex_counts)}")
print(f"Min vertices in a polygon: {min(vertex_counts)}")

# Vertex count distribution
from collections import Counter
dist = Counter(vertex_counts)
print("\nVertex count distribution:")
for count in sorted(dist.keys()):
    print(f"  {count} vertices: {dist[count]} polygons")

# Render and save
rendered = render_frame(result_frame, 320, 200)
cv2.imwrite('test_fresh_output.png', cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
print("\nSaved: test_fresh_output.png")
