#!/usr/bin/env python3
"""
Run both pipelines on test images and videos, saving comparisons.
"""
import sys, os, time
sys.path.insert(0, '/home/cdeck/dev/rotoscope')

# Force fresh imports
for mod in list(sys.modules.keys()):
    if mod.startswith('src'):
        del sys.modules[mod]

import numpy as np
import cv2

from src.pipeline import VideoToPolygonPipeline, SAM2Pipeline
from src.rasterize import render_frame, render_frame_with_outlines
from src.encoder import encode_to_file

INPUTS = '/home/cdeck/dev/rotoscope/samples/inputs'
OUTPUTS = '/home/cdeck/dev/rotoscope/samples/outputs'
SAM_CKPT = '/home/cdeck/dev/rotoscope/sam2_repo/checkpoints/sam2.1_hiera_tiny.pt'
SAM_CFG = 'configs/sam2.1/sam2.1_hiera_t.yaml'

os.makedirs(OUTPUTS, exist_ok=True)

# ============================================================
# IMAGE TESTS
# ============================================================

def test_image(name, path):
    """Process one image through both pipelines."""
    print(f'\n{"="*60}')
    print(f'IMAGE: {name}')
    print(f'{"="*60}')

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    print(f'  Source: {img.shape[1]}x{img.shape[0]}')

    # Save resized original
    orig = cv2.resize(img, (320, 200), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'{OUTPUTS}/{name}_original.png',
                cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))

    # ---- Colour-only pipeline ----
    t0 = time.time()
    color_pipe = VideoToPolygonPipeline(
        width=320, height=200, num_colors=16,
        style='another_world', min_polygon_area=30,
    )
    frame_c, palette_c = color_pipe.process_image(img)
    dt = time.time() - t0

    nv = sum(p.vertex_count for p in frame_c.polygons)
    print(f'  Colour pipeline: {len(frame_c.polygons)} polys, {nv} verts ({dt:.2f}s)')

    rendered_c = render_frame(frame_c, 320, 200)
    cv2.imwrite(f'{OUTPUTS}/{name}_color.png',
                cv2.cvtColor(rendered_c, cv2.COLOR_RGB2BGR))

    # with outlines
    rendered_co = render_frame_with_outlines(frame_c, 320, 200,
                                             outline_color=(255, 255, 255),
                                             outline_thickness=1)
    cv2.imwrite(f'{OUTPUTS}/{name}_color_outlines.png',
                cv2.cvtColor(rendered_co, cv2.COLOR_RGB2BGR))

    # .roto size
    encode_to_file([frame_c], f'{OUTPUTS}/{name}_color.roto', 320, 200)
    roto_sz = os.path.getsize(f'{OUTPUTS}/{name}_color.roto')
    print(f'  .roto: {roto_sz} bytes')

    # ---- SAM 2 pipeline ----
    t0 = time.time()
    sam_pipe = SAM2Pipeline(
        sam_checkpoint=SAM_CKPT, sam_config=SAM_CFG,
        width=320, height=200, style='another_world',
        min_polygon_area=50, max_objects=8,
        points_per_side=24, device='cpu',
    )
    frame_s, palette_s = sam_pipe.process_image(img)
    dt = time.time() - t0

    nv = sum(p.vertex_count for p in frame_s.polygons)
    print(f'  SAM 2 pipeline:  {len(frame_s.polygons)} polys, {nv} verts ({dt:.2f}s)')

    rendered_s = render_frame(frame_s, 320, 200)
    cv2.imwrite(f'{OUTPUTS}/{name}_sam2.png',
                cv2.cvtColor(rendered_s, cv2.COLOR_RGB2BGR))

    rendered_so = render_frame_with_outlines(frame_s, 320, 200,
                                             outline_color=(255, 255, 255),
                                             outline_thickness=1)
    cv2.imwrite(f'{OUTPUTS}/{name}_sam2_outlines.png',
                cv2.cvtColor(rendered_so, cv2.COLOR_RGB2BGR))

    encode_to_file([frame_s], f'{OUTPUTS}/{name}_sam2.roto', 320, 200)
    roto_sz = os.path.getsize(f'{OUTPUTS}/{name}_sam2.roto')
    print(f'  .roto: {roto_sz} bytes')


# ============================================================
# VIDEO TESTS (colour-only, fast)
# ============================================================

def test_video_color(name, path, max_frames=30, frame_skip=2):
    """Process a video through the colour pipeline."""
    print(f'\n{"="*60}')
    print(f'VIDEO (colour): {name}')
    print(f'{"="*60}')

    t0 = time.time()
    pipe = VideoToPolygonPipeline(
        width=320, height=200, num_colors=16,
        style='another_world', min_polygon_area=30,
    )
    frames = list(pipe.process_video(path, max_frames=max_frames, frame_skip=frame_skip))
    dt = time.time() - t0

    total_p = sum(len(f.polygons) for f in frames)
    total_v = sum(sum(p.vertex_count for p in f.polygons) for f in frames)
    print(f'  {len(frames)} frames, {total_p} polygons, {total_v} vertices ({dt:.1f}s)')

    # Save .roto
    encode_to_file(frames, f'{OUTPUTS}/{name}_color.roto', 320, 200, 12.0)
    roto_sz = os.path.getsize(f'{OUTPUTS}/{name}_color.roto')
    input_sz = os.path.getsize(path)
    print(f'  Input: {input_sz/1024:.0f} KB → .roto: {roto_sz/1024:.1f} KB ({input_sz/roto_sz:.0f}x)')

    # Save GIF
    from src.rasterize import save_animation_gif
    save_animation_gif(frames, f'{OUTPUTS}/{name}_color.gif', 320, 200, fps=12.0)
    gif_sz = os.path.getsize(f'{OUTPUTS}/{name}_color.gif')
    print(f'  GIF: {gif_sz/1024:.1f} KB')

    # Save first+last frame PNGs
    for idx in [0, len(frames)-1]:
        r = render_frame(frames[idx], 320, 200)
        cv2.imwrite(f'{OUTPUTS}/{name}_color_f{idx:02d}.png',
                    cv2.cvtColor(r, cv2.COLOR_RGB2BGR))


# ============================================================
# RUN
# ============================================================

if __name__ == '__main__':
    # Images
    for name, fn in [('landscape', 'landscape.jpg'),
                     ('city', 'city.jpg'),
                     ('portrait', 'portrait.jpg')]:
        test_image(name, f'{INPUTS}/{fn}')

    # Videos (colour-only for speed)
    for name, fn, mf, fs in [('walk', 'walk.mp4', 30, 2),
                              ('lester_yt1', '/home/cdeck/dev/rotoscope/lester/results/youtube1/footage.mp4', 50, 3)]:
        test_video_color(name, f'{INPUTS}/{fn}', max_frames=mf, frame_skip=fs)

    print(f'\n{"="*60}')
    print('ALL DONE. Outputs in samples/outputs/')
    print(f'{"="*60}')
