#!/usr/bin/env python3
"""
Demo script for the rotoscope pipeline.

Usage:
    python demo.py <input_image> [output_image]
    python demo.py --video <input_video> [output.roto]
"""
import sys
import argparse
from pathlib import Path


def demo_image(input_path: str, output_path: str = None,
               style: str = 'another_world',
               width: int = 320, height: int = 200,
               show_outlines: bool = False):
    """Process a single image and display/save results."""
    import numpy as np

    # Import our modules
    from src.pipeline import VideoToPolygonPipeline
    from src.rasterize import render_frame, render_frame_with_outlines

    # Try to load image
    try:
        import cv2
        img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    except ImportError:
        from PIL import Image
        img = np.array(Image.open(input_path).convert('RGB'))

    print(f"Input image: {img.shape}")

    # Process
    pipeline = VideoToPolygonPipeline(
        width=width,
        height=height,
        style=style,
        min_polygon_area=30
    )

    frame, palette = pipeline.process_image(img)

    print(f"Generated {len(frame.polygons)} polygons")
    print(f"Palette: {len(palette)} colors")

    # Calculate stats
    total_vertices = sum(p.vertex_count for p in frame.polygons)
    print(f"Total vertices: {total_vertices}")

    # Estimate compressed size
    # Rough estimate: 4 bytes per vertex + 4 bytes per polygon header
    estimated_bytes = total_vertices * 4 + len(frame.polygons) * 4 + 48  # palette
    print(f"Estimated frame size: {estimated_bytes} bytes ({estimated_bytes/1024:.1f} KB)")

    # Render
    if show_outlines:
        rendered = render_frame_with_outlines(frame, width, height)
    else:
        rendered = render_frame(frame, width, height)

    # Save or display
    if output_path:
        try:
            import cv2
            cv2.imwrite(output_path, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        except ImportError:
            from PIL import Image
            Image.fromarray(rendered).save(output_path)
        print(f"Saved to: {output_path}")
    else:
        # Try to display
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Original (resized)
            try:
                import cv2
                resized = cv2.resize(img, (width, height))
            except ImportError:
                from PIL import Image
                resized = np.array(Image.fromarray(img).resize((width, height)))

            axes[0].imshow(resized)
            axes[0].set_title(f'Original ({width}x{height})')
            axes[0].axis('off')

            # Polygon version
            axes[1].imshow(rendered)
            axes[1].set_title(f'Polygons: {len(frame.polygons)} ({total_vertices} vertices)')
            axes[1].axis('off')

            plt.tight_layout()
            plt.savefig('demo_output.png', dpi=150)
            print("Saved comparison to: demo_output.png")
            plt.show()

        except ImportError:
            print("matplotlib not available, saving to demo_output.png")
            try:
                import cv2
                cv2.imwrite('demo_output.png', cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
            except ImportError:
                from PIL import Image
                Image.fromarray(rendered).save('demo_output.png')

    return frame


def demo_video(input_path: str, output_path: str = None,
               style: str = 'another_world',
               width: int = 320, height: int = 200,
               max_frames: int = 100,
               frame_skip: int = 1):
    """Process a video and encode to .roto format."""
    from src.pipeline import VideoToPolygonPipeline
    from src.encoder import encode_to_file
    from src.rasterize import save_animation_gif

    print(f"Processing video: {input_path}")
    print(f"Style: {style}, Size: {width}x{height}")
    print(f"Max frames: {max_frames}, Skip: {frame_skip}")

    pipeline = VideoToPolygonPipeline(
        width=width,
        height=height,
        style=style,
        min_polygon_area=30
    )

    frames = []
    for i, frame in enumerate(pipeline.process_video(input_path, max_frames, frame_skip)):
        frames.append(frame)
        if (i + 1) % 10 == 0:
            print(f"  Processed frame {i + 1}...")

    print(f"Total frames: {len(frames)}")

    # Calculate stats
    total_polygons = sum(len(f.polygons) for f in frames)
    total_vertices = sum(sum(p.vertex_count for p in f.polygons) for f in frames)
    avg_polygons = total_polygons / len(frames) if frames else 0
    avg_vertices = total_vertices / len(frames) if frames else 0

    print(f"Average polygons per frame: {avg_polygons:.1f}")
    print(f"Average vertices per frame: {avg_vertices:.1f}")

    # Save
    if output_path:
        if output_path.endswith('.roto'):
            encode_to_file(frames, output_path, width, height, fps=24.0)
            file_size = Path(output_path).stat().st_size
            print(f"Saved .roto file: {output_path} ({file_size} bytes, {file_size/1024:.1f} KB)")
        elif output_path.endswith('.gif'):
            save_animation_gif(frames, output_path, width, height, fps=12.0)
            file_size = Path(output_path).stat().st_size
            print(f"Saved GIF: {output_path} ({file_size/1024:.1f} KB)")
        else:
            print(f"Unknown output format: {output_path}")
    else:
        # Default: save as GIF
        output_path = 'demo_output.gif'
        save_animation_gif(frames, output_path, width, height, fps=12.0)
        file_size = Path(output_path).stat().st_size
        print(f"Saved GIF: {output_path} ({file_size/1024:.1f} KB)")

    return frames


def main():
    parser = argparse.ArgumentParser(description='Rotoscope demo')
    parser.add_argument('input', help='Input image or video path')
    parser.add_argument('output', nargs='?', help='Output path')
    parser.add_argument('--video', action='store_true', help='Process as video')
    parser.add_argument('--style', default='another_world',
                       choices=['detailed', 'moderate', 'stylized', 'another_world', 'extreme'],
                       help='Simplification style')
    parser.add_argument('--width', type=int, default=320, help='Output width')
    parser.add_argument('--height', type=int, default=200, help='Output height')
    parser.add_argument('--max-frames', type=int, default=100, help='Max frames for video')
    parser.add_argument('--frame-skip', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--outlines', action='store_true', help='Show polygon outlines')

    args = parser.parse_args()

    if args.video or args.input.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        demo_video(
            args.input,
            args.output,
            style=args.style,
            width=args.width,
            height=args.height,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip
        )
    else:
        demo_image(
            args.input,
            args.output,
            style=args.style,
            width=args.width,
            height=args.height,
            show_outlines=args.outlines
        )


if __name__ == '__main__':
    main()
