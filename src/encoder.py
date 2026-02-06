"""
.roto format encoder - converts polygon frames to binary format.
"""
import struct
import io
from typing import List, BinaryIO
import numpy as np

from .polygon import Polygon, Frame, PolygonDelta, compute_frame_delta


# Magic number and version
ROTO_MAGIC = b'ROTO'
ROTO_VERSION = 0x0001

# Frame types
FRAME_KEYFRAME = 0x00
FRAME_DELTA = 0x01

# Delta opcodes
OP_NOOP = 0x00
OP_MOVE = 0x01
OP_MORPH = 0x02
OP_RECOLOR = 0x03
OP_BIRTH = 0x04
OP_DEATH = 0x05


class RotoEncoder:
    """Encoder for .roto polygon animation format."""

    def __init__(self, width: int = 320, height: int = 200, fps: float = 24.0):
        self.width = width
        self.height = height
        self.fps = fps
        self.frames: List[Frame] = []
        self.keyframe_interval = 48  # Keyframe every 2 seconds at 24fps

    def add_frame(self, frame: Frame):
        """Add a frame to the animation."""
        self.frames.append(frame)

    def encode(self, output: BinaryIO):
        """
        Encode all frames to binary .roto format.

        Args:
            output: Binary file object to write to
        """
        # Determine keyframe positions
        keyframe_indices = list(range(0, len(self.frames), self.keyframe_interval))
        if len(self.frames) - 1 not in keyframe_indices:
            keyframe_indices.append(len(self.frames) - 1)

        # Build frame data in memory first to calculate offsets
        frame_data = io.BytesIO()
        keyframe_offsets = []

        prev_frame = None
        for i, frame in enumerate(self.frames):
            is_keyframe = i in keyframe_indices

            if is_keyframe:
                keyframe_offsets.append((i, frame_data.tell()))
                self._encode_keyframe(frame, frame_data)
                prev_frame = frame
            else:
                self._encode_delta_frame(prev_frame, frame, frame_data)
                prev_frame = frame

        # Write header
        frame_data_bytes = frame_data.getvalue()
        header_size = 0x20
        keyframe_index_size = len(keyframe_offsets) * 8

        output.write(ROTO_MAGIC)
        output.write(struct.pack('<H', ROTO_VERSION))
        output.write(struct.pack('<H', self.width))
        output.write(struct.pack('<H', self.height))
        output.write(struct.pack('<H', int(self.fps * 100)))
        output.write(struct.pack('<I', len(self.frames)))
        output.write(struct.pack('<I', len(keyframe_offsets)))
        output.write(struct.pack('<I', header_size))  # Offset to keyframe index
        output.write(struct.pack('<I', header_size + keyframe_index_size))  # Offset to frame data
        output.write(struct.pack('<I', 0))  # Audio offset (not implemented)

        # Write keyframe index
        for frame_idx, offset in keyframe_offsets:
            output.write(struct.pack('<I', frame_idx))
            output.write(struct.pack('<I', offset))

        # Write frame data
        output.write(frame_data_bytes)

    def _encode_keyframe(self, frame: Frame, output: BinaryIO):
        """Encode a keyframe with full polygon data."""
        output.write(struct.pack('B', FRAME_KEYFRAME))
        output.write(struct.pack('<H', len(frame.polygons)))

        # Write palette (16 colors × 3 bytes)
        if frame.palette is not None:
            output.write(frame.palette.tobytes())
        else:
            output.write(bytes(48))  # Default black palette

        # Write each polygon
        for polygon in frame.polygons:
            self._encode_polygon(polygon, output)

    def _encode_polygon(self, polygon: Polygon, output: BinaryIO):
        """Encode a single polygon."""
        output.write(struct.pack('<H', polygon.id))

        # Handle large polygons by splitting or truncating
        vertex_count = min(polygon.vertex_count, 255)
        output.write(struct.pack('B', vertex_count))
        output.write(struct.pack('B', polygon.color_index & 0x0F))

        # Encode vertices (2 bytes each: 10-bit x + 6-bit padding, or packed)
        # For simplicity, using 2 bytes per coordinate (4 bytes per vertex)
        for x, y in polygon.vertices[:vertex_count]:
            output.write(struct.pack('<hh', int(x), int(y)))

    def _encode_delta_frame(self, prev_frame: Frame, curr_frame: Frame, output: BinaryIO):
        """Encode a delta frame with only changes from previous frame."""
        deltas = compute_frame_delta(prev_frame, curr_frame)

        output.write(struct.pack('B', FRAME_DELTA))
        output.write(struct.pack('<H', len(deltas)))

        for delta in deltas:
            self._encode_delta(delta, output)

    def _encode_delta(self, delta: PolygonDelta, output: BinaryIO):
        """Encode a single delta operation."""
        if delta.operation == 'move':
            output.write(struct.pack('B', OP_MOVE))
            output.write(struct.pack('<H', delta.polygon_id))
            output.write(struct.pack('<hh', delta.dx, delta.dy))

        elif delta.operation == 'morph':
            output.write(struct.pack('B', OP_MORPH))
            output.write(struct.pack('<H', delta.polygon_id))
            vertex_count = len(delta.vertex_deltas)
            output.write(struct.pack('B', vertex_count))
            # Pack vertex deltas as signed bytes
            for dx, dy in delta.vertex_deltas:
                output.write(struct.pack('bb', int(dx), int(dy)))

        elif delta.operation == 'recolor':
            output.write(struct.pack('B', OP_RECOLOR))
            output.write(struct.pack('<H', delta.polygon_id))
            output.write(struct.pack('B', delta.new_color))

        elif delta.operation == 'birth':
            output.write(struct.pack('B', OP_BIRTH))
            self._encode_polygon(delta.new_polygon, output)

        elif delta.operation == 'death':
            output.write(struct.pack('B', OP_DEATH))
            output.write(struct.pack('<H', delta.polygon_id))


def encode_to_file(frames: List[Frame], filepath: str,
                   width: int = 320, height: int = 200, fps: float = 24.0):
    """
    Convenience function to encode frames to a .roto file.

    Args:
        frames: List of Frame objects
        filepath: Output file path
        width: Video width
        height: Video height
        fps: Frames per second
    """
    encoder = RotoEncoder(width, height, fps)
    for frame in frames:
        encoder.add_frame(frame)

    with open(filepath, 'wb') as f:
        encoder.encode(f)
