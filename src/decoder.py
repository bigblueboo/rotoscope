"""
.roto format decoder - reads binary format and produces frames.
"""
import struct
from typing import List, Dict, BinaryIO, Optional, Iterator, Tuple
import numpy as np

from .polygon import Polygon, Frame


# Constants from encoder
ROTO_MAGIC = b'ROTO'
FRAME_KEYFRAME = 0x00
FRAME_DELTA = 0x01
OP_NOOP = 0x00
OP_MOVE = 0x01
OP_MORPH = 0x02
OP_RECOLOR = 0x03
OP_BIRTH = 0x04
OP_DEATH = 0x05


class RotoDecoder:
    """Decoder for .roto polygon animation format."""

    def __init__(self):
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.total_frames = 0
        self.keyframe_index: List[Tuple[int, int]] = []
        self.frame_data_offset = 0
        self._file: Optional[BinaryIO] = None
        self._current_frame: Optional[Frame] = None
        self._current_polygons: Dict[int, Polygon] = {}
        self._current_palette: Optional[np.ndarray] = None

    def open(self, filepath: str):
        """Open a .roto file for reading."""
        self._file = open(filepath, 'rb')
        self._read_header()

    def close(self):
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _read_header(self):
        """Read and parse the file header."""
        magic = self._file.read(4)
        if magic != ROTO_MAGIC:
            raise ValueError(f"Invalid .roto file: expected {ROTO_MAGIC}, got {magic}")

        version = struct.unpack('<H', self._file.read(2))[0]
        self.width = struct.unpack('<H', self._file.read(2))[0]
        self.height = struct.unpack('<H', self._file.read(2))[0]
        fps_encoded = struct.unpack('<H', self._file.read(2))[0]
        self.fps = fps_encoded / 100.0
        self.total_frames = struct.unpack('<I', self._file.read(4))[0]
        keyframe_count = struct.unpack('<I', self._file.read(4))[0]
        keyframe_index_offset = struct.unpack('<I', self._file.read(4))[0]
        self.frame_data_offset = struct.unpack('<I', self._file.read(4))[0]
        _audio_offset = struct.unpack('<I', self._file.read(4))[0]

        # Read keyframe index
        self._file.seek(keyframe_index_offset)
        self.keyframe_index = []
        for _ in range(keyframe_count):
            frame_idx = struct.unpack('<I', self._file.read(4))[0]
            offset = struct.unpack('<I', self._file.read(4))[0]
            self.keyframe_index.append((frame_idx, offset))

    def seek_frame(self, frame_idx: int):
        """
        Seek to a specific frame.

        For efficiency, seeks to the nearest keyframe and decodes forward.
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise ValueError(f"Frame {frame_idx} out of range [0, {self.total_frames})")

        # Find the nearest keyframe at or before this frame
        keyframe_idx = 0
        keyframe_offset = 0
        for kf_idx, kf_offset in self.keyframe_index:
            if kf_idx <= frame_idx:
                keyframe_idx = kf_idx
                keyframe_offset = kf_offset
            else:
                break

        # Seek to keyframe
        self._file.seek(self.frame_data_offset + keyframe_offset)
        self._current_polygons.clear()

        # Decode frames from keyframe to target
        for i in range(keyframe_idx, frame_idx + 1):
            self._decode_next_frame(i)

    def _decode_next_frame(self, frame_idx: int) -> Frame:
        """Decode the next frame from current file position."""
        frame_type = struct.unpack('B', self._file.read(1))[0]

        if frame_type == FRAME_KEYFRAME:
            return self._decode_keyframe(frame_idx)
        elif frame_type == FRAME_DELTA:
            return self._decode_delta_frame(frame_idx)
        else:
            raise ValueError(f"Unknown frame type: {frame_type}")

    def _decode_keyframe(self, frame_idx: int) -> Frame:
        """Decode a keyframe."""
        polygon_count = struct.unpack('<H', self._file.read(2))[0]

        # Read palette
        palette_data = self._file.read(48)
        palette = np.frombuffer(palette_data, dtype=np.uint8).reshape(16, 3)
        self._current_palette = palette

        # Clear and read polygons
        self._current_polygons.clear()
        for _ in range(polygon_count):
            polygon = self._decode_polygon()
            self._current_polygons[polygon.id] = polygon

        frame = Frame(
            index=frame_idx,
            polygons=list(self._current_polygons.values()),
            palette=palette.copy()
        )
        self._current_frame = frame
        return frame

    def _decode_polygon(self) -> Polygon:
        """Decode a single polygon."""
        polygon_id = struct.unpack('<H', self._file.read(2))[0]
        vertex_count = struct.unpack('B', self._file.read(1))[0]
        color_index = struct.unpack('B', self._file.read(1))[0] & 0x0F

        vertices = []
        for _ in range(vertex_count):
            x, y = struct.unpack('<hh', self._file.read(4))
            vertices.append([x, y])

        return Polygon(
            id=polygon_id,
            vertices=np.array(vertices, dtype=np.int16),
            color_index=color_index
        )

    def _decode_delta_frame(self, frame_idx: int) -> Frame:
        """Decode a delta frame."""
        operation_count = struct.unpack('<H', self._file.read(2))[0]

        for _ in range(operation_count):
            opcode = struct.unpack('B', self._file.read(1))[0]

            if opcode == OP_MOVE:
                polygon_id = struct.unpack('<H', self._file.read(2))[0]
                dx, dy = struct.unpack('<hh', self._file.read(4))
                if polygon_id in self._current_polygons:
                    poly = self._current_polygons[polygon_id]
                    self._current_polygons[polygon_id] = poly.translate(dx, dy)

            elif opcode == OP_MORPH:
                polygon_id = struct.unpack('<H', self._file.read(2))[0]
                vertex_count = struct.unpack('B', self._file.read(1))[0]
                deltas = []
                for _ in range(vertex_count):
                    dx, dy = struct.unpack('bb', self._file.read(2))
                    deltas.append([dx, dy])
                if polygon_id in self._current_polygons:
                    poly = self._current_polygons[polygon_id]
                    new_vertices = poly.vertices + np.array(deltas, dtype=np.int16)
                    self._current_polygons[polygon_id] = Polygon(
                        id=polygon_id,
                        vertices=new_vertices,
                        color_index=poly.color_index
                    )

            elif opcode == OP_RECOLOR:
                polygon_id = struct.unpack('<H', self._file.read(2))[0]
                new_color = struct.unpack('B', self._file.read(1))[0]
                if polygon_id in self._current_polygons:
                    poly = self._current_polygons[polygon_id]
                    self._current_polygons[polygon_id] = Polygon(
                        id=polygon_id,
                        vertices=poly.vertices.copy(),
                        color_index=new_color
                    )

            elif opcode == OP_BIRTH:
                polygon = self._decode_polygon()
                self._current_polygons[polygon.id] = polygon

            elif opcode == OP_DEATH:
                polygon_id = struct.unpack('<H', self._file.read(2))[0]
                self._current_polygons.pop(polygon_id, None)

        frame = Frame(
            index=frame_idx,
            polygons=list(self._current_polygons.values()),
            palette=self._current_palette.copy() if self._current_palette is not None else None
        )
        self._current_frame = frame
        return frame

    def frames(self) -> Iterator[Frame]:
        """Iterate through all frames."""
        self._file.seek(self.frame_data_offset)
        self._current_polygons.clear()

        for i in range(self.total_frames):
            yield self._decode_next_frame(i)

    def get_frame(self, frame_idx: int) -> Frame:
        """Get a specific frame by index."""
        self.seek_frame(frame_idx)
        return self._current_frame


def decode_file(filepath: str) -> List[Frame]:
    """
    Convenience function to decode all frames from a .roto file.

    Args:
        filepath: Path to .roto file

    Returns:
        List of all frames
    """
    with RotoDecoder() as decoder:
        decoder.open(filepath)
        return list(decoder.frames())
