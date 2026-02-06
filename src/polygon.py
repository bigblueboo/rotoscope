"""
Polygon data structures and operations.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Polygon:
    """A single polygon with vertices and color."""
    id: int
    vertices: np.ndarray  # Shape: (N, 2) - x, y coordinates
    color_index: int  # Index into 16-color palette

    def __post_init__(self):
        if isinstance(self.vertices, list):
            self.vertices = np.array(self.vertices, dtype=np.int16)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    def bounds(self) -> Tuple[int, int, int, int]:
        """Return bounding box (x_min, y_min, x_max, y_max)."""
        if len(self.vertices) == 0:
            return (0, 0, 0, 0)
        x_min, y_min = self.vertices.min(axis=0)
        x_max, y_max = self.vertices.max(axis=0)
        return (int(x_min), int(y_min), int(x_max), int(y_max))

    def centroid(self) -> Tuple[float, float]:
        """Return the centroid of the polygon."""
        if len(self.vertices) == 0:
            return (0.0, 0.0)
        return tuple(self.vertices.mean(axis=0))

    def area(self) -> float:
        """Calculate polygon area using shoelace formula."""
        n = len(self.vertices)
        if n < 3:
            return 0.0
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def translate(self, dx: int, dy: int) -> 'Polygon':
        """Return a new polygon translated by (dx, dy)."""
        return Polygon(
            id=self.id,
            vertices=self.vertices + np.array([dx, dy]),
            color_index=self.color_index
        )

    def delta_from(self, other: 'Polygon') -> Optional[np.ndarray]:
        """
        Compute vertex deltas from another polygon.
        Returns None if vertex counts don't match.
        """
        if self.vertex_count != other.vertex_count:
            return None
        return self.vertices - other.vertices


@dataclass
class Frame:
    """A single frame containing polygons and palette."""
    index: int
    polygons: List[Polygon] = field(default_factory=list)
    palette: Optional[np.ndarray] = None  # Shape: (16, 3) RGB values

    def add_polygon(self, polygon: Polygon):
        self.polygons.append(polygon)

    def get_polygon(self, polygon_id: int) -> Optional[Polygon]:
        for p in self.polygons:
            if p.id == polygon_id:
                return p
        return None

    def polygon_ids(self) -> List[int]:
        return [p.id for p in self.polygons]


@dataclass
class PolygonDelta:
    """Represents a change to a polygon between frames."""
    polygon_id: int
    operation: str  # 'move', 'morph', 'recolor', 'birth', 'death'

    # For 'move': uniform translation
    dx: int = 0
    dy: int = 0

    # For 'morph': per-vertex deltas
    vertex_deltas: Optional[np.ndarray] = None

    # For 'recolor': new color index
    new_color: Optional[int] = None

    # For 'birth': full polygon data
    new_polygon: Optional[Polygon] = None


def compute_frame_delta(prev_frame: Frame, curr_frame: Frame) -> List[PolygonDelta]:
    """
    Compute the delta operations needed to transform prev_frame into curr_frame.
    """
    deltas = []
    prev_ids = set(prev_frame.polygon_ids())
    curr_ids = set(curr_frame.polygon_ids())

    # Deaths: polygons that disappeared
    for pid in prev_ids - curr_ids:
        deltas.append(PolygonDelta(polygon_id=pid, operation='death'))

    # Births: new polygons
    for pid in curr_ids - prev_ids:
        polygon = curr_frame.get_polygon(pid)
        deltas.append(PolygonDelta(
            polygon_id=pid,
            operation='birth',
            new_polygon=polygon
        ))

    # Changes: polygons that exist in both frames
    for pid in prev_ids & curr_ids:
        prev_poly = prev_frame.get_polygon(pid)
        curr_poly = curr_frame.get_polygon(pid)

        # Check for color change
        if prev_poly.color_index != curr_poly.color_index:
            deltas.append(PolygonDelta(
                polygon_id=pid,
                operation='recolor',
                new_color=curr_poly.color_index
            ))

        # Check for vertex changes
        if prev_poly.vertex_count == curr_poly.vertex_count:
            vertex_delta = curr_poly.delta_from(prev_poly)

            if vertex_delta is not None and not np.allclose(vertex_delta, 0):
                # Check if it's a uniform translation
                if np.all(vertex_delta == vertex_delta[0]):
                    dx, dy = vertex_delta[0]
                    deltas.append(PolygonDelta(
                        polygon_id=pid,
                        operation='move',
                        dx=int(dx),
                        dy=int(dy)
                    ))
                else:
                    deltas.append(PolygonDelta(
                        polygon_id=pid,
                        operation='morph',
                        vertex_deltas=vertex_delta.astype(np.int8)
                    ))
        else:
            # Vertex count changed - treat as death + birth
            deltas.append(PolygonDelta(polygon_id=pid, operation='death'))
            deltas.append(PolygonDelta(
                polygon_id=pid,
                operation='birth',
                new_polygon=curr_poly
            ))

    return deltas
