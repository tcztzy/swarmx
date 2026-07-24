#!/usr/bin/env python3
"""Rebuild the SwarmX SVG from the raster design with OpenCV.

Run with:
  uv run --with opencv-contrib-python-headless --with numpy \
    python scripts/rebuild-icon.py \
    --input assets/swarmx-icon-concept-tech.png \
    --output packages/desktop/src/renderer/public/app-icon.svg \
    --typescript-output packages/desktop/src/renderer/src/app-icon-data.ts \
    --debug-dir /tmp/swarmx-icon-cv
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SVG_SIZE = 1024
RIBBON_GAP = 6
RIBBON_SCALE = 1.1
# Segment order is P0-P2, P2-P3, P3-P4, P4-P5. Read in the visual
# circulation direction (P5 to P0), the shared template thickens gradually.
RIBBON_SEGMENT_WIDTHS = (70.0, 60.0, 50.0, 40.0)
# Positive values move the shared ribbon template down; negative values move it up.
TEMPLATE_RIBBON_Y_OFFSET = 28
NEIGHBORS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class Component:
    label: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    mask: np.ndarray


@dataclass(frozen=True)
class Geometry:
    source_size: tuple[int, int]
    ribbon_observed_points: np.ndarray
    ribbon_outline_points: np.ndarray
    ribbon_source_points: np.ndarray
    ribbon_svg_points: np.ndarray
    ribbon_polygon_svg_points: np.ndarray
    ribbon_segment_widths: tuple[float, ...]
    ribbon_clearance: float
    center_inner_radii: tuple[float, float]
    background_box: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--typescript-output", type=Path)
    parser.add_argument("--debug-dir", type=Path)
    return parser.parse_args()


def saturated_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 70)).astype(np.uint8) * 255


def connected_components(mask: np.ndarray) -> list[Component]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    components: list[Component] = []
    for label in range(1, count):
        components.append(
            Component(
                label=label,
                area=int(stats[label, cv2.CC_STAT_AREA]),
                bbox=(
                    int(stats[label, 0]),
                    int(stats[label, 1]),
                    int(stats[label, 2]),
                    int(stats[label, 3]),
                ),
                centroid=(
                    float(centroids[label, 0]),
                    float(centroids[label, 1]),
                ),
                mask=((labels == label).astype(np.uint8) * 255),
            )
        )
    return components


def classify_components(
    components: list[Component], image_area: int
) -> tuple[Component, list[Component]]:
    large = [
        component for component in components if component.area > image_area * 0.04
    ]
    if len(large) != 4:
        raise ValueError(f"expected 4 large colored components, found {len(large)}")

    frame = max(large, key=lambda component: component.bbox[3])
    ribbons = [component for component in large if component.label != frame.label]
    ribbons.sort(key=lambda component: component.centroid[1])

    if len(ribbons) != 3:
        raise ValueError(f"expected 3 ribbons, found {len(ribbons)}")
    return frame, ribbons


def skeleton_longest_path(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    skeleton = cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    nodes: set[tuple[int, int]] = {
        (int(node[0]), int(node[1])) for node in np.argwhere(skeleton > 0)
    }
    endpoints = [
        point
        for point in nodes
        if sum((point[0] + dy, point[1] + dx) in nodes for dy, dx in NEIGHBORS) == 1
    ]
    if len(endpoints) < 2:
        raise ValueError("ribbon skeleton does not have two endpoints")

    longest: list[tuple[int, int]] = []
    for start in endpoints:
        queue = deque([start])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        distance = {start: 0}
        while queue:
            point = queue.popleft()
            for dy, dx in NEIGHBORS:
                neighbor = (point[0] + dy, point[1] + dx)
                if neighbor in nodes and neighbor not in parent:
                    parent[neighbor] = point
                    distance[neighbor] = distance[point] + 1
                    queue.append(neighbor)

        end = max(endpoints, key=lambda point: distance.get(point, -1))
        path: list[tuple[int, int]] = []
        current: tuple[int, int] | None = end
        while current is not None:
            path.append((current[1], current[0]))
            current = parent.get(current)
        if len(path) > len(longest):
            longest = path[::-1]

    return np.asarray(longest, dtype=np.float32), skeleton


def approximate_path(
    path: np.ndarray, source_width: int, target_vertices: int = 6
) -> np.ndarray:
    preferred_epsilon = source_width * 0.016
    candidates: list[tuple[float, np.ndarray]] = []
    for epsilon in np.linspace(source_width * 0.005, source_width * 0.03, 151):
        approximation = cv2.approxPolyDP(
            path.reshape(-1, 1, 2), float(epsilon), False
        ).reshape(-1, 2)
        if len(approximation) == target_vertices:
            candidates.append((abs(epsilon - preferred_epsilon), approximation))

    if not candidates:
        raise ValueError(
            f"could not reduce ribbon skeleton to {target_vertices} points"
        )
    points = min(candidates, key=lambda candidate: candidate[0])[1]

    # The canonical path starts at the lower-left outer tail and ends at the
    # inner-right tail.
    if points[0, 0] > points[-1, 0]:
        points = points[::-1]
    return points


def ordered_ribbon_outline(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    points = polygon_with_vertices(contour, 10).astype(np.float64)

    rightmost = np.flatnonzero(points[:, 0] == points[:, 0].max())
    start = int(rightmost[np.argmin(points[rightmost, 1])])
    points = np.roll(points, -start, axis=0)
    if points[1, 1] > points[-1, 1]:
        points = np.concatenate([points[:1], points[:0:-1]])
    return points


def edge_support(
    points: np.ndarray, first: int, second: int, normal: np.ndarray
) -> float:
    return float(np.mean(points[[first, second]] @ normal))


def intersect_lines(
    first_normal: np.ndarray,
    first_support: float,
    second_normal: np.ndarray,
    second_support: float,
) -> np.ndarray:
    return np.linalg.solve(
        np.stack([first_normal, second_normal]),
        np.asarray([first_support, second_support]),
    )


def constrain_ribbon_path(
    mask: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the exact centerline and the two oblique cut lines.

    The ten-point outline consists of paired boundaries around four fixed
    directions: vertical, -30, +30, vertical. Averaging each boundary pair
    gives the centerline. The start and end cuts are fixed at +30 and -30,
    creating the cyclic continuation into the rotated ribbon copies.
    """
    if observed.shape != (6, 2):
        raise ValueError(f"expected six ribbon points, found {observed.shape}")

    outline = ordered_ribbon_outline(mask)
    root_three_over_two = math.sqrt(3) / 2
    up_right_normal = np.asarray([0.5, root_three_over_two])
    down_right_normal = np.asarray([-0.5, root_three_over_two])

    left_x = float(
        np.mean(
            [
                np.mean(outline[[2, 3], 0]),
                np.mean(outline[[4, 5], 0]),
            ]
        )
    )
    right_x = float(
        np.mean(
            [
                np.mean(outline[[9, 0], 0]),
                np.mean(outline[[7, 8], 0]),
            ]
        )
    )
    up_right_support = float(
        np.mean(
            [
                edge_support(outline, 1, 2, up_right_normal),
                edge_support(outline, 5, 6, up_right_normal),
            ]
        )
    )
    down_right_support = float(
        np.mean(
            [
                edge_support(outline, 0, 1, down_right_normal),
                edge_support(outline, 6, 7, down_right_normal),
            ]
        )
    )
    start_cap_support = edge_support(outline, 3, 4, down_right_normal)
    end_cap_support = edge_support(outline, 8, 9, up_right_normal)

    p0 = intersect_lines(
        np.asarray([1.0, 0.0]),
        left_x,
        down_right_normal,
        start_cap_support,
    )
    p2 = intersect_lines(
        np.asarray([1.0, 0.0]),
        left_x,
        up_right_normal,
        up_right_support,
    )
    p3 = intersect_lines(
        up_right_normal,
        up_right_support,
        down_right_normal,
        down_right_support,
    )
    p4 = intersect_lines(
        np.asarray([1.0, 0.0]),
        right_x,
        down_right_normal,
        down_right_support,
    )
    p5 = intersect_lines(
        np.asarray([1.0, 0.0]),
        right_x,
        up_right_normal,
        end_cap_support,
    )
    p1 = np.asarray([left_x, observed[1, 1]])
    centerline = np.stack([p0, p1, p2, p3, p4, p5])
    return centerline, outline


def interior_angle(before: np.ndarray, vertex: np.ndarray, after: np.ndarray) -> float:
    incoming = before - vertex
    outgoing = after - vertex
    cosine = np.dot(incoming, outgoing) / (
        np.linalg.norm(incoming) * np.linalg.norm(outgoing)
    )
    return math.degrees(math.acos(float(np.clip(cosine, -1, 1))))


def rotate_vector(vector: np.ndarray, degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    return rotation @ vector


def parallel_error(first: np.ndarray, second: np.ndarray) -> float:
    determinant = first[0] * second[1] - first[1] * second[0]
    return abs(float(determinant)) / (
        float(np.linalg.norm(first)) * float(np.linalg.norm(second))
    )


def validate_ribbon_constraints(points: np.ndarray) -> None:
    if not np.allclose(points[:3, 0], points[0, 0], atol=1e-9):
        raise ValueError("P0, P1, and P2 must share one x coordinate")
    if not math.isclose(points[4, 0], points[5, 0], abs_tol=1e-9):
        raise ValueError("P4 and P5 must share one x coordinate")

    angles = [
        interior_angle(points[index - 1], points[index], points[index + 1])
        for index in (2, 3, 4)
    ]
    if not np.allclose(angles, 120, atol=1e-9):
        raise ValueError(f"ribbon angles must be 120 degrees, found {angles}")

    root_three_over_two = math.sqrt(3) / 2
    start_cap = np.asarray([root_three_over_two, 0.5])
    end_cap = np.asarray([root_three_over_two, -0.5])
    final_segment = points[5] - points[4]
    cycle_errors = [
        parallel_error(start_cap, rotate_vector(final_segment, 120)),
        parallel_error(end_cap, rotate_vector(final_segment, 240)),
    ]
    if not np.allclose(cycle_errors, 0, atol=1e-12):
        raise ValueError("ribbon end cuts must continue the two rotated final segments")


def polygon_with_vertices(contour: np.ndarray, vertices: int) -> np.ndarray:
    perimeter = cv2.arcLength(contour, True)
    candidates: list[tuple[float, np.ndarray]] = []
    for factor in np.linspace(0.005, 0.03, 101):
        approximation = cv2.approxPolyDP(
            contour, float(factor * perimeter), True
        ).reshape(-1, 2)
        if len(approximation) == vertices:
            candidates.append((abs(factor - 0.012), approximation))
    if not candidates:
        raise ValueError(f"could not approximate contour with {vertices} vertices")
    return min(candidates, key=lambda candidate: candidate[0])[1]


def detect_background_box(image: np.ndarray) -> tuple[int, int, int, int]:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (grayscale > 3).astype(np.uint8) * 255
    components = connected_components(mask)
    component = max(components, key=lambda item: item.area)
    return component.bbox


def scale_points_precisely(
    points: np.ndarray, source_size: tuple[int, int]
) -> np.ndarray:
    width, height = source_size
    scale = np.asarray([SVG_SIZE / width, SVG_SIZE / height])
    return points * scale


def rotate_points(points: np.ndarray, degrees: float, center: np.ndarray) -> np.ndarray:
    angle = math.radians(degrees)
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    return center + (points - center) @ rotation.T


def scale_from_center(
    points: np.ndarray, factor: float, center: np.ndarray
) -> np.ndarray:
    return center + (points - center) * factor


def point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    segment = end - start
    amount = float(
        np.clip(
            np.dot(point - start, segment) / np.dot(segment, segment),
            0,
            1,
        )
    )
    return float(np.linalg.norm(point - (start + amount * segment)))


def cross_product(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    first = first_end - first_start
    second = second_end - second_start
    denominator = cross_product(first, second)
    if not math.isclose(denominator, 0, abs_tol=1e-12):
        delta = second_start - first_start
        first_amount = cross_product(delta, second) / denominator
        second_amount = cross_product(delta, first) / denominator
        if 0 <= first_amount <= 1 and 0 <= second_amount <= 1:
            return 0

    return min(
        point_segment_distance(first_start, second_start, second_end),
        point_segment_distance(first_end, second_start, second_end),
        point_segment_distance(second_start, first_start, first_end),
        point_segment_distance(second_end, first_start, first_end),
    )


def polygon_distance(first: np.ndarray, second: np.ndarray) -> float:
    if (
        cv2.pointPolygonTest(second.astype(np.float32), tuple(first[0]), False) >= 0
        or cv2.pointPolygonTest(first.astype(np.float32), tuple(second[0]), False) >= 0
    ):
        return 0

    distances = [
        segment_distance(
            first[first_index],
            first[(first_index + 1) % len(first)],
            second[second_index],
            second[(second_index + 1) % len(second)],
        )
        for first_index in range(len(first))
        for second_index in range(len(second))
    ]
    return min(distances)


def variable_width_ribbon_outline(
    points: np.ndarray, segment_widths: tuple[float, ...]
) -> np.ndarray:
    """Build one filled ribbon from the exact four-segment centerline."""
    centerline = points[[0, 2, 3, 4, 5]]
    if len(segment_widths) != len(centerline) - 1:
        raise ValueError("one width is required for each ribbon segment")

    directions = np.diff(centerline, axis=0)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    half_widths = np.asarray(segment_widths) / 2

    root_three_over_two = math.sqrt(3) / 2
    start_cap_normal = np.asarray([-0.5, root_three_over_two])
    end_cap_normal = np.asarray([0.5, root_three_over_two])

    left = [
        intersect_lines(
            normals[0],
            float(np.dot(normals[0], centerline[0]) + half_widths[0]),
            start_cap_normal,
            float(np.dot(start_cap_normal, centerline[0])),
        )
    ]
    right = [
        intersect_lines(
            normals[0],
            float(np.dot(normals[0], centerline[0]) - half_widths[0]),
            start_cap_normal,
            float(np.dot(start_cap_normal, centerline[0])),
        )
    ]

    for index in range(1, len(centerline) - 1):
        vertex = centerline[index]
        left.append(
            intersect_lines(
                normals[index - 1],
                float(np.dot(normals[index - 1], vertex) + half_widths[index - 1]),
                normals[index],
                float(np.dot(normals[index], vertex) + half_widths[index]),
            )
        )
        right.append(
            intersect_lines(
                normals[index - 1],
                float(np.dot(normals[index - 1], vertex) - half_widths[index - 1]),
                normals[index],
                float(np.dot(normals[index], vertex) - half_widths[index]),
            )
        )

    left.append(
        intersect_lines(
            normals[-1],
            float(np.dot(normals[-1], centerline[-1]) + half_widths[-1]),
            end_cap_normal,
            float(np.dot(end_cap_normal, centerline[-1])),
        )
    )
    right.append(
        intersect_lines(
            normals[-1],
            float(np.dot(normals[-1], centerline[-1]) - half_widths[-1]),
            end_cap_normal,
            float(np.dot(end_cap_normal, centerline[-1])),
        )
    )
    return np.asarray([*left, *reversed(right)])


def center_inner_radii(
    points: np.ndarray, segment_widths: tuple[float, ...]
) -> tuple[float, float]:
    center = np.asarray([SVG_SIZE / 2, SVG_SIZE / 2])
    centerline = points[[0, 2, 3, 4, 5]]
    radii = []
    for index in (2, 3):
        start = centerline[index]
        end = centerline[index + 1]
        distance = abs(cross_product(end - start, center - start)) / float(
            np.linalg.norm(end - start)
        )
        radii.append(distance - segment_widths[index] / 2)
    return (radii[0], radii[1])


def extract_geometry(image: np.ndarray) -> tuple[Geometry, dict[str, np.ndarray]]:
    height, width = image.shape[:2]
    color_mask = saturated_mask(image)
    components = connected_components(color_mask)
    frame, ribbons = classify_components(components, width * height)

    canonical = ribbons[0]
    ribbon_path, ribbon_skeleton = skeleton_longest_path(canonical.mask)
    ribbon_observed_points = approximate_path(ribbon_path, width)
    ribbon_source_points, ribbon_outline_points = constrain_ribbon_path(
        canonical.mask, ribbon_observed_points
    )
    validate_ribbon_constraints(ribbon_source_points)
    ribbon_svg_points = scale_points_precisely(ribbon_source_points, (width, height))
    ribbon_svg_points[:, 1] += TEMPLATE_RIBBON_Y_OFFSET
    center = np.asarray([SVG_SIZE / 2, SVG_SIZE / 2])
    ribbon_svg_points = scale_from_center(ribbon_svg_points, RIBBON_SCALE, center)
    ribbon_segment_widths = tuple(
        round(width * RIBBON_SCALE, 6) for width in RIBBON_SEGMENT_WIDTHS
    )
    ribbon_polygon_svg_points = variable_width_ribbon_outline(
        ribbon_svg_points, ribbon_segment_widths
    )
    ribbon_clearance = polygon_distance(
        ribbon_polygon_svg_points,
        rotate_points(ribbon_polygon_svg_points, 120, center),
    )
    inner_radii = center_inner_radii(ribbon_svg_points, ribbon_segment_widths)

    box = detect_background_box(image)
    background_box = (
        round(box[0] * SVG_SIZE / width),
        round(box[1] * SVG_SIZE / height),
        round(box[2] * SVG_SIZE / width),
        round(box[3] * SVG_SIZE / height),
    )
    geometry = Geometry(
        source_size=(width, height),
        ribbon_observed_points=ribbon_observed_points,
        ribbon_outline_points=ribbon_outline_points,
        ribbon_source_points=ribbon_source_points,
        ribbon_svg_points=ribbon_svg_points,
        ribbon_polygon_svg_points=ribbon_polygon_svg_points,
        ribbon_segment_widths=ribbon_segment_widths,
        ribbon_clearance=ribbon_clearance,
        center_inner_radii=inner_radii,
        background_box=background_box,
    )
    masks = {
        "all": color_mask,
        "frame": frame.mask,
        "cyan": ribbons[0].mask,
        "blue": ribbons[1].mask,
        "violet": ribbons[2].mask,
        "ribbon_skeleton": ribbon_skeleton,
    }
    return geometry, masks


def svg_path(points: np.ndarray, *, close: bool = False) -> str:
    def format_number(value: float) -> str:
        rounded = round(float(value), 6)
        if rounded.is_integer():
            return str(int(rounded))
        return f"{rounded:.6f}".rstrip("0").rstrip(".")

    coordinates = " ".join(f"{format_number(x)} {format_number(y)}" for x, y in points)
    return f"M{coordinates}{'Z' if close else ''}"


def render_svg(
    geometry: Geometry,
    *,
    include_background: bool = True,
    grayscale: bool = False,
) -> str:
    ribbon = svg_path(geometry.ribbon_polygon_svg_points, close=True)
    background_x, background_y, background_width, background_height = (
        geometry.background_box
    )
    if grayscale:
        background_colors = ("#282828", "#151515", "#080808")
        light_colors = ("#f5f5f5", "#b5b5b5", "#6d6d6d")
        dark_colors = ("#d8d8d8", "#8d8d8d", "#474747")
        description = "SwarmX grayscale icon with three circulating folded ribbons."
    else:
        background_colors = ("#111a38", "#080e22", "#03050c")
        light_colors = ("#20f2e8", "#35b9ff", "#9a76ff")
        dark_colors = ("#19cfe8", "#2365f2", "#6548ee")
        description = "SwarmX application icon with three circulating folded ribbons."

    if not include_background:
        description = description.removesuffix(".") + " on a transparent background."
    background_definition = (
        f"""    <radialGradient id="background" cx="50%" cy="44%" r="74%">
      <stop offset="0" stop-color="{background_colors[0]}"/>
      <stop offset="0.6" stop-color="{background_colors[1]}"/>
      <stop offset="1" stop-color="{background_colors[2]}"/>
    </radialGradient>
"""
        if include_background
        else ""
    )
    background_panel = (
        f"""  <rect
    id="background-panel"
    x="{background_x}"
    y="{background_y}"
    width="{background_width}"
    height="{background_height}"
    rx="220"
    fill="url(#background)"
  />

"""
        if include_background
        else ""
    )

    return f"""<svg
  xmlns="http://www.w3.org/2000/svg"
  width="{SVG_SIZE}"
  height="{SVG_SIZE}"
  viewBox="0 0 {SVG_SIZE} {SVG_SIZE}"
  role="img"
  aria-labelledby="swarmx-title swarmx-description"
>
  <title id="swarmx-title">SwarmX</title>
  <desc id="swarmx-description">
    {description}
  </desc>

  <defs>
{background_definition}\
    <linearGradient id="ribbon-gradient-0" x1="0" y1="0" x2="1" y2="1">
      <stop class="ribbon-light phase-0" offset="0" color="{light_colors[0]}" stop-color="currentColor"/>
      <stop class="ribbon-dark phase-0" offset="1" color="{dark_colors[0]}" stop-color="currentColor"/>
    </linearGradient>
    <linearGradient id="ribbon-gradient-1" x1="0" y1="0" x2="1" y2="1">
      <stop class="ribbon-light phase-1" offset="0" color="{light_colors[1]}" stop-color="currentColor"/>
      <stop class="ribbon-dark phase-1" offset="1" color="{dark_colors[1]}" stop-color="currentColor"/>
    </linearGradient>
    <linearGradient id="ribbon-gradient-2" x1="0" y1="0" x2="1" y2="1">
      <stop class="ribbon-light phase-2" offset="0" color="{light_colors[2]}" stop-color="currentColor"/>
      <stop class="ribbon-dark phase-2" offset="1" color="{dark_colors[2]}" stop-color="currentColor"/>
    </linearGradient>
    <path id="ribbon" d="{ribbon}"/>
  </defs>

  <style>
    :root {{
      --swarm-cycle: 3600ms;
    }}

    .ribbon-light,
    .ribbon-dark {{
      animation-duration: var(--swarm-cycle);
      animation-iteration-count: infinite;
      animation-timing-function: linear;
    }}

    .ribbon-light {{
      animation-name: cycle-light;
    }}

    .ribbon-dark {{
      animation-name: cycle-dark;
    }}

    .phase-1 {{
      animation-delay: -1200ms;
    }}

    .phase-2 {{
      animation-delay: -2400ms;
    }}

    @keyframes cycle-light {{
      0%,
      100% {{
        color: {light_colors[0]};
      }}

      33.333% {{
        color: {light_colors[1]};
      }}

      66.667% {{
        color: {light_colors[2]};
      }}
    }}

    @keyframes cycle-dark {{
      0%,
      100% {{
        color: {dark_colors[0]};
      }}

      33.333% {{
        color: {dark_colors[1]};
      }}

      66.667% {{
        color: {dark_colors[2]};
      }}
    }}

    @media (prefers-reduced-motion: reduce) {{
      .ribbon-light,
      .ribbon-dark {{
        animation: none;
      }}
    }}
  </style>

{background_panel}\
  <g id="ribbons">
    <use
      id="ribbon-0"
      class="ribbon"
      href="#ribbon"
      fill="url(#ribbon-gradient-0)"
    />
    <use
      id="ribbon-1"
      class="ribbon"
      href="#ribbon"
      fill="url(#ribbon-gradient-1)"
      transform="rotate(120 512 512)"
    />
    <use
      id="ribbon-2"
      class="ribbon"
      href="#ribbon"
      fill="url(#ribbon-gradient-2)"
      transform="rotate(240 512 512)"
    />
  </g>
</svg>
"""


def render_monochrome_svg(geometry: Geometry, *, color: str, label: str) -> str:
    ribbon = svg_path(geometry.ribbon_polygon_svg_points, close=True)
    return f"""<svg
  xmlns="http://www.w3.org/2000/svg"
  width="{SVG_SIZE}"
  height="{SVG_SIZE}"
  viewBox="0 0 {SVG_SIZE} {SVG_SIZE}"
  role="img"
  aria-labelledby="swarmx-title swarmx-description"
>
  <title id="swarmx-title">SwarmX</title>
  <desc id="swarmx-description">
    SwarmX {label} monochrome icon on a transparent background.
  </desc>

  <defs>
    <path id="ribbon" d="{ribbon}"/>
  </defs>

  <g id="ribbons" fill="{color}">
    <use id="ribbon-0" href="#ribbon"/>
    <use id="ribbon-1" href="#ribbon" transform="rotate(120 512 512)"/>
    <use id="ribbon-2" href="#ribbon" transform="rotate(240 512 512)"/>
  </g>
</svg>
"""


def render_svg_variants(geometry: Geometry, output: Path) -> dict[Path, str]:
    stem = output.stem
    return {
        output.with_name(f"{stem}-transparent.svg"): render_svg(
            geometry, include_background=False
        ),
        output.with_name(f"{stem}-grayscale.svg"): render_svg(geometry, grayscale=True),
        output.with_name(f"{stem}-monochrome-light.svg"): (
            render_monochrome_svg(geometry, color="#ffffff", label="white")
        ),
        output.with_name(f"{stem}-monochrome-dark.svg"): (
            render_monochrome_svg(geometry, color="#0a0a0a", label="black")
        ),
    }


def render_typescript(svg: str) -> str:
    escaped_svg = (
        svg.rstrip().replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    )
    return (
        f"const APP_ICON_SVG = `{escaped_svg}`;\n\n"
        "export const APP_ICON_URL = "
        "`data:image/svg+xml;charset=utf-8,${encodeURIComponent(APP_ICON_SVG)}`;\n"
    )


def save_debug(
    directory: Path,
    image: np.ndarray,
    geometry: Geometry,
    masks: dict[str, np.ndarray],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        cv2.imwrite(str(directory / f"{name}.png"), mask)

    overlay = image.copy()
    cv2.polylines(
        overlay,
        [np.rint(geometry.ribbon_observed_points).astype(np.int32)],
        False,
        (0, 215, 255),
        3,
    )
    cv2.polylines(
        overlay,
        [np.rint(geometry.ribbon_outline_points).astype(np.int32)],
        True,
        (255, 255, 255),
        2,
    )
    cv2.polylines(
        overlay,
        [np.rint(geometry.ribbon_source_points).astype(np.int32)],
        False,
        (0, 255, 0),
        4,
    )
    for index, point in enumerate(geometry.ribbon_source_points):
        center = tuple(np.rint(point).astype(int))
        cv2.circle(overlay, center, 8, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            str(index),
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(directory / "geometry-overlay.png"), overlay)

    report = {
        "source_size": geometry.source_size,
        "ribbon_observed_points": geometry.ribbon_observed_points.tolist(),
        "ribbon_outline_points": geometry.ribbon_outline_points.tolist(),
        "ribbon_source_points": geometry.ribbon_source_points.tolist(),
        "ribbon_svg_points": geometry.ribbon_svg_points.tolist(),
        "ribbon_polygon_svg_points": (geometry.ribbon_polygon_svg_points.tolist()),
        "ribbon_angles": [
            interior_angle(
                geometry.ribbon_source_points[index - 1],
                geometry.ribbon_source_points[index],
                geometry.ribbon_source_points[index + 1],
            )
            for index in (2, 3, 4)
        ],
        "ribbon_cap_angles": {"start": 30, "end": -30},
        "cyclic_relations": {
            "start_cap_parallel_to": "rotate(120, P4-P5)",
            "end_cap_parallel_to": "rotate(240, P4-P5)",
        },
        "ribbon_segment_widths": geometry.ribbon_segment_widths,
        "ribbon_clearance": geometry.ribbon_clearance,
        "ribbon_gap_target_met": geometry.ribbon_clearance >= RIBBON_GAP,
        "center_inner_radii": geometry.center_inner_radii,
        "center_inner_radius_delta": abs(
            geometry.center_inner_radii[0] - geometry.center_inner_radii[1]
        ),
        "ribbon_scale": RIBBON_SCALE,
        "template_ribbon_y_offset": TEMPLATE_RIBBON_Y_OFFSET,
        "background_box": geometry.background_box,
    }
    (directory / "geometry.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(args.input)

    geometry, masks = extract_geometry(image)
    svg = render_svg(geometry)
    args.output.write_text(svg, encoding="utf-8")
    variants = render_svg_variants(geometry, args.output)
    for path, content in variants.items():
        path.write_text(content, encoding="utf-8")
    if args.typescript_output:
        args.typescript_output.write_text(render_typescript(svg), encoding="utf-8")
    if args.debug_dir:
        save_debug(args.debug_dir, image, geometry, masks)

    print(f"ribbon points: {np.round(geometry.ribbon_svg_points, 3).tolist()}")
    print(f"ribbon segment widths: {geometry.ribbon_segment_widths}")
    print(f"ribbon clearance: {geometry.ribbon_clearance:.3f}")
    print(f"ribbon gap target met: {geometry.ribbon_clearance >= RIBBON_GAP}")
    print(
        "center inner radii: "
        f"{tuple(round(value, 3) for value in geometry.center_inner_radii)}"
    )
    print(f"ribbon scale: {RIBBON_SCALE}")
    print(f"template ribbon y offset: {TEMPLATE_RIBBON_Y_OFFSET}")
    print(args.output)
    for path in variants:
        print(path)
    if args.typescript_output:
        print(args.typescript_output)


if __name__ == "__main__":
    main()
