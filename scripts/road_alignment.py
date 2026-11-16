"""Estimate a visible ground corridor from perspective and appearance across frames."""

from dataclasses import dataclass

import cv2
import numpy as np


# Densely sample near the lower edge, where the final tread must meet the footage.
ROWS = np.linspace(0.48, 0.98, 15)


@dataclass
class Road:
    # Each row is [image y, left edge x, right edge x], in normalized units.
    edges: np.ndarray

    def at(self, y):
        rows, left, right = self.edges.T
        return np.interp(y, rows, left), np.interp(y, rows, right)

    def texture_point(self):
        y = float(self.edges[-1, 0] * 0.72 + self.edges[0, 0] * 0.28)
        left, right = self.at(y)
        return (float((left + right) / 2), y)


def road_from_points(points):
    if len(points) < 2 or not all(0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 for x, y, width in points):
        raise ValueError("Supply at least two road points as normalized X Y WIDTH triples")
    center, y, width = np.array(sorted(points, key=lambda point: point[1])).T
    if np.any(np.diff(y) <= 0.001) or y[-1] < 0.80 or y[0] > 0.77:
        raise ValueError("Use distinct rows and mark both a near point and a farther point")
    return Road(np.column_stack((y, center - width / 2, center + width / 2)))


def row_candidates(integral, y, width, height):
    """Score smooth strips against their immediate surroundings on both sides."""
    centers = np.arange(2, width - 2, 3)
    depth = (y - ROWS[0]) / (ROWS[-1] - ROWS[0])
    spans = width * np.geomspace(0.018 + 0.018 * depth, 0.125 + 0.59 * depth, 24)
    x, span = np.broadcast_arrays(centers[:, None], spans[None, :])
    left = np.rint(x - span / 2).astype(int)
    right = np.rint(x + span / 2).astype(int)
    shoulder = np.maximum(4, np.rint(span * 0.23).astype(int))
    valid = (left >= 2) & (right <= width - 2)
    x, span, left, right, shoulder = (a[valid] for a in (x, span, left, right, shoulder))
    top = max(0, round(y * height) - 3)
    bottom = min(height, round(y * height) + 4)
    sums = integral[bottom] - integral[top]

    def mean(lo, hi):
        return (sums[hi] - sums[lo]) / ((bottom - top) * (hi - lo)[:, None])

    inside = mean(left, right)
    a = mean(np.maximum(0, left - shoulder), left)
    b = mean(right, np.minimum(width, right + shoulder))
    weights = np.array([0.55, 1.0, 1.0])
    contrast = np.linalg.norm((inside[:, :3] - a[:, :3]) * weights, axis=1)
    other = np.linalg.norm((inside[:, :3] - b[:, :3]) * weights, axis=1)
    edge_band = np.clip(np.rint(span * 0.10).astype(int), 3, 9)
    left_edge = mean(left, left + edge_band)[:, :3] - mean(np.maximum(0, left - edge_band), left)[:, :3]
    right_edge = mean(right - edge_band, right)[:, :3] - mean(right, np.minimum(width, right + edge_band))[:, :3]
    boundary = np.minimum(np.linalg.norm(left_edge * weights, axis=1), np.linalg.norm(right_edge * weights, axis=1))
    # Texture and colour contrast favour a path bounded by vegetation or ground.
    cross_color = np.minimum(np.linalg.norm(inside[:, 1:3] - a[:, 1:3], axis=1),
                             np.linalg.norm(inside[:, 1:3] - b[:, 1:3], axis=1))
    band_texture = inside[:, 3]
    chroma_spread = np.sqrt(np.maximum(0, (inside[:, 4:6] - inside[:, 1:3] ** 2).sum(axis=1)))
    gain = np.minimum(contrast, other) * 0.75 + np.minimum(np.maximum(contrast, other), 37) * 0.25
    score = np.minimum(gain, 47) + 0.25 * np.minimum(cross_color, 26)
    score += np.clip(0.36 * (np.minimum(a[:, 3], b[:, 3]) - band_texture), -8, 7)
    score -= 0.14 * band_texture
    score -= np.maximum(0, np.abs(contrast - other) - 18) * 0.20
    score += np.minimum(18, boundary) * 0.65 - np.maximum(0, 6 - boundary) * 2.2
    chroma = np.linalg.norm(inside[:, 1:3] - 128, axis=1)
    score -= np.maximum(0, chroma - 33) * 0.53
    score -= np.maximum(0, chroma_spread - 13) * 1.05
    order = np.argsort(score)[::-1]
    kept = []
    for index in order:
        if score[index] < 8 or len(kept) == 135:
            break
        if chroma[index] <= 58 and chroma_spread[index] <= 32 and all(
            abs(x[index] - x[j]) >= 4 or abs(span[index] - span[j]) >= max(5, span[index] * 0.16)
            for j in kept
        ):
            kept.append(index)
    return np.column_stack((x[kept] / width, span[kept] / width, score[kept], boundary[kept]))


def detect(image):
    """Return (road, evidence) or None when no stable corridor is visible."""
    height, width = 216, 384
    small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    luma = lab[..., 0]
    texture = cv2.absdiff(luma, cv2.GaussianBlur(luma, (0, 0), 1.8))
    integral = cv2.integral(np.dstack((lab, texture, lab[..., 1:3] ** 2)))
    candidates = [row_candidates(integral, y, width, height) for y in ROWS[::-1]]
    if any(not len(row) for row in candidates[:7]):
        return None

    totals = candidates[0][:, 2] - 14
    parents = []
    best = None
    for depth, (near, far) in enumerate(zip(candidates, candidates[1:]), start=1):
        if not len(far):
            break
        dx = np.abs(near[:, 0, None] - far[:, 0])
        dw = near[:, 1, None] - far[:, 1]
        penalty = dx * 58 + np.abs(dw - 0.012) * 65 + np.maximum(0, -dw) * 110
        compatible = (dx <= 0.115) & (dw < 0.17) & (dw > -0.052)
        progression = np.where(compatible, totals[:, None] - penalty, -np.inf)
        parent = progression.argmax(axis=0)
        parents.append(parent)
        totals = progression[parent, np.arange(len(far))] + far[:, 2] - 16.0
        winner = int(totals.argmax())
        if depth >= 7 and np.isfinite(totals[winner]):
            value = totals[winner] + depth * 2.0
            if best is None or value > best[0]:
                best = (value, depth, winner)
    if best is None:
        return None

    _, depth, index = best
    selected = [candidates[depth][index]]
    for row in range(depth, 0, -1):
        index = parents[row - 1][index]
        selected.append(candidates[row - 1][index])
    center, span, score, boundary = np.asarray(selected).T
    if np.mean(score) < 21 or np.quantile(score, 0.25) < 13:
        return None
    if np.mean(boundary >= 5) < 0.57:
        return None
    # Width should increase towards the viewer and stay distinct from isolated thin lines.
    if span[-1] < 0.075 or np.median(span[-3:]) < np.median(span[:3]) * 1.42:
        return None
    y = ROWS[-len(center):]
    fit = np.polynomial.Polynomial.fit(y, center, 2)
    if np.max(np.abs(fit(y) - center)) > 0.078:
        return None
    center = 0.7 * fit(y) + 0.3 * center
    span = 0.55 * np.polynomial.Polynomial.fit(y, span, 2)(y) + 0.45 * span
    return Road(np.column_stack((y, center - span / 2, center + span / 2)))


def select_track(images):
    hits = []
    for frame, image in images:
        match = detect(image)
        if match is not None:
            hits.append((frame, match))
    if len(hits) < 2:
        return []
    comparison = np.array([0.73, 0.84, 0.95])
    centers = np.array([np.mean(road.at(comparison), axis=0) for _, road in hits])
    support = np.max(np.abs(centers[:, None, :] - centers[None, :, :]), axis=2) < 0.22
    consensus = np.argmax(support.sum(axis=1))
    kept = [item for item, keep in zip(hits, support[consensus]) if keep]
    if len(kept) < max(2, (len(images) + 1) // 2):
        return []
    return kept


def road_for_frame(track, frame):
    if not track:
        return None
    frames = [time for time, _ in track]
    index = int(np.searchsorted(frames, frame, side="right"))
    before, first = track[max(0, index - 1)]
    after, second = track[min(index, len(track) - 1)]
    if before == after:
        return first
    fraction = (frame - before) / (after - before)
    start = (1 - fraction) * first.edges[0, 0] + fraction * second.edges[0, 0]
    y = np.linspace(start, 0.98, len(ROWS))
    edges = np.array(first.at(y)) * (1 - fraction) + np.array(second.at(y)) * fraction
    return Road(np.column_stack((y, edges.T)))
