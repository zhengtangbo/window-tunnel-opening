import math

import cv2
import numpy as np


FPS = 30
REFERENCE_FRAMES = 270
OPEN_END = 208
WAVE_START = 180
WAVE_END = 207
ANGLES = [-2.58, -1.90, -1.26, -0.62, 0.03, 0.68, 1.32, 1.96, 2.59]


def path_center(z):
    return -0.95 * math.sin(z * 0.21) + 0.10 * math.sin(z * 0.68)


def smooth(lo, hi, value):
    t = np.clip((value - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def make_geometry():
    rng = np.random.default_rng(20261114)
    panels = []
    for row in range(9):
        for lane, base_angle in enumerate(ANGLES):
            a = base_angle + rng.uniform(-0.055, 0.055) + 0.075 * math.sin(row * 0.85)
            z = 2.23 + row * 1.94 + rng.uniform(-0.16, 0.17)
            center = np.array([path_center(z) + 2.85 * math.sin(a), 2.42 * math.cos(a) + 0.24, z])
            tangent = np.array([2.85 * math.cos(a), -2.42 * math.sin(a), 0.0])
            tangent /= np.linalg.norm(tangent)
            longitudinal = np.array([0.0, 0.0, 1.0])
            roll = rng.uniform(-0.18, 0.18)
            across = tangent * math.cos(roll) + longitudinal * math.sin(roll)
            along = longitudinal * math.cos(roll) - tangent * math.sin(roll)
            half_w = rng.uniform(0.53, 0.76)
            half_l = rng.uniform(0.62, 0.81)
            corners = np.array([
                center - across * half_w - along * half_l,
                center + across * half_w - along * half_l,
                center + across * half_w + along * half_l,
                center - across * half_w + along * half_l,
            ])
            # Light travels first around the front arch, then further into the tunnel.
            delay = 0.14 + row * 0.56 + lane * 0.114 + rng.uniform(-0.05, 0.08)
            panels.append((z, lane, row, center, corners, delay))

    steps = []
    for i in range(32):
        z = 1.18 + 0.61 * i
        x = path_center(z)
        y = -1.68
        steps.append((z, np.array([
            [x - 0.63, y, z - 0.23], [x + 0.63, y, z - 0.23],
            [x + 0.63, y, z + 0.23], [x - 0.63, y, z + 0.23],
        ])))
    return panels, steps


class Opening:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.size = (width, height)
        self.light_size = (width // 4, height // 4)
        self.panels, self.steps = make_geometry()
        self.y, self.x = np.ogrid[:height, :width]
        nx = (self.x / width - 0.5) * 2.0
        ny = (self.y / height - 0.5) * 2.0
        self.vignette = np.asarray(np.clip(1.045 - 0.165 * (nx * nx + ny * ny), 0.64, 1.0), np.float32)
        rng = np.random.default_rng(56)
        self.noise = sum(
            amplitude * cv2.resize(rng.random((rows, cols), dtype=np.float32), self.size, interpolation=cv2.INTER_CUBIC)
            for cols, rows, amplitude in [(10, 6, 0.16), (31, 17, 0.056), (72, 41, 0.019), (173, 97, 0.008)]
        )
        self.scan_field = self.y / height + self.noise + 0.035 * np.sin(self.x / width * 10.0 + 0.8)
        self.scan_field = self.scan_field.astype(np.float32)
        self.scan_tilt = (0.014 * (self.x / width - 0.5)).astype(np.float32)

    def camera(self, t):
        travel = -1.20 + 0.77 * t + 0.63 * smooth(0.9, 5.8, t)
        x = path_center(travel + 0.50) + 0.08
        yaw = -0.025 + 0.135 * smooth(0.7, 6.1, t)
        pitch = 0.021 + 0.036 * smooth(1.2, 5.8, t)
        roll = -0.028 * math.sin(t * 0.52)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        matrix = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) @ np.array(
            [[1, 0, 0], [0, cp, -sp], [0, sp, cp]]
        ) @ np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
        return np.array([x, 0.0, travel]), matrix

    def project(self, points, camera, clip=True):
        location, matrix = camera
        points = (points - location) @ matrix.T
        if np.any(points[:, 2] < 0.16):
            return None
        normalized = points[:, :2] / points[:, 2, None]
        focal = max(0.555 * self.w, 0.42 * self.h)
        screen = normalized * np.array([focal, -focal]) + np.array([0.50 * self.w, 0.47 * self.h])
        if clip and (
            np.max(screen[:, 0]) < -self.w * 0.09 or np.min(screen[:, 0]) > self.w * 1.09
            or np.max(screen[:, 1]) < -self.h * 0.09 or np.min(screen[:, 1]) > self.h * 1.09
        ):
            return None
        return screen

    def paint_panel(self, scene, source, polygon, intensity, lane, row, floor=False, glint=0.0, on_road=False):
        pad = 2
        x0, y0 = np.maximum(np.floor(polygon.min(axis=0)).astype(int) - pad, 0)
        x1, y1 = np.minimum(np.ceil(polygon.max(axis=0)).astype(int) + pad, [self.w, self.h])
        if x1 <= x0 or y1 <= y0:
            return
        target = scene[y0:y1, x0:x1]
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        points = np.round((polygon - [x0, y0]) * 8).astype(np.int32)
        cv2.fillConvexPoly(mask, points, 255, cv2.LINE_AA, shift=3)
        if on_road:
            values = source[y0:y1, x0:x1].astype(np.float32) * intensity
        elif floor:
            # A fixed sample of the source's stone path keeps the treads stable in motion.
            stone = self.stone[:, (row % 4) * 12:(row % 4) * 12 + 46]
            height, width = stone.shape[:2]
            transform = cv2.getPerspectiveTransform(
                np.float32([[0, height - 1], [width - 1, height - 1], [width - 1, 0], [0, 0]]),
                (polygon - [x0, y0]).astype(np.float32),
            )
            tex = cv2.warpPerspective(stone, transform, (x1 - x0, y1 - y0), borderMode=cv2.BORDER_REPLICATE)
            values = tex.astype(np.float32) * np.array([0.60, 0.63, 0.69], dtype=np.float32) * intensity
        else:
            dx = int(self.w * (0.017 * math.sin(row * 1.81 + lane * 0.88) + 0.013 * (lane - 4) / 4))
            dy = int(self.h * (0.023 * math.cos(row * 0.66 + lane * 1.32)))
            sx, sy = np.arange(x0, x1), np.arange(y0, y1)
            sx = sx + (dx * np.minimum(sx, self.w - 1 - sx) / (np.minimum(sx, self.w - 1 - sx) + abs(dx) + 1)).astype(int)
            sy = sy + (dy * np.minimum(sy, self.h - 1 - sy) / (np.minimum(sy, self.h - 1 - sy) + abs(dy) + 1)).astype(int)
            tex = source[sy[:, None], sx]
            values = tex.astype(np.float32) * intensity
            values *= np.array([1.035, 1.005, 0.975], dtype=np.float32)
            values += (20.0 if 2 <= lane <= 6 else 7.0) * intensity
        a = mask.astype(np.float32)[..., None] / 255.0
        if glint > 0.0:
            # Brief light across the inside of the sky-facing rim as each pane wakes up.
            upper = np.argsort(points[:, 1])[:2]
            rim = np.zeros_like(mask)
            cv2.line(rim, points[upper[0]], points[upper[1]], 255, max(1, round(self.w / 680)), cv2.LINE_AA, shift=3)
            softened = cv2.GaussianBlur(rim, (0, 0), self.w / 1550)
            shine = (0.53 * rim.astype(np.float32) + 0.47 * softened) * (glint / 255.0)
            values += shine[..., None] * np.array([64, 84, 88], np.float32)
        target[:] = np.clip(target + (values - target) * a, 0, 255).astype(np.uint8)

    def paint_beam(self, haze, polygon, intensity, lane, row, focus=0.50):
        center = np.clip(polygon.mean(axis=0), [-self.w * 0.05, -self.h * 0.05], [self.w * 1.05, self.h * 1.05])
        direction = np.array([self.w * (focus + 0.11 * math.sin(lane * 1.24)), self.h * 0.80]) - center
        direction[1] = max(self.h * 0.18, direction[1])
        direction /= max(np.linalg.norm(direction), 1.0)
        side = np.array([-direction[1], direction[0]])
        width = np.clip(np.ptp(polygon @ side) * 0.28, self.w * 0.014, self.w * 0.081)
        length = np.clip(np.ptp(polygon @ direction) * 2.35, self.h * 0.26, self.h * 1.1)
        start = center - direction * min(length * 0.21, self.h * 0.12)
        end = center + direction * length
        corners = np.array([start - side * width, start + side * width, end + side * width * 0.42, end - side * width * 0.42]) / 4.0
        lw, lh = self.light_size
        lo = np.maximum(np.floor(corners.min(axis=0)).astype(int) - 2, 0)
        hi = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 2, [lw, lh])
        x0, y0 = lo
        x1, y1 = hi
        if x1 <= x0 or y1 <= y0:
            return
        patch = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(patch, np.round((corners - lo) * 8).astype(np.int32), 255, cv2.LINE_AA, shift=3)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = ((xx * 4 - start[0]) * direction[0] + (yy * 4 - start[1]) * direction[1]) / length
        fade = np.maximum(0.0, 1.0 - np.maximum(0.0, dist - 0.11)) ** 1.9
        strength = (0.32 if lane in (2, 3, 6) else 0.10) * intensity * (0.82 + 0.14 * math.sin(row * 2.0 + lane))
        haze[y0:y1, x0:x1] += patch * (strength / 255.0) * fade

    def road_windows(self, polygon, road, camera, z):
        depth = max(0, z - camera[0][2])
        top = road.edges[0, 0]
        y = top + (1 - top) / (1 + 0.45 * depth) ** 1.20
        left, right = road.at(y)
        axis = self.project(np.array([[path_center(z), 0.0, z]]), camera, clip=False)
        if axis is None:
            return polygon
        # Bend the window banks around the ground corridor, raising or lowering the
        # convergence point with it. The ceiling may remain wider than the road.
        offset = np.clip((left + right) * self.w / 2 - axis[0, 0], -0.33 * self.w, 0.33 * self.w)
        shift_y = np.clip(top - 0.49, -0.09, 0.16) * self.h
        return polygon + [offset, shift_y]

    def paint_road_treads(self, scene, source, road, t):
        phase = t * 0.62
        top = road.edges[0, 0]
        for slot in range(18, -1, -1):
            # Space slices in depth so the visible gaps and treads converge together.
            distance = slot - (phase % 1)
            near, far = top + (1.0 - top) / (1 + 0.48 * np.maximum(-0.20, [distance, distance + 0.69])) ** 1.24
            if far >= 1.0 or near - far < 0.0025:
                continue
            y = np.repeat([near, far], 2)
            left, right = road.at(y)
            x = (left + right) * 0.5 + (right - left) * np.array([-0.46, 0.46, 0.46, -0.46])
            polygon = np.column_stack((x * self.w, y * self.h))
            delay = 0.13 + (slot + math.floor(phase)) * 0.145
            on = float(smooth(delay, delay + 0.36, t))
            depth_shade = 0.64 + 0.30 * math.exp(-0.19 * max(0, distance))
            if on > 0.0:
                self.paint_panel(scene, source, polygon, on * depth_shade, 0, slot, on_road=True)

    def render(self, source, frame, road=None):
        t = frame / FPS
        camera = self.camera(t)
        scene = np.zeros_like(source)
        haze = np.zeros(self.light_size[::-1], np.float32)
        focus = 0.50 if road is None else sum(road.at(0.79)) / 2
        for z, lane, row, _, corners, delay in reversed(self.panels):
            rise = 0.39 if row else 0.29
            on = float(smooth(delay, delay + rise, t))
            if on <= 0.0:
                continue
            polygon = self.project(corners, camera, clip=road is None)
            if polygon is None:
                continue
            depth = z - camera[0][2]
            if road is not None:
                polygon = self.road_windows(polygon, road, camera, z)
                center = np.rint(polygon.mean(axis=0)).astype(int)
                x, y = np.clip(center, [0, 0], [self.w - 1, self.h - 1])
                sky_light = float(smooth(47, 178, np.mean(source[max(0, y - 2):y + 3, max(0, x - 2):x + 3])))
            else:
                sky_light = 0.0
            intensity = on * (0.65 + 0.30 * math.exp(-0.06 * depth))
            glint = 0.0
            beam = intensity
            if lane in (2, 3, 5, 6):
                age = t - delay
                pulse = float(smooth(0.07, 0.22, age) * (1.0 - smooth(0.27, 0.66, age)))
                glint = pulse * math.exp(-depth * 0.11)
                beam *= 1.21 + 0.34 * glint
            self.paint_panel(scene, source, polygon, intensity, lane, row, glint=glint)
            self.paint_beam(haze, polygon, beam * (1.0 + 0.40 * sky_light), lane, row, focus=focus)

        if road is not None:
            self.paint_road_treads(scene, source, road, t)
        for index, (z, corners) in reversed(list(enumerate(self.steps if road is None else ()))):
            on = float(smooth(0.07 + index * 0.12, 0.37 + index * 0.12, t))
            if on <= 0.0:
                continue
            polygon = self.project(corners, camera)
            if polygon is not None:
                depth = z - camera[0][2]
                light = (0.68 + 0.25 * math.exp(-0.13 * depth)) * (1.0 - smooth(4.8, 10.4, depth))
                # A small reflection follows the row of windows onto the first treads.
                ripple = math.exp(-((t - 0.72 - index * 0.20) / 0.36) ** 2)
                light *= 1.0 + 0.17 * ripple
                self.paint_panel(scene, source, polygon, on * light, 0, index, floor=True)

        scene = (scene * self.vignette[..., None]).astype(np.uint8)
        haze = 0.63 * cv2.GaussianBlur(haze, (0, 0), self.w * 0.004) + 0.37 * cv2.GaussianBlur(haze, (0, 0), self.w * 0.010)
        haze = cv2.resize(haze, self.size, interpolation=cv2.INTER_LINEAR)
        haze = np.clip(haze * self.vignette, 0.0, 0.60)
        opaque = np.ones((self.h, self.w), np.float32)
        edge = np.zeros_like(opaque)
        if frame >= WAVE_START:
            p = (frame - WAVE_START) / (WAVE_END - WAVE_START)
            wave = -0.005 + 1.25 * (0.26 * p + 0.74 * float(smooth(0.0, 1.0, p)))
            distance = self.scan_field - wave
            # Slightly bend the sweeping edge along source-image detail.
            small = cv2.resize(source, self.light_size, interpolation=cv2.INTER_AREA)
            luma = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            local = cv2.GaussianBlur(luma, (0, 0), self.w * 0.002)
            wide = cv2.GaussianBlur(luma, (0, 0), self.w * 0.009)
            terrain = cv2.resize(local - wide, self.size, interpolation=cv2.INTER_LINEAR)
            distance += (0.17 * terrain + self.scan_tilt) * math.sin(math.pi * p)
            opaque = smooth(-0.006, 0.007, distance).astype(np.float32)
            envelope = float(smooth(0.0, 0.14, p) * (1.0 - smooth(0.86, 1.0, p)))
            line = np.exp(-np.square(distance / 0.0023))
            bloom = np.exp(-np.square(distance / 0.014))
            broad = np.exp(-np.square(distance / 0.037))
            edge = envelope * (line * 0.77 + bloom * 0.22 + broad * 0.25)
            lead = distance - 0.15 - 0.031 * np.sin(self.x / self.w * 54.0 - p * 5.0)
            flicker = (np.exp(-np.square(lead / 0.003)) * 0.51 + np.exp(-np.square(lead / 0.015)) * 0.12)
            islands = smooth(0.12, 0.71, 0.5 + 0.5 * np.sin(self.x / self.w * 20.0 + p * 2.2))
            flicker *= islands.astype(np.float32) * 0.78
            edge += flicker * smooth(5.0, 72.0, scene.max(axis=2)) * (1.0 - smooth(0.54, 0.83, p))
            haze *= (0.48 + opaque * 0.52) * (1.0 - smooth(0.56, 1.0, p))

        light_alpha = np.clip(haze + edge * (1.0 - haze), 0.0, 1.0)
        light_color = np.empty_like(scene)
        light_color[:] = (216, 226, 215)
        bright = edge > 0.014
        light_color[bright] = (248, 251, 231)
        after_scene = source + (scene.astype(np.float32) - source) * opaque[..., None]
        composite = after_scene + (light_color.astype(np.float32) - after_scene) * light_alpha[..., None]
        return scene, opaque, light_color, light_alpha, np.rint(composite).astype(np.uint8)
