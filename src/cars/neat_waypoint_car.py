"""NEAT waypoint-following car."""
import math
import pygame
from global_vars import *
from cars.base_car import AbstractCar
from cars.neat_car_base import NeatCar

REACH_RADIUS = 40


def _signed_angle_to_target(car, tx, ty):
    rad = math.radians(car.angle)
    hx, hy = -math.sin(rad), -math.cos(rad)
    dx, dy = tx - car.x, ty - car.y
    dist = math.hypot(dx, dy) or 1e-9
    wx, wy = dx / dist, dy / dist
    dot = max(-1.0, min(1.0, hx * wx + hy * wy))
    angle = math.degrees(math.acos(dot))
    cross = hx * wy - hy * wx
    # pygame: y-down flips the cross-product meaning
    return angle if cross < 0 else -angle


def _dist_to_target(car, tx, ty):
    return math.hypot(tx - car.x, ty - car.y)


def _curve_sharpness_at_idx(path, idx):
    n = len(path)
    if n < 3 or idx <= 0 or idx >= n - 1:
        return 0.0
    v1x = path[idx][0] - path[idx - 1][0]
    v1y = path[idx][1] - path[idx - 1][1]
    v2x = path[idx + 1][0] - path[idx][0]
    v2y = path[idx + 1][1] - path[idx][1]
    m1 = math.hypot(v1x, v1y) or 1e-9
    m2 = math.hypot(v2x, v2y) or 1e-9
    dot = (v1x * v2x + v1y * v2y) / (m1 * m2)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


class NeatWaypointCar(NeatCar):
    IMG = GREEN_CAR
    START_POS = (180, 200)

    def __init__(self, net, max_vel=4, rotation_vel=4, path=None):
        super().__init__(net, max_vel, rotation_vel)
        self.path = list(path) if path else list(PATH)
        self.current_point = 0

    def reset(self):
        AbstractCar.reset(self)
        self.current_point = 0
        self.waypoints_reached = 0
        self.reset_neat_state()

    def next_level(self, level):
        """Reset for next level (used by game.py handle_collision)."""
        self.reset()
        self.vel = self.max_vel + (level - 1) * 0.02

    def get_nn_inputs(self):
        n = len(self.path)
        if n == 0:
            return [0.0] * 5

        # ── WRAP waypoints: after the last, go back to the first ──
        # This matches DTGreenCar behaviour and prevents the car from
        # targeting FINISH_POSITION as a point (which caused circling).
        idx = self.current_point % n
        tx, ty = self.path[idx]

        dist = _dist_to_target(self, tx, ty)
        angle = _signed_angle_to_target(self, tx, ty)
        sharp = _curve_sharpness_at_idx(self.path, idx)

        # Look ahead to next waypoint (also wrapped)
        next_sharp = 0.0
        next_idx = (idx + 1) % n
        if next_idx != idx:  # avoid single-waypoint path
            next_sharp = _curve_sharpness_at_idx(self.path, next_idx)

        max_dist = 600.0
        return [
            min(dist / max_dist, 1.0),
            angle / 180.0,
            sharp / 180.0,
            next_sharp / 180.0,
            self.vel / self.max_vel,
        ]

    def step(self, verbose=False):
        n = len(self.path)
        if n > 0:
            # ── WRAP: target waypoint index modulo path length ──
            idx = self.current_point % n
            tx, ty = self.path[idx]
            cx = self.x + CAR_SIZE[0]
            cy = self.y + CAR_SIZE[1]
            if math.hypot(cx - tx, cy - ty) < REACH_RADIUS:
                self.current_point += 1
                self.waypoints_reached += 1
        super().step(verbose)

    def draw_points(self, win):
        for i, pt in enumerate(self.path):
            color = (0, 200, 0) if i == (self.current_point % len(self.path)) else (0, 100, 0)
            pygame.draw.circle(win, color, pt, 6 if i == (self.current_point % len(self.path)) else 4)

    def draw(self, win):
        AbstractCar.draw(self, win)
        self.draw_points(win)