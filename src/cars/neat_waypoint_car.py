"""NEAT waypoint-following car.

FIX: não herda ComputerCar. Path manual. next_level() adicionado.
"""
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
    return angle if cross >= 0 else -angle


def _dist_to_target(car, tx, ty):
    return math.hypot(tx - car.x, ty - car.y)


def _curve_sharpness_at_idx(path, idx):
    n = len(path)
    if n < 3 or idx <= 0 or idx >= n - 1:
        return 0.0
    v1x, v1y = path[idx][0] - path[idx - 1][0], path[idx][1] - path[idx - 1][1]
    v2x, v2y = path[idx + 1][0] - path[idx][0], path[idx + 1][1] - path[idx][1]
    m1 = math.hypot(v1x, v1y) or 1e-9
    m2 = math.hypot(v2x, v2y) or 1e-9
    dot = (v1x * v2x + v1y * v2y) / (m1 * m2)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


class NeatWaypointCar(NeatCar):
    IMG = GREEN_CAR
    START_POS = (180, 200)

    def __init__(self, net, max_vel=4, rotation_vel=4, path=None):
        AbstractCar.__init__(self, max_vel, rotation_vel)
        self.net = net
        self.path = list(path) if path else list(PATH)
        self.current_point = 0
        self.vel = 0
        self.alive = True
        self.off_track_frames = 0
        self.total_off_track_frames = 0
        self.distance_traveled = 0.0
        self.in_track_distance = 0.0
        self.prev_pos = (self.x, self.y)
        self.stuck_frames = 0
        self.frame_count = 0
        self.finish_reached = False
        self.waypoints_reached = 0
        self._last_outputs = (0.0, 0.0)

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
        if self.current_point < len(self.path):
            tx, ty = self.path[self.current_point]
            dist = _dist_to_target(self, tx, ty)
            angle = _signed_angle_to_target(self, tx, ty)
            sharp = _curve_sharpness_at_idx(self.path, self.current_point)
            next_sharp = 0.0
            if self.current_point < len(self.path) - 1:
                next_sharp = _curve_sharpness_at_idx(self.path, self.current_point + 1)
        else:
            fx, fy = FINISH_POSITION
            dist = _dist_to_target(self, fx, fy)
            angle = _signed_angle_to_target(self, fx, fy)
            sharp = 0.0
            next_sharp = 0.0

        max_dist = 600.0
        return [
            min(dist / max_dist, 1.0),
            angle / 180.0,
            sharp / 180.0,
            next_sharp / 180.0,
            self.vel / self.max_vel,
        ]

    def step(self, verbose=False):
        if self.current_point < len(self.path):
            tx, ty = self.path[self.current_point]
            rect = pygame.Rect(self.x, self.y, self.img.get_width(), self.img.get_height())
            if rect.collidepoint(tx, ty):
                self.current_point += 1
                self.waypoints_reached += 1
        super().step(verbose)

    def draw_points(self, win):
        for i, pt in enumerate(self.path):
            color = (0, 200, 0) if i == self.current_point else (0, 100, 0)
            pygame.draw.circle(win, color, pt, 6 if i == self.current_point else 4)

    def draw(self, win):
        AbstractCar.draw(self, win)
        self.draw_points(win)