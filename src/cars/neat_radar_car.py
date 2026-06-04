"""NEAT radar-based car.

Inputs (normalizados):
  1..5  radar distances [-90°, -45°, 0°, +45°, +90°] relative to heading  [0,1]
  6     current velocity                                              [0,1]

Outputs:
  0  steering  (-1 .. 1)
  1  throttle  (-1 .. 1)
"""
import math
import pygame
from global_vars import *
from cars.base_car import AbstractCar
from cars.neat_car_base import NeatCar
from radar_utils import cast_radars


class NeatRadarCar(NeatCar):
    IMG = PURPLE_CAR
    START_POS = (150, 200)
    RADAR_ANGLES = [-90, -45, 0, 45, 90]
    RADAR_MAX_RANGE = 200

    def __init__(self, net, max_vel=4, rotation_vel=4):
        super().__init__(net, max_vel, rotation_vel)

    def reset(self):
        AbstractCar.reset(self)
        self.reset_neat_state()

    def next_level(self, level):
        """Reset for next level (used by game.py handle_collision)."""
        self.reset()
        self.vel = self.max_vel + (level - 1) * 0.02

    def get_nn_inputs(self):
        radar_vals = cast_radars(self, self.RADAR_ANGLES, self.RADAR_MAX_RANGE)
        norm_vel = self.vel / self.max_vel
        return radar_vals + [norm_vel]

    def draw(self, win):
        AbstractCar.draw(self, win)
        cx = self.x + CAR_SIZE[0]
        cy = self.y + CAR_SIZE[1]
        for a in self.RADAR_ANGLES:
            rad = math.radians(self.angle + a)
            fx = cx + -math.sin(rad) * self.RADAR_MAX_RANGE
            fy = cy + -math.cos(rad) * self.RADAR_MAX_RANGE
            pygame.draw.line(win, (255, 255, 0), (cx, cy), (fx, fy), 1)