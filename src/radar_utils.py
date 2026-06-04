"""Radar / ray-casting utilities for the NEAT radar car.

Replaces the old 3-point colour sensors with proper distance radars
that measure distance to the track border (TRACK_BORDER_MASK).
"""
import math
from global_vars import TRACK_BORDER_MASK, WIDTH, HEIGHT, CAR_SIZE


def cast_radar(car, angle_offset, max_range, step=3):
    """Cast a single ray and return normalized distance in [0,1].

    Parameters
    ----------
    car : AbstractCar
        Must have .x, .y, .angle.
    angle_offset : float
        Degrees relative to car heading (0 = straight ahead).
    max_range : int
        Maximum radar reach in pixels.
    step : int
        Pixel step for marching (trade-off speed vs precision).

    Returns
    -------
    float
        0.0 = touching border immediately, 1.0 = no border within max_range.
    """
    # In this game's coord system: angle=0 is UP (north).
    # x decreases when turning left (90°), y decreases when going up (0°).
    rad = math.radians(car.angle + angle_offset)
    dir_x = -math.sin(rad)
    dir_y = -math.cos(rad)

    # Car centre (base_car uses x,y as top-left; CAR_SIZE = half w/h)
    cx = car.x + CAR_SIZE[0]
    cy = car.y + CAR_SIZE[1]

    for dist in range(0, max_range, step):
        px = int(cx + dir_x * dist)
        py = int(cy + dir_y * dist)

        if px < 0 or px >= WIDTH or py < 0 or py >= HEIGHT:
            return 1.0

        if TRACK_BORDER_MASK.get_at((px, py)):
            return dist / max_range

    return 1.0


def cast_radars(car, angles, max_range, step=3):
    """Cast multiple radars and return list of normalized distances."""
    return [cast_radar(car, a, max_range, step) for a in angles]
