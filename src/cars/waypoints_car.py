# TODO - Change this file: it is using an older type of car based on decision trees

import math
import pygame

from global_vars import *
from cars.base_car import *
from decision_trees import *

"""
The green car DT works on a sensors dict with these keys:

  'dist_to_wp'    – pixels to current waypoint
  'angle_to_wp'   – signed degrees: car heading → waypoint direction
                    positive = waypoint is LEFT, negative = RIGHT
  'curve_sharpness' – exterior angle at current waypoint (0=straight, 180=U-turn)
  'at_waypoint'   – bool: dist_to_wp < REACH_RADIUS

Actions (methods on DTGreenCar):
  'advance_waypoint'  – step current_point forward, then steer+move
  'steer_and_go'      – correct heading, accelerate to max_vel
  'cruise'            – correct heading, hold mid speed (approaching curve)
  'brake_and_steer'   – correct heading, reduce speed (sharp curve close ahead)
"""

# ══════════════════════════════════════════════════════════════════════════════
#  Geometry helpers — all car-relative
# ══════════════════════════════════════════════════════════════════════════════

def _signed_angle_to_wp(car):
    """
    Signed angle (degrees) from car heading to the direction of current waypoint.
    Positive → waypoint is to the LEFT.   Negative → to the RIGHT.
    Returns 0 if no path.
    """
    if not car.path or car.current_point >= len(car.path):
        return 0.0
    # car heading unit vector (pygame: y grows downward, angle=0 is north)
    rad = math.radians(car.angle)
    hx, hy = -math.sin(rad), -math.cos(rad)
    # vector to waypoint
    tx, ty = car.path[car.current_point]
    dx, dy = tx - car.x, ty - car.y
    dist = math.hypot(dx, dy) or 1e-9
    wx, wy = dx / dist, dy / dist
    # signed angle via cross & dot
    dot   = max(-1.0, min(1.0, hx*wx + hy*wy))
    angle = math.degrees(math.acos(dot))
    cross = hx*wy - hy*wx          # z of cross product
    return angle if cross >= 0 else -angle


def _dist_to_wp(car):
    if not car.path or car.current_point >= len(car.path):
        return 0.0
    tx, ty = car.path[car.current_point]
    return math.hypot(tx - car.x, ty - car.y)


def _curve_sharpness(car):
    """
    Automatically compute the turning angle at current_point (degrees).
    0° = straight ahead.  180° = U-turn.
    """
    path = car.path
    n    = len(path)
    idx  = car.current_point
    if n < 3 or idx == 0 or idx >= n - 1:
        return 0.0
    prev = path[idx - 1]
    curr = path[idx]
    nxt  = path[idx + 1]
    v1x, v1y = curr[0]-prev[0], curr[1]-prev[1]
    v2x, v2y = nxt[0]-curr[0],  nxt[1]-curr[1]
    m1 = math.hypot(v1x, v1y) or 1e-9
    m2 = math.hypot(v2x, v2y) or 1e-9
    dot = (v1x*v2x + v1y*v2y) / (m1 * m2)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


# ══════════════════════════════════════════════════════════════════════════════
#  Sensor builder — called every frame inside DTGreenCar.step()
# ══════════════════════════════════════════════════════════════════════════════

REACH_RADIUS   = 40     # px — close enough to "touch" a waypoint
TIGHT_CURVE    = 45.0   # ° — start slowing
SHARP_CURVE    = 80.0   # ° — brake hard
BRAKE_HORIZON  = 120    # px — how far ahead to look for curves
CRUISE_SPEED   = 2.5    # px/frame — comfortable corner speed
SLOW_SPEED     = 1.8    # px/frame — minimum when braking
MAX_VEL        = 4


def build_sensors(car):
    dist  = _dist_to_wp(car)
    angle = _signed_angle_to_wp(car)
    sharp = _curve_sharpness(car)
    
    # Look ahead to NEXT waypoint for curve prediction when close to current
    next_sharp = 0.0
    if car.current_point < len(car.path) - 1:
        # Temporarily advance to check next curve
        saved = car.current_point
        car.current_point += 1
        next_sharp = _curve_sharpness(car)
        car.current_point = saved
    
    return {
        'at_waypoint':       dist < REACH_RADIUS,
        'dist_to_wp':        dist,
        'angle_to_wp':       abs(angle),
        'angle_to_wp_signed': angle,
        'curve_sharpness':   sharp,
        'next_curve_sharp':  next_sharp,
        'curve_close':       (sharp > TIGHT_CURVE or next_sharp > TIGHT_CURVE) and dist < BRAKE_HORIZON,
        'sharp_curve_close': (sharp > SHARP_CURVE or next_sharp > SHARP_CURVE) and dist < BRAKE_HORIZON,
        'going_too_fast':    car.vel > CRUISE_SPEED,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  The decision tree (using the lab's Boolean node)
#
#  Logic:
#    at_waypoint?
#      yes → advance_waypoint
#      no  →
#        sharp_curve_close?
#          yes → brake_and_steer
#          no  →
#            curve_close?
#              yes → cruise
#              no  → steer_and_go
#
#  (steer_and_go also handles "not aligned" because the steer step is
#   always executed inside every action — see DTGreenCar below)
# ══════════════════════════════════════════════════════════════════════════════

GREEN_CAR_TREE = Boolean('at_waypoint',
    yesNode=Action('advance_waypoint'),
    noNode=Boolean('sharp_curve_close',
        yesNode=Action('brake_and_steer'),
        noNode=Boolean('curve_close',
            yesNode=Action('cruise'),
            noNode=Action('steer_and_go')
        )
    )
)

"""
Contains:
  • All existing classes (AbstractCar, PlayerCar, ComputerCar, GameInfo, etc.)
  • DTGreenCar  – waypoint-following car controlled by GREEN_CAR_TREE
  • run_race()  – main loop with optional verbose mode
─────────────────────────────────────────────────────────────────────────────
"""

# ── helpers ───────────────────────────────────────────────────────────────────

def draw(win, images, player_car, computer_car, game_info):
    for img, pos in images:
        win.blit(img, pos)
    win.blit(MAIN_FONT.render(f'Level {game_info.level}', 1, WHITE),
             (10, HEIGHT - 134))
    win.blit(MAIN_FONT.render(f'Time  {game_info.get_level_time()}', 1, WHITE),
             (10, HEIGHT - 90))
    win.blit(MAIN_FONT.render(f'Vel   {round(computer_car.vel, 1)} px/f', 1, WHITE),
             (10, HEIGHT - 46))
    player_car.draw(win)
    computer_car.draw(win)
    pygame.display.update()

# ── DTGreenCar ────────────────────────────────────────────────────────────────

class DTGreenCar(DTCar,ComputerCar):
    """
    Decision-tree controlled green car.

    Inherits action methods from GreenCarActions and the DT dispatch
    machinery from DTCar.  Path is the shared waypoint list.
    """
    IMG       = GREEN_CAR
    START_POS = (180, 200)
    DT        = GREEN_CAR_TREE

    def __init__(self, max_vel, rotation_vel, path=None):
        super().__init__(max_vel, rotation_vel)
        self.path          = list(path) if path else list(PATH)
        self.current_point = 0
        self.vel           = 0

    def step(self, verbose=False):
        if self.current_point >= len(self.path):
            return  # finished path
        
        sensor_data = build_sensors(self)
        action = self.DT.decide(sensor_data)
        self._last_action = action
        eval('self.' + action + '()')

        if verbose:
            print(f"\tGreen: {self.verbose_state()}")

    # ── verbose summary ───────────────────────────────────────────────────────
    def verbose_state(self):
        dist   = _dist_to_wp(self)
        angle  = _signed_angle_to_wp(self)
        sharp  = _curve_sharpness(self)
        return (f"action={self._last_action:<18}  "
                f"wp={self.current_point:>2}/{len(self.path)-1}  "
                f"dist={dist:>6.1f}px  "
                f"angle_to_wp={angle:>+7.1f}°  "
                f"curve={sharp:>5.1f}°  "
                f"vel={self.vel:>4.2f}")

    def draw_points(self, win):
        for i, pt in enumerate(self.path):
            color = (0, 200, 0) if i == self.current_point else (0, 100, 0)
            pygame.draw.circle(win, color, pt, 6 if i == self.current_point else 4)

    def draw(self, win):
        AbstractCar.draw(self, win)
        self.draw_points(win)

    def next_level(self, level):
        self.reset()
        self.vel           = self.max_vel + (level - 1) * 0.02
        self.current_point = 0

    # ── actions functions ───────────────────────────────────────────────────────

    def advance_waypoint(self):
        """Move to next waypoint, but don't steer yet — let next frame handle it."""
        self.current_point += 1
        if self.current_point >= len(self.path):
            self.vel = 0  # stop at end
            return
        # Gentle acceleration out of waypoint, don't snap to max
        self.vel = min(self.vel + self.acceleration, self.max_vel * 0.7)
        self.calculate_angle()
        self.move()

    def steer_and_go(self):
        """Steer toward waypoint and accelerate smoothly."""
        if self.current_point >= len(self.path):
            return
        self.calculate_angle()
        # Smooth acceleration instead of instant max
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def cruise(self):
        """Maintain cruise speed through moderate curve."""
        if self.current_point >= len(self.path):
            return
        self.calculate_angle()
        # Smooth deceleration to cruise speed
        if self.vel > CRUISE_SPEED:
            self.vel = max(self.vel - self.acceleration * 0.5, CRUISE_SPEED)
        else:
            self.vel = min(self.vel + self.acceleration, CRUISE_SPEED)
        self.move()

    def brake_and_steer(self):
        """Slow down for sharp curve."""
        if self.current_point >= len(self.path):
            return
        self.calculate_angle()
        # Smooth braking
        self.vel = max(self.vel - self.acceleration, SLOW_SPEED)
        self.move()