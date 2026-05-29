"""Base class for all NEAT-controlled cars.

Correções críticas:
  • Fricção aplicada quando não há aceleração/travagem activa.
  • Boundary kill — morre se sair do ecrã ou ficar off-track > 30 frames.
  • total_off_track: acumulador que nunca diminui (para penalização de atalhos).
  • rotate() sem ruído aleatório do AbstractCar.
  • move() resolve sempre para AbstractCar.move().

Outras Correções:
  • Fricção, boundary kill, rotate limpo, move() via AbstractCar.
  • OFF-TRACK kill: 60 frames consecutivos (era 30, demasiado agressivo para gen 0).
  • GRACE PERIOD: primeiros 15 frames não contam off-track (evita kill instantâneo
    se o carro nascer perto da borda).
  • total_off_track_frames acumulado (nunca diminui).
  • in_track_distance: só conta quando na pista.
"""
import math
from global_vars import *
from cars.base_car import AbstractCar
from noise_utils import add_sensor_noise, add_actuator_noise


class NeatCar(AbstractCar):
    OFF_TRACK_KILL = 60
    GRACE_PERIOD = 15
    STUCK_KILL = 180

    def __init__(self, net, max_vel=4, rotation_vel=4):
        super().__init__(max_vel, rotation_vel)
        self.net = net
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

    def get_nn_inputs(self):
        raise NotImplementedError

    def reset_neat_state(self):
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

    def rotate(self, left=False, right=False):
        if left and right:
            return
        elif left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel
        self.angle = int(self.angle) % 360

    def _apply_friction(self):
        if abs(self.vel) > 0.05:
            self.vel *= 0.94
        else:
            self.vel = 0.0

    def _check_boundaries(self):
        margin = 20
        if (self.x < -margin or self.x > WIDTH + margin or
                self.y < -margin or self.y > HEIGHT + margin):
            self.alive = False
            return

        if self.frame_count < self.GRACE_PERIOD:
            return

        if self.collide(TRACK_BORDER_MASK) is not None:
            self.off_track_frames += 1
            self.total_off_track_frames += 1
        else:
            self.off_track_frames = 0

        if self.off_track_frames > self.OFF_TRACK_KILL:
            self.alive = False

    def _check_finish_line(self):
        # Car center
        cx = self.x + CAR_SIZE[0]
        cy = self.y + CAR_SIZE[1]

        # Check if center point is within finish mask bounds
        fx, fy = FINISH_POSITION
        fw, fh = FINISH.get_width(), FINISH.get_height()
        if not (fx <= cx < fx + fw and fy <= cy < fy + fh):
            return False

        # Check if center point is actually on the finish mask
        if not FINISH_MASK.get_at((int(cx - fx), int(cy - fy))):
            return False

        # CRITICAL: only count finish if car is currently in-track
        if self.off_track_frames > 0:
            return False

        return True

    def step(self, verbose=False):
        if not self.alive:
            return

        self.frame_count += 1

        raw_inputs = self.get_nn_inputs()
        inputs = [add_sensor_noise(v) for v in raw_inputs]
        outputs = self.net.activate(inputs)
        steer_raw = outputs[0]
        throttle_raw = outputs[1] if len(outputs) > 1 else 0.0

        steer = add_actuator_noise(steer_raw)
        throttle = add_actuator_noise(throttle_raw)
        self._last_outputs = (steer, throttle)

        if steer < -0.15:
            self.rotate(left=True)
        elif steer > 0.15:
            self.rotate(right=True)

        if throttle > 0.15:
            self.vel = min(self.vel + self.acceleration, self.max_vel)
        elif throttle < -0.15:
            self.vel = max(self.vel - self.acceleration, -self.max_vel // 2)
        else:
            self._apply_friction()

        AbstractCar.move(self)

        dx = self.x - self.prev_pos[0]
        dy = self.y - self.prev_pos[1]
        step_dist = math.hypot(dx, dy)
        self.distance_traveled += step_dist
        if self.off_track_frames == 0:
            self.in_track_distance += step_dist
        self.prev_pos = (self.x, self.y)

        self._check_boundaries()

        # Finish line check — center-based and in-track required
        if self._check_finish_line():
            self.finish_reached = True
            self.alive = False

        if abs(self.vel) < 0.3:
            self.stuck_frames += 1
        else:
            self.stuck_frames = 0

        if self.stuck_frames > self.STUCK_KILL:
            self.alive = False

        if verbose and self.frame_count % 10 == 0:
            print(f"\\t{self.__class__.__name__}: "
                  f"steer={steer:+.2f} thr={throttle:+.2f} "
                  f"vel={self.vel:.2f} off={self.off_track_frames} total_off={self.total_off_track_frames} "
                  f"pos=({self.x:.0f},{self.y:.0f})")

    def next_level(self, level):
        self.reset()
        self.reset_neat_state()
        self.vel = self.max_vel + (level - 1) * 0.02