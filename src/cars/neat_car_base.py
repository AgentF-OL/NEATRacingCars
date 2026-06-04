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

    OFF_TRACK_KILL = 90  # was 60 — gives 1.5s to recover instead of 1s
    GRACE_PERIOD = 30  # was 10 — enough time to settle on spawn
    WAYPOINT_STUCK = 900  # was 600 — 15s without progress before dying
    CIRCLE_RADIUS = 20  # was 12  → bounding box of 40 px
    CIRCLE_HISTORY = 120

    def __init__(self, net, max_vel=4, rotation_vel=4):
        super().__init__(max_vel, rotation_vel)
        self.net = net
        self._pos_history = []
        self._last_wp_frame = 0
        self._last_wp_count = 0
        self.reset_neat_state()

    def get_nn_inputs(self):
        raise NotImplementedError

    def reset_neat_state(self):
        self.alive = True
        self.off_track_frames = 0
        self.total_off_track_frames = 0
        self.distance_traveled = 0.0
        self.in_track_distance = 0.0
        self.prev_pos = (self.x, self.y)
        self.frame_count = 0
        self.finish_reached = False
        self.waypoints_reached = 0
        self._last_outputs = (0.0, 0.0)
        self._pos_history = []
        self._last_wp_frame = 0
        self._last_wp_count = 0
        self._last_wp_pos = (self.x, self.y)  # ← NEW: pos when last wp was hit
        self._last_finish_dist = None

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

        # Deteção pelo CENTRO — mais estável para aprendizagem
        cx = int(self.x + CAR_SIZE[0])
        cy = int(self.y + CAR_SIZE[1])

        if 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
            if TRACK_BORDER_MASK.get_at((cx, cy)):
                self.off_track_frames += 1
                self.total_off_track_frames += 1
            else:
                self.off_track_frames = 0
        else:
            # Centro fora do ecrã considera-se off-track
            self.off_track_frames += 1
            self.total_off_track_frames += 1

        if self.off_track_frames > self.OFF_TRACK_KILL:
            self.alive = False

    def _check_circular(self):
        cx = self.x + CAR_SIZE[0]
        cy = self.y + CAR_SIZE[1]

        # Always buffer the current position
        self._pos_history.append((cx, cy))
        if len(self._pos_history) > self.CIRCLE_HISTORY:
            self._pos_history.pop(0)

        # Not enough history yet → can't judge
        if len(self._pos_history) < self.CIRCLE_HISTORY:
            return

        # True stuck/spinning test: has the car stayed inside a small box
        # for the entire window?  A car driving the track spreads out;
        # a spinner or a jammed car stays in one place.
        xs = [p[0] for p in self._pos_history]
        ys = [p[1] for p in self._pos_history]
        if (max(xs) - min(xs)) <= self.CIRCLE_RADIUS * 2 and \
                (max(ys) - min(ys)) <= self.CIRCLE_RADIUS * 2:
            self.alive = False

    def _check_waypoint_stuck(self):
        """Kill only when the car is *actually* stuck.

        Rules:
          • Hitting a waypoint resets the timer (unchanged).
          • After the last waypoint we track progress toward the finish line.
          • If no waypoint has been hit for a long time BUT the car has moved
            a meaningful distance, it is approaching a far-away waypoint on a
            long straight → reset the timer.
          • Radar cars (no path) fall back to a pure distance-based check.
        """
        current_wp = getattr(self, 'current_point', 0)
        total_wp = len(getattr(self, 'path', []))

        # ── 1. Waypoint advance → reset everything ──
        if current_wp > self._last_wp_count:
            self._last_wp_count = current_wp
            self._last_wp_frame = self.frame_count
            self._last_wp_pos = (self.x, self.y)
            self._last_finish_dist = None
            return

        frames_since = self.frame_count - self._last_wp_frame

        # ── 2. Radar cars or cars without waypoints → distance-only check ──
        if total_wp == 0:
            if frames_since > self.WAYPOINT_STUCK:
                dist_moved = math.hypot(self.x - self._last_wp_pos[0],
                                        self.y - self._last_wp_pos[1])
                if dist_moved > 120:  # still moving, give more time
                    self._last_wp_frame = self.frame_count
                    self._last_wp_pos = (self.x, self.y)
                else:
                    self.alive = False
            return

        # ── 3. All waypoints passed → monitor finish-line approach ──
        if current_wp >= total_wp:
            if frames_since > self.WAYPOINT_STUCK:
                fx, fy = FINISH_POSITION
                dist_to_finish = math.hypot(self.x - fx, self.y - fy)
                if self._last_finish_dist is None:
                    self._last_finish_dist = dist_to_finish
                # Kill only if we are NOT getting closer to the finish
                if dist_to_finish >= self._last_finish_dist - 10:
                    self.alive = False
                else:
                    # Making progress → reset timer and keep driving
                    self._last_finish_dist = dist_to_finish
                    self._last_wp_frame = self.frame_count
            return

        # ── 4. Normal operation: long time without waypoint progress ──
        if frames_since > self.WAYPOINT_STUCK:
            # Grace rule: if the car has travelled a decent distance since the
            # last waypoint hit, it is probably on a long straight or a wide
            # corner and simply hasn't reached the next radius yet.
            dist_moved = math.hypot(self.x - self._last_wp_pos[0],
                                    self.y - self._last_wp_pos[1])
            if dist_moved > 120:  # ~120 px is generous but safe
                self._last_wp_frame = self.frame_count
                self._last_wp_pos = (self.x, self.y)
                return
            self.alive = False

    def _check_finish_line(self):
        cx = self.x + CAR_SIZE[0]
        cy = self.y + CAR_SIZE[1]
        fx, fy = FINISH_POSITION
        fw, fh = FINISH.get_width(), FINISH.get_height()
        if not (fx <= cx < fx + fw and fy <= cy < fy + fh):
            return False
        if not FINISH_MASK.get_at((int(cx - fx), int(cy - fy))):
            return False
        if self.off_track_frames > 0:
            return False

        # ── NEW: require most waypoints completed before finish counts ──
        if hasattr(self, 'path') and hasattr(self, 'current_point'):
            min_required = int(len(self.path) * 0.85)  # must complete 85% of waypoints
            if self.current_point < min_required:
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

        # ── Controlo CONTÍNUO (sem zona morta) ──
        # Qualquer sinal não-nulo produz movimento — essencial para geração 0
        if abs(steer) > 0.02:
            self.angle += steer * self.rotation_vel * 2.5
            self.angle = int(self.angle) % 360

        # Aceleração/travagem contínua com ganho aumentado
        self.vel += throttle * self.acceleration * 2.0
        self.vel = max(-self.max_vel * 0.5, min(self.vel, self.max_vel))

        # Fricção apenas quando throttle neutro
        if abs(throttle) < 0.05:
            self._apply_friction()

        AbstractCar.move(self)

        # Atualizar distâncias
        dx = self.x - self.prev_pos[0]
        dy = self.y - self.prev_pos[1]
        step_dist = math.hypot(dx, dy)
        self.distance_traveled += step_dist
        if self.off_track_frames == 0:
            self.in_track_distance += step_dist
        self.prev_pos = (self.x, self.y)

        self._check_boundaries()
        self._check_circular()

        if self._check_finish_line():
            self.finish_reached = True
            self.alive = False

        self._check_waypoint_stuck()

        # NOTA: Removemos o kill por stuck_frames (velocidade baixa).
        # O limite de tempo (max_frames) e o time_bonus no fitness
        # lidam com carros imóveis de forma mais suave.

        if verbose and self.frame_count % 30 == 0:
            print(f"\\t{self.__class__.__name__}: "
                  f"steer={steer:+.2f} thr={throttle:+.2f} "
                  f"vel={self.vel:.2f} off={self.off_track_frames} "
                  f"wp={getattr(self, 'current_point', 0)} "
                  f"pos=({self.x:.0f},{self.y:.0f})")

    def next_level(self, level):
        self.reset()
        self.reset_neat_state()
        self.vel = self.max_vel + (level - 1) * 0.02