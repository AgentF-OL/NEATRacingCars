import pybresenham
import neat
import math

import config
from global_vars import *
from cars.base_car import *

class RadarCar(AbstractCar):
    IMG = RED_CAR
    START_POS = (150, 200)

    def __init__(self, max_vel, rotation_vel, network: neat.nn.FeedForwardNetwork):
        super().__init__(max_vel, rotation_vel)
        self.network = network
        self.max_radar_distance = config.MAX_RADAR_DISTANCE
        self.stop_radar_at = RED # Track border from the track mask
        self.radar_end_radius = 3
        self.radar_angles = {
            "left": -150,
            "left_middle": -120,
            "middle": -90,
            "right_middle": -60,
            "right": -30
        }
        self.radar_distances = {
            "left": self.max_radar_distance,
            "left_middle": self.max_radar_distance,
            "middle": self.max_radar_distance,
            "right_middle": self.max_radar_distance,
            "right": self.max_radar_distance
        }
        self.distance_traveled = 0.0
        self.time_taken = 0.0
        self.is_alive = True

    # x,y will be the center of the car and the starting point of all the radar lines
    def draw(self, win, font):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)
        x, y = int(self.x + CAR_SIZE[0]), int(self.y + CAR_SIZE[1])
        self.compute_radar_distances()
        for radar_name, radar_angle in self.radar_angles.items():
            distance = self.radar_distances[radar_name]
            final_angle = math.radians((radar_angle - self.angle) % 360)
            x_radar = x + distance * math.cos(final_angle)
            y_radar = y + distance * math.sin(final_angle)
            pygame.draw.line(win, WHITE, (x, y), (x_radar, y_radar), 1)
            pygame.draw.circle(win, GREEN, (x_radar, y_radar), self.radar_end_radius, width=self.radar_end_radius)
            self.radar_distances[radar_name] = distance

    def draw_stats(self, win, font):
        fitness = config.FITNESS_FUN_RADARS_CAR(self)
        x, y = (11/20) * win.get_width(), win.get_height() / 2 + 10
        x_tab = x + 20
        newline = 50
        blit_text_at(win, font, "RED:", x, y)
        blit_text_at(win, font, f"Vel={self.vel}", x_tab, y + newline)
        blit_text_at(win, font, f"Acc={self.acceleration}", x_tab, y + 2 * newline)
        blit_text_at(win, font, f"Angle={self.angle}", x_tab, y + 3 * newline)
        blit_text_at(win, font, f"Fitness={fitness}", x_tab, y + 4 * newline)
        pygame.display.update()

    def compute_radar_distances(self):
        x, y = int(self.x + CAR_SIZE[0]), int(self.y + CAR_SIZE[1])
        for radar_name, radar_angle in self.radar_angles.items():
            final_angle = math.radians((radar_angle - self.angle) % 360)
            x_radar = x + self.max_radar_distance * math.cos(final_angle)
            y_radar = y + self.max_radar_distance * math.sin(final_angle)
            x_radar, y_radar, distance = self.cut_radar_if_on_border(x, y, x_radar, y_radar)
            x_radar, y_radar, distance = self.add_sensor_noise(x, y, x_radar, y_radar)
            self.radar_distances[radar_name] = distance

    def cut_radar_if_on_border(self, x, y, x_radar, y_radar):
        out_of_bounds = False
        for x_point, y_point in pybresenham.line(x, y, x_radar, y_radar):
            if x_point < 0:
                x_point = 0
                out_of_bounds = True
            if x_point >= WIDTH:
                x_point = WIDTH - 1
                out_of_bounds = True
            if y_point < 0:
                y_point = 0
                out_of_bounds = True
            if y_point >= HEIGHT:
                y_point = HEIGHT - 1
                out_of_bounds = True
            if out_of_bounds or TRACK_MASK.get_at((x_point, y_point)) == self.stop_radar_at:
                return x_point, y_point, math.hypot(x - x_point, y - y_point)
        return x_radar, y_radar, self.max_radar_distance
    
    def add_sensor_noise(self, x, y, x_radar, y_radar):
        x_radar += random.randint(-config.NOISE_RADARS_CAR, config.NOISE_RADARS_CAR)
        y_radar += random.randint(-config.NOISE_RADARS_CAR, config.NOISE_RADARS_CAR)
        if x_radar < 0: x_radar = 0
        if x_radar >= WIDTH: x_radar = WIDTH - 1
        if y_radar < 0: y_radar = 0
        if y_radar >= HEIGHT: y_radar = HEIGHT - 1
        return x_radar, y_radar, math.hypot(x - x_radar, y - y_radar)

    def step(self, verbose=False, stdout_verbose=False):
        outputs = self.network.activate(self.radar_distances.values())
        vel = abs(outputs[0]) * self.max_vel # from interval [-1, 1] to [0, max_vel]
        left_output = outputs[1]
        right_output = outputs[2]
        self.vel = min(vel, self.max_vel)
        left = left_output > 0.0
        right = right_output > 0.0
        self.rotate(left=left, right=right)
        self.move()
        self.distance_traveled += self.vel
        self.time_taken += 1
        self.check_liveness()
        if stdout_verbose: print(self)
        if verbose: self.draw_stats(WIN, MAIN_FONT)

    def check_liveness(self):
        if self.is_alive and not self.liveness_conditions():
            self.is_alive = False

    def liveness_conditions(self):
        return not (self.out_of_bounds() or self.collide(TRACK_BORDER_MASK) or self.collide(FINISH_MASK) or self.in_grass())

    def out_of_bounds(self):
        return self.x < 0 or self.x >= WIDTH or self.y < 0 or self.y >= HEIGHT

    def in_grass(self):
        return TRACK_MASK.get_at((int(self.x), int(self.y))) == YELLOW

    def next_level(self, level):
        self.reset()
        self.distance_traveled = 0.0
        self.time_taken = 0.0
        self.is_alive = True

    def __str__(self):
        return "\tRed: " + super().__str__()
