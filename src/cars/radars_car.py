import pybresenham

from global_vars import *
from cars.base_car import *

class RadarCar(AbstractCar):
    IMG = RED_CAR
    START_POS = (150, 200)

    def __init__(self, max_vel, rotation_vel):
        super().__init__(max_vel, rotation_vel)
        self.max_radar_distance = 100.0
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

    # x,y will be the center of the car and the starting point of all the radar lines
    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)
        x, y = int(self.x + CAR_SIZE[0]), int(self.y + CAR_SIZE[1])
        for radar_name, radar_angle in self.radar_angles.items():
            final_angle = math.radians((self.angle + radar_angle) % 360)
            x_radar = x + self.max_radar_distance * math.cos(final_angle)
            y_radar = y + self.max_radar_distance * math.sin(final_angle)
            x_radar, y_radar, distance = self.cut_line_if_on_border(x, y, x_radar, y_radar)
            pygame.draw.line(win, WHITE, (x, y), (x_radar, y_radar), 1)
            pygame.draw.circle(win, GREEN, (x_radar, y_radar), self.radar_end_radius, width=self.radar_end_radius)
            self.radar_distances[radar_name] = distance

    def cut_line_if_on_border(self, x, y, x_radar, y_radar):
        for x_point, y_point in pybresenham.line(x, y, x_radar, y_radar):
            if TRACK_MASK.get_at((x_point, y_point)) == self.stop_radar_at:
                return x_point, y_point, math.hypot(x - x_point, y - y_point)
        return x_radar, y_radar, self.max_radar_distance

    def step(self, verbose=False):
        pass

    def next_level(self, level):
        self.reset()
