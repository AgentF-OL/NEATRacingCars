import random
import math
import numpy

from global_vars import *

class AbstractCar:
    
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x,self.y=self.START_POS
        self.acceleration=1
        ##################### Sensor creation ######################
        self.index_bullet = {0: "top_left", 1: "top_right", 2: "bottom_left", 3: "bottom_right",
                             4: "top_bow_left", 5: "top_bow_middle", 6: "top_bow_right", 7: "bottom"}
        dx = int(CAR_SIZE[0])
        dy = int(CAR_SIZE[1])
        self.bow_x = 10
        self.bow_y = 10
        self.bow_y_factor = 3
        self.bullets = {
            "top_left": (self.x - dx, self.y - dy),
            "top_right": (self.x + dx, self.y - dy),
            "bottom_left": (self.x - dx, self.y + dy),
            "bottom_right": (self.x + dx, self.y + dy),
            "top_bow_left": (self.x - dx - self.bow_x, self.y - dy - self.bow_y),
            "top_bow_middle": (self.x, self.y - dy - self.bow_y_factor * self.bow_y),
            "top_bow_right": (self.x + dx + self.bow_x, self.y - dy - self.bow_y),
            "bottom": (self.x, self.y + dy + self.bow_y_factor * self.bow_y)
        }

    def rotate(self, left=False, right=False):
        if left and right:
            return
        elif left:
            self.angle += self.rotation_vel + random.choice(range(-1,1))
        elif right:
            self.angle -= self.rotation_vel + random.choice(range(-1,1))
        self.angle = int(self.angle) % 360
        
    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()
        
    def move_backwards(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel//2)
        self.move()
    
    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal
        
    
    def collide(self,mask,x=0,y=0):
            car_mask=pygame.mask.from_surface(self.img)
            offset=(int(self.x-x),int(self.y-y))
            poi=mask.overlap(car_mask,offset)
            return poi
        
    def reset(self):
        self.x,self.y=self.START_POS
        self.angle=0
        self.vel=0

    # x,y will be the center of the car
    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)
        x, y = int(self.x + CAR_SIZE[0]), int(self.y + CAR_SIZE[1])
        dx, dy = CAR_SIZE[0], CAR_SIZE[1]
        alfa = (self.angle) * math.pi / 180
        mrot = numpy.array([[math.cos(alfa), -math.sin(alfa)], [math.sin(alfa), math.cos(alfa)]])
        pts = numpy.array([
            [-dx, -dy], [dx, -dy], [-dx, dy], [dx, dy],
            [- dx - self.bow_x, - dy - self.bow_y],
            [0, - dy - self.bow_y_factor * self.bow_y],
            [dx + self.bow_x, - dy - self.bow_y],
            [0, dy + self.bow_y_factor * self.bow_y]
        ])
        npts = numpy.dot(pts, mrot)
        for index in range(len(npts)):
            i = npts[index]
            if 0<=i[0]+x<WIDTH and 0<=i[1]+y<HEIGHT:
                self.bullets[self.index_bullet[index]] = (int(i[0] + x), int(i[1] + y))
                pygame.draw.circle(win, TRACK_MASK.get_at((int(i[0] + x), int(i[1] + y))), \
                                                          (int(i[0] + x), int(i[1] + y)), 2, 2)
                
class ComputerCar(AbstractCar):
    IMG = GREEN_CAR
    START_POS = (150, 200)
    
    def __init__(self, max_vel, rotation_vel, path=[]):
        super().__init__(max_vel, rotation_vel)
        self.path=path
        self.current_point=0
        self.vel=self.max_vel
             
    def calculate_angle(self):
        target_x, target_y = self.path[self.current_point]
        x_diff = target_x - self.x
        y_diff = target_y - self.y

        if y_diff == 0:
            desired_radian_angle = (1 if x_diff<0 else -1)*math.pi / 2
        else:
            desired_radian_angle = math.atan(x_diff / y_diff)

        if target_y > self.y:
            desired_radian_angle += math.pi

        difference_in_angle = self.angle - math.degrees(desired_radian_angle)
        if difference_in_angle >= 180:
            difference_in_angle -= 360
            
        elif difference_in_angle <= -180:
            difference_in_angle += 360

        if difference_in_angle > 0:
            self.angle -= min(self.rotation_vel, abs(difference_in_angle))
        else:
            self.angle += min(self.rotation_vel, abs(difference_in_angle))

    def update_path_point(self):
        target = self.path[self.current_point]
        rect = pygame.Rect(
            self.x, self.y, self.img.get_width(), self.img.get_height())
        if rect.collidepoint(*target):
            self.current_point += 1
            
    def move(self):
        if self.current_point >= len(self.path):
            return

        self.calculate_angle()
        self.update_path_point()
        super().move()
            
    def next_level(self,level):
        self.reset()
        self.vel=self.max_vel+(level-1)*0.02
        self.current_point=0
        
    def draw_points(self,win):
        for point in self.path:
            pygame.draw.circle(win,(255,0,0),point,5)
        
    def draw(self,win):
        super().draw(win)
                
# TODO - Remove this type of car since this game does not used decision tree based cars
class DTCar(AbstractCar):
    DT=None
    
    def __init__(self, max_vel, rotation_vel):
        super().__init__(max_vel, rotation_vel)
        self.sensors=dict()
        self.update_sensors()
        self._last_action = "none" # For verbose logging

    def update_sensors(self):
        return None
    
    def step(self):
        sensor_data = self.update_sensors()
        action = self.DT.decide(sensor_data)
        self._last_action = action
        eval('self.'+action+'()')
        
    def next_level(self,level):
        self.reset()
        self.vel=self.max_vel+(level-1)*0.02