import time

from global_vars import *

def handle_collision(computer_car, computer_car2,game_info):
    computer_finish_poi_collide = computer_car.collide(
        FINISH_MASK, *FINISH_POSITION)
    if computer_finish_poi_collide != None:
        blit_text_center(WIN,MAIN_FONT,"YOU LOST!")
        pygame.display.update()
        pygame.time.wait(5000)
        game_info.reset()
        computer_car.next_level(1)
        return True

    computer_finish_poi_collide = computer_car2.collide(
        FINISH_MASK, *FINISH_POSITION)
    if computer_finish_poi_collide != None:
        blit_text_center(WIN,MAIN_FONT,"YOU LOST!")
        pygame.display.update()
        pygame.time.wait(5000)
        game_info.reset()
        computer_car2.next_level(1)
        return True
    return False


class GameInfo:
    LEVELS = 10

    def __init__(self, level=1):
        self.level = level
        self.started = False
        self.level_start_time = 0

    def next_level(self):
        self.level += 1
        self.started = False

    def reset(self):
        self.level = 1
        self.started = False
        self.level_start_time = 0

    def game_finished(self):
        return self.level > self.LEVELS

    def start_level(self):
        self.started = True
        self.level_start_time = time.time()

    def get_level_time(self):
        if not self.started:
            return 0
        return round(time.time() - self.level_start_time)