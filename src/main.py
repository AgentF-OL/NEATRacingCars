import sys
import neat
from collections.abc import Callable

import config as config
from game import *
from cars.waypoints_car import *
from cars.radars_car import *
from neat_cars.adapter import (neat_train_radars_car, neat_load_radars_car)
from utils import *

def neat_train_or_load_model(
        train_model: Callable[[], neat.nn.FeedForwardNetwork],
        load_model: Callable[[], neat.nn.FeedForwardNetwork],
        train: bool = True) -> neat.nn.FeedForwardNetwork:
    if train:
        return train_model()
    else:
        return load_model()

network_radars_car: neat.nn.FeedForwardNetwork = neat_train_or_load_model(
    neat_train_radars_car,
    neat_load_radars_car,
    config.TRAIN_RADARS_CAR
)

pygame.init()

green_car=DTGreenCar(4,4)
red_car=RadarCar(4, 4, network_radars_car)
game_info=GameInfo()
run=True
clock = pygame.time.Clock()
iters = -1

while run:
    iters += 1
    print("Clock:", iters, "-")
    clock.tick(FPS)
    draw(WIN,images,green_car,red_car,game_info)
    
    while not game_info.started:
        blit_text_center(WIN, MAIN_FONT, f'Press any key to start level {game_info.level}!')
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run=False
                pygame.quit()
                sys.exit()
            if event.type==pygame.KEYDOWN:
                game_info.start_level()
            
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run=False
            break
                            
    green_car.step(verbose=config.VERBOSE)
    red_car.step(verbose=config.VERBOSE, stdout_verbose=config.STDOUT_VERBOSE)
    
    if handle_collision(red_car, green_car, game_info):
        draw(WIN,images,green_car,red_car,game_info)
    
    if game_info.game_finished():
        blit_text_center(WIN,MAIN_FONT,"YOU WON!")
        pygame.time.wait(5000)
        game_info.reset()
        green_car.reset()
        red_car.reset()

pygame.quit()