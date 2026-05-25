import numpy
import pygame
import math
import time
import random
import sys

from game import *
from cars.waypoints_car import *
from cars.radars_car import *
from utils import *

pygame.init()

green_car=DTGreenCar(4,4)
purple_car=AntecipateCar(4,4)
game_info=GameInfo()
run=True
clock = pygame.time.Clock()
iters = -1

while run:
    iters += 1
    print("Clock:", iters, "-")
    clock.tick(FPS)
    draw(WIN,images,green_car,purple_car,game_info)
    
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
                            
    green_car.step(verbose=True)
    purple_car.step(verbose=True)
    
    if handle_collision(green_car, purple_car,game_info):
        draw(WIN,images,green_car,purple_car,game_info)
    
    if game_info.game_finished():
        blit_text_center(WIN,MAIN_FONT,"YOU WON!")
        pygame.time.wait(5000)
        game_info.reset()
        green_car.reset()
        purple_car.next_level(1)

pygame.quit()