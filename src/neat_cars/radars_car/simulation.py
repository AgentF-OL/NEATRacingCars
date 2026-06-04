from typing import Callable
import random
import neat

import config
from global_vars import *
from cars.radars_car import RadarCar

def run_radars_car_simulation(
        car: RadarCar,
        fitness_fun: Callable[[tuple[float, float, float, dict[str, float]], tuple[float, float, float, dict[str, float]]], float],
        steps: int = 1000) -> float:
    assert(steps > 0)
    curr_step = 0
    fitness_sum = 0.0
    car.x, car.y = random_start_pos(car)
    #draw_car(car)
    while curr_step < steps:
        before: tuple[float, float, float, dict[str, float]] = (car.x, car.y, car.angle, car.radar_distances.copy())
        car.compute_radar_distances()
        car.step(verbose=False, stdout_verbose=False)
        #car.step(verbose=True, stdout_verbose=False)
        after: tuple[float, float, float, dict[str, float]] = (car.x, car.y, car.angle, car.radar_distances)
        fitness_sum += fitness_fun(before, after)
        #draw_car(car)
        curr_step += 1
    return fitness_sum / curr_step

def random_start_pos(car: RadarCar) -> tuple[int, int]:
    while True:
        x = random.randint(int(CAR_SIZE[0]), WIDTH - 1 - int(CAR_SIZE[0]))
        y = random.randint(int(CAR_SIZE[1]), HEIGHT - 1 - int(CAR_SIZE[1]))
        car.x, car.y = x, y
        if car.collide(TRACK_BORDER_MASK) == None and TRACK_MASK.get_at((x, y)) != YELLOW: return x, y

def simulate(cars: list[RadarCar], genomes: list[neat.DefaultGenome], generation: int) -> None:
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        for car in cars:
            car.step()

        cars_alive: int = 0
        for i, car in enumerate(cars):
            if car.is_alive:
                cars_alive += 1
                genomes[i][1].fitness += config.FITNESS_FUN_RADARS_CAR(car)

        if cars_alive == 0:
            break

        draw_images()
        for car in cars:
            if car.is_alive:
                car.draw(WIN, MAIN_FONT)
        blit_text_center(WIN, MAIN_FONT, f"Generation: {generation}")

        pygame.display.update()
        clock.tick(FPS)
