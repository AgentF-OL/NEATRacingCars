import config
from global_vars import *
from cars.radars_car import RadarCar

def tight_curves_fitness(car: RadarCar) -> float:
    return car.distance_traveled / 50