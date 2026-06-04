import os
import neat
import typing as t

from neat_cars.radars_car.experiment import (train as train_radars_car, load as load_radars_car)

def neat_train_radars_car() -> neat.nn.FeedForwardNetwork:
    return train_radars_car(__get_config_file__())

def neat_load_radars_car() -> neat.nn.FeedForwardNetwork:
    return load_radars_car(__get_config_file__())

def __get_config_file__() -> str:
    local_dir: t.Union[t.Any, str] = os.path.dirname(__file__)
    config_path: str = os.path.join(local_dir, 'radars_car/config.ini')
    return config_path