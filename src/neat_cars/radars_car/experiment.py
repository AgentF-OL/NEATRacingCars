import os
import neat
import multiprocessing as mp
import pickle

import config as cfg
import neat_cars.visualize as visualize
import neat_cars.radars_car.simulation as simulation
from cars.radars_car import RadarCar

GENERATION = [0]

def train(config_file: str) -> neat.nn.FeedForwardNetwork:
    config: neat.Config = load_config(config_file)

    filename_prefix: str = f"../{cfg.MODEL_DIR_RADARS_CAR}/"
    make_out_dir(filename_prefix)
    
    population: neat.Population = neat.Population(config, seed=cfg.SEED)
    stats: neat.StatisticsReporter = neat.StatisticsReporter()
    population.add_reporter(neat.StdOutReporter(True)) # Show progress in the terminal
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5, filename_prefix=f"{filename_prefix}neat-checkpoint-"))

    # Speed up the experiment by running the fitness evaluation in parallel between CPU cores
    #with neat.ParallelEvaluator(mp.cpu_count(), eval_genomes) as evaluator:
    #    winner: neat.DefaultGenome = population.run(evaluator.evaluate, cfg.N_GENERATIONS_RADARS_CAR)
    winner: neat.DefaultGenome = population.run(eval_genomes, cfg.N_GENERATIONS_RADARS_CAR)
    GENERATION[0] = 0
    print(f'\nBest genome:\n{winner!s}')
    save(winner)
    winner_network: neat.nn.FeedForwardNetwork = neat.nn.FeedForwardNetwork.create(winner, config)

    node_names: dict[int, str] = {
        -5: "left",
        -4: "left_middle",
        -3: "middle",
        -2: "right_middle",
        -1: "right",
        0: "vel",
        1: "left",
        2: "right"
    }
    visualize.draw_net(config, winner, True, node_names=node_names, filename=f"{filename_prefix}winner_net_unpruned")
    visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True, filename=f"{filename_prefix}winner_net_pruned")
    visualize.plot_stats(stats, ylog=False, view=True, filename=f"{filename_prefix}avg_fitness.svg")
    visualize.plot_species(stats, view=True, filename=f"{filename_prefix}speciation.svg")

    return winner_network

def load(config_file: str) -> neat.nn.FeedForwardNetwork:
    filename: str = f"../{cfg.MODEL_DIR_RADARS_CAR}/winner"
    config: neat.Config = load_config(config_file)
    with open(filename, "rb") as f:
        genome = pickle.load(f)
    return neat.nn.FeedForwardNetwork.create(genome, config)

def save(winner: neat.DefaultGenome) -> None:
    filename: str = f"../{cfg.MODEL_DIR_RADARS_CAR}/winner"
    with open(filename, "wb") as f:
        pickle.dump(winner, f)

def load_config(config_file: str) -> neat.Config:
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_file
    )

def eval_genomes(genomes: list[neat.DefaultGenome], config: neat.Config) -> None:
    genome: neat.DefaultGenome
    cars: list[RadarCar] = []
    print(f"Generation: {GENERATION[0]}")
    for _, genome in genomes:
        genome.fitness = 0.0
        network: neat.nn.FeedForwardNetwork = neat.nn.FeedForwardNetwork.create(genome, config)
        cars.append(RadarCar(4, 4, network))
    simulation.simulate(cars, genomes, GENERATION[0])
    GENERATION[0] += 1

def eval_fitness(network: neat.nn.FeedForwardNetwork) -> float:
    return simulation.run_radars_car_simulation(
        RadarCar(4, 4, network),
        cfg.FITNESS_FUN_RADARS_CAR,
        cfg.TOTAL_STEPS_SIMULATION_RADARS_CAR
    )

def make_out_dir(filename_prefix: str) -> None:
    if not os.path.exists(filename_prefix):
        os.makedirs(filename_prefix)