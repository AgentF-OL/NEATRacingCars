"""Unified training script for both waypoints and radars.

Usage:
    python src/train_unified.py --strategy waypoints --generations 100
    python src/train_unified.py --strategy radars --generations 100

Organizes outputs under results/{strategy}/ with checkpoints, winners, logs, and plots.
"""
import os
import sys
import argparse
import neat
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluator import eval_genomes
from cars.neat_waypoint_car import NeatWaypointCar
from cars.neat_radar_car import NeatRadarCar
from generation_logger import GenerationLogger
from analysis_utils import plot_stats, draw_net


def train(strategy: str, generations: int, results_root: str = 'results'):
    results_dir = os.path.join(results_root, strategy)
    os.makedirs(os.path.join(results_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, '..', 'config', f'neat_{strategy}.cfg')

    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_path)

    p = neat.Population(config)

    # Reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Custom logger (CSV + per-gen winners)
    gen_logger = GenerationLogger(results_dir, strategy)
    p.add_reporter(gen_logger)

    # Checkpoints every 10 generations
    p.add_reporter(neat.Checkpointer(
        10,
        filename_prefix=os.path.join(results_dir, 'checkpoints', 'gen_')
    ))

    # Car class selection
    car_class = NeatWaypointCar if strategy == 'waypoints' else NeatRadarCar

    # Run evolution
    winner = p.run(
        lambda genomes, cfg: eval_genomes(genomes, cfg, car_class),
        n=generations
    )

    # Save final artifacts
    gen_logger.save_final(config)

    plot_stats(stats, ylog=False, view=False,
               filename=os.path.join(results_dir, 'plots', 'fitness.svg'))
    draw_net(config, winner, view=False,
             filename=os.path.join(results_dir, 'plots', 'network.svg'))

    return winner, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NEAT racing cars')
    parser.add_argument('--strategy', choices=['waypoints', 'radars'], required=True)
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--results', type=str, default='results')
    args = parser.parse_args()

    pygame.init()
    winner, stats = train(args.strategy, args.generations, args.results)