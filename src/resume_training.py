"""Resume training from a NEAT checkpoint.

Usage:
    python src/resume_training.py results/waypoints/checkpoints/gen_0050.pkl --generations 50
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


def resume(checkpoint_path: str, additional_generations: int):
    parts = os.path.normpath(checkpoint_path).split(os.sep)
    try:
        strategy = parts[parts.index('results') + 1]
    except ValueError:
        strategy = 'unknown'
    results_dir = os.path.join('results', strategy)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

    pygame.init()
    p = neat.Checkpointer.restore_checkpoint(checkpoint_path)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    gen_logger = GenerationLogger(results_dir, strategy)
    p.add_reporter(gen_logger)

    p.add_reporter(neat.Checkpointer(
        10,
        filename_prefix=os.path.join(results_dir, 'checkpoints', 'gen_')
    ))

    car_class = NeatWaypointCar if strategy == 'waypoints' else NeatRadarCar

    winner = p.run(
        lambda genomes, cfg: eval_genomes(genomes, cfg, car_class),
        n=additional_generations
    )

    gen_logger.save_final(p.config)

    plot_stats(stats, ylog=False, view=False,
               filename=os.path.join(results_dir, 'plots', 'fitness.svg'))
    draw_net(p.config, winner, view=False,
             filename=os.path.join(results_dir, 'plots', 'network.svg'))

    print(f"Resumed training complete. Final winner saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Path to .pkl checkpoint')
    parser.add_argument('--generations', type=int, default=50)
    args = parser.parse_args()
    resume(args.checkpoint, args.generations)