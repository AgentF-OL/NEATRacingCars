"""Demo script: load a trained winner and watch it race in pygame.

Usage:
    python src/demo_winner.py waypoints --gen 42    # best of generation 42
    python src/demo_winner.py waypoints --final      # final best overall
    python src/demo_winner.py radars --gen 10

    Melhorias:
  • Mostra mensagem quando o carro morre (stuck/off-track/finish).
  • Auto-resets o carro automaticamente para continuar a demo.
  • Não crasha com next_level faltando.
"""
import sys
import os
import pickle
import argparse
import neat
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from global_vars import *
from game import GameInfo
from cars.neat_waypoint_car import NeatWaypointCar
from cars.neat_radar_car import NeatRadarCar


def main(strategy, gen=None, use_final=False, winner_path=None):
    if winner_path is None:
        results_dir = os.path.join('results', strategy)
        if use_final:
            winner_path = os.path.join(results_dir, 'winners', 'winner_final.pkl')
        elif gen is not None:
            winner_path = os.path.join(results_dir, 'winners', f'winner_gen_{gen:03d}.pkl')
        else:
            print("Provide --gen N, --final, or --path")
            sys.exit(1)

    if not os.path.exists(winner_path):
        print(f"Winner not found: {winner_path}")
        sys.exit(1)

    config_path = os.path.join('config', f'neat_{strategy}.cfg')
    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_path)

    with open(winner_path, 'rb') as f:
        winner = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(winner, config)
    car = NeatWaypointCar(net) if strategy == 'waypoints' else NeatRadarCar(net)

    game_info = GameInfo()
    run = True
    clock = pygame.time.Clock()
    last_death_reason = ""
    valid_finishes = 0
    invalid_finishes = 0
    total_runs = 0

    print(f"Running demo for {strategy} — genome {winner.key} (fitness: {winner.fitness:.2f})")
    print("Controls: SPACE = reset | Q = quit")

    while run:
        clock.tick(FPS)

        for img, pos in images:
            WIN.blit(img, pos)
        level_text = MAIN_FONT.render(f'Demo — {strategy}', 1, WHITE)
        WIN.blit(level_text, (10, 10))
        car.draw(WIN)

        status_lines = [
            f"Genome: {winner.key}  Fitness: {winner.fitness:.1f}",
            f"Vel: {car.vel:.2f}  Angle: {car.angle}  WP: {getattr(car, 'current_point', 0)}",
            f"Off-track: {car.off_track_frames}f (total: {car.total_off_track_frames})",
            f"Frames: {car.frame_count}  Alive: {car.alive}",
            f"Runs: {total_runs}  Valid: {valid_finishes}  Invalid: {invalid_finishes}",
        ]
        if last_death_reason:
            status_lines.append(last_death_reason)

        for i, line in enumerate(status_lines):
            color = YELLOW if i == len(status_lines) - 1 and last_death_reason else WHITE
            surf = MAIN_FONT.render(line, 1, color)
            WIN.blit(surf, (10, 50 + i * 30))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    run = False
                if event.key == pygame.K_SPACE:
                    car.reset()
                    car.reset_neat_state()
                    last_death_reason = ""

        if not car.alive:
            total_runs += 1
            if car.finish_reached:
                if car.total_off_track_frames == 0:
                    last_death_reason = "FINISH LINE REACHED — VALID!"
                    valid_finishes += 1
                    blit_text_center(WIN, MAIN_FONT, "VALID FINISH!")
                else:
                    last_death_reason = f"FINISH — INVALID (off-track {car.total_off_track_frames}f)"
                    invalid_finishes += 1
                    blit_text_center(WIN, MAIN_FONT, "INVALID FINISH!")
            elif car.total_off_track_frames > 0:
                last_death_reason = f"DIED: OFF-TRACK ({car.total_off_track_frames}f)"
            else:
                last_death_reason = "DIED: STUCK (no movement)"

            print(f"[DEMO] {last_death_reason}")
            pygame.display.update()
            pygame.time.wait(1500)
            car.reset()
            car.reset_neat_state()
            continue

        car.step(verbose=False)

    pygame.quit()
    print(f"\\nDemo stats: {total_runs} runs, {valid_finishes} valid, {invalid_finishes} invalid")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('strategy', choices=['waypoints', 'radars'])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gen', type=int)
    group.add_argument('--final', action='store_true')
    parser.add_argument('--path', help='Direct path to a .pkl winner')
    args = parser.parse_args()
    main(args.strategy, args.gen, args.final, args.path)