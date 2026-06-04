"""Demo script: load a trained winner and watch it race in pygame.

Runs N consecutive trials and prints aggregate statistics so you can
see the TRUE performance of the genome, not just one noisy sample.
"""
import sys
import os
import pickle
import argparse
import neat
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ═════════════════════════════════════════════════════════════════
#  1. Import global_vars FIRST to trigger config load
#  2. Only THEN import car classes so they see the new FINISH/PATH
# ═════════════════════════════════════════════════════════════════
import global_vars  # noqa: F401
from global_vars import *
from game import GameInfo

# Car classes imported AFTER global_vars has loaded config
from cars.neat_waypoint_car import NeatWaypointCar
from cars.neat_radar_car import NeatRadarCar

# Apply custom start position / angle
if hasattr(global_vars, 'CUSTOM_START_POS') and global_vars.CUSTOM_START_POS:
    NeatWaypointCar.START_POS = global_vars.CUSTOM_START_POS
    NeatRadarCar.START_POS = global_vars.CUSTOM_START_POS

if hasattr(global_vars, 'CUSTOM_START_ANGLE'):
    NeatWaypointCar.START_ANGLE = global_vars.CUSTOM_START_ANGLE
    NeatRadarCar.START_ANGLE = global_vars.CUSTOM_START_ANGLE


def run_single_trial(strategy, net, config, verbose=False):
    """Run one trial and return stats dict."""
    car = NeatWaypointCar(net) if strategy == 'waypoints' else NeatRadarCar(net)

    clock = pygame.time.Clock()
    total_frames = 0

    while car.alive and total_frames < 60 * 60:
        clock.tick(FPS)

        for img, pos in images:
            WIN.blit(img, pos)
        level_text = MAIN_FONT.render(f'Demo — {strategy}', 1, WHITE)
        WIN.blit(level_text, (10, 10))
        car.draw(WIN)

        wp = getattr(car, 'current_point', 0)
        total_wp = len(getattr(car, 'path', []))

        if verbose:
            status_lines = [
                f"WP: {wp}/{total_wp}  Vel: {car.vel:.2f}  Angle: {car.angle}",
                f"Off-track: {car.off_track_frames}f (total: {car.total_off_track_frames})",
                f"Frames: {car.frame_count}  Alive: {car.alive}",
            ]
            for i, line in enumerate(status_lines):
                surf = MAIN_FONT.render(line, 1, WHITE)
                WIN.blit(surf, (10, 50 + i * 30))
            pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None  # signal abort

        car.step(verbose=False)
        total_frames += 1

    # Determine outcome
    if car.finish_reached:
        if car.total_off_track_frames == 0:
            outcome = "VALID"
        else:
            outcome = f"DIRTY(off{car.total_off_track_frames})"
    else:
        outcome = f"DIED(wp{wp}/{total_wp})"

    return {
        'outcome': outcome,
        'frames': total_frames,
        'wp': wp,
        'total_wp': total_wp,
        'off_track': car.total_off_track_frames,
        'finish_reached': car.finish_reached,
    }


def main(strategy, gen=None, use_final=False, winner_path=None, num_trials=30):
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
    neat_config = neat.Config(neat.DefaultGenome,
                              neat.DefaultReproduction,
                              neat.DefaultSpeciesSet,
                              neat.DefaultStagnation,
                              config_path)

    with open(winner_path, 'rb') as f:
        winner = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(winner, neat_config)

    print(f"Running {num_trials} trials for {strategy} — genome {winner.key}")
    print(f"Training fitness: {winner.fitness:.2f}")
    print("Controls: Q = quit after current trial | SPACE = next trial immediately")
    print("-" * 50)

    results = []
    trial = 0
    run = True

    while run and trial < num_trials:
        trial += 1
        result = run_single_trial(strategy, net, neat_config, verbose=True)
        if result is None:
            break

        results.append(result)
        print(f"Trial {trial:2d}: {result['outcome']:20s} "
              f"frames={result['frames']:4d}  wp={result['wp']}/{result['total_wp']}")

        # Brief pause between trials, check for quit
        pygame.time.wait(300)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    run = False

    pygame.quit()

    # ── Aggregate statistics ──
    if not results:
        return

    valid = sum(1 for r in results if r['outcome'] == 'VALID')
    dirty = sum(1 for r in results if r['outcome'].startswith('DIRTY'))
    died = len(results) - valid - dirty

    finish_times = [r['frames'] for r in results if r['finish_reached']]
    off_tracks = [r['off_track'] for r in results if r['finish_reached']]

    print("\n" + "=" * 50)
    print("AGGREGATE RESULTS")
    print("=" * 50)
    print(f"Total trials: {len(results)}")
    print(f"  VALID finishes:   {valid} ({100*valid/len(results):.1f}%)")
    print(f"  DIRTY finishes:   {dirty} ({100*dirty/len(results):.1f}%)")
    print(f"  Died before end:  {died} ({100*died/len(results):.1f}%)")
    if finish_times:
        print(f"\nFinish times (frames):")
        print(f"  Mean: {sum(finish_times)/len(finish_times):.0f}")
        print(f"  Min:  {min(finish_times)}")
        print(f"  Max:  {max(finish_times)}")
    if off_tracks:
        print(f"\nOff-track frames (among finishers):")
        print(f"  Mean: {sum(off_tracks)/len(off_tracks):.1f}")
        print(f"  Min:  {min(off_tracks)}")
        print(f"  Max:  {max(off_tracks)}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('strategy', choices=['waypoints', 'radars'])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gen', type=int)
    group.add_argument('--final', action='store_true')
    parser.add_argument('--path', help='Direct path to a .pkl winner')
    parser.add_argument('--trials', type=int, default=30, help='Number of demo trials')
    args = parser.parse_args()
    main(args.strategy, args.gen, args.final, args.path, args.trials)