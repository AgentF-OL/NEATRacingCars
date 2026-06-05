"""Race mode: load any number of NEAT winners and watch them compete in real-time.

Usage examples:
    python src/demo_race.py --waypoints final --radars final
    python src/demo_race.py --waypoints gen_868 --radars gen_042
    python src/demo_race.py --waypoints path/to/custom.pkl
    python src/demo_race.py --waypoints gen_868 --waypoints gen_900 --radars final
"""
import sys
import os
import pickle
import argparse
import math

import pygame
import neat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ═════════════════════════════════════════════════════════════════
#  1. global_vars FIRST (triggers track_config.json load)
# ═════════════════════════════════════════════════════════════════
import global_vars  # noqa: F401
from global_vars import *
from game import GameInfo

from cars.neat_waypoint_car import NeatWaypointCar
from cars.neat_radar_car import NeatRadarCar


# ═════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════
def load_winner(strategy, spec, config):
    """spec can be: 'final', 'gen_NNN', or an absolute/relative file path."""
    if spec == "final":
        path = os.path.join("results", "waypoints", "winners", "winner_final.pkl")
    elif spec.startswith("gen_"):
        path = os.path.join("results", strategy, "winners", f"winner_{spec}.pkl")
    else:
        path = spec

    if not os.path.exists(path):
        raise FileNotFoundError(f"Winner not found: {path}")

    with open(path, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return net, genome


def apply_track_config():
    """Push custom start pos/angle into both car classes."""
    if hasattr(global_vars, "CUSTOM_START_POS") and global_vars.CUSTOM_START_POS:
        NeatWaypointCar.START_POS = global_vars.CUSTOM_START_POS
        NeatRadarCar.START_POS = global_vars.CUSTOM_START_POS
    if hasattr(global_vars, "CUSTOM_START_ANGLE"):
        NeatWaypointCar.START_ANGLE = global_vars.CUSTOM_START_ANGLE
        NeatRadarCar.START_ANGLE = global_vars.CUSTOM_START_ANGLE


def _make_car(strategy, net, index, total):
    """Create car and apply a small x-offset so they don't spawn exactly on top."""
    if strategy == "waypoints":
        car = NeatWaypointCar(net)
    else:
        car = NeatRadarCar(net)

    if total > 1:
        offset = (index - (total - 1) / 2) * 22
        car.x += offset
        car.prev_pos = (car.x, car.y)

    return car


def _draw_hud(win, cars, game_info, fonts):
    """Draw small info boxes near each car."""
    for i, (car, label) in enumerate(cars):
        x = int(car.x + CAR_SIZE[0])
        y = int(car.y + CAR_SIZE[1])

        # Background tag
        tag = f"{label}  wp:{getattr(car,'current_point',0)}  v:{car.vel:.1f}"
        if not car.alive:
            tag += "  [DEAD]"
        elif car.finish_reached:
            tag += "  [FINISH]"

        surf = fonts["small"].render(tag, True, (255, 255, 255))
        bg = pygame.Surface((surf.get_width() + 8, surf.get_height() + 4))
        bg.fill((30, 30, 30))
        bg.set_alpha(180)
        win.blit(bg, (x + 12, y - 10))
        win.blit(surf, (x + 16, y - 8))


# ═════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="NEAT car race demo")
    parser.add_argument(
        "--waypoints", action="append", metavar="SPEC",
        help='Waypoints winner: "final", "gen_NNN", or path to .pkl'
    )
    parser.add_argument(
        "--radars", action="append", metavar="SPEC",
        help='Radars winner: "final", "gen_NNN", or path to .pkl'
    )
    args = parser.parse_args()

    if not args.waypoints and not args.radars:
        print("Provide at least one car, e.g. --waypoints final --radars final")
        sys.exit(1)

    # ── Load NEAT configs ──
    configs = {}
    for strategy in ("waypoints", "radars"):
        cfg_path = os.path.join("config", f"neat_{strategy}.cfg")
        if os.path.exists(cfg_path):
            configs[strategy] = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                cfg_path,
            )

    # ── Apply track setup (finish line, waypoints, start pos/angle) ──
    _apply_track_config()

    # ── Build car list ──
    car_specs = []   # list of (strategy, net, genome, label)
    for strategy, specs in (("waypoints", args.waypoints), ("radars", args.radars)):
        if not specs:
            continue
        if strategy not in configs:
            print(f"Missing config for {strategy}")
            continue
        cfg = configs[strategy]
        for spec in specs:
            try:
                net, genome = _load_winner(strategy, spec, cfg)
                label = f"{strategy[:3].upper()}-{genome.key}"
                car_specs.append((strategy, net, genome, label))
            except Exception as e:
                print(f"[warn] Could not load {strategy} '{spec}': {e}")

    if not car_specs:
        print("No cars could be loaded.")
        sys.exit(1)

    # ── Pygame init ──
    pygame.init()
    pygame.font.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT Race")
    clock = pygame.time.Clock()

    fonts = {
        "main": MAIN_FONT,
        "small": pygame.font.SysFont("consolas", 18),
    }

    # ── Race loop ──
    def reset_race():
        cars = []
        for i, (strategy, net, genome, label) in enumerate(car_specs):
            car = _make_car(strategy, net, i, len(car_specs))
            cars.append((car, label))
        return cars, GameInfo()

    cars, game_info = reset_race()
    running = True
    winner = None
    winner_frame = 0
    frame_count = 0
    paused = False

    while running:
        clock.tick(FPS)
        if not paused:
            frame_count += 1

        # ── Events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    # Restart race with same cars
                    cars, game_info = reset_race()
                    winner = None
                    winner_frame = 0
                    frame_count = 0

        if paused:
            # Still draw, just don't step
            pass

        # ── Step all alive cars ──
        if not paused:
            for car, _ in cars:
                if car.alive and not car.finish_reached:
                    car.step(verbose=False)

        # ── Check winner ──
        if winner is None:
            for car, label in cars:
                if car.finish_reached:
                    winner = label
                    winner_frame = frame_count
                    break

        # ── Draw ──
        for img, pos in images:
            win.blit(img, pos)

        # Draw finish line (already rotated by global_vars if needed)
        win.blit(FINISH, FINISH_POSITION)

        # Draw waypoints if any car uses them
        for car, _ in cars:
            if hasattr(car, "draw_points"):
                car.draw_points(win)

        # Draw cars
        for car, _ in cars:
            car.draw(win)

        # HUD
        _draw_hud(win, cars, game_info, fonts)

        # Top overlay
        top_text = f"Frame: {frame_count}  |  Press SPACE=Pause  R=Restart  ESC=Quit"
        if winner:
            top_text += f"  |  WINNER: {winner} @ frame {winner_frame}"
        elif paused:
            top_text += "  |  PAUSED"
        surf = fonts["small"].render(top_text, True, (255, 255, 0))
        win.blit(surf, (10, 10))

        pygame.display.update()

        # End condition: all dead or finished
        all_done = all(not c.alive or c.finish_reached for c, _ in cars)
        if all_done and winner is not None:
            # Keep showing for a moment, then allow restart
            pass

    pygame.quit()
    print(f"\nRace over. Winner: {winner} at frame {winner_frame}" if winner else "\nRace aborted.")


if __name__ == "__main__":
    main()