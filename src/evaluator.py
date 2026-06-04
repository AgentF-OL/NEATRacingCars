"""Genome evaluation / fitness function.

Core principles:
  • Time is ONLY a cost — every frame alive burns fitness linearly.
  • Finish bonus is a flat reward multiplied by inverse elapsed time.
    Faster lap = exponentially more fitness.
  • Off-track is penalised from the very first frame (no 60-frame grace).
  • No benefit to surviving past the finish line — finish ends the episode.
"""

import math
import neat
import numpy as np

def eval_genome(genome, config, car_class, max_frames=60 * 60, trials=1):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    total_fitness = 0.0

    for trial in range(trials):
        car = car_class(net)
        car.reset()

        if trial > 0:
            car.x += (trial - 1) * 8
            car.prev_pos = (car.x, car.y)

        frames = 0
        while car.alive and frames < max_frames:
            car.step()
            frames += 1

        is_waypoint_car = hasattr(car, 'current_point')
        n_wp = len(car.path) if is_waypoint_car else 0

        # ── 1. WAYPOINT PROGRESS ──
        if is_waypoint_car:
            wp_raw = min(car.current_point, n_wp) * 250
            # Heavy off-track decay: even 1 frame off-track hurts progress reward
            off_ratio = min(car.total_off_track_frames / 200.0, 1.0)
            progress_bonus = wp_raw * (1.0 - off_ratio * 0.9)
        else:
            progress_bonus = car.in_track_distance * 0.3

        # ── 2. IN-TRACK DISTANCE ──
        dist_score = car.in_track_distance * 0.15

        # ── 3. TIME PENALTY (every frame costs, no exceptions) ──
        time_penalty = frames * 0.8

        # ── 4. SPEED BONUS (quadratic average speed) ──
        speed_bonus = 0.0
        if frames > 0:
            avg_speed = car.in_track_distance / frames
            speed_bonus = (avg_speed ** 2) * 60.0

        # ── 5. EFFICIENCY (waypoints per frame, capped) ──
        efficiency_bonus = 0.0
        if frames > 0 and is_waypoint_car:
            wp_per_frame = min(car.current_point, n_wp) / frames
            efficiency_bonus = wp_per_frame * 1000.0

        # ── 6. FINISH BONUS (flat + inverse-time multiplier) ──
        finish_bonus = 0
        if car.finish_reached:
            wp_ratio = (min(car.current_point, n_wp) / n_wp) if n_wp > 0 else 1.0
            if wp_ratio >= 0.85:
                # Base reward for finishing at all
                base_finish = 2000

                # INVERSE TIME MULTIPLIER: faster = much more
                # elapsed_time factor: 1.0 at 0 frames, 0.0 at max_frames
                # We use a power curve so that 600f >> 3000f
                elapsed = frames
                time_factor = max(0.0, 1.0 - (elapsed / max_frames)) ** 2.5

                if car.total_off_track_frames == 0:
                    # Clean finish: full multiplier
                    finish_bonus = base_finish * (1.0 + 4.0 * time_factor)
                elif car.total_off_track_frames < 10:
                    # Slightly dirty: reduced multiplier
                    finish_bonus = base_finish * (0.5 + 1.5 * time_factor)
                else:
                    # Dirty finish: minimal reward, effectively a penalty
                    finish_bonus = base_finish * 0.1

        # ── 7. OFF-TRACK PENALTY (linear from frame 1, no grace) ──
        # Every single off-track frame costs. No safe threshold.
        off_penalty = car.total_off_track_frames * 2.0

        # ── 8. IDLE PENALTY ──
        idle_penalty = 0
        if car.in_track_distance < 1.0 and frames > 100:
            idle_penalty = 80

        fitness = (progress_bonus + dist_score + finish_bonus
                   + speed_bonus + efficiency_bonus
                   - time_penalty - off_penalty - idle_penalty)

        fitness = max(0.01, fitness)
        total_fitness += fitness

    return total_fitness / trials


def eval_genomes(genomes, config, car_class):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, car_class)
# Robust evaluation:
#   • 3 trials per genome (noise averages out)
#   • Fitness = minimum of trials (conservative — must perform consistently)
#   • Alternatively: mean minus std_dev (penalizes inconsistency)

#def eval_genome(genome, config, car_class, max_frames=60 * 60, trials=3):
#     net = neat.nn.FeedForwardNetwork.create(genome, config)
#     trial_fitnesses = []
#
#     for trial in range(trials):
#         car = car_class(net)
#         car.reset()
#
#         # Slightly varied starting position per trial
#         if trial > 0:
#             car.x += (trial - 1) * 4
#             car.prev_pos = (car.x, car.y)
#
#         frames = 0
#         while car.alive and frames < max_frames:
#             car.step()
#             frames += 1
#
#         is_waypoint_car = hasattr(car, 'current_point')
#         n_wp = len(car.path) if is_waypoint_car else 0
#
#         # ── 1. WAYPOINT PROGRESS ──
#         if is_waypoint_car:
#             wp_raw = min(car.current_point, n_wp) * 250
#             off_ratio = min(car.total_off_track_frames / 200.0, 1.0)
#             progress_bonus = wp_raw * (1.0 - off_ratio * 0.9)
#         else:
#             progress_bonus = car.in_track_distance * 0.3
#
#         # ── 2. IN-TRACK DISTANCE ──
#         dist_score = car.in_track_distance * 0.15
#
#         # ── 3. TIME PENALTY ──
#         time_penalty = frames * 0.8
#
#         # ── 4. SPEED BONUS ──
#         speed_bonus = 0.0
#         if frames > 0:
#             avg_speed = car.in_track_distance / frames
#             speed_bonus = (avg_speed ** 2) * 60.0
#
#         # ── 5. EFFICIENCY ──
#         efficiency_bonus = 0.0
#         if frames > 0 and is_waypoint_car:
#             wp_per_frame = min(car.current_point, n_wp) / frames
#             efficiency_bonus = wp_per_frame * 1000.0
#
#         # ── 6. FINISH BONUS (clean = jackpot, dirty = peanuts) ──
#         finish_bonus = 0
#         if car.finish_reached:
#             wp_ratio = (min(car.current_point, n_wp) / n_wp) if n_wp > 0 else 1.0
#             if wp_ratio >= 0.85:
#                 if car.total_off_track_frames == 0:
#                     finish_bonus = 1000 + max(0, max_frames - frames) * 3.0
#                 elif car.total_off_track_frames < 10:
#                     finish_bonus = 300 + max(0, max_frames - frames) * 1.0
#                 else:
#                     finish_bonus = 25
#
#         # ── 7. OFF-TRACK PENALTY (linear from frame 1) ──
#         off_penalty = car.total_off_track_frames * 2.0
#
#         # ── 8. IDLE PENALTY ──
#         idle_penalty = 0
#         if car.in_track_distance < 1.0 and frames > 100:
#             idle_penalty = 80
#
#         fitness = (progress_bonus + dist_score + finish_bonus
#                    + speed_bonus + efficiency_bonus
#                    - time_penalty - off_penalty - idle_penalty)
#
#         fitness = max(0.01, fitness)
#         trial_fitnesses.append(fitness)
#
#     # Conservative: use the WORST trial (genome must be robust)
#     # Alternative: return np.mean(trial_fitnesses) - np.std(trial_fitnesses)
#     return max(trial_fitnesses)