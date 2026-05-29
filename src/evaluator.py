"""Genome evaluation / fitness function.

Reescrito para EVITAR CORTA-MATO:
  • Distância só conta quando IN-TRACK (in_track_distance).
  • Progresso de waypoints é anulado/reduzido se houver off-track.
  • Penalização pesada por total_off_track_frames (acumulado).
  • Bonus por corrida limpa (zero off-track).
  • Kill mais rápido (30 frames off-track consecutivos).

  Anti-cheat measures:
  • Finish line ONLY counts if car is in-track when crossing.
  • Distance only counts in-track (in_track_distance).
  • Heavy quadratic penalty for total off-track time.
  • Bonus for clean runs (zero off-track).
  • Progress bonus reduced if off-track ratio is high.
"""


import math
import neat


def eval_genome(genome, config, car_class, max_frames=60 * 60, trials=1):
    """Evaluate a single genome over `trials` race attempts."""
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

        # ── Fitness composition ──
        is_waypoint_car = hasattr(car, 'current_point')

        # 1. Progress: waypoints only count if mostly clean
        if is_waypoint_car:
            wp_raw = car.current_point * 200
            off_track_ratio = car.total_off_track_frames / (frames + 1)
            progress_bonus = wp_raw * max(0, 1.0 - off_track_ratio * 5.0)
        else:
            progress_bonus = car.in_track_distance * 0.8

        # 2. Distance: ONLY in-track distance
        dist_score = car.in_track_distance * 0.5

        # 3. Finish bonus — ONLY if in-track when crossing (enforced in _check_finish_line)
        # and cleanliness bonus
        if car.finish_reached:
            cleanliness = max(0, 1.0 - car.total_off_track_frames / 100.0)
            finish_bonus = 5000 * cleanliness
            finish_bonus += max(0, (max_frames - frames)) * 0.5
        else:
            finish_bonus = 0

        # 4. Clean run bonus
        clean_bonus = 1000 if car.total_off_track_frames == 0 else 0

        # 5. Off-track penalty: QUADRATIC
        off_penalty = (car.total_off_track_frames ** 2) * 2.5

        # 6. Idle / stuck penalty
        idle_penalty = 0
        if car.in_track_distance < 30:
            idle_penalty = 1500

        # 7. Velocity efficiency
        efficiency_bonus = 0
        if frames > 0:
            avg_speed = car.in_track_distance / frames
            efficiency_bonus = avg_speed * 50

        fitness = (progress_bonus + dist_score + finish_bonus + clean_bonus
                   + efficiency_bonus - off_penalty - idle_penalty)

        fitness = max(0.01, fitness)
        total_fitness += fitness

    return total_fitness / trials


def eval_genomes(genomes, config, car_class):
    """Callback for NEAT-Python `Population.run()`."""
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, car_class)