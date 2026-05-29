"""Custom NEAT reporter that logs per-generation stats to CSV and saves best-of-gen genomes.

Compatible with neat-python's BaseReporter interface.
"""
import os
import csv
import pickle
from datetime import datetime
import neat


class GenerationLogger(neat.reporting.BaseReporter):
    """Logs generation stats to CSV and saves the best genome of each generation."""

    def __init__(self, results_dir, strategy_name):
        self.results_dir = results_dir
        self.strategy_name = strategy_name
        self.winners_dir = os.path.join(results_dir, 'winners')
        self.logs_dir = os.path.join(results_dir, 'logs')
        os.makedirs(self.winners_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.csv_path = os.path.join(self.logs_dir, 'generations.csv')
        self._init_csv()

        self.best_ever_fitness = -float('inf')
        self.best_ever_genome = None
        self.best_ever_gen = -1
        self.current_generation = 0  # tracked manually

    def _init_csv(self):
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'generation', 'best_id', 'best_fitness', 'avg_fitness',
                'std_fitness', 'species_count', 'best_nodes', 'best_connections',
                'timestamp'
            ])

    # ── Required BaseReporter interface ──
    def start_generation(self, generation):
        self.current_generation = generation

    def end_generation(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species_set, best_genome):
        """Called after evaluation of a generation."""
        generation = self.current_generation

        fitnesses = [g.fitness for g in population.values()]
        best_fitness = best_genome.fitness
        avg_fitness = sum(fitnesses) / len(fitnesses)
        std_fitness = (sum((f - avg_fitness) ** 2 for f in fitnesses) / len(fitnesses)) ** 0.5
        species_count = len(species_set.species)
        best_nodes = len(best_genome.nodes)
        best_connections = len(best_genome.connections)

        # Save best-of-generation
        gen_winner_path = os.path.join(
            self.winners_dir, f'winner_gen_{generation:03d}.pkl'
        )
        with open(gen_winner_path, 'wb') as f:
            pickle.dump(best_genome, f)

        # Track best ever
        if best_fitness > self.best_ever_fitness:
            self.best_ever_fitness = best_fitness
            self.best_ever_genome = best_genome
            self.best_ever_gen = generation

        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                generation, best_genome.key, f'{best_fitness:.6f}',
                f'{avg_fitness:.6f}', f'{std_fitness:.6f}', species_count,
                best_nodes, best_connections, datetime.now().isoformat()
            ])

        print(f"[GEN {generation:03d}] best={best_fitness:.2f} avg={avg_fitness:.2f} "
              f"std={std_fitness:.2f} species={species_count} "
              f"size=({best_nodes},{best_connections}) id={best_genome.key}")

    def post_reproduction(self, config, population, species_set):
        pass

    def complete_extinction(self):
        print("[WARN] Population extinction event!")

    def found_solution(self, config, generation, best):
        print(f"[INFO] Solution found at generation {generation} (fitness={best.fitness:.2f})")

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass

    def save_final(self, config):
        """Save the overall best genome and print summary."""
        final_path = os.path.join(self.winners_dir, 'winner_final.pkl')
        with open(final_path, 'wb') as f:
            pickle.dump(self.best_ever_genome, f)

        print("\\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best overall fitness: {self.best_ever_fitness:.6f}")
        print(f"Achieved at generation: {self.best_ever_gen}")
        print(f"Best genome ID: {self.best_ever_genome.key}")
        print(f"Final winner saved to: {final_path}")
        print(f"Per-generation winners in: {self.winners_dir}")
        print(f"CSV log: {self.csv_path}")
        print("=" * 60)