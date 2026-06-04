"""Utilities for plotting fitness and visualising network topology.

Compatible with neat-python's StatisticsReporter.
If `visualize.py` from the book examples is present, it is used for
the network graph; otherwise a simple matplotlib fallback is provided.
"""
import os
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def plot_stats(statistics, ylog=False, view=False, filename='fitness.svg'):
    """Plot best and average fitness per generation."""
    if plt is None:
        print("[analysis] matplotlib not installed, skipping plot_stats")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = statistics.get_fitness_mean()

    plt.figure(figsize=(10, 6))
    plt.plot(generation, best_fitness, 'b-', label="best")
    plt.plot(generation, avg_fitness, 'r-', label="average")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="best")
    plt.grid(True)
    if ylog:
        plt.gca().set_yscale('symlog')
    plt.tight_layout()
    plt.savefig(filename)
    if view:
        plt.show()
    plt.close()
    print(f"[analysis] Saved fitness plot to {filename}")


def draw_net(config, genome, view=False, filename=None):
    """Render genome topology to SVG/PNG."""
    # Try the book's visualize module first
    try:
        import visualize
        visualize.draw_net(config, genome, view=view, filename=filename)
        print(f"[analysis] Saved network plot to {filename}")
        return
    except Exception:
        pass

    # Fallback: just save a text description
    if filename:
        txt = filename.replace('.svg', '.txt').replace('.png', '.txt')
        with open(txt, 'w') as f:
            f.write(str(genome))
        print(f"[analysis] Saved genome text to {txt}")
