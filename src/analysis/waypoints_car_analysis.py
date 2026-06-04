"""Analysis script for waypoint NEAT training results.

Generates:
  • Fitness progression plot (best + average per generation)
  • Network complexity plot (nodes + connections over time)
  • Comparison table for the report

Usage:
    python src/analysis/waypoints_car_analysis.py
"""
import os
import csv
import matplotlib.pyplot as plt


def analyze(results_dir='results/waypoints'):
    csv_path = os.path.join(results_dir, 'logs', 'generations.csv')
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    generations, best, avg, std, nodes, conns = [], [], [], [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            generations.append(int(row['generation']))
            best.append(float(row['best_fitness']))
            avg.append(float(row['avg_fitness']))
            std.append(float(row['std_fitness']))
            nodes.append(int(row['best_nodes']))
            conns.append(int(row['best_connections']))

    # Plot fitness
    plt.figure(figsize=(10, 5))
    plt.plot(generations, best, 'b-', label='Best fitness')
    plt.plot(generations, avg, 'r-', label='Average fitness')
    plt.fill_between(generations,
                     [a - s for a, s in zip(avg, std)],
                     [a + s for a, s in zip(avg, std)],
                     color='red', alpha=0.15, label='±1 std dev')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Waypoint NEAT — Fitness Evolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(results_dir, 'plots', 'fitness_detailed.svg')
    plt.savefig(out)
    print(f"Saved fitness plot to {out}")
    plt.close()

    # Plot complexity
    plt.figure(figsize=(10, 5))
    plt.plot(generations, nodes, 'g-', label='Nodes')
    plt.plot(generations, conns, 'm-', label='Connections')
    plt.xlabel('Generation')
    plt.ylabel('Count')
    plt.title('Waypoint NEAT — Network Complexity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(results_dir, 'plots', 'complexity.svg')
    plt.savefig(out)
    print(f"Saved complexity plot to {out}")
    plt.close()

    # Summary for report
    best_idx = best.index(max(best))
    print("\n=== Waypoints Summary ===")
    print(f"Total generations: {len(generations)}")
    print(f"Best fitness: {best[best_idx]:.2f} at gen {generations[best_idx]}")
    print(f"Final avg fitness: {avg[-1]:.2f}")
    print(f"Best network size: {nodes[best_idx]} nodes, {conns[best_idx]} connections")


if __name__ == '__main__':
    analyze()