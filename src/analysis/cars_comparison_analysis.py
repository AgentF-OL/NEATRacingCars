"""Compare waypoint vs radar NEAT strategies.

Generates side-by-side fitness plots and a summary table.

Usage:
    python src/analysis/cars_comparison_analysis.py
"""
import os
import csv
import matplotlib.pyplot as plt


def load_csv(path):
    generations, best, avg = [], [], []
    if not os.path.exists(path):
        return None, None, None
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            generations.append(int(row['generation']))
            best.append(float(row['best_fitness']))
            avg.append(float(row['avg_fitness']))
    return generations, best, avg


def compare(results_root='results'):
    wp_csv = os.path.join(results_root, 'waypoints', 'logs', 'generations.csv')
    rd_csv = os.path.join(results_root, 'radars', 'logs', 'generations.csv')

    g_wp, b_wp, a_wp = load_csv(wp_csv)
    g_rd, b_rd, a_rd = load_csv(rd_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if g_wp:
        axes[0].plot(g_wp, b_wp, 'b-', label='Waypoints best')
        axes[0].plot(g_wp, a_wp, 'b--', alpha=0.6, label='Waypoints avg')
    if g_rd:
        axes[0].plot(g_rd, b_rd, 'r-', label='Radars best')
        axes[0].plot(g_rd, a_rd, 'r--', alpha=0.6, label='Radars avg')

    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Fitness')
    axes[0].set_title('Fitness Comparison')
    axes[0].legend()
    axes[0].grid(True)

    # Bar chart: best final fitness
    labels, bests = [], []
    if b_wp:
        labels.append('Waypoints')
        bests.append(max(b_wp))
    if b_rd:
        labels.append('Radars')
        bests.append(max(b_rd))

    axes[1].bar(labels, bests, color=['blue', 'red'])
    axes[1].set_ylabel('Best Fitness')
    axes[1].set_title('Best Overall Fitness')
    axes[1].grid(True, axis='y')

    plt.tight_layout()
    out = os.path.join(results_root, 'comparison.svg')
    plt.savefig(out)
    print(f"Saved comparison plot to {out}")
    plt.close()

    # Text summary
    print("\n=== Comparison Summary ===")
    if b_wp:
        print(f"Waypoints: best={max(b_wp):.2f} at gen {g_wp[b_wp.index(max(b_wp))]}")
    if b_rd:
        print(f"Radars:    best={max(b_rd):.2f} at gen {g_rd[b_rd.index(max(b_rd))]}")


if __name__ == '__main__':
    compare()