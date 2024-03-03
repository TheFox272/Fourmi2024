import sys
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    tests = re.findall(r'nbp = (\d+), Cycle: (\d+) / (\d+), Temps total : ([\d.]+) sec', data)
    misc = re.findall(r'Temps total affichage : ([\d.]+) sec, Temps total calcul fourmis : ([\d.]+) sec, Temps total evapo phéromones : ([\d.]+) sec', data)

    return tests, misc


def calculate_speedup(tests):
    speedup_data = {}
    for test in tests:
        num_cpus = int(test[0])
        total_time = float(test[3])
        speedup_data[num_cpus] = total_time

    # Calculate speedup
    if 1 in speedup_data:
        baseline_time = speedup_data[1]
        speedup = {num_cpus: baseline_time / total_time for num_cpus, total_time in speedup_data.items()}
        t_total = speedup_data[1]
        return speedup, t_total
    else:
        print("Error: Data for 1 CPU not found. Ensure the data file contains information for 1 CPU.")
        return {}


def plot_speedup(speedup, misc_data, t_total, output_path=None):
    if not speedup:
        print("Error: Cannot plot speedup. No valid data.")
        return

    num_cpus = list(speedup.keys())
    speedup_values = list(speedup.values())

    # Create theorical speedup
    t_affichage = float(misc_data[0][0])
    t_calcul_fourmis = float(misc_data[0][1])
    t_evapo_pheromones = float(misc_data[0][2])
    x_th = np.arange(min(num_cpus), max(num_cpus) + 1)
    y_th = [1] + [t_total / (max(t_affichage, t_calcul_fourmis / (x-1)) + t_evapo_pheromones) for x in x_th[1:]]

    plt.figure(figsize=(10, 6))
    plt.plot(num_cpus, speedup_values, marker='o', label='Speedup')
    plt.plot(x_th, y_th, linestyle='--', color='red', label='Theorical speedup')

    plt.title('Speedup vs Number of CPUs')
    plt.xlabel('Number of (virtual) CPUs')
    plt.xticks(num_cpus)
    plt.ylabel('Speedup')
    plt.legend()

    if output_path:
        plt.savefig(output_path)
        print(f"Graph saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <data_file_path> <output_path>")
        sys.exit(1)

    data_file_path = sys.argv[1]
    output_path = sys.argv[2]

    tests_data, misc_data = parse_data(data_file_path)
    speedup_data, t_total = calculate_speedup(tests_data)
    plot_speedup(speedup_data, misc_data, t_total, output_path)
