"""
Module managing an ant colony in a labyrinth.
"""
import sys
import time
import numpy as np
import maze
import colony as colony
import pheromone
import pygame as pg
from mpi4py import MPI

# MPI initialization
globCom = MPI.COMM_WORLD.Dup()
nbp = globCom.size
rank = globCom.rank
name = MPI.Get_processor_name()


def print_stats(in_one_line=True):
    """
    Print the current stats of the simulation
    """
    first_char = "\r"
    if not in_one_line:
        first_char += "END OF SIMULATION" + '-' * 200 + "\nnbp = 2, "
    sys.stdout.write(f"{first_char}Cycle: {cycle + 1} / {max_exec_cycle}, "
                     f"Temps total : {round(t_total, 5)} sec, "
                     f"Temps total affichage : {round(t_display, 5)} sec, "
                     f"Temps total comm de P0 : {round(t_comm, 5)} sec, "
                     f"Temps total attente de P0 : {round(t_wait, 5)} sec, "
                     f"Temps total comm vers P0 : {round(t_comm_ants, 5)} sec, "
                     f"Temps total evapo phéromones : {round(t_pheromon, 5)} sec, "
                     f"FPS : {1. / (end - deb):6.2f}, nourriture : {food_counter[0]:7d} ")
    sys.stdout.flush()


def exit_function():
    """
    Function to exit the program
    """
    # End the pygame module
    pg.quit()
    # Tell the other processes to exit as well
    globCom.Send([np.array([True], dtype=np.bool8), MPI.BOOL], dest=1)
    print_stats(False)
    print("")
    # Exit the program
    exit(0)


# Initialize the pygame module
pg.init()

# Initialize the size of the labyrinth
size_laby = np.array([25, 25], dtype=np.uint32)
if len(sys.argv) > 2:
    size_laby[0] = int(sys.argv[1])
    size_laby[1] = int(sys.argv[2])
# Define the resolution of the labyrinth
resolution = size_laby[1] * 8, size_laby[0] * 8
screen = pg.display.set_mode(resolution)

# Send the size of the labyrinth to the other processes
globCom.Send([size_laby, MPI.UINT32_T], dest=1)

# Define the number of ants, the other processes will compute it locally so no need to send it
nb_ants = size_laby[0] * size_laby[1] // 4

# Initialize the max_life constant
max_life = np.array([1000], dtype=np.uint32)
if len(sys.argv) > 3:
    max_life[0] = int(sys.argv[3])

# Send the max_life constant to the other processes
globCom.Send([max_life, MPI.UINT32_T], dest=1)

# Initialize the position of the food and the nest
pos_food = size_laby[0] - 1, size_laby[1] - 1
pos_nest = 0, 0

# Send pos_food and pos_nest to the other processes
pos_food_and_nest = np.array([pos_food[0], pos_food[1], pos_nest[0], pos_nest[1]], dtype=np.uint32)
globCom.Send([pos_food_and_nest, MPI.UINT32_T], dest=1)

# Initialize the alpha and beta constants, which won't be needed by the other processes
alpha = 0.9
beta = 0.99
if len(sys.argv) > 4:
    alpha = float(sys.argv[4])
if len(sys.argv) > 5:
    beta = float(sys.argv[5])

# Initialize the maze
a_maze = maze.Maze(size_laby, 12345)
# Send the maze to the other processes, to make sure they all have the same in case we modify its seed
globCom.Send([a_maze.maze, MPI.INT8_T], dest=1)

# Initialize the ants, pheromones and the maze display, no need to send them to the other processes
ants = colony.Colony_display(nb_ants, max_life[0])
pherom = pheromone.Pheromon(size_laby, pos_food, alpha, beta)

# Display the maze
mazeImg = a_maze.display()

# Initialize the food counter
food_counter = np.zeros(1, dtype=np.uint32)

# Initialize the timers
t_display = 0.
t_comm = 0.
t_wait = 0.
t_comm_ants = 0.
t_pheromon = 0.
t_total = 0.

snapshop_taken = False

# Initialize the maximum number of cycles of the simulation
max_exec_cycle = 5000
if len(sys.argv) > 6:
    max_exec_cycle = int(sys.argv[6])

t1_total = time.time()
for cycle in range(max_exec_cycle):
    for event in pg.event.get():
        if event.type == pg.QUIT:
            exit_function()

    deb = time.time()
    t1_comm = time.time()

    # Tell the other processes to continue computing
    globCom.Send([np.array([False], dtype=np.bool8), MPI.BOOL], dest=1)
    # Send the pheromones to the other processes
    globCom.Send([pherom.pheromon, MPI.DOUBLE], dest=1)

    t_comm += time.time() - t1_comm
    t1_display = time.time()

    # Display everything
    pherom.display(screen)
    screen.blit(mazeImg, (0, 0))
    ants.display(screen)
    pg.display.update()

    t_display += time.time() - t1_display
    t1_wait = time.time()

    # Receive the food counter & pheromones from the other processes
    globCom.Recv([food_counter, MPI.UINT32_T], source=1)
    t_wait += time.time() - t1_wait
    t1_comm_ants = time.time()
    globCom.Recv([pherom.pheromon, MPI.DOUBLE], source=1)

    # Receive the ants info from the other processes
    globCom.Recv([ants.max_life, MPI.INT32_T], source=1)
    globCom.Recv([ants.age, MPI.INT64_T], source=1)
    globCom.Recv([ants.historic_path, MPI.INT16_T], source=1)
    globCom.Recv([ants.directions, MPI.INT8_T], source=1)

    t_comm_ants += time.time() - t1_comm_ants
    t1_pheromon = time.time()

    # Compute the evaporation of the pheromones
    pherom.do_evaporation(pos_food)

    t_pheromon += time.time() - t1_pheromon
    end = time.time()

    if food_counter[0] == 1 and not snapshop_taken:
        pg.image.save(screen, "MyFirstFood.png")
        snapshop_taken = True

    # Update the timers
    t_total = time.time() - t1_total

    # Print the current stats of the simulation
    print_stats()

    # pg.time.wait(500)  # To slow down the display

exit_function()
