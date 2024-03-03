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
        first_char += "END OF SIMULATION" + '-' * 200 + f"\nnbp = {nbp}, "
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
    globCom.Bcast([np.array([True], dtype=np.bool8), MPI.BOOL], root=0)
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
globCom.Bcast([size_laby, MPI.UINT32_T], root=0)

# Define the number of ants, the other processes will compute it locally so no need to send it
nb_ants = size_laby[0] * size_laby[1] // 4

# No ants ants will be computed locally
nb_ants_local = np.zeros(1, dtype=np.uint32)

# Builds the array of the number of ants for each process, for later Gatherv
recvcounts = np.empty(nbp, dtype=np.uint32)
globCom.Allgather([np.array([nb_ants_local], dtype=np.uint32), MPI.UINT32_T], [recvcounts, MPI.UINT32_T])
displacements = np.empty(nbp, dtype=np.uint32)
displacements = np.cumsum(recvcounts) - recvcounts

# Initialize the max_life constant
max_life = np.array([1000], dtype=np.uint32)
if len(sys.argv) > 3:
    max_life[0] = int(sys.argv[3])

# Send the max_life constant to the other processes
globCom.Bcast([max_life, MPI.UINT32_T], root=0)

# Defines a recvcounts that will be used for the Gatherv of the ants' historic_path
historic_recvcounts = recvcounts.copy() * (max_life + 1) * 2
displacements_recvcounts = displacements.copy() * (max_life + 1) * 2

# Initialize the position of the food and the nest
pos_food = size_laby[0] - 1, size_laby[1] - 1
pos_nest = 0, 0

# Send pos_food and pos_nest to the other processes
pos_food_and_nest = np.array([pos_food[0], pos_food[1], pos_nest[0], pos_nest[1]], dtype=np.uint32)
globCom.Bcast([pos_food_and_nest, MPI.UINT32_T], root=0)

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
globCom.Bcast([a_maze.maze, MPI.INT8_T], root=0)

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
    globCom.Bcast([np.array([False], dtype=np.bool8), MPI.BOOL], root=0)
    # Send the pheromones to the other processes
    globCom.Bcast([pherom.pheromon, MPI.DOUBLE], root=0)

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
    globCom.Reduce([food_counter[0], MPI.UINT32_T], [food_counter, MPI.UINT32_T], op=MPI.SUM, root=0)
    t_wait += time.time() - t1_wait
    t1_comm_ants = time.time()
    null_pheromons = pherom.pheromon.copy() * 0
    globCom.Reduce([null_pheromons, MPI.DOUBLE], [pherom.pheromon, MPI.DOUBLE], op=MPI.MAX, root=0)

    # Receive the ants info from the other processes
    globCom.Gatherv([np.empty(0, np.int32), MPI.INT32_T], [ants.max_life, recvcounts, displacements, MPI.INT32_T], root=0)
    globCom.Gatherv([np.empty(0, np.int64), MPI.INT64_T], [ants.age, recvcounts, displacements, MPI.INT64_T], root=0)
    globCom.Gatherv([np.empty(0, np.int16), MPI.INT16_T], [ants.historic_path, historic_recvcounts, displacements_recvcounts, MPI.INT16_T], root=0)
    globCom.Gatherv([np.empty(0, np.int8), MPI.INT8_T], [ants.directions, recvcounts, displacements, MPI.INT8_T], root=0)

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
