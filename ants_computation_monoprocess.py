import numpy as np
import colony as colony
import pheromone
from mpi4py import MPI

# MPI initialization
globCom = MPI.COMM_WORLD.Dup()
nbp     = globCom.size
rank    = globCom.rank
name    = MPI.Get_processor_name()

# Receive size of labyrinth from master
size_laby = np.zeros(2, dtype=np.uint32)
globCom.Recv([size_laby, MPI.UINT32_T], source=0)

# Compute the number of ants
nb_ants = (size_laby[0] * size_laby[1]) // 4

# Receive max_life constant from master
max_life = np.zeros(1, dtype=np.uint32)
globCom.Recv([max_life, MPI.UINT32_T], source=0)

# Receive pos_food_and_nest constant from master
pos_food_and_nest = np.zeros(4, dtype=np.uint32)
globCom.Recv([pos_food_and_nest, MPI.UINT32_T], source=0)
pos_food = pos_food_and_nest[0], pos_food_and_nest[1]
pos_nest = pos_food_and_nest[2], pos_food_and_nest[3]

# Receive maze from master
a_maze = np.zeros(size_laby, dtype=np.int8)
globCom.Recv([a_maze, MPI.INT8_T], source=0)

# Initialize the ants, pheromones, and food_counter
ants = colony.Colony_compute(nb_ants, pos_nest, max_life[0])
pherom = pheromone.Pheromon(size_laby, pos_food)
food_counter = np.zeros(1, dtype=np.uint32)

# Initialize the exit_value
exit_value = np.zeros(1, dtype=np.bool8)
globCom.Recv([exit_value, MPI.BOOL], source=0)

# Main loop, will compute the ants and send the results to the master until the master tells it to exit
while not exit_value[0]:
    # Receive the pheromones from the master
    globCom.Recv([pherom.pheromon, MPI.DOUBLE], source=0)

    # Compute the ants
    food_counter[0] += ants.advance(a_maze, pos_food, pos_nest, pherom)

    # Send the resulting food_counter and pheromones to the master
    globCom.Send([food_counter[0], MPI.UINT32_T], dest=0)
    globCom.Send([pherom.pheromon, MPI.DOUBLE], dest=0)

    # Send the ants info to the master
    globCom.Send([ants.max_life, MPI.INT32_T], dest=0)
    globCom.Send([ants.age, MPI.INT64_T], dest=0)
    globCom.Send([ants.historic_path, MPI.INT16_T], dest=0)
    globCom.Send([ants.directions, MPI.INT8_T], dest=0)

    # Receive the exit_value from the master
    globCom.Recv([exit_value, MPI.BOOL], source=0)
