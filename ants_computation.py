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
globCom.Bcast([size_laby, MPI.UINT32_T], root=0)

# Compute the local number of ants
nb_ants = (size_laby[0] * size_laby[1]) // 4
nb_ants_local = np.array([nb_ants // (nbp-1)], dtype=np.uint32)
if rank <= nb_ants - (nbp-1)*nb_ants_local:
    nb_ants_local[0] += 1

# Builds the array of the number of ants for each process, for later Gatherv
recvcounts = np.empty(nbp, dtype=np.uint32)
globCom.Allgather([np.array([nb_ants_local], dtype=np.uint32), MPI.UINT32_T], [recvcounts, MPI.UINT32_T])
displacements = np.empty(nbp, dtype=np.uint32)
displacements = np.cumsum(recvcounts) - recvcounts

# Receive max_life constant from master
max_life = np.zeros(1, dtype=np.uint32)
globCom.Bcast([max_life, MPI.UINT32_T], root=0)

# Receive pos_food_and_nest constant from master
pos_food_and_nest = np.zeros(4, dtype=np.uint32)
globCom.Bcast([pos_food_and_nest, MPI.UINT32_T], root=0)
pos_food = pos_food_and_nest[0], pos_food_and_nest[1]
pos_nest = pos_food_and_nest[2], pos_food_and_nest[3]

# Receive maze from master
a_maze = np.zeros(size_laby, dtype=np.int8)
globCom.Bcast([a_maze, MPI.INT8_T], root=0)

# Initialize the ants, pheromones, and food_counter
ants = colony.Colony_compute(nb_ants_local[0], pos_nest, max_life[0], 1 + displacements[rank])
pherom = pheromone.Pheromon(size_laby, pos_food)
food_counter = np.zeros(1, dtype=np.uint32)

# Initialize the exit_value
exit_value = np.zeros(1, dtype=np.bool8)
globCom.Bcast([exit_value, MPI.BOOL], root=0)

# Main loop, will compute the ants and send the results to the master until the master tells it to exit
while not exit_value[0]:
    # Receive the pheromones from the master
    globCom.Bcast([pherom.pheromon, MPI.DOUBLE], root=0)

    # Compute the ants
    food_counter[0] = ants.advance(a_maze, pos_food, pos_nest, pherom)

    # Send the resulting food_counter and pheromones to the master
    globCom.Reduce([food_counter[0], MPI.UINT32_T], None, op=MPI.SUM, root=0)
    globCom.Reduce([pherom.pheromon, MPI.DOUBLE], None, op=MPI.MAX, root=0)

    # Send the ants info to the master
    globCom.Gatherv([ants.max_life, MPI.INT32_T], None, root=0)
    globCom.Gatherv([ants.age, MPI.INT64_T], None, root=0)
    globCom.Gatherv([ants.historic_path, MPI.INT16_T], None, root=0)
    globCom.Gatherv([ants.directions, MPI.INT8_T], None, root=0)

    # Receive the exit_value from the master
    globCom.Bcast([exit_value, MPI.BOOL], root=0)
