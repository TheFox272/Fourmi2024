#!/bin/bash

N_TEST=1
BASIC_PARAMS="50 50 2000000 0.9 0.99 5000"
OUTPUT_FILE="results/memory_tests_output.txt"
GRAPH_OUTPUT_FILE="graphs/memory_graph.png"

run_command() {
  echo "Running test $N_TEST..."
  ((N_TEST++))
  echo "$1"
  echo "" >> $OUTPUT_FILE
  echo "\$ $1" >> $OUTPUT_FILE
  eval "$1" | tail -n 1 >> $OUTPUT_FILE
  echo "Waiting for cache to clean itself..."
  sleep 5
}

# Make sure numpy won't parallelize the computation in your back
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Running memory tests..."
mkdir -p results
echo "MEMORY TESTS :" > $OUTPUT_FILE

run_command "mpirun -np 1 python3 ants.py $BASIC_PARAMS"

run_command "mpirun -np 1 python3 ants_display_monoprocess.py $BASIC_PARAMS : -np 1 python3 ants_computation_monoprocess.py"

run_command "mpirun -np 1 python3 ants_display.py $BASIC_PARAMS : -np 2 python3 ants_computation.py"

run_command "mpirun -np 1 python3 ants_display.py $BASIC_PARAMS : -np 3 python3 ants_computation.py"

echo ""
echo "Done, you can find the results in $OUTPUT_FILE"
echo ""

echo "Building graph..."
python3 build_graph.py $OUTPUT_FILE $GRAPH_OUTPUT_FILE
