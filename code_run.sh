#!/bin/sh

# The following code may be run on the GPU
echo "Beginning Imports From Bash"
pip install xplique
pip install openturns

echo "Calling Main Script From Bash"
python3 main.py

# The following code needs to be executed on the CPU.
echo "Generating MuFidelity Values From Bash"
python3 mu_fid.py
