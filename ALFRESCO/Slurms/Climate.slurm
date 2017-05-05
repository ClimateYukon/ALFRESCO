#!/bin/sh
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --account=snap
#SBATCH --mail-type=all
#SBATCH --mail-user=jschroder@alaska.edu
#SBATCH -p main

source activate Climate


python /workspace/Shared/Users/jschroder/Github/Climate/Climate/New_climate/V21_extract.py
