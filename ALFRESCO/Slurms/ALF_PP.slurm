#!/bin/sh
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --account=snap
#SBATCH --mail-type=all
#SBATCH --mail-user=jschroder@alaska.edu
#SBATCH -p main

source ~/py27/bin/activate


python /workspace/Shared/Users/jschroder/Github/Alfresco_Calibration/Calib_launcher.py -p '/atlas_scratch/apbennett/IEM_AR5/' -shp '/workspace/Shared/Users/jschroder/Data_off/LCC_AIEM.shp' -field 'LCC_ID' -name 'LCC_Name' -o '/atlas_scratch/jschroder/IEM_2017_shp/'
