#!/bin/sh
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --account=snap
#SBATCH --mail-type=all
#SBATCH --mail-user=jschroder@alaska.edu
#SBATCH -p main

source ~/py27/bin/activate

python /workspace/Shared/Users/jschroder/Github/ALFRESCO/Calib_historical.py -p '/atlas_scratch/apbennett/IEM/FinalCalib/' -shp '/workspace/Shared/Users/jschroder/TMP/IEM_Boreal/IEM_Boreal.shp' -field 'OBJECTID' -name 'Name' -o '/atlas_scratch/jschroder/IEM_2017_SouthBoreal/'

python /workspace/Shared/Users/jschroder/Github/ALFRESCO/Calib_launcher_AR4.py -p '/atlas_scratch/apbennett/IEM/FinalCalib/' -shp '/workspace/Shared/Users/jschroder/TMP/IEM_Boreal/IEM_Boreal.shp' -field 'OBJECTID' -name 'Name' -o '/atlas_scratch/jschroder/IEM_2017_SouthBoreal/'
