#!/bin/sh
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --account=snap
#SBATCH --mail-type=all
#SBATCH --mail-user=jschroder@alaska.edu
#SBATCH -p main

source ~/py27/bin/activate


python /workspace/Shared/Users/jschroder/Github/Alfresco_Calibration/Calib_launcher.py -p '/atlas_scratch/apbennett/IEM_AR5/' -shp '/workspace/Shared/Users/jschroder/Alec_Alfresco_PP_improvements/Data/Shapefile/Boreal_Tundra_CombinedDomains.shp' -field 'Id' -name 'Name' -o '/atlas_scratch/jschroder/IEM_2017_boreal/'
