#!/bin/sh
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --account=snap
#SBATCH --mail-type=all
#SBATCH --mail-user=jschroder@alaska.edu
#SBATCH -p main

source ~/py27/bin/activate

python /workspace/Shared/Users/jschroder/Github/Alfresco_Calibration/Calib_historical.py -p '/atlas_scratch/apbennett/IEM/FinalCalib/' -shp '/workspace/Shared/Tech_Projects/ALF_bird_habitat/project_data/NA_TerrestrialProtectedAreas/NA_TPA_simplified.shp' -field 'OBJECTID' -name 'MGT_AGENCY' -o '/atlas_scratch/jschroder/2017_IEM_birds_simple_polygon/'

python /workspace/Shared/Users/jschroder/Github/Alfresco_Calibration/Calib_launcher_AR4.py -p '/atlas_scratch/apbennett/IEM/FinalCalib/' -shp '/workspace/Shared/Tech_Projects/ALF_bird_habitat/project_data/NA_TerrestrialProtectedAreas/NA_TPA_simplified.shp' -field 'OBJECTID' -name 'MGT_AGENCY' -o '/atlas_scratch/jschroder/2017_IEM_birds_simple_polygon/'
