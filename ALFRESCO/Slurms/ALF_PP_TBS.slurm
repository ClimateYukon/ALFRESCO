#!/bin/sh
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --account=snap
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jschroder@alaska.edu
#SBATCH -p main
source activate try


python /workspace/Shared/Users/jschroder/Github/ALFRESCO/ALFRESCO/PP_launcher.py -p /atlas_scratch/apbennett/Calibration/HighCalib/FMO_Calibrated/CCSM4_rcp85 -shp /workspace/Shared/Users/jschroder/TMP/IEM_Boreal/IEM_Boreal.shp -field OBJECTID -name Name -o /atlas_scratch/jschroder/ALF_outputs/TBS