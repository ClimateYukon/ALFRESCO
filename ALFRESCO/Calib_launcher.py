# import library
import alfresco_postprocessing as ap
from tinydb import TinyDB, Query
import os, argparse, json
from alfresco_postprocessing import Calib_plotting as CP

parser = argparse.ArgumentParser( description='' )
parser.add_argument( "-p", "--base_path", action='store', dest='base_path', type=str, help="path to output directory" )
parser.add_argument( "-shp", "--shapefile", action='store', dest='shp', type=str, help="full path to the subdomains shapefile used in subsetting" )
parser.add_argument( "-field", "--field_name", action='store', dest='id_field', type=str, help="field name in shp that defines subdomains" )
parser.add_argument( "-name", "--name", action='store', dest='name', type=str, help="field name in shp that defines subdomains name" )
parser.add_argument( "-o", "--output", action='store', dest='out', type=str, help="output path" )

args = parser.parse_args()

ncores = 32

base_path = args.base_path

historical_maps_path = '/Data/Base_Data/ALFRESCO/ALFRESCO_Master_Dataset/ALFRESCO_Model_Input_Datasets/AK_CAN_Inputs/Fire'
metrics = [ 'veg_counts','avg_fire_size','number_of_fires','all_fire_sizes','total_area_burned' ]
out = args.out

if not os.path.exists( out ):
    os.makedirs( out )

json_path = os.path.join(out,'JSON')
if not os.path.exists( json_path ):
    os.makedirs( json_path )

obs_json_fn = os.path.join( json_path, 'Observed.json' )
subdomains_fn = args.shp
id_field = args.id_field
name_field = args.name

#run historical
pp_hist = ap.run_postprocessing_historical( historical_maps_path, obs_json_fn, ncores, ap.veg_name_dict, subdomains_fn, id_field, name_field)
pp_hist.close()
models = ['CCSM4_rcp85', 'MRI-CGCM3_rcp85']

#run all modeled
for dirs in models :
    print dirs

    print 'processing %s' %dirs

    suffix = dirs

    if 'CCSM4' in suffix :
        suffix = suffix.replace('CCSM4','NCAR-CCSM4')
    else : pass
    
    output_path = os.path.join(out , suffix)
    if not os.path.exists( output_path ):
        os.makedirs( output_path )

    mod_json_fn = os.path.join( json_path,'_'.join([ suffix + '.json'  ]))
    maps_path = os.path.join(base_path, dirs, 'Maps')
    # PostProcess using shapefile-derived rasterized subdomains.
    pp = ap.run_postprocessing( maps_path, mod_json_fn, ncores , ap.veg_name_dict ,subdomains_fn, id_field, name_field )

    # # Output to CSV files for researcher ease-of-use
    _ = ap.to_csvs( pp, metrics, output_path, suffix )

    # # close the database
    pp.close() 

    CP.launcher( suffix , out )
