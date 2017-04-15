# import library
import alfresco_postprocessing as ap
from tinydb import TinyDB, Query
import os, argparse, json


parser = argparse.ArgumentParser( description='' )
parser.add_argument( "-p", "--base_path", action='store', dest='base_path', type=str, help="path to output directory" )
parser.add_argument( "-shp", "--shapefile", action='store', dest='shp', type=str, help="full path to the subdomains shapefile used in subsetting" )
parser.add_argument( "-field", "--field_name", action='store', dest='id_field', type=str, help="field name in shp that defines subdomains" )
parser.add_argument( "-name", "--name", action='store', dest='name', type=str, help="field name in shp that defines subdomains name" )
parser.add_argument( "-o", "--output", action='store', dest='out', type=str, help="output path" )

args = parser.parse_args()
base_path = args.base_path
subdomains_fn = args.shp
id_field = args.id_field
name_field = args.name
out = args.out
ncores = 32

historical_maps_path = '/workspace/Shared/Users/jschroder/TMP/FireHistory/'
metrics = ['avg_fire_size','number_of_fires','all_fire_sizes','total_area_burned' ]

if not os.path.exists( out ):
    os.makedirs( out )

json_path = os.path.join(out,'JSON')
if not os.path.exists( json_path ):
    os.makedirs( json_path )

obs_json_fn = os.path.join( json_path, 'Observed2.json' )

suffix = 'historical'
#run historical
pp_hist = ap.run_postprocessing_historical( historical_maps_path, obs_json_fn, ncores, ap.veg_name_dict, subdomains_fn, id_field, name_field)
_ = ap.to_csvs( pp_hist, metrics, out, suffix , observed=True)
pp_hist.close()


