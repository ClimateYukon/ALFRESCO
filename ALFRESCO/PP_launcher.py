#Official ALFRESCO PP launcher, needs to be updated to include csv TRUE/FALSE and historical TRUE/FALSE
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
ncores=32
base_path = args.base_path
csv=True
hist=True

if not os.path.exists( args.out ):
    os.makedirs( args.out )

json_path = os.path.join(args.out,'JSON')
if not os.path.exists( json_path ):
    os.makedirs( json_path )
csv_path = os.path.join(args.out,'CSV')
if not os.path.exists( csv_path ):
    os.makedirs( csv_path )


# obs_json_fn = '/atlas_scratch/jschroder/ALF_outputs/PP_2017-04-24-12-51/JSON/historical.json'
historical_maps_path='/workspace/Shared/Users/jschroder/FireHistory'
obs_json_fn = os.path.join( json_path, 'Observed.json' )
#run historical
if hist==True:
    pp_hist = ap.run_postprocessing_historical( historical_maps_path, obs_json_fn, ncores, ap.veg_name_dict, args.shp, args.id_field, args.name)
    pp_hist.close()

metrics = ['avg_fire_size','number_of_fires','all_fire_sizes','total_area_burned' ,"veg_counts" , "severity_counts"]
suffix = os.path.split(base_path)[1]

mod_json_fn = os.path.join( json_path,'_'.join([ suffix + '.json'  ]))
maps_path = os.path.join(base_path,  'Maps')

pp = ap.run_postprocessing( maps_path, mod_json_fn, ncores , ap.veg_name_dict ,args.shp, args.id_field, args.name )

if csv==True:
	_ = ap.to_csvs( pp, metrics, csv_path, suffix )

pp.close()

_plot = ap.launcher_SERDP( obs_json_fn , args.out, suffix , args.out)
