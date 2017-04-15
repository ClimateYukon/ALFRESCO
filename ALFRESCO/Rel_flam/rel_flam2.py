import numpy as np
import pandas as pd
from pathos import multiprocessing
import rasterio

def prep_firescar( fn ):
	import rasterio
	array1 = rasterio.open( fn ).read( 3 )
	array1 = np.where( array1 > -2147483647, 1, 0 )
	return array1

def get_repnum( fn ):
	''' 
	based on the current ALFRESCO FireScar naming convention,
	return the replicate number
	'''
	return os.path.basename( fn ).split( '_' )[-2]

def sum_firescars2( firescar_list, ncores ):
	pool = multiprocessing.Pool( processes=ncores, maxtasksperchild=2 )

	# tmp_rst = rasterio.open( firescar_list[0] )
	# tmp_arr = tmp_rst.read( 3 )

	# groupby the replicate number
	firescar_series = pd.Series( firescar_list )
	repgrouper = firescar_series.apply( get_repnum )
	firescar_groups = [ j.tolist() for i,j in firescar_series.groupby( repgrouper ) ]

	repsums = [ np.sum( pool.map( lambda fn: prep_firescar(fn), group ), axis=0 ) for group in firescar_groups ]
	pool.close()
	sum_arr = np.sum( repsums, axis=0 )
	return sum_arr


def relative_flammability( args ):

	'''
	run relative flammability.
	Arguments:
		firescar_list = [list] string paths to all GeoTiff FireScar outputs to be processed
		output_filename = [str] path to output relative flammability filename to be generated. 
						* only GTiff supported. *
		ncores = [int] number of cores to use if None multiprocessing.cpu_count() used.
		mask_arr = [numpy.ndarray] numpy ndarray with dimensions matching the rasters' arrays
					listed in firescar_list and masked where 1=dontmask 0=mask (this is opposite
					numpy mask behavior, but follows common GIS patterns ) * THIS MAY CHANGE. *
		mask_value = [numeric] single numeric value determining the value of the newly masked out
					regions. If None, the nodata value from the firescar outputs will be used and 
					if this is an invalid value, it will be set to -9999.
		crs=[dict] rasterio-compatible crs dict object i.e.: {'init':'epsg:3338'}
	
	Returns:
		output_filename, with the side effect of the relative flammability raster being written to 
		disk in that location.
	'''

	if not os.path.exists( args.output_path ):
		os.makedirs( args.output_path )

	firescar_list = [ os.path.join( root, fn ) for root, subs, files in os.walk( args.input_path ) for fn in files if 'FireScar_' in fn and fn.endswith('.tif') ]
	tmp_rst = rasterio.open( firescar_list[0] )


	if args.ncores == None:
		args.ncores = multiprocessing.cpu_count() - 1

	# ALTERNATIVE SUM FIRESCARS in VERSION 2
	out = sum_firescars2( firescar_list, ncores=args.ncores )

	# calculate the relative flammability
	relative_flammability = ( out / len( firescar_list ) )
	mask_value = None
	if mask_value == None:
		mask_value = tmp_rst.nodata
		if mask_value == None or mask_value == '':
			print( 'setting mask_value to -9999')
			mask_value = -9999

	# if mask_arr:
	# 	relative_flammability[ mask_arr == 0 ] = mask_value

	meta = tmp_rst.meta
	# pop out transform to overcome warning
	if 'transform' in meta.keys():
		_ = meta.pop( 'transform' )

	meta.update( compress='lzw', count=1, dtype='float32', nodata=mask_value )
	crs={'init':'epsg:3338'}
	if crs:
		meta.update( crs=crs )

	output_filename = os.path.join( args.output_path, 'alfresco_relative_flammability_' + args.model + '_' + args.scenario + '_' + str(1901) + '_' + str(2100) + '.tif' )

	with rasterio.open( output_filename, 'w', **meta ) as out:
		out.write( relative_flammability.astype( np.float32 ), 1 )

	return output_filename


if __name__ == '__main__':
	from itertools import groupby
	import glob, os, sys, re, rasterio
	from pathos import multiprocessing as mp
	import numpy as np
	import scipy as sp
	import argparse

	
	parser = argparse.ArgumentParser( description='program to calculate Relative Vegetation Change from ALFRESCO Veg outputs' )
	parser.add_argument( '-p', '--input_path', action='store', dest='input_path', type=str, help='path to ALFRESCO output Maps directory' )
	parser.add_argument( '-o', '--output_path', action='store', dest='output_path', type=str, help='path to output directory' )
	parser.add_argument( '-m', '--model', action='store', dest='model', type=str, help='model name' )
	parser.add_argument( '-s', '--scenario', action='store', dest='scenario', type=str, help='scenario' )
	parser.add_argument( '-n', '--ncores', action='store', dest='ncores', type=int, help='number of cores to utilize' )

	args = parser.parse_args()
	print( ' running %s : scenario %s' % ( args.model, args.scenario ) )
	_ = relative_flammability( args )





