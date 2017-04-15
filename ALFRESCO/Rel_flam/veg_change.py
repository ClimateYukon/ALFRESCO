#!/usr/bin/env python

# # calculate vegetation resilience counts through time
def get_rep_num( x ):
	'''return rep number from firescar filename'''
	base = os.path.basename( x )
	return base.split( '_' )[ 1 ]
def count_transitions( arr_list ):
	''' 
	takes list of numpy ndarrays of integers and returns the number of 
	shifts in values in the series. arr_list is expected to be in 
	chronological order.
	'''
	import numpy as np
	arr_list = np.array([ np.where( arr != 0, 1, 0 ) for arr in np.diff( np.array( arr_list ), axis=0 ) ])
	return np.sum( arr_list, axis=0 )
def open_raster( fn, band=1 ):
	''' remove mem leaks from stale file handles '''
	import rasterio
	with rasterio.open( fn ) as out:
		arr = out.read( band )
	return arr
def relative_veg_change( veg_list, ncpus=32 ):
	'''
	opens list of vegetation filenames into 2-d numpy
	ndarrays and counts the number of transitons in vegetation 
	occur by pixel through the series. 
	Arguments:
		veg_list:[list] list of paths to the vegetation output files
					from the ALFRESCO Fire Model. * expects filenames in 
					chronological order *
	Returns:
		2-D numpy.ndarray of transition counts across the list of 
		filenames passed.
	'''

	pool = mp.Pool( 32 )
	arr_list = pool.map(open_raster,veg_list )
	pool.close()
	pool.join()

	# arr_list = mp_map( open_raster, veg_list, nproc=ncpus )
	return count_transitions( arr_list )
def main( args ):
	'''
	run relative flammability with the input args dict from argparse
	'''
	import numpy as np
	if not os.path.exists( args.output_path ):
		os.makedirs( output_path )

	# list, sort, group by replicate
	veg_list = [ os.path.join( root, fn ) for root, subs, files in os.walk( args.input_path ) for fn in files if 'Veg_' in fn and fn.endswith( '.tif' ) ]
	year_list = range( args.begin_year, args.end_year + 1 )
	veg_list = [ i for i in veg_list if int( os.path.basename( i ).split('_')[ len( os.path.basename( i ).split( '_' ) )-1 ].split( '.' )[0] ) in year_list ]
	veg_sorted = sorted( veg_list, key=lambda x: get_rep_num( x ) )
	veg_grouped = [ list( g ) for k, g in groupby( veg_sorted, key=lambda x: get_rep_num( x ) ) ]
	
	# calculate relative vegetation change -- parallel
	# final = mp_map( relative_veg_change, veg_grouped, nproc=int( args.ncpus ) )
	final = [ relative_veg_change( v, int(args.ncpus) ) for v in veg_grouped ]
	final = np.sum( final, axis=0 ) / np.float( len(veg_list) )

	# set dtype to float32 and round it
	final = final.astype( np.float32 )
	final = np.around( final, 4 ) 

	# mask the data with the out-of-bounds of Veg --> 255
	with rasterio.open( veg_list[0] ) as rst:
		arr = rst.read(1)
		final[ arr == 255 ] = -9999

	# write it out
	meta = rasterio.open( veg_list[ 0 ] ).meta
	meta.update( compress='lzw', dtype=np.float32, crs={ 'init':'EPSG:3338' }, nodata=-9999 )
	output_filename = os.path.join( args.output_path, 'alfresco_relative_vegetation_change_counts_' + args.model + '_' + args.scenario + '_' + str(args.begin_year) + '_' + str(args.end_year) + '.tif' )
	with rasterio.open( output_filename, 'w', **meta ) as out:
		out.write( final, 1 )
	return output_filename

if __name__ == '__main__':
	from itertools import groupby
	import glob, os, sys, re, rasterio
	from pathos import multiprocessing as mp
	import numpy as np
	import scipy as sp
	import argparse
	# # # TESTING
	# # input args
	# input_path = '/atlas_scratch/apbennett/IEM_AR5/'
	# output_path = '/atlas_scratch/jschroder/ALFRESCO_IEM_DERIVED_DEC2016'

	# script_path = '/workspace/UA/malindgren/repos/alfresco-calibration/alfresco_postprocessing/bin/alfresco_relative_vegetation_change.py'
	# model = ['CCSM4' , 'GFDL-CM3' , 'GISS-E2-R' , 'IPSL-CM5A-LR' , 'MRI-CGCM3' ]
	# scenario = ['rcp45' , 'rcp60' , 'rcp85']
	# ncpus = 1
	# begin_year = 1901
	# end_year = 1999
	
	# class hold:
	# 	def __init__( self, input_path, output_path, scenario, model, script_path, ncpus, begin_year, end_year):
	# 		self.input_path = input_path
	# 		self.output_path = output_path
	# 		self.scenario = scenario
	# 		self.model = model
	# 		self.script_path = script_path
	# 		self.ncpus = ncpus
	# 		self.begin_year = begin_year
	# 		self.end_year = end_year
				
	# args = hold( input_path, output_path, scenario, model, script_path, ncpus, begin_year, end_year )
	# # # END TESTING
	
	parser = argparse.ArgumentParser( description='program to calculate Relative Vegetation Change from ALFRESCO Veg outputs' )
	parser.add_argument( '-p', '--input_path', action='store', dest='input_path', type=str, help='path to ALFRESCO output Maps directory' )
	parser.add_argument( '-o', '--output_path', action='store', dest='output_path', type=str, help='path to output directory' )
	parser.add_argument( '-m', '--model', action='store', dest='model', type=str, help='model name' )
	parser.add_argument( '-s', '--scenario', action='store', dest='scenario', type=str, help='scenario' )
	parser.add_argument( '-n', '--ncpus', action='store', dest='ncpus', type=int, help='number of cores to utilize' )
	parser.add_argument( '-by', '--begin_year', action='store', dest='begin_year', type=int, help='beginning year in the range' )
	parser.add_argument( '-ey', '--end_year', action='store', dest='end_year', type=int, help='ending year in the range' )
	
	args = parser.parse_args()
	print( ' running %s : scenario %s' % ( args.model, args.scenario ) )
	_ = main( args )

