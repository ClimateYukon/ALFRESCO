#!/usr/bin/env python

import pandas as pd
import os
import scipy.stats
import json

def best_rep_calculation(alf_dict , hist_dict , output_path , domain_dict) :


	#################Processing ALFRESCO output data##########################################
	#Generate a dictionnary that will hold both subdomains dataframe fitting the historical data so between 1950 and 2011
	alf = {}
	for i , sub_domain in domain_dict.iteritems(): 
		tmp = alf_dict['total_area_burned'][str(i)]
		tmp.index = tmp.index.astype(int)
		tmp = tmp.ix[1950:2011]
		alf[sub_domain] = tmp


	#################Processing historical data##########################################
	#Grab the historical data passed as global variable
	hist_df = hist_dict['total_area_burned']

	# col = domain_dict.keys()
	hist_df.columns= domain_dict.values()


	################# Best reap actual calculation ##########################################

	#Everything happens here, compare historical with each columns of each subdomains and compute the correlation using Pearson, Scipy was easier to use in that case
	data = {
		hist_k: {
			k: scipy.stats.spearmanr(hist_df[hist_k], v)[0]
			for k,v in alf[hist_k].iteritems()
		}
		for hist_k,hist_v in hist_df.iteritems()
	}

	#Small fumction to make the last step cleaner, just return the name of the best rep with the Rho
	def get_max_dict(dictionnary):
		max_dict = max(dictionnary.iterkeys(),key=lambda k:dictionnary[k])
		Rho = dictionnary[max_dict]
		return max_dict , Rho

	#Build the dictionnary storing both name and Rho for the best rep for the two subdomains.
	Best_rep_dic = {keys:get_max_dict(data[keys]) for keys in data.iterkeys() }
	output_json = os.path.join( output_path , 'Best_rep.txt')

	json.dump(Best_rep_dic,open(output_json,'w'))
	return Best_rep_dic

def fire_metrics_to_dict( alf_dict, metrics ):
	'''
	take an ALFRESCO Post Processing generated JSON file
	read in via json.load( filename ) and using the keys within
	break the data into a csv of the output fire metrics. 
		reps are the columns and years are the rows.
	Notes:
		current supported fire metrics include:
			[ 'fire_counts', 'avg_fire_size', 'number_of_fires', 'total_area_burned' ]
		Others are not yet supported.
	'''
	from collections import defaultdict
	
	def rec_dd():
		''' simple way to nest defaultdict's recursively '''
		from collections import defaultdict
		return defaultdict( rec_dd )


	metric_dic = {}

	for metric in metrics:
		metric_dic[metric]={}

		reps = alf_dict.keys()
		years = alf_dict[ reps[0] ].keys()
		domains = alf_dict[ reps[0] ][ years[0] ][ metric ].keys()
		out = defaultdict( rec_dd )
		
		for domain in domains:
			for rep in reps:
				for year in years:
					try:
						out[ rep ][ year ] = alf_dict[ rep ][ year ][ metric ][ domain ]

					except:
						pass

			metric_dic[metric][domain] = {}
			df = pd.DataFrame( out )
			colnames = [ int( col ) for col in df.columns ]
			colnames.sort()
			colnames = [ unicode( i ) for i in colnames ]
			df = df.reindex_axis( colnames, axis=1 )
			colnames = [ '_'.join([ 'rep', str(int(i)+1) ]) for i in df.columns ]
			df.columns = colnames
			metric_dic[metric][domain] = df

	return metric_dic

def metrics_domains_to_csv_obs( hist_dict, metrics ):
	'''
	take a HISTORICAL (OBSERVED) ALFRESCO Post Processing generated JSON file
	read in via json.load( filename ) and using the keys within
	break the data into a csv of the desired metric. 
		domains are the columns and years are the rows.
	Notes:
	 if the metric returned is a single value across each year:domain
	 combination, then the output csv will be single values in each of the
	 'cells' or if there are multiples returned it will be a square brace list
	 of those values.  ie: [ 1, 23, 900, 43, 56 ]
	'''

	metric_dic = {}
	for metric in metrics :
		years = hist_dict.keys()
		# subset out a specific metric
		dat = { key:value[ metric ] for key,value in hist_dict.iteritems() }
		# convert to a dataframe with years in the rows and domains in the columns
		df = pd.DataFrame( dat ).T
		df.index = [ int( ind ) for ind in df.index ]
		metric_dic[metric] = df

	return metric_dic











