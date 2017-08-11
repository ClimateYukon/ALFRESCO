import pandas as pd
import os
import alfresco_postprocessing as ap

def best_rep2( modplot, obsplot, domain, method='spearman' ):
	'''
	calculate correlation between replicates and historical to find which one most
	highly correlates with the observed data series for the same timestep and temoral period.
	Arguments:
	----------
	modplot = [ alfresco_postprocessing.Plot ] modeled data input JSON records file
	obsplot = [ alfresco_postprocessing.Plot ] observed data input JSON records file
	domain = [ str ] the Name of the domain to use in determining the 'best' replicate
	method = [str] one of 'pearson', 'kendall', 'spearman'
	Returns:
	--------
	dict with the best replicate number as the key and the correlation value as the value/\.
	'''
	mod_tab_dict = modplot.get_metric_dataframes( 'total_area_burned' )
	mod_df = mod_tab_dict[ domain ]
	obs_tab_dict = obsplot.get_metric_dataframes( 'total_area_burned' )
	obs = obs_tab_dict[ domain ][ 'observed' ]
	years = obs.index[:-2]
	mod_df = mod_df.ix[ years, : ]
	corr = pd.Series({i:mod_df[i].corr( obs, method=method ) for i in mod_df.columns })
	a = corr.sort_values(ascending=False)
	return  [str(i) + ' ' + str(v) for i,v in a.iteritems()]


obs_json = '/atlas_scratch/jschroder/ALF_outputs/PP_2017-07-19-09-03_all_polygons/JSON/Observed.json'
mod_base = '/atlas_scratch/jschroder/ALF_outputs/PP_2017-07-19-09-03_all_polygons/JSON/'
models = ('NCAR-CCSM4_rcp85', 'MRI-CGCM3_rcp85')
domains = ['Boreal','Tundra']
obsplot = ap.Plot(obs_json , 'observed', 'historical')

for i in models :

	obs_tab_dict = obsplot.get_metric_dataframes( 'total_area_burned' )
	mod = os.path.join(mod_base, i + '.json')
	modplot = ap.Plot(mod , os.path.basename(mod).split("_")[0],os.path.basename(mod.split("_")[1][:4]))
	ls = [best_rep2(modplot,obsplot,domain) for domain in domains]
	df = pd.DataFrame({'Boreal': ls[0],'Tundra' : ls[1]})
	csv = os.path.join('/workspace/Shared/Users/jschroder/TMP/', '{}_bestrep.csv'.format(i))
	df.to_csv(csv)
