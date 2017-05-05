
import pandas as pd
import numpy as np
import glob, os, ast, sys,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
pd.options.mode.chained_assignment = None  # default='warn'
import alfresco_postprocessing as ap
import seaborn.apionly as sns


rcParams[ 'xtick.direction' ] = 'out'
rcParams[ 'ytick.direction' ] = 'out'
rcParams[ 'xtick.labelsize' ] = 'small'
rcParams[ 'ytick.labelsize' ] = 'small'
rcParams[ 'figure.titlesize' ] = 'small'
rcParams[ 'axes.titlesize' ] = 'small'
rcParams[ 'axes.spines.top' ] = 'False'
rcParams[ 'axes.spines.right' ] = 'False'
rcParams[ 'savefig.dpi' ] = 150
rcParams[ 'figure.figsize'] = 14 , 8
year_range = (1950,2100)



class Scenario( object ):
	'''
	class for storing data attributes and methods to abstract some of the
	ugliness of plotting the ALFRESCO Post Processing outputs.
	'''
	def __init__( self, json_fn, model, scenario, caption , color,*args, **kwargs ):
		'''
		Arguments:
		----------
		json_fn = [str] path to the alfresco_postprocessing output TinyDB JSON database file
		model = [str] name of the model being processed (used in naming)
		scenario = [str] name of the scenario being processed (used in naming)

		Returns:
		--------
		object of type alfresco_postprocessing.Plot
				
		'''
		from tinydb import TinyDB
		self.json_fn = json_fn
		self.db = TinyDB( self.json_fn )
		self.records = self.db.all()
		self.model = '_'.join(model.split('_')[0:-1]).upper()
		self.scenario = scenario
		self.years = self._get_years()
		self.replicates = self._get_replicates()
		self.domains = self._get_domains()
		self.caption = caption
		self.metrics = self._get_metric_names()
		self.color = color
		self.mscenario = model.split('_')[-1].upper()
		self.patch = mpatches.Patch([], [], linewidth=1.2, color= self.color , label=self.caption )
		self.line = mlines.Line2D([], [], linewidth=1.2, color=self.color, label= self.caption )
		for metric in self.metrics:
			setattr(self, metric, self.get_metric_dataframes(metric))
		self.CD_ratio = self._get_veg_ratios()

	def _get_years( self ):
		if 'fire_year' in self.records[0].keys():
			years = np.unique( [ rec['fire_year'] for rec in self.records ] ).astype( np.int )
		else :
			years = np.unique( [ rec['year'] for rec in self.records ] ).astype( np.int )

		years.sort()
		return years.astype( str )
	def _get_replicates( self ):
		replicates = np.unique( [ rec['replicate'] for rec in self.records ] )
		replicates.sort()
		return replicates
	def _get_domains( self ):
		record = self.records[0]
		metric = record.keys()[0]
		return record[ metric ].keys()
	def _get_metric_names( self ) :
		record = self.records[0]
		metric = [value for value in record if 'year' not in value if 'rep' not in value]
		return metric
	def get_metric_dataframes( self , metric_name ):
		'''
		output a dict of pandas.DataFrame objects representing the 
		data of type metric_name in key:value pairs of 
		domainname:corresponding_DataFrame

		Arguments:
		----------
		metric_name = [str] metric name to be converted to pandas DataFrame obj(s).

		Returns:
		--------
		dict of pandas DataFrame objects from the output alfresco TinyDB json file
		for the desired metric_name
		'''
		from collections import defaultdict
		if 'fire_year' in self.records[0].keys():
			metric_select = ap.get_metric_json( self.db, metric_name )
		else :
			metric_select = ap.get_metric_json_hist( self.db, metric_name )

		panel = pd.Panel( metric_select )

		dd = defaultdict( lambda: defaultdict( lambda: defaultdict ) )
		for domain in self.domains :

			if metric_name != 'veg_counts': # fire
				dd[ domain ] = panel[ :, domain, : ]
				dd[ domain ].index = dd[ domain ].index.astype(int)

			if metric_name == 'veg_counts': # veg
				df = panel[ :, domain, : ]
				vegtypes = sorted( df.ix[0,0].keys() )
				new_panel = pd.Panel( df.to_dict() )
				for vegtype in vegtypes:
					# subset the data again into vegetation types
					dd[ domain ][ vegtype ] = new_panel[ :, vegtype, : ]
					dd[ domain ][vegtype].index = dd[ domain ][vegtype].index.astype(int)
		return dict(dd)
	def _get_veg_ratios( self, year_range = (1950,2100), group1=['White Spruce', 'Black Spruce'], group2=['Deciduous'] ):
		'''
		calculate ratios from lists of veg types.
		'''
		dd = {}
		for domain in self.domains  :

			try :
				begin,end = year_range
				agg1 = sum([ self.veg_counts[ domain ][ i ].ix[begin:end] for i in group1 ])
				agg2 = sum([ self.veg_counts[ domain ][ i ].ix[begin:end]for i in group2 ])
				dd[ domain ] = agg1/agg2
			except : "check %s" %domain
		return dd


def upcase( word ):
	_tmp = [i.title() for i in word.split('_')]
	_tmp = " ".join(_tmp)
	return _tmp

def fill_in(ax , df ,colors ,low_percentile = 5 , high_percentile = 95 , alpha = 0.2 ) :
	
	x = df.index.unique()

	ax.fill_between(x, df.groupby(df.index).apply(np.percentile, low_percentile ), \
	df.groupby(df.index).apply(np.percentile, high_percentile), alpha= alpha, color=colors)

	return ax

def df_processing2(dataframe , std_arg = False , cumsum_arg = False , *args):

	def _process_df(df , std_arg , cumsum_arg):

		if cumsum_arg == True :
			df = df.apply( np.cumsum, axis=0 )
		else : pass

		if std_arg == True :
			df['std'] = df.std(axis=1)
		else : pass
			
		return df

	_tmp = _process_df( dataframe , std_arg , cumsum_arg) 
	return _tmp

def underscore_fix(string) :
	string = string.replace("_"," ")
	return string

def ticks(ax , decade=False) :

	if decade == False :

		# Getting ticks every ten years
		n = 10 # every n ticks... from the existing set of all
		ticks = ax.xaxis.get_ticklocs()
		ticklabels = [ l.get_text() for l in ax.xaxis.get_ticklabels() ]
		ax.xaxis.set_ticks( ticks[::n] )
		ax.xaxis.set_ticklabels( ticklabels[::n] )
		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()
	else : 

		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()

	return ax


def compare_metric(mod_obj , observed , output_path , pdf, model , graph_variable, year_range , domain , cumsum=True , *args):
	#This plot compare the cumulative area burn for managed, unmanaged and historical period


	begin, end = year_range

	if len(mod_obj)==1 :
		sub = '\n {} - {} \n'.format(mod_obj[0].model,mod_obj[0].mscenario)
	else :
		_init= [' '.join([scen_arg.model,scen_arg.mscenario,scen_arg.scenario]) for scen_arg in mod_obj]
		sub = [' vs '.join(string) for string in [_init]]

	if cumsum == True :
		plot_title = 'ALFRESCO Cumulative Sum of {} {}-{} \n'.format(upcase(graph_variable), begin, end) + '\n' + sub[0] + '\n' + '\n {}'.format(underscore_fix(domain))
	else :
		plot_title = 'ALFRESCO Annual {} {}-{} \n'.format(upcase(graph_variable), begin, end) + '\n' + sub[0] + '\n' + '\n {}'.format(underscore_fix(domain))

	#Handling the historical, oserved data
	if cumsum == True :
		obs_domain = np.cumsum( observed.__dict__[graph_variable][domain].ix[begin : ] )
	else : 
		obs_domain = observed.__dict__[graph_variable][domain].ix[begin : ]


	fig, ax = plt.subplots() 
	_ = [df_processing2( scen_arg.__dict__[graph_variable][domain].ix[begin : end], False, cumsum ).plot(ax=ax,legend=False, color = scen_arg.color, title=plot_title,lw = 0.7,alpha=0.1, grid=False )for scen_arg in mod_obj]
	_ = [df_processing2( scen_arg.__dict__[graph_variable][domain].ix[begin : end], False, cumsum ).mean(axis=1).plot(ax=ax,legend=False, color = scen_arg.color, title=plot_title,lw = 0.7,  grid=False )for scen_arg in mod_obj]

	obs_domain.plot( ax=ax,legend=False, color=observed.color, grid=False, label= "observed" ,lw = 1)

	#Create label for axis
	plt.xlabel( 'Years' )
	if graph_variable == 'avg_fire_size' :
		ylabel ='Average Fire Size ('+'$\mathregular{km^2}$' + ')' 

	elif graph_variable == 'number_of_fires' :
		ylabel = 'Number of Fires'

	elif graph_variable == 'total_area_burned' :
		ylabel = 'Area Burned in ('+'$\mathregular{km^2}$' + ')'

	else : 'Error with Title'
	plt.ylabel( ylabel )
	ax = ticks(ax , decade=True)

	#have to pass the scenario object so they are avalaible for color definition
	replicate = mlines.Line2D([], [], linewidth=1.2, color='0.75', label= 'Replicates' )

	plt.legend(handles = [mod.line for mod in mod_obj] +[observed.line, replicate],fontsize='medium',loc='best',borderaxespad=0.,ncol=1,frameon=False)

	if cumsum == True :
		output_filename = os.path.join( output_path, domain , '_'.join([ 'alfresco_lines_cumsum',domain,graph_variable, model , str(begin), str(end)]) + '.png' )
	else : 
		output_filename = os.path.join( output_path, domain , '_'.join([ 'alfresco_lines_annual',domain,graph_variable, model , str(begin), str(end)]) + '.png' )

	plt.savefig( output_filename )
	pdf.savefig()
	plt.close()


def compare_vegcounts(mod_obj  , observed , output_path , pdf, model , graph_variable,year_range , domain , *args):


	begin, end = year_range #subset the dataframes to the years of interest

	for veg_name in mod_obj[0].veg_counts[domain].keys():



		if len(mod_obj)==1 :
			sub = '\n {} - {} \n'.format(mod_obj[0].model,mod_obj[0].mscenario)
		else :
			_init= [' '.join([scen_arg.model,scen_arg.mscenario,scen_arg.scenario]) for scen_arg in mod_obj]
			sub = [' vs '.join(string) for string in [_init]]


		plot_title = 'ALFRESCO Vegetation Annual {} {}-{} \n'.format(veg_name, begin, end) + '\n' + sub[0] + '\n' + '\n {}'.format(underscore_fix(domain))



		# Plot the average value by condition and date
		
		fig, ax = plt.subplots()
		_ = [df_processing2( scen_arg.__dict__['veg_counts'][domain][veg_name].ix[begin : end] ).mean(axis=1).plot(ax=ax,legend=False, color = scen_arg.color, title=plot_title,lw = 0.7,  grid=False )for scen_arg in mod_obj]

		ax = ticks(ax , decade=True)
		
		#Create label for axis
		plt.xlabel( 'Year' )
		plt.ylabel( 'Area Covered ('+'$\mathregular{km^2}$' + ')' )


		_ = [fill_in(ax , df_processing2( scen_arg.__dict__['veg_counts'][domain][veg_name].ix[begin : end]) , scen_arg.color ,low_percentile = 5 , high_percentile = 95) for scen_arg in mod_obj]
		plt.legend(handles = [mod.line for mod in mod_obj],fontsize='medium',loc='best',borderaxespad=0.,ncol=1,frameon=False)

		output_filename = os.path.join( output_path, domain , '_'.join([ 'alfresco_annual_areaveg_line',model, domain, veg_name.replace(' ', '' ), str(begin), str(end) ]) + '.png' ) 

		plt.savefig( output_filename )
		pdf.savefig()
		plt.close()



def CD_ratio(mod_obj , observed , output_path , pdf, model , graph_variable, year_range , domain , *args):

	begin, end = year_range


	if len(mod_obj)==1 :
		sub = '\n {} - {} \n'.format(mod_obj[0].model,mod_obj[0].mscenario)
	else :
		_init= [' '.join([scen_arg.model,scen_arg.mscenario,scen_arg.scenario]) for scen_arg in mod_obj]
		sub = [' vs '.join(string) for string in [_init]]

	plot_title = 'ALFRESCO Conifer:Deciduous Ratios {}-{} \n'.format( begin, end) + '\n' + sub[0] + '\n' + '\n {}'.format(underscore_fix(domain))

	fig, ax = plt.subplots()
	_ = [df_processing2( scen_arg.__dict__['CD_ratio'][domain].ix[begin : end] ).mean(axis=1).plot(ax=ax,legend=False, color = scen_arg.color, title=plot_title,lw = 0.7,  grid=False )for scen_arg in mod_obj]


	ax = ticks(ax, decade=True)
	
	#Create label for axis
	plt.xlabel( 'Year' )
	plt.ylabel( 'C:D Ratio' )

	_ = [fill_in(ax , df_processing2( scen_arg.__dict__['CD_ratio'][domain].ix[begin : end]) , scen_arg.color ,low_percentile = 5 , high_percentile = 95) for scen_arg in mod_obj]

	plt.legend(handles = [mod.line for mod in mod_obj],fontsize='medium',loc='best',borderaxespad=0.,ncol=1,frameon=False)

	output_filename = os.path.join( output_path, domain , '_'.join([ 'alfresco_CD_ratio',domain, model, str(begin), str(end) ]) + '.png' ) 

	plt.savefig( output_filename )
	pdf.savefig()
	plt.close()



def launcher_SERDP(obs_json_fn, paths, labels, colors, model , out ) :
	print 'launching'
	mod_obj_fn = [os.path.join(path , 'JSON' , model + '.json' ) for path in paths  ]
	mod_obj = [Scenario( json, model, label, model , color) for json, label , model, color in zip(mod_obj_fn, labels ,[model]*len(mod_obj_fn), colors)]

	hist_obj = Scenario( obs_json_fn, model, 'Observed', "Historical", '#B22222' )

	output_path = os.path.join( out , 'Plots_SERDP' , model )

	for domain in mod_obj[0].domains:
		 
		try:
		    os.makedirs(os.path.join( output_path, domain))
		except OSError:
		    pass


		pdf = os.path.join( output_path, '_'.join([ model, domain ,'plots']) + '.pdf' )

		with PdfPages(pdf) as pdf:
			try :
				_ = [compare_metric(mod_obj , hist_obj , output_path , pdf, model , metric, year_range ,domain , cumsum=False) for metric in mod_obj[0].metrics if metric not in [ 'veg_counts' , 'all_fire_sizes' , 'severity_counts']]
			except : pass

			try :
				compare_metric(mod_obj , hist_obj , output_path , pdf, model , 'total_area_burned', year_range ,domain, cumsum=True)
			except : pass			
			try :	
				CD_ratio(mod_obj , hist_obj , output_path , pdf, model , 'veg_counts', year_range, domain)
			except : pass				
			try :
				compare_vegcounts(mod_obj  , hist_obj , output_path , pdf, model , 'veg_counts', year_range, domain)
			except : pass


























path2 = '/atlas_scratch/jschroder/ALF_outputs/PP_2017-05-01-12-23'

path1 = '/atlas_scratch/jschroder/ALF_outputs/PP_2017-04-26-16-49'
obs_json_fn = '/atlas_scratch/jschroder/ALF_outputs/PP_2017-05-01-12-23/JSON/Observed.json'
labels = ['AR5 V2','AR5 V2.1']
colors = ['#6c2436','#00ad5e']
out = '/workspace/Shared/Users/jschroder/TMP/try/'
models = os.listdir('/atlas_scratch/jschroder/ALF_outputs/PP_2017-05-01-12-23/JSON/')

models = models[:-1]

models = [m[:-5]for m in models]


paths = [path1]+[path2]
from pathos.multiprocessing import ProcessingPool
pool = ProcessingPool(nodes=15)
r = pool.map(launcher_SERDP ,[obs_json_fn]*15,[paths]*15,[labels]*15,[colors]*15,models,[out]*15)
