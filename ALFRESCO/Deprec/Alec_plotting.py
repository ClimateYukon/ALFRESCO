
import alfresco_postprocessing as ap
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
		for domain in self.domains:
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
		return dd

def get_veg_ratios( veg_dd, domain ,year_range = (1950,2100), group1=['White Spruce', 'Black Spruce'], group2=['Deciduous'] ):
	'''
	calculate ratios from lists of veg types.
	'''
	begin,end = year_range
	agg1 = sum([ veg_dd[ domain ][ i ].ix[begin:end] for i in group1 ])
	agg2 = sum([ veg_dd[ domain ][ i ].ix[begin:end]for i in group2 ])
	return agg1 / agg2

def fill_in(ax , df ,colors ,low_percentile = 5 , high_percentile = 95 , alpha = 0.2 ) :
	
	x = df.date.unique()

	for cond, cond_df in df.groupby("condition"):

		ax.fill_between(x, cond_df.groupby("date").value.apply(np.percentile, low_percentile ), \
		cond_df.groupby("date").value.apply(np.percentile, high_percentile), alpha= alpha, color=colors[cond])

	return ax

def df_processing(dictionnary , std_arg = False , cumsum_arg = False , *args):

	def _process_df(scen_arg , df , std_arg , cumsum_arg):

		if cumsum_arg == True :
			df = df.apply( np.cumsum, axis=0 )
		else : pass

		df['date'] = df.index
		df['scenario']= scen_arg


		if std_arg == True :
			df = pd.melt(df, id_vars=["date", "scenario",'std'], var_name="condition")
		else :
			df = pd.melt(df, id_vars=["date", "scenario"], var_name="condition")

		return df

	_tmp = [_process_df( k , v , std_arg , cumsum_arg) for k , v in dictionnary.iteritems()]

	df = pd.concat(_tmp, ignore_index= True)
	df = df.drop('condition', 1)
	df = df.rename(columns = {'scenario':'condition'})
	df = df.sort_values(by=['condition','date'])
	df = df.reset_index(drop=True)

	return df

def decade_plot(scenario1 , scenario2 , observed , output_path , pdf, model , graph_variable, year_range=(1950,2100) , *args):
	# take a graphvariable, average over reps for a year and sums it over a decade.

	for domain in scenario1.domains :

		begin, end = year_range 
		end = end-1

		if graph_variable == 'avg_fire_size' :
			plot_title = 'Average Size of Fire per Decade %d-%d \n ALFRESCO, %s, %s, %s Domain' % ( begin, end, scenario1.model,scenario1.mscenario,domain )
			ylabel ='Average Fire Size ('+'$\mathregular{km^2}$' + ')' 

		elif graph_variable == 'number_of_fires' :
			plot_title = 'Total Number of Fires per Decade %d-%d \n ALFRESCO, %s, %s, %s Domain' % ( begin, end, scenario1.model,scenario1.mscenario,domain )
			ylabel = 'Number of Fires'

		elif graph_variable == 'total_area_burned' :
			plot_title = 'Total Area Burned %d-%d \n ALFRESCO, %s, %s, %s Domain' % ( begin, end, scenario1.model,scenario1.mscenario,domain )
			ylabel = 'Area Burned in ('+'$\mathregular{km^2}$' + ')'

		else : 'Error with Title'

		def std_calc(df):
			df['std'] = np.nan
			for i in range( begin , end , 10) :
				std = df.ix[ i : i + 9 ].sum( axis = 0 ).std()
				df.set_value((range( i , i + 10 ) ), 'std', std )
			return df

		#Handling the historical, oserved data
		obs_domain = observed.__dict__[graph_variable][domain].ix[begin : 2009]

		data = {scen_arg.scenario :std_calc(scen_arg.__dict__[graph_variable][domain].ix[begin : end]) for scen_arg in [scenario1,scenario2]}

		df = df_processing(data , std_arg = True)

		df = df.groupby(["condition", "date"]).mean().unstack("condition")

		#help to create those as the yerr is pretty sensitive to changes, had to create a 0 columns for std.
		errors = df['std'].groupby(df.index // 10 * 10).last() #all the value are similar anyway so last, first or mean would do
		errors['hist'] = 0
		means = df.value.groupby(df.index // 10 * 10).sum()
		means['hist'] = obs_domain.groupby(obs_domain.index // 10 * 10).sum()

		#plotting
		ax = means.plot(kind='bar',yerr= errors.values.T, error_kw={'ecolor':'grey','linewidth':1},legend=False, color = [scenario1.color,scenario2.color,observed.color], title=plot_title,  grid=False, width=0.8 )

		#Create label for axis
		plt.ylabel( ylabel )
		plt.xlabel( 'Decade' )
		plt.ylim(ymin=0 ) 

		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()
		
		plt.legend(handles = [ scenario1.patch , scenario2.patch , observed.patch],fontsize='medium',loc='best',borderaxespad=0.,ncol=1,frameon=False)

		output_filename = os.path.join( output_path, '_'.join([ 'alfresco', domain,graph_variable,'decade', model , str(begin), str(end)]) + '.png' )

		plt.savefig( output_filename )
		pdf.savefig()
		plt.close()

def compare_area_burned(scenario1 , scenario2 , observed , output_path , pdf, model , graph_variable, year_range=(1950,2100) , *args):
	#This plot compare the cumulative area burn for managed, unmanaged and historical period

	for domain in scenario1.domains :

		begin, end = year_range

		#Set some Style and settings for the plots

		plot_title = 'Cumulative Sum of Annual Area Burned %d-%d \n ALFRESCO, %s, %s, %s Domain' % ( begin, end, scenario1.model,scenario1.mscenario,domain )

		#Handling the historical, oserved data
		obs_domain = np.cumsum( observed.__dict__[graph_variable][domain].ix[begin : 2009] )
		
		data = {scen_arg.scenario :scen_arg.__dict__[graph_variable][domain].ix[begin : end] for scen_arg in [scenario1,scenario2]}

		df =df_processing(data , cumsum_arg = True)

		tmp = df.groupby(["condition", "date"]).mean().unstack("condition")
		tmp = tmp.value[[scenario1.scenario, scenario2.scenario]]

		#checking if colors_list list work
		ax = tmp.plot(legend=False, color = [scenario1.color,scenario2.color], title=plot_title,lw = 1.2,  grid=False)

		obs_domain.plot( ax=ax,legend=False, linestyle= '--', color=observed.color, grid=False, label= "observed" ,lw = 1.1)

		#Create label for axis
		plt.xlabel( 'Year' )
		plt.ylabel( 'Area burned in ('+'$\mathregular{km^2}$' + ')' )

		# Get a reference to the x-points corresponding to the dates and the the colors_list
		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()

		#have to pass the scenario object so they are avalaible for color definition
		fill_in(ax , df ,{'scenario1' : scenario1.color,'scenario2':scenario2.color} ,low_percentile = 5 , high_percentile = 95)

		plt.legend(handles = [ scenario1.line , scenario2.line , observed.line],fontsize='medium',loc='best',borderaxespad=0.,ncol=1,frameon=False)

		output_filename = os.path.join( output_path, '_'.join([ 'alfresco_annual_areaburned_compared_lines',domain, model , str(begin), str(end)]) + '.png' )

		plt.savefig( output_filename )
		pdf.savefig()
		plt.close()

def compare_vegcounts(scenario1 , scenario2 , observed , output_path , pdf, model ,veg_name_dict, graph_variable, year_range=(1950,2100) , *args):
	for domain in scenario1.domains :
		begin, end = year_range #subset the dataframes to the years of interest

		for veg_name in veg_name_dict.itervalues():

			plot_title = "Annual %s Coverage %s-%s \n ALFRESCO, %s, %s, %s Domain" \
				% ( veg_name, str(begin), str(end),scenario1.model,scenario1.mscenario,domain )

			data = {scen_arg.scenario :scen_arg.__dict__[graph_variable][domain][veg_name].ix[begin : end] for scen_arg in [scenario1,scenario2]}

			df =df_processing(data)	

			# Plot the average value by condition and date
			ax = df.groupby(["condition", "date"]).mean().unstack("condition").plot(legend=False, color = [scenario1.color,scenario2.color], title=plot_title,lw = 1,  grid=False )

			ax.get_xaxis().tick_bottom()
			ax.get_yaxis().tick_left()
			
			#Create label for axis
			plt.xlabel( 'Year' )
			plt.ylabel( 'Area covered in ('+'$\mathregular{km^2}$' + ')' )

			fill_in(ax , df ,{'scenario1' : scenario1.color,'scenario2':scenario2.color} ,low_percentile = 5 , high_percentile = 95)

			plt.legend(handles = [ scenario1.line , scenario2.line ],fontsize='medium',loc='best',borderaxespad=0.,ncol=1,frameon=False)

			output_filename = os.path.join( output_path, '_'.join([ 'alfresco_annual_areaveg_compared_lines',domain, model, veg_name.replace(' ', '' ), str(begin), str(end) ]) + '.png' ) 

			plt.savefig( output_filename )
			pdf.savefig()
			plt.close()

def compare_firesize(scenario1 , scenario2 , observed , output_path , pdf, model ,graph_variable, year_range=(1950,2100) , buff=False):
	#This graph will be about producing a comparative graph of fire size between managed and unmanaged in order to see if it changes with management
	for domain in scenario1.domains :
		begin, end = year_range #subset the dataframes to the years of interest

		plot_title = 'Average Size of Fire %d-%d \n ALFRESCO, %s, %s, %s Domain' % ( begin, end, scenario1.model,scenario1.mscenario,domain )

		data = {scen_arg.scenario :scen_arg.__dict__[graph_variable][domain].ix[begin : end] for scen_arg in [scenario1,scenario2]}

		df = df_processing(data)	

		# Plot the average value by condition and date
		ax = df.groupby(["condition", "date"]).mean().unstack("condition").plot(legend=False, color = [scenario1.color,scenario2.color], title=plot_title,lw = 1,  grid=False )

		#Create label for axis
		plt.xlabel( 'Year' )
		plt.ylabel( 'Average fire size ('+'$\mathregular{km^2}$' + ')' )

		#Create the buffer around the mean value to display the rep dispersion
		if buff==True :
			fill_in(ax , df ,{'scenario1' : scenario1.color,'scenario2':scenario2.color} ,low_percentile = 5 , high_percentile = 95)
		else: 
			pass
		
		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()
		
		plt.legend(handles = [ scenario1.line , scenario2.line ],fontsize='medium',loc='best',borderaxespad=0.,ncol=1,frameon=False)

		if buff==True :
			output_filename = os.path.join( output_path, '_'.join([ 'alfresco_avgfiresize_compared_buff',domain, model , str(begin), str(end)]) + '.png' )

		else :
			output_filename = os.path.join( output_path, '_'.join([ 'alfresco_avgfiresize_compared', domain,model , str(begin), str(end) ]) + '.png' )

		plt.savefig( output_filename )
		pdf.savefig()
		plt.close()

def compare_numberoffires(scenario1 , scenario2 , observed , output_path , pdf, model , graph_variable,year_range=(1950,2100) , buff=True):
	#This graph will be about producing a comparative graph of fire size between managed and unmanaged in order to see if it changes with management
	for domain in scenario1.domains :
		begin, end = year_range #subset the dataframes to the years of interest


		plot_title = 'Cumulative Number of Fires %d-%d \n ALFRESCO, %s, %s, %s Domain' % ( begin, end, scenario1.model,scenario1.mscenario,domain )

		data = {scen_arg.scenario :scen_arg.__dict__[graph_variable][domain].ix[begin : end] for scen_arg in [scenario1,scenario2]}

		df = df_processing(data , cumsum_arg = True)	

		ax = df.groupby(["condition", "date"]).mean().unstack("condition").plot(legend=False, color = [scenario1.color,scenario2.color], title=plot_title,lw = 0.7,  grid=False )


		#Create label for axis
		plt.xlabel( 'Year' )
		plt.ylabel( 'Number of fires' )

		# Get a reference to the x-points corresponding to the dates and the the colors_list
		x = df.date.unique()

		#Create the buffer around the mean value to display the rep dispersion
		if buff==True :
			fill_in(ax , df ,{'scenario1' : scenario1.color,'scenario2':scenario2.color} ,low_percentile = 5 , high_percentile = 95)
		else: 
			pass

		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()
		
		plt.legend(handles = [ scenario1.line , scenario2.line ],fontsize='medium',loc='best',borderaxespad=0.,ncol=1,frameon=False)

		if buff==True :
			output_filename = os.path.join( output_path, '_'.join([ 'alfresco_numberoffires_cum_compared_buff',domain, model , str(begin), str(end)]) + '.png' )

		else :
			output_filename = os.path.join( output_path, '_'.join([ 'alfresco_numberoffires_cum_compared', domain,model , str(begin), str(end)]) + '.png' )

		plt.savefig( output_filename )
		pdf.savefig()
		plt.close()

def compare_cab_vs_fs(scenario1 , scenario2 , observed , output_path , pdf, model , graph_variable, year_range=(1950,2100) , *args):
	#This graph shows the cumulative area burnt by fire size, managed and unmanaged scenario are compared on the same plot
	#Mainly based on Michael's code https://github.com/ua-snap/alfresco-calibration/blob/cavm/alfresco_postprocessing_plotting.py#L252
	
	begin, end = year_range

	for domain in scenario1.domains :
		fig, ax = plt.subplots() 

		def wrangling(df , color , scenario):
			if scenario!= 'observed' :

				df_list = []
				for col in df.columns[1:]:
					mod_sorted = sorted( [ j for i in df[ col ].astype(str) for j in ast.literal_eval(i) ] )
					mod_cumsum = np.cumsum( mod_sorted )
					replicate = [ col for i in range( len( mod_sorted ) ) ]
					df_list.append( pd.DataFrame( {'mod_sorted':mod_sorted, 'mod_cumsum':mod_cumsum, 'replicate':replicate} ) )
				mod_melted = pd.concat( df_list )	
				mod_melted.groupby( 'replicate' ).apply( lambda x: plt.plot( x['mod_sorted'], x['mod_cumsum'], color=color, alpha=0.5, lw=1) )

			else :
				mod_sorted = sorted( [ j for i in df[df.columns[0]] for j in ast.literal_eval(i) ] )
				mod_cumsum = np.cumsum( mod_sorted )
				plt.plot( mod_sorted, mod_cumsum, color=color, alpha=0.5, lw=1)


		wrangling(scenario1.__dict__[graph_variable][domain].ix[begin : end] , scenario1.color ,'scenario1')
		wrangling(scenario2.__dict__[graph_variable][domain].ix[begin : end] , scenario2.color , 'scenario2')
		wrangling(observed.__dict__[graph_variable][domain].ix[begin : end] , observed.color , 'observed')

		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()

		#Create label for axis
		plt.ylabel( 'Area burned in ('+'$\mathregular{km^2}$' + ')' )
		plt.xlabel( 'Fire size ('+'$\mathregular{km^2}$' + ')' )

		fig.suptitle('Cumulative Area Burned vs. Fire Sizes %d-%d \n ALFRESCO, %s, %s, %s Domain' % ( begin, end, scenario1.model,scenario1.mscenario,domain ))
	
		plt.legend(handles = [ scenario1.line , scenario2.line,observed.line ],fontsize='medium',loc='best',borderaxespad=0.,ncol=1,frameon=False)

		output_filename = os.path.join( output_path, '_'.join([ 'alfresco_cab_vs_fs',domain, model , str(begin), str(end)]) + '.png' )

		plt.savefig( output_filename )
		pdf.savefig()
		plt.close()

def CD_ratio(scenario1 , scenario2 , observed , output_path , pdf, model , graph_variable, year_range=(1950,2100) , *args):

	begin, end = year_range

	for domain in scenario1.domains :

		plot_title = 'Conifer:Deciduous Ratios %d-%d \n ALFRESCO, %s, %s, %s Domain' % ( begin, end, scenario1.model,scenario1.mscenario,domain )


		data = {scen_arg.scenario : get_veg_ratios( scen_arg.__dict__[graph_variable], domain ) for scen_arg in [scenario1,scenario2] }


		df =df_processing( data )	
		ax = df.groupby(["condition", "date"]).mean().unstack("condition").plot(legend=False, color = [scenario1.color,scenario2.color], title=plot_title,lw = 1,  grid=False )

		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()
		
		#Create label for axis
		plt.xlabel( 'Year' )
		plt.ylabel( 'C:D Ratio' )

		fill_in(ax , df ,{'scenario1' : scenario1.color,'scenario2':scenario2.color} ,low_percentile = 5 , high_percentile = 95)

		plt.legend(handles = [ scenario1.line , scenario2.line ],fontsize='medium',loc='best',borderaxespad=0.,ncol=1,frameon=False)

		output_filename = os.path.join( output_path, '_'.join([ 'alfresco_CD_ratio',domain, model, str(begin), str(end) ]) + '.png' ) 

		plt.savefig( output_filename )
		pdf.savefig()
		plt.close()


def launcher(model) :

	from collections import defaultdict


	# json_list = [os.path.join(data_path, 'JSON','_'.join([ 'ALF',model,'PreFMO']) + '.json' ),os.path.join(data_path,'JSON','_'.join([ 'ALF',model]) + '.json' ), os.path.join(data_path,'JSON','Observed' + '.json' )]
	json_list = [glob.glob(os.path.join(prefmopath,'JSON','_'.join([ 'ALF',model]) + '*.json' ))[0],
	os.path.join(data_path,'JSON','_'.join([ 'ALF',model]) + '.json' )
	, os.path.join(data_path,'JSON','Observed' + '.json' )]
	scenario1 = Scenario( json_list[0], model, 'scenario1', "PreFMO calibration" , '#db5757')
	scenario2 = Scenario( json_list[1], model, 'scenario2', "Post FMO calibration" , '#4a71b5')
	observed = Scenario( json_list[2], model, 'Observed', "Historical", '#162f3b' )

	#################################################################################################################
	########################################      FIX as ALF observed v3.1 is broken ################################

	hist_dict = defaultdict( lambda: defaultdict( lambda: defaultdict ) )
	#hardwired as a quick fix! those three metrics are wrong in v3.1
	nof = pd.read_csv('/workspace/Shared/Users/jschroder/ALFRESCO_SERDP/ALFRESCO_EPA/RCP85/output_csvs/alfresco_observed_historical_serdp_numberoffires_all_domains_1917_2011.csv', index_col=0)
	hist_dict['number_of_fires']['Boreal'] = nof['1'].to_frame(name='Observed')
	hist_dict['number_of_fires']['Tundra'] = nof['2'].to_frame(name='Observed')
	observed.number_of_fires = hist_dict['number_of_fires']


	avgfs = pd.read_csv('/workspace/Shared/Users/jschroder/ALFRESCO_SERDP/ALFRESCO_EPA/RCP85/output_csvs/alfresco_observed_historical_serdp_avgfiresize_all_domains_1917_2011.csv', index_col=0)
	hist_dict['avg_fire_size']['Boreal'] = avgfs['1'].to_frame(name='Observed')
	hist_dict['avg_fire_size']['Tundra'] = avgfs['2'].to_frame(name='Observed')
	observed.avg_fire_size = hist_dict['avg_fire_size']

	afs = pd.read_csv('/workspace/Shared/Users/jschroder/ALFRESCO_SERDP/ALFRESCO_EPA/RCP85/output_csvs/alfresco_observed_historical_serdp_allfiresizes_all_domains_1917_2011.csv', index_col=0)
	hist_dict['all_fire_sizes']['Boreal'] = afs['1'].to_frame(name='Observed')
	hist_dict['all_fire_sizes']['Tundra'] = afs['2'].to_frame(name='Observed')
	observed.all_fire_sizes = hist_dict['all_fire_sizes']

	del hist_dict

	###############################################################################################################
	###############################################################################################################
	###############################################################################################################
	###############################################################################################################



	output_path = os.path.join( visual , model ) #for production

	if not os.path.exists( output_path ):
		os.makedirs( output_path )

	pdf = os.path.join( visual, '_'.join([ model,'plots']) + '.pdf' )

	with PdfPages(pdf) as pdf:

		for metric in scenario1.metrics :
			if metric not in [ 'veg_counts' , 'all_fire_sizes']:
				decade_plot(scenario1 , scenario2 , observed , output_path , pdf, model , metric)

			else : pass
		
		compare_area_burned(scenario1 , scenario2 , observed , output_path , pdf, model , 'total_area_burned' )
		compare_vegcounts(scenario1 , scenario2 , observed , output_path , pdf, model, veg_name_dict , 'veg_counts' )
		compare_firesize( scenario1 , scenario2 , observed , output_path , pdf, model , 'avg_fire_size' )
		compare_numberoffires( scenario1 , scenario2 , observed , output_path , pdf, model , 'number_of_fires' )
		compare_cab_vs_fs(scenario1 , scenario2 , observed , output_path , pdf, model , 'all_fire_sizes' )
		CD_ratio(scenario1 , scenario2 , observed , output_path , pdf, model, 'veg_counts' )


if __name__ == '__main__':
	import pandas as pd
	import numpy as np
	import glob, os, ast, sys,argparse
	import numpy as np
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	import matplotlib.patches as mpatches
	import matplotlib.lines as mlines
	from alfresco_postprocessing import plot
	from matplotlib.backends.backend_pdf import PdfPages
	from matplotlib import rcParams
	from pathos import multiprocessing as mp
	pd.options.mode.chained_assignment = None  # default='warn'


	parser = argparse.ArgumentParser( description='' )
	parser.add_argument( "-p", "--path", action='store', dest='path', type=str, help="model path" )
	args = parser.parse_args()
	prefmopath = args.path


	data_path = '/atlas_scratch/jschroder/Calibration_AR4_AR5'

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
	# rcParams[ 'savefig.bbox'] = 'tight'
	# rcParams[ 'savefig.pad_inches'] = 0.2


	visual = os.path.join( prefmopath , 'Plots' )
	if not os.path.exists( visual ):
		os.mkdir( visual )

	models = [
		'cccma_cgcm3_1_sresa1b']


	veg_name_dict = {'BlackSpruce':'Black Spruce',
				'WhiteSpruce':'White Spruce',
				'Deciduous':'Deciduous',
				'ShrubTundra':'Shrub Tundra',
				'GraminoidTundra':'Graminoid Tundra',
				#6:'Wetland Tundra',
				#7:'Barren lichen-moss',
				#8:'Temperate Rainforest'
				}

	
	scenarios = [ 'scenario_1', 'scenario_2']

	# pool = mp.Pool( len(models) )

	# pool.map( launcher,models)
	# pool.close()
	launcher(models[0])





