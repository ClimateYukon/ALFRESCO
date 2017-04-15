
def decade_plot( model , graph_variable, year_range,output_path,pdf):
	# take a graphvariable, average over reps for a year and sums it over a decade.
	
	sns.set(style="whitegrid")

	begin, end = year_range #subset the dataframes to the years of interest
	end = end-1

	figsize = ( 14, 8 )

	if graph_variable == 'avg_fire_size' :
		plot_title = 'Average Size of Fire per Decade %d-%d in %s domain\n ALFRESCO, %s, RCP 8.5' % ( begin, end,domain_name, model )
		ylabel ='Average fire size ('+'$\mathregular{km^2}$' + ')' 

	elif graph_variable == 'number_of_fires' :
		plot_title = 'Total Number of Fires per Decade %d-%d %s domain\n ALFRESCO, %s, RCP 8.5' % ( begin, end,domain_name, model )
		ylabel = 'Number of fires'

	elif graph_variable == 'total_area_burned' :
		plot_title = 'Total Area Burned %d-%d %s domain\n ALFRESCO, %s, RCP 8.5' % ( begin, end,domain_name, model )
		ylabel = 'Area burned in ('+'$\mathregular{km^2}$' + ')'



	#Handling the historical, oserved data
	obs_tab = pd.read_csv( glob.glob( os.path.join( input_path, '*'.join([ 'alfresco', 'historical',graph_variable.replace('_',''), '.csv' ]) ) )[0], index_col=0 )
	obs_domain = obs_tab[ str(domain_num) ]
	obs_domain = obs_domain.ix[ begin: 2009]

	frames = []
	for scenario in scenarios :
		i = glob.glob( os.path.join( input_path,graph_variable, '*'.join([ 'alfresco',graph_variable.replace('_',''), domain_name,model, scenario, '.csv' ]) ) )[0]

		tab = pd.read_csv( i, index_col=0 ).ix[begin:end]

		#Standard deviation calculation happens here, in case needed to change for quantile
		tab['std'] = np.nan
		for i in range( begin , end+1 , 10) :
			std = tab.ix[ i : i + 9 ].sum( axis = 0 ).std()
			tab.set_value((range( i , i + 10 ) ), 'std', std )

		tab['date'] = tab.index
		tab['scenario']= scenario
		tab = pd.melt(tab, id_vars=["date", "scenario",'std'], var_name="condition")
		frames.append(tab)

	#Complete dataframe creation with both scenario in long fata form
	df = pd.concat(frames, ignore_index= True)
	df = df.drop('condition', 1)
	df = df.rename(columns = {'scenario':'condition'})
	df = df.sort_values(by=['condition','date'])
	df = df.reset_index(drop=True)
	df = df.groupby(["condition", "date"]).mean().unstack("condition")

	#help to create those as the yerr is pretty sensitive to changes, had to create a 0 columns for std.
	errors = df['std'].groupby(df.index // 10 * 10).last() #all the value are similar anyway so last, first or mean would do
	errors['hist'] = 0
	means = df.value.groupby(df.index // 10 * 10).sum()
	means['hist'] = obs_domain.groupby(obs_domain.index // 10 * 10).sum()

	#plotting
	ax = means.plot(kind='bar',yerr= errors.values.T, error_kw={'ecolor':'grey','linewidth':1},legend=False, color = colors_list, title=plot_title, figsize=figsize, grid=False, width=0.8 )

	#Create label for axis
	plt.ylabel( ylabel )
	plt.xlabel( 'Decade' )
	plt.ylim(ymin=0 ) 

	blu_patch = mpatches.Patch([], [], linewidth=1.2, color=colors_list[0], label='Without FMO' )
	green_patch = mpatches.Patch([], [], linewidth=1.2, color= colors_list[1] , label='With FMO' )
	#red_patch = mpatches.Patch([], [], linewidth=1.2, color=colors_list[1], label='Future FMO' )
	gren_patch = mpatches.Patch([], [], linewidth=1.2, color=colors_list[2], label='Historical' )
	plt.legend(handles = [ blu_patch,green_patch ,gren_patch],loc='best',ncol=1, shadow=True, fancybox=True)


	output_filename = os.path.join( output_path, '_'.join([ 'alfresco', graph_variable,domain_name,'decade', model , str(begin), str(end),model ]) + '.png' )
	print "Writing %s to disk" %output_filename
	sns.despine()
	plt.savefig( output_filename )
	pdf.savefig()
	plt.close()

def compare_area_burned(model , year_range,output_path,pdf):
	#This plot compare the cumulative area burn for managed, unmanaged and historical period
	sns.set(style="whitegrid")
	begin, end = year_range #subset the dataframes to the years of interest
	graph_variable = 'total_area_burned'
	#Set some Style and settings for the plots
	figsize = ( 14 ,8 )
	plot_title = 'Cumulative Sum of Annual Area Burned %d-%d %s domain\n ALFRESCO, %s, RCP 8.5' % ( begin, end,domain_name,model)

	#Handling the historical, oserved data
	obs_tab = pd.read_csv( glob.glob( os.path.join( input_path, '*'.join([ 'alfresco', 'historical',  graph_variable.replace('_',''), '.csv' ]) ) )[0], index_col=0 )
	obs_domain = obs_tab[ str(domain_num) ]
	obs_domain = obs_domain.ix[ begin:2009 ]
	obs_domain = np.cumsum( obs_domain )


	frames = []
	# Cleaning and adding some fields for each individual scenario's dataframe for concat
	for scenario in scenarios :
		i = glob.glob( os.path.join( input_path,graph_variable, '*'.join([ 'alfresco',graph_variable.replace('_',''), domain_name,model, scenario, '.csv' ]) ) )[0]

		tab = pd.read_csv( i, index_col=0 ).ix[begin:end]
		tab = tab.apply( np.cumsum, axis=0 )
		tab['date'] = tab.index
		tab['scenario']= scenario
		tab = pd.melt(tab, id_vars=["date", "scenario"], var_name="condition")
		frames.append(tab)

	#Complete dataframe creation with both scenario in long fata form
	df = pd.concat(frames, ignore_index= True)
	df = df.drop('condition', 1)
	df = df.rename(columns = {'scenario':'condition'})
	df = df.sort_values(by=['condition','date'])
	df = df.reset_index(drop=True)
	
	tmp = df.groupby(["condition", "date"]).mean().unstack("condition")
	#tmp = tmp.value[['scenario_1', 'scenario_3', 'scenario_2']]
	tmp = tmp.value[['scenario_1', 'scenario_2']]
	colors = [colors_list[0],colors_list[1],colors_list[2]]
	
	#checking if colors_list list work
	ax = tmp.plot(legend=False, color = colors, title=plot_title,lw = 1.2, figsize=figsize, grid=False)


	#We have to plot scenario_3 first so we need to change the colors order

	#checking if colors_list list work
	obs_domain.plot( legend=False, linestyle= '--', color=colors[2], grid=False, label= "observed" ,lw = 1.1)

	#Create label for axis
	plt.xlabel( 'Years' )
	plt.ylabel( 'Area burned in ('+'$\mathregular{km^2}$' + ')'  )

	# Get a reference to the x-points corresponding to the dates and the the colors_list
	x = df.date.unique()
	palette = sns.color_palette()

	#Create the buffer around the mean value to display the rep dispersion
	for cond, cond_df in df.groupby("condition"):
		low = cond_df.groupby("date").value.apply(np.percentile, 5)
		high = cond_df.groupby("date").value.apply(np.percentile, 95)
		if cond == 'scenario_2' : i = 1
		elif cond == 'scenario_1' : i =0
		ax.fill_between(x, low, high, alpha=.2, color=colors_list[i])


	# build and display legend
	
	blu_patch = mlines.Line2D([], [], linewidth=1.2, color=colors[0], label='Without FMO' )
	green_patch = mlines.Line2D([], [], linewidth=1.2, color= colors[1] , label='With FMO' )
	#red_patch = mlines.Line2D([], [], linewidth=1.2, color=colors[1], label='Future FMO' )
	ired_patch = mlines.Line2D([], [], ls='--', linewidth=1, color=colors[2], label='Historical' )
	#Setting legend
	plt.legend(handles = [blu_patch,green_patch,ired_patch,],loc="best",ncol=1, shadow=True, fancybox=True)
	sns.despine()

	output_filename = os.path.join( output_path, '_'.join([ 'alfresco_annual_areaburned_compared_lines',domain_name, model , str(begin), str(end),model ]) + '.png' )
	print "Writing %s to disk" %output_filename
	plt.savefig( output_filename )
	pdf.savefig()
	plt.close()

def compare_vegcounts(model ,veg_name_dict, year_range,output_path,pdf):
	sns.set(style="whitegrid")
	begin, end = year_range #subset the dataframes to the years of interest

	#Set some Style and settings for the plots
	figsize = ( 14, 8 )
	graph_variable = 'veg_counts'
	fig, ax = plt.subplots(figsize=figsize, facecolor = 'w' ) 
	for veg_num, veg_name in veg_name_dict.iteritems():

		plot_title = "Annual %s Coverage %s-%s %s domain\n ALFRESCO, %s, RCP 8.5"\
			% ( veg_name, str(begin), str(end),domain_name,model)

		frames = []
		# Cleaning and adding some fields for each individual scenario's dataframe for concat
		for scenario in scenarios :
			i = glob.glob( os.path.join( input_path,graph_variable, '*'.join([ 'alfresco',graph_variable.replace('_',''), domain_name,veg_num,model, scenario, '.csv' ]) ) )[0]

			tab = pd.read_csv( i, index_col=0 ).ix[begin:end]
			#tab = tab.apply( np.cumsum, axis=0 )
			tab['date'] = tab.index
			tab['scenario']= scenario
			tab = pd.melt(tab, id_vars=["date", "scenario"], var_name="condition")
			frames.append(tab)
		
		#Complete dataframe creation with both scenario in long fata form
		df = pd.concat(frames, ignore_index= True)
		df = df.drop('condition', 1)
		df = df.rename(columns = {'scenario':'condition'})
		df = df.sort_values(by=['condition','date'])
		df = df.reset_index(drop=True)

		# Plot the average value by condition and date
		ax = df.groupby(["condition", "date"]).mean().unstack("condition").plot(legend=False, color = colors_list, title=plot_title,lw = 1, figsize=figsize, grid=False )


		#Create label for axis
		plt.xlabel( 'Years' )
		plt.ylabel( 'Area covered in ('+'$\mathregular{km^2}$' + ')'  )

		# Get a reference to the x-points corresponding to the dates and the the colors_list
		x = df.date.unique()


		#Create the buffer around the mean value to display the rep dispersion
		for cond, cond_df in df.groupby("condition"):
			low = cond_df.groupby("date").value.apply(np.percentile, 5)
			high = cond_df.groupby("date").value.apply(np.percentile, 95)
			if cond == 'scenario_2' : i = 1
			elif cond == 'scenario_1' : i =0
			ax.fill_between(x, low, high, alpha=.2, color=colors_list[i])


		# build and display legend
		blu_patch = mlines.Line2D([], [], linewidth=1.2, color=colors_list[0], label='Without FMO' )
		green_patch = mlines.Line2D([], [], linewidth=1.2, color= colors_list[1] , label='With FMO' )
		#red_patch = mlines.Line2D([], [], linewidth=1.2, color=colors_list[1], label='Future FMO' )
		
		# if veg_name in ['GraminoidTundra','BlackSpruce','ShrubTundra'] :
		# 	plt.legend(handles = [#blu_patch,green_patch,red_patch],loc="bottom right",ncol=1, shadow=True, fancybox=True) bbox_to_anchor=[0, 1],
		# else :
		plt.legend(handles = [blu_patch,green_patch],loc="best",ncol=1, shadow=True, fancybox=True)

		sns.despine()

		output_filename = os.path.join( output_path, '_'.join([ 'alfresco_annual_areaveg_compared_lines',domain_name, model, veg_name.replace(' ', '' ), domain_name.replace(' ', '' ), str(begin), str(end),model ]) + '.png' ) 
		print "Writing %s to disk" %output_filename
		plt.savefig( output_filename )
		pdf.savefig()
		plt.close()

def compare_firesize( model , year_range ,output_path, pdf,buff=False,):
	#This graph will be about producing a comparative graph of fire size between managed and unmanaged in order to see if it changes with management
	sns.set(style="whitegrid")
	graph_variable = 'avg_fire_size'

	begin, end = year_range #subset the dataframes to the years of interest

	plot_title = 'Average Size of Fire %d-%d %s domain\n ALFRESCO, %s, RCP 8.5' % ( begin, end,domain_name, model )
	figsize = ( 14, 8 )


	frames = []
	# Cleaning and adding some fields for each individual scenario's dataframe for concat
	for scenario in scenarios :
		i = glob.glob( os.path.join( input_path,graph_variable, '*'.join([ 'alfresco',graph_variable.replace('_',''), domain_name,model, scenario, '.csv' ]) ) )[0]
		tab = pd.read_csv( i, index_col=0 ).ix[begin:end]
		#tab = tab.apply( np.cumsum, axis=0 )
		tab['date'] = tab.index
		tab['scenario']= scenario
		tab = pd.melt(tab, id_vars=["date", "scenario"], var_name="condition")
		frames.append(tab)

	#Complete dataframe creation with both scenario in long fata form
	df = pd.concat(frames, ignore_index= True)
	df = df.drop('condition', 1)
	df = df.rename(columns = {'scenario':'condition'})
	df = df.sort_values(by=['condition','date'])
	df = df.reset_index(drop=True)

	# Plot the average value by condition and date
	ax = df.groupby(["condition", "date"]).mean().unstack("condition").plot(legend=False, color = colors_list, title=plot_title,lw = 1, figsize=figsize, grid=False )


	#Create label for axis
	plt.xlabel( 'Years' )
	plt.ylabel( 'Average fire size ('+'$\mathregular{km^2}$' + ')' )

	# Get a reference to the x-points corresponding to the dates and the the colors_list
	x = df.date.unique()


	#Create the buffer around the mean value to display the rep dispersion
	if buff==True :
		for cond, cond_df in df.groupby("condition"):
			low = cond_df.groupby("date").value.apply(np.percentile, 5)
			high = cond_df.groupby("date").value.apply(np.percentile, 95)
			if cond == 'scenario_1' : i = 0
			elif cond == 'scenario_2' : i =1
			ax.fill_between(x, low, high, alpha=.2, color=colors_list[i])
	else: 
		pass

	# build and display legend

	blu_patch = mlines.Line2D([], [], linewidth=1.2, color=colors_list[0], label='Without FMO' )
	green_patch = mlines.Line2D([], [], linewidth=1.2, color= colors_list[1] , label='With FMO' )
	#red_patch = mlines.Line2D([], [], linewidth=1.2, color=colors_list[1], label='Future FMO' )

	plt.legend(handles = [blu_patch,green_patch],loc="best", ncol=1, shadow=True, fancybox=True)
	#plt.show()

	if buff==True :
		output_filename = os.path.join( output_path, '_'.join([ 'alfresco_avgfiresize_compared_buff', domain_name ,model , str(begin), str(end),model]) + '.png' )

	else :
		output_filename = os.path.join( output_path, '_'.join([ 'alfresco_avgfiresize_compared', model , str(begin), str(end),model ]) + '.png' )
	print "Writing %s to disk" %output_filename
	sns.despine()
	plt.savefig( output_filename )
	pdf.savefig()
	plt.close()

def compare_numberoffires( model , year_range ,output_path, pdf,buff=True):
	#This graph will be about producing a comparative graph of fire size between managed and unmanaged in order to see if it changes with management
	sns.set(style="whitegrid")
	graph_variable = 'number_of_fires'

	begin, end = year_range #subset the dataframes to the years of interest


	plot_title = 'Cumulative Number of Fires %d-%d %s domain\n ALFRESCO, %s, RCP 8.5' % ( begin, end,domain_name, model )
	figsize = ( 14, 8 )


	frames = []
	# Cleaning and adding some fields for each individual scenario's dataframe for concat

	for scenario in scenarios :
		i = glob.glob( os.path.join( input_path,graph_variable, '*'.join([ 'alfresco',graph_variable.replace('_',''), domain_name,model, scenario, '.csv' ]) ) )[0]
		tab = pd.read_csv( i, index_col=0 ).ix[begin:end]
		tab = tab.apply( np.cumsum, axis=0 )
		tab['date'] = tab.index
		tab['scenario']= scenario
		tab = pd.melt(tab, id_vars=["date", "scenario"], var_name="condition")
		frames.append(tab)

	#Complete dataframe creation with both scenario in long fata form
	df = pd.concat(frames, ignore_index= True)
	df = df.drop('condition', 1)
	df = df.rename(columns = {'scenario':'condition'})
	df = df.sort_values(by=['condition','date'])
	df = df.reset_index(drop=True)


	ax = df.groupby(["condition", "date"]).mean().unstack("condition").plot(legend=False, color = colors_list, title=plot_title,lw = 0.7, figsize=figsize, grid=False )


	#Create label for axis
	plt.xlabel( 'Years' )
	plt.ylabel( 'Number of fires' )

	# Get a reference to the x-points corresponding to the dates and the the colors_list
	x = df.date.unique()


	#Create the buffer around the mean value to display the rep dispersion
	if buff==True :
		for cond, cond_df in df.groupby("condition"):
			low = cond_df.groupby("date").value.apply(np.percentile, 5)
			high = cond_df.groupby("date").value.apply(np.percentile, 95)
			if cond == 'scenario_2' : i = 1
			elif cond == 'scenario_1' : i =0
			ax.fill_between(x, low, high, alpha=.2, color=colors_list[i])
	else: 
		pass

	blu_patch = mlines.Line2D([], [], linewidth=1.2, color=colors_list[0], label='Without FMO' )
	green_patch = mlines.Line2D([], [], linewidth=1.2, color= colors_list[1] , label='With FMO' )
	#red_patch = mlines.Line2D([], [], linewidth=1.2, color=colors_list[1], label='Future FMO' )

	plt.legend(handles = [blu_patch,green_patch],loc="best",ncol=1, shadow=True, fancybox=True)
	#plt.show()

	if buff==True :
		output_filename = os.path.join( output_path, '_'.join([ 'alfresco_numberoffires_cum_compared_buff',domain_name, model , str(begin), str(end),model]) + '.png' )

	else :
		output_filename = os.path.join( output_path, '_'.join([ 'alfresco_numberoffires_cum_compared', domain_name,model , str(begin), str(end),model ]) + '.png' )

	print "Writing %s to disk" %output_filename
	sns.despine()
	plt.savefig( output_filename )
	pdf.savefig()
	plt.close()

def compare_cab_vs_fs(model , year_range,output_path,pdf):
	#This graph shows the cumulative area burnt by fire size, managed and unmanaged scenario are compared on the same plot
	#Mainly based on Michael's code https://github.com/ua-snap/alfresco-calibration/blob/cavm/alfresco_postprocessing_plotting.py#L252
	sns.set(style="whitegrid")
	begin, end = year_range
	figsize = (14,8)

	fig, ax = plt.subplots(figsize=figsize) 
	graph_variable = 'all_fire_sizes'
	for color, scenario in zip(colors_list , scenarios) :
		l = glob.glob( os.path.join( input_path,graph_variable, '*'.join([ 'alfresco',graph_variable.replace('_',''),domain_name , model, scenario,'.csv' ]) ) )[0]

		modeled = pd.read_csv( l, index_col=0 )

		df_list = []
		for col in modeled.columns[1:]:
			mod_sorted = sorted( [ j for i in  modeled[ col ] for j in ast.literal_eval(i) ] )
			mod_cumsum = np.cumsum( mod_sorted )
			replicate = [ col for i in range( len( mod_sorted ) ) ]
			df_list.append( pd.DataFrame( {'mod_sorted':mod_sorted, 'mod_cumsum':mod_cumsum, 'replicate':replicate} ) )

		# melt the ragged arrays with a concat -- dirty way
		mod_melted = pd.concat( df_list )	
		mod_melted.groupby( 'replicate' ).apply( lambda x: plt.plot( x['mod_sorted'], x['mod_cumsum'], color=color, alpha=0.5, lw=1) )


	sns.set_style( 'whitegrid', {'ytick.major.size': 7, 'xtick.major.size': 7} )

	# build and display legend
	blu_patch = mlines.Line2D([], [], linewidth=1.2, color=colors_list[0], label='Without FMO' )
	green_patch = mlines.Line2D([], [], linewidth=1.2, color= colors_list[1] , label='With FMO' )
	#red_patch = mlines.Line2D([], [], linewidth=1.2, color=colors_list[1], label='Future FMO')
	#Create label for axis
	plt.ylabel( 'Area burned in ('+'$\mathregular{km^2}$' + ')'  )
	plt.xlabel( 'Fire size ('+'$\mathregular{km^2}$' + ')' )

	fig.suptitle('Cumulative Area Burned vs. Fire Sizes %d-%d %s domain\n ALFRESCO, %s, RCP 8.5' \
						% ( begin, end,domain_name, model))
	plt.legend( handles=[blu_patch,green_patch], frameon=False, loc=0 )


	sns.despine()

	output_filename = os.path.join( output_path, '_'.join([ 'alfresco_cab_vs_fs', domain_name,model , str(begin), str(end),model ]) + '.png' )
	print "Writing %s to disk" %output_filename
	plt.savefig( output_filename )
	pdf.savefig()
	plt.close()

def plot(model):
	output_path = os.path.join( visu , model ) #for production

	if not os.path.exists( output_path ):
		os.mkdir( output_path )


	pdf_output =  os.path.join( visu, '_'.join([ domain_name, model,'plots','RCP85' ]) + '.pdf' )
	with PdfPages(pdf_output) as pdf:
		for metric in metrics[:-2]:
			decade_plot(model , metric, year_range,output_path,pdf)

		compare_area_burned(model,year_range,output_path,pdf)
		compare_vegcounts(model, veg_name_dict,year_range,output_path,pdf)
		compare_firesize( model,year_range,output_path,pdf)
		compare_numberoffires( model,year_range,output_path,pdf)
		compare_cab_vs_fs(model,year_range,output_path,pdf)

if __name__ == '__main__':
	import pandas as pd
	import numpy as np
	import geopandas as gpd
	import glob, os, json, ast
	import numpy as np
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	import matplotlib.patches as mpatches
	import matplotlib.lines as mlines
	import seaborn as sns
	from pathos import multiprocessing as mp
	from collections import OrderedDict
	from matplotlib.backends.backend_pdf import PdfPages

	
	data_path = '/workspace/Shared/Users/jschroder/ALFRESCO_EPA/RCP85'
	os.chdir(data_path)

	input_path = 'output_csvs'
	visu = 'Visual_outputs'

	if not os.path.exists( os.path.join( data_path , visu ) ):
		os.mkdir( os.path.join(data_path,visu) )
	
	metrics = [ 'avg_fire_size' , 'number_of_fires' , 'total_area_burned' , 'veg_counts' , 'all_fire_sizes']

	# for metric in metrics :
	# 	l = glob.glob(os.path.join(data_path, input_path, metric, '*.csv'))
	
	# 	for i in l :
	# 		if os.path.splitext( os.path.basename( i ) )[0].split( '_' )[-3] == 'rcp85' :
	# 			a = i.replace('rcp85','rcp85_scenario_2')
	# 			os.rename(i, a)
	# 		elif os.path.splitext( os.path.basename( i ) )[0].split( '_' )[-3] == 'NoFMO':
	# 			a = i.replace('NoFMO','scenario_1')
	# 			os.rename(i, a)
	# 		elif os.path.splitext( os.path.basename( i ) )[0].split( '_' )[-3] == 'AltFMO':
	# 			a = i.replace('AltFMO','scenario_3')
	# 			os.rename(i, a)
	# 		else : pass
		

	#subdomains_path = '/workspace/jschroder/ALFRESCO_SERDP/Domains/AOI_SERDP.shp'

	#Hardwired for debugging since SERDP project just have one domain, easy to switch back
	for domain_num,domain_name in zip([1,2],['Boreal','Tundra']):

		models = ['GISS-E2-R', 'GFDL-CM3', 'IPSL-CM5A-LR', 'MRI-CGCM3', 'CCSM4']
		#models = ['GISS-E2-R']
		#Order matters as the decade_plot takes metric[:-2] as input
		metrics = [ 'avg_fire_size' , 'number_of_fires' , 'total_area_burned' , 'veg_counts' , 'all_fire_sizes']

		veg_name_dict = {'BlackSpruce':'Black Spruce',
					'WhiteSpruce':'White Spruce',
					'Deciduous':'Deciduous',
					'ShrubTundra':'Shrub Tundra',
					'GraminoidTundra':'Graminoid Tundra',
					#'Wetland : Tundra',
					#'Barren : lichen-moss',
					#'Temperate : Rainforest'
					}

		storage = sns.color_palette('deep',7)
		colors_list = [storage[1],storage[2],sns.xkcd_rgb["charcoal"]]
		
		scenarios = [ 'scenario_1', 'scenario_2']
		year_range = (1950,2100)

		pool = mp.Pool( 32 )

		#Use partial in order to be able to pass output_path argument to the mapped function
		pool.map( plot,models)
		pool.close()
		pool.join()
