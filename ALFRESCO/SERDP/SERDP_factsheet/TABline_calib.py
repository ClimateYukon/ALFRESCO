def generate( ax, meanline, lower, upper, label, linecolor, rangecolor, alpha, line_zorder, linestyle, *args, **kwargs ):
	'''
	overlay lines and ranges...
	ax = mpl axes object to use for plotting lines and ranges
	meanline = pandas.Series with index of years and summarized line values (mean)
	lower = pandas.Series with index of years and summarized line values (min) for fill_between
	upper = pandas.Series with index of years and summarized line values (max) for fill_between
	linecolor = matplotlib compatible color for the line object
	rangecolor = matplotlib compatible color for the fill_between object
	alpha = transparency level
	'''
	# plot line
	years = meanline.index.astype( int )
	ax.plot( np.array(years), np.array(meanline), lw=0.8, linestyle=linestyle, label=label, color=linecolor, alpha=1, zorder=line_zorder )
	# fill between axes
	ax.fill_between( np.array(years), np.array(lower), np.array(upper), facecolor=rangecolor, alpha=alpha, linewidth=0.0, label='range' )
	return ax



def calib_ALF_lines(graph_variable):


	begin = 1950
	end = 2100
	figsize = ( 8, 5.5 )
	plot_title = 'Annual Area Burned %d-%d \n ALFRESCO, %s - NCAR-%s, RCP 8.5' % ( begin, end,models[0],models[1])


	df_dic = {}
	dic = {}
	obs_tab = pd.read_csv( glob.glob( os.path.join( boreal_path, '*'.join([ 'alfresco', 'historical',  graph_variable.replace('_',''), '.csv' ]) ) )[0], index_col=0 )
	obs_domain = obs_tab[ str(domain_num) ]
	obs_domain = obs_domain.ix[ begin:2009 ]


	for model in models :
		frames = []
		for scenario in scenarios :
			i = glob.glob( os.path.join( boreal_path,graph_variable, '*'.join([ 'alfresco',graph_variable.replace('_',''), 'Boreal',model, scenario, '.csv' ]) ) )[0]
			tab = pd.read_csv( i, index_col=0 ).ix[begin:end]
			# tab = tab.apply( np.cumsum, axis=0 )
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

		tmp = tmp.value[['scenario_2', 'scenario_3']]
		tmp.columns = ['scenario_2_%s'%model,'scenario_3_%s'%model]
		df_dic[model] = df
		dic[model] = tmp

	dif = pd.concat([dic[models[0]],dic[models[1]]], axis=1)

	fig, ax1 = plt.subplots()
	linestyle=['-','--','-','--']


	for col, ls in zip(dif.columns, linestyle):
		if 'CCSM4' in col : 
			color = colors_list[1]
			dif[col].plot(ax=ax1,legend=False, color = color, title=plot_title, figsize=figsize, grid=False ,lw = 1.2, ls=ls)
		else :
			color = colors_list[0]
			dif[col].plot(ax=ax1,legend=False, color = color, title=plot_title, figsize=figsize, grid=False ,lw = 1.2, ls=ls)


	obs_domain.plot(ax=ax1, legend=False, linestyle= '--', color=colors_list[2], grid=False, label= "observed" ,lw = 1.1)
	plt.xlabel( 'Year' )

	#Create label for axis
	# y-axis acreage 
	y1, y2 = ax1.get_ylim()
	x1, x2 = ax1.get_xlim()

	# make new axis
	ax2 = ax1.twinx()

	# set it to the acreage in the same limit bounds
	acre_conv = 247.105
	ax2.set_ylim( y1*acre_conv, y2*acre_conv )


	ax2.set_xlim( x1, x2 )

	#spines and ticks control
	ax1.get_xaxis().tick_bottom()
	ax1.get_yaxis().tick_left()
	ax2.get_yaxis().tick_right()

	ax2.get_yaxis().get_major_formatter().set_useOffset(False)
	ax1.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
	ax2.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

	plt.xlabel( 'Year' )
	ax1.set_ylabel( ''+' Area Burned ($\mathregular{km^2}$)' + ''  )
	ax2.set_ylabel( 'Area Burned (acres)' )

	d = mlines.Line2D([], [],ls='--',  linewidth=1.2, color=colors_list[1], label='NCAR-CCSM4 Alternative FMPO' )
	a = mlines.Line2D([], [],linewidth=1.2, color= colors_list[0] , label='MRI-CGCM3 Curent FMPO' )
	b = mlines.Line2D([], [],ls='--',  linewidth=1.2, color=colors_list[0], label='MRI-CGCM3 Alternative FMPO' )
	c = mlines.Line2D([], [], linewidth=1.2, color= colors_list[1] , label='NCAR-CCSM4 Current FMPO' )

	e = mlines.Line2D([], [],ls='--', linewidth=1.2, color=colors_list[2], label='Historical' )

	plt.legend(handles = [a,b,c,d, e],handlelength=4, fontsize='medium',loc='upper left',borderaxespad=0.,ncol=1,frameon=False)
	# plt.legend(handles = [ green_patch , red_patch],handlelength=4, fontsize='medium',loc='upper left',borderaxespad=0.,ncol=1)


	output_filename = os.path.join( output_path, '_'.join([ 'NEW','alfresco', graph_variable,'annual_lines', model , str(begin), str(end),'Calib' ]) + '.png' )
	print "Writing %s to disk" %output_filename
	# sns.despine()
	plt.savefig( output_filename,figsize=figsize, dpi=600, bbox_inches='tight', pad_inches=0.2  )
	plt.close()

def bimodal(graph_variable):

	begin = 1950
	end = 2100
	figsize = ( 11, 9 )

	df_dic = {}
	dic = {}
	obs_tab = pd.read_csv( glob.glob( os.path.join( input_path, '*'.join([ 'alfresco', 'historical',  graph_variable.replace('_',''), '.csv' ]) ) )[0], index_col=0 )
	obs_domain = obs_tab[ str(domain_num) ]
	obs_domain = obs_domain.ix[ begin:2009 ]
	obs_domain = np.cumsum( obs_domain )

	for model in models :
		frames = []
		for scenario in scenarios :
			i = glob.glob( os.path.join( input_path,graph_variable, '*'.join([ 'alfresco',graph_variable.replace('_',''), model, scenario, '.csv' ]) ) )[0]
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

		tmp = tmp.value[['scenario_2', 'scenario_3']]
		tmp.columns = ['scenario_2_%s'%model,'scenario_3_%s'%model]
		df_dic[model] = df
		dic[model] = tmp
	
	dif = pd.concat([dic[models[0]],dic[models[1]]], axis=1)


	fig, ax1 = plt.subplots()
	linestyle=['-','--','-','--']
	for col, ls in zip(dif.columns, linestyle):
		if 'CCSM4' in col : 
			color = colors_list[1]
			dif[col].plot(ax=ax1,legend=False, color = color, figsize=figsize, grid=False ,lw = 1.2, ls=ls)
		else :
			color = colors_list[0]
			dif[col].plot(ax=ax1,legend=False, color = color, figsize=figsize, grid=False ,lw = 1.2, ls=ls)


	obs_domain.plot(ax=ax1, legend=False, linestyle= '--', color=colors_list[2], grid=False, label= "observed" ,lw = 1.1)
	#Create label for axis

	plt.xlabel( 'Year' )

	x = df.date.unique()
	for model,color in zip(models,[colors_list[0],colors_list[1]]) :
		for cond, cond_df in df_dic[model].groupby("condition"):
			low = cond_df.groupby("date").value.apply(np.percentile, 5)
			high = cond_df.groupby("date").value.apply(np.percentile, 95)
			# if cond == 'scenario_2' : i = 1
			# elif cond == 'scenario_3' : i =0
			ax1.fill_between(x, low, high, alpha=.2, color=color)

	#Create label for axis
	# y-axis acreage 
	y1, y2 = ax1.get_ylim()
	x1, x2 = ax1.get_xlim()

	# make new axis
	ax2 = ax1.twinx()

	# set it to the acreage in the same limit bounds
	acre_conv = 247.105
	ax2.set_ylim( y1*acre_conv, y2*acre_conv )


	ax2.set_xlim( x1, x2 )

	# ax2 spines

	ax1.get_xaxis().tick_bottom()
	ax1.get_yaxis().tick_left()
	ax2.get_yaxis().tick_right()



	ax2.get_yaxis().get_major_formatter().set_useOffset(False)
	ax1.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
	ax2.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

	plt.xlabel( 'Year' )
	ax1.set_ylabel( ''+' Area Burned ($\mathregular{km^2}$)' + ''  )
	ax2.set_ylabel( 'Area Burned (acres)' )

	d = mlines.Line2D([], [],ls='--',  linewidth=1.2, color=colors_list[1], label='NCAR-CCSM4 Alternative FMPO' )
	a = mlines.Line2D([], [],linewidth=1.2, color= colors_list[0] , label='MRI-CGCM3 Current FMPO' )
	b = mlines.Line2D([], [],ls='--',  linewidth=1.2, color=colors_list[0], label='MRI-CGCM3 Alternative FMPO' )
	c = mlines.Line2D([], [], linewidth=1.2, color= colors_list[1] , label='NCAR-CCSM4 Current FMPO' )

	e = mlines.Line2D([], [],ls='--', linewidth=1.2, color=colors_list[2], label='Historical' )

	plt.legend(handles = [a,b,c,d ,e],handlelength=4, fontsize='medium',loc='upper left',borderaxespad=0.,ncol=1,frameon=False)


	output_filename = os.path.join( output_path, '_'.join([ 'NEW2','alfresco', graph_variable,'bimodal', models[0],models[1] , str(begin), str(end)]) + '.png' )
	print "Writing %s to disk" %output_filename
	# sns.despine()
	plt.savefig( output_filename ,figsize=figsize, dpi=400, bbox_inches='tight', pad_inches=0.2 )
	# plt.savefig( '/workspace/Shared/Tech_Projects/SERDP/project_data/ALFRESCO/SERPD_run/AMY_SERDP_2016/ALFRESCO_CD_ratios_'+'overlay'+'_'+scenario+'_iqr.png', figsize=figsize, dpi=600, bbox_inches='tight', pad_inches=0.2 )

	plt.close()


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
	from collections import OrderedDict
	from matplotlib.backends.backend_pdf import PdfPages
	from matplotlib import rcParams
	domain_num = 1
	domain_name = 'AOI_SERDP'
	output_path = '/workspace/Shared/Tech_Projects/SERDP/project_data/ALFRESCO/SERPD_run/AMY_SERDP_2016/'
	input_path = '/workspace/Shared/Tech_Projects/SERDP/project_data/ALFRESCO/SERPD_run/output_csvs/'

	boreal_path = '/workspace/Shared/Tech_Projects/SERDP/project_data/ALFRESCO/SERDP_Boreal_run/output_csvs'

	rcParams[ 'xtick.direction' ] = 'out'
	rcParams[ 'ytick.direction' ] = 'out'
	rcParams[ 'xtick.labelsize' ] = 'small'
	rcParams[ 'ytick.labelsize' ] = 'small'
	rcParams[ 'figure.titlesize' ]  = 'medium'
	rcParams[ 'axes.titlesize' ] = 'medium'


	models = ['MRI-CGCM3', 'CCSM4']
	scenarios = ['scenario_2','scenario_3']
	graph_variable = 'total_area_burned'
	#storage = sns.color_palette('deep',7)
	pale_red = '#d9544d'
	denim_blue = '#3b5b92'
	charcoal = '#36454F'

	colors_list = [denim_blue,pale_red,charcoal]

	#calib_ALF_lines(graph_variable)
	# calib_ALF(graph_variable,model = models[0])
	bimodal(graph_variable)