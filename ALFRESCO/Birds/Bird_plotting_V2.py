
def decade_plot( domain_name , graph_variable, year_range=(1950,2100)):
    # take a graphvariable, average over reps for a year and sums it over a decade.
    
    sns.set(style="whitegrid")

    begin, end = year_range #subset the dataframes to the years of interest
    end = end-1

    figsize = ( 14, 8 )

    if graph_variable == 'avg_fire_size' :
        plot_title = 'Average Size of Fire per Decade %d-%d \n ALFRESCO, %s, CMIP3' % ( begin, end, domain_name )
        ylabel ='Average fire size ('+'$\mathregular{km^2}$' + ')' 

    elif graph_variable == 'number_of_fires' :
        plot_title = 'Total Number of Fires per Decade %d-%d \n ALFRESCO, %s, CMIP3' % ( begin, end, domain_name )
        ylabel = 'Number of fires'

    elif graph_variable == 'total_area_burned' :
        plot_title = 'Total Area Burned %d-%d \n ALFRESCO, %s, CMIP3' % ( begin, end, domain_name )
        ylabel = 'Area burned in ('+'$\mathregular{km^2}$' + ')'



    #Handling the historical, oserved data
    obs_tab = pd.read_csv( glob.glob( os.path.join( input_path, 'historical', graph_variable,  '*'.join([ 'firehistory',graph_variable.replace('_',''), domain_name, '.csv' ]) ) )[0], index_col=0 )

    obs_domain = obs_tab.ix[ begin: 2009]

    frames = []
    for model in models :
        print os.path.join( input_path, model,graph_variable, '*'.join([ 'alfresco',graph_variable.replace('_',''),domain_name, model,'.csv' ]) )
        i = glob.glob( os.path.join( input_path,model,graph_variable, '*'.join([ 'alfresco',graph_variable.replace('_',''),domain_name, model,'.csv' ]) ) )[0]
        tab = pd.read_csv( i, index_col=0 ).ix[begin:end]

        #Standard deviation calculation happens here, in case needed to change for quantile
        tab['std'] = np.nan
        for i in range( begin , end+1 , 10) :
            std = tab.ix[ i : i + 9 ].sum( axis = 0 ).std()
            tab.set_value((range( i , i + 10 ) ), 'std', std )

        tab['date'] = tab.index
        tab['model']= model
        tab = pd.melt(tab, id_vars=["date", "model",'std'], var_name="condition")
        frames.append(tab)

    #Complete dataframe creation with both model in long fata form
    df = pd.concat(frames, ignore_index= True)
    df = df.drop('condition', 1)
    df = df.rename(columns = {'model':'condition'})
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

    #blu_patch = mpatches.Patch([], [], linewidth=1.2, color=colors_list[0], label='Without fire management' )
    green_patch = mpatches.Patch([], [], linewidth=1.2, color= colors_list[0] , label=models[0] )
    red_patch = mpatches.Patch([], [], linewidth=1.2, color=colors_list[1], label=models[1] )
    gren_patch = mpatches.Patch([], [], linewidth=1.2, color=colors_list[2], label='Historical' )
    plt.legend(handles = [ green_patch , red_patch, gren_patch],loc='best',ncol=1, shadow=True, fancybox=True)


    output_filename = os.path.join( output_path ,  '_'.join([ 'alfresco', graph_variable,'decade', domain_name , str(begin), str(end),model ]) + '.png' )
    print "Writing %s to disk" %output_filename
    sns.despine()
    plt.savefig( output_filename )
    pdf.savefig()
    plt.close()

def compare_area_burned(domain_name ,graph_variable, year_range=(1950,2100)):
    #This plot compare the cumulative area burn for managed, unmanaged and historical period
    sns.set(style="whitegrid")
    begin, end = year_range #subset the dataframes to the years of interest
    graph_variable = 'total_area_burned'
    #Set some Style and settings for the plots
    figsize = ( 14 ,8 )
    plot_title = 'Cumulative Sum of Annual Area Burned %d-%d \n ALFRESCO, %s, CMIP3' % ( begin, end,domain_name)

    #Handling the historical, oserved data
    obs_tab = pd.read_csv( glob.glob( os.path.join( input_path , 'historical', graph_variable, '*'.join([ 'firehistory',  graph_variable.replace('_',''),domain_name, '.csv' ]) ) )[0], index_col=0 )

    obs_domain = obs_tab.ix[ begin:2009 ]
    obs_domain = np.cumsum( obs_domain )


    frames = []
    # Cleaning and adding some fields for each individual model's dataframe for concat
    for model in models :
        i = glob.glob( os.path.join( input_path , model, graph_variable, '*'.join([ 'alfresco',graph_variable.replace('_',''), domain_name, model,'.csv' ]) ) )[0]
        tab = pd.read_csv( i, index_col=0 ).ix[begin:end]
        tab = tab.apply( np.cumsum, axis=0 )
        tab['date'] = tab.index
        tab['model']= model
        tab = pd.melt(tab, id_vars=["date", "model"], var_name="condition")
        frames.append(tab)

    #Complete dataframe creation with both model in long fata form
    df = pd.concat(frames, ignore_index= True)
    df = df.drop('condition', 1)
    df = df.rename(columns = {'model':'condition'})
    df = df.sort_values(by=['condition','date'])
    df = df.reset_index(drop=True)
    
    tmp = df.groupby(["condition", "date"]).mean().unstack("condition")


    
    #checking if colors_list list work
    ax = tmp.plot(legend=False, color = colors_list, title=plot_title,lw = 1.2, figsize=figsize, grid=False)


    #We have to plot model_3 first so we need to change the colors order

    #checking if colors_list list work
    obs_domain.plot(ax=ax, legend=False, linestyle= '--', color=colors_list[2], grid=False, label= "observed" ,lw = 1.1)

    #Create label for axis
    plt.xlabel( 'Years' )
    plt.ylabel( 'Area burned in ('+'$\mathregular{km^2}$' + ')'  )

    # Get a reference to the x-points corresponding to the dates and the the colors_list
    x = df.date.unique()


    #Create the buffer around the mean value to display the rep dispersion
    for cond, cond_df in df.groupby("condition"):
        low = cond_df.groupby("date").value.apply(np.percentile, 5)
        high = cond_df.groupby("date").value.apply(np.percentile, 95)
        if cond == models[0] : i = 0
        elif cond == models[1] : i =1
        ax.fill_between(x, low, high, alpha=.2, color=colors_list[i])


    # build and display legend
    
    #blu_patch = mlines.Line2D([], [], linewidth=1.2, color=colors[0], label='Without fire management' )
    green_patch = mlines.Line2D([], [], linewidth=1.2, color= colors_list[0] , label=models[0] )
    red_patch = mlines.Line2D([], [], linewidth=1.2, color=colors_list[1], label=models[1] )
    ired_patch = mlines.Line2D([], [], ls='--', linewidth=1, color=colors_list[2], label='Historical' )
    #Setting legend
    plt.legend(handles = [green_patch,red_patch,ired_patch,],loc="best",ncol=1, shadow=True, fancybox=True)
    sns.despine()

    output_filename = os.path.join( output_path, '_'.join([ 'alfresco_annual_areaburned_compared_lines', model , str(begin), str(end),model ]) + '.png' )
    print "Writing %s to disk" %output_filename
    plt.savefig( output_filename )
    pdf.savefig()
    plt.close()

def compare_vegcounts(model ,veg_name_dict, year_range=(1950,2100)):
    sns.set(style="whitegrid")
    begin, end = year_range #subset the dataframes to the years of interest

    #Set some Style and settings for the plots
    figsize = ( 14, 8 )
    graph_variable = 'veg_counts'
    fig, ax = plt.subplots(figsize=figsize, facecolor = 'w' ) 
    for veg_num, veg_name in veg_name_dict.iteritems():
        try :
            plot_title = "Annual %s Coverage %s-%s \n ALFRESCO, %s, CMIP3"\
                % ( veg_name, str(begin), str(end),domain_name)

            frames = []
            # Cleaning and adding some fields for each individual model's dataframe for concat
            for model in models :

                i = glob.glob( os.path.join( input_path, model , graph_variable, '*'.join(['alfresco',graph_variable.replace('_',''),domain_name,veg_num,  model,'.csv' ]) ) )[0]
                tab = pd.read_csv( i, index_col=0 ).ix[begin:end]
                #tab = tab.apply( np.cumsum, axis=0 )
                tab['date'] = tab.index
                tab['model']= model
                tab = pd.melt(tab, id_vars=["date", "model"], var_name="condition")
                frames.append(tab)
            
            #Complete dataframe creation with both model in long fata form
            df = pd.concat(frames, ignore_index= True)
            df = df.drop('condition', 1)
            df = df.rename(columns = {'model':'condition'})
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
                if cond == models[0] : i = 0
                elif cond == models[1] : i =1
                ax.fill_between(x, low, high, alpha=.2, color=colors_list[i])

            # build and display legend
            #blu_patch = mlines.Line2D([], [], linewidth=1.2, color=colors_list[0], label='Without fire management' )
            green_patch = mlines.Line2D([], [], linewidth=1.2, color= colors_list[0] , label=models[0] )
            red_patch = mlines.Line2D([], [], linewidth=1.2, color=colors_list[1], label=models[1] )
            
            # if veg_name in ['GraminoidTundra','BlackSpruce','ShrubTundra'] :
            #   plt.legend(handles = [#blu_patch,green_patch,red_patch],loc="bottom right",ncol=1, shadow=True, fancybox=True) bbox_to_anchor=[0, 1],
            # else :
            plt.legend(handles = [green_patch,red_patch],loc="best",ncol=1, shadow=True, fancybox=True)

            sns.despine()

            output_filename = os.path.join( output_path, '_'.join([ 'alfresco_annual_areaveg_compared_lines', model, veg_name.replace(' ', '' ), domain_name.replace(' ', '' ), str(begin), str(end),model ]) + '.png' ) 
            print "Writing %s to disk" %output_filename
            plt.savefig( output_filename )
            pdf.savefig()
            plt.close()
        except : print 'oups'

def compare_firesize( model , year_range=(1950,2100) , buff=False):
    #This graph will be about producing a comparative graph of fire size between managed and unmanaged in order to see if it changes with management
    sns.set(style="whitegrid")
    graph_variable = 'avg_fire_size'

    begin, end = year_range #subset the dataframes to the years of interest

    plot_title = 'Average Size of Fire %d-%d \n ALFRESCO, %s, CMIP3' % ( begin, end, domain_name )
    figsize = ( 14, 8 )


    frames = []
    # Cleaning and adding some fields for each individual model's dataframe for concat
    for model in models :
        i = glob.glob( os.path.join( input_path, model,graph_variable, '*'.join(['alfresco',graph_variable.replace('_',''), domain_name, model,'.csv' ]) ) )[0]
        tab = pd.read_csv( i, index_col=0 ).ix[begin:end]
        #tab = tab.apply( np.cumsum, axis=0 )
        tab['date'] = tab.index
        tab['model']= model
        tab = pd.melt(tab, id_vars=["date", "model"], var_name="condition")
        frames.append(tab)

    #Complete dataframe creation with both model in long fata form
    df = pd.concat(frames, ignore_index= True)
    df = df.drop('condition', 1)
    df = df.rename(columns = {'model':'condition'})
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
            if cond == models[0] : i = 0
            elif cond == models[1] : i =1
            ax.fill_between(x, low, high, alpha=.2, color=colors[i])

    else: 
        pass

    # build and display legend

    #blu_patch = mlines.Line2D([], [], linewidth=1.2, color=colors_list[0], label='Without fire management' )
    green_patch = mlines.Line2D([], [], linewidth=1.2, color= colors_list[0] , label=models[0] )
    red_patch = mlines.Line2D([], [], linewidth=1.2, color=colors_list[1], label=models[1] )

    plt.legend(handles = [green_patch,red_patch],loc="best", ncol=1, shadow=True, fancybox=True)
    #plt.show()

    if buff==True :
        output_filename = os.path.join( output_path, '_'.join([ 'alfresco_avgfiresize_compared_buff', model , str(begin), str(end),model]) + '.png' )

    else :
        output_filename = os.path.join( output_path, '_'.join([ 'alfresco_avgfiresize_compared', model , str(begin), str(end),model ]) + '.png' )
    print "Writing %s to disk" %output_filename
    sns.despine()
    plt.savefig( output_filename )
    pdf.savefig()
    plt.close()

def compare_numberoffires( model , year_range=(1950,2100) , buff=True):
    #This graph will be about producing a comparative graph of fire size between managed and unmanaged in order to see if it changes with management
    sns.set(style="whitegrid")
    graph_variable = 'number_of_fires'

    begin, end = year_range #subset the dataframes to the years of interest


    plot_title = 'Cumulative Number of Fires %d-%d \n ALFRESCO, %s, CMIP3' % ( begin, end, domain_name )
    figsize = ( 14, 8 )


    frames = []
    # Cleaning and adding some fields for each individual model's dataframe for concat

    for model in models :
        i = glob.glob( os.path.join( input_path, model ,graph_variable, '*'.join([ 'alfresco',graph_variable.replace('_',''), domain_name, model, '.csv' ]) ) )[0]
        tab = pd.read_csv( i, index_col=0 ).ix[begin:end]
        tab = tab.apply( np.cumsum, axis=0 )
        tab['date'] = tab.index
        tab['model']= model
        tab = pd.melt(tab, id_vars=["date", "model"], var_name="condition")
        frames.append(tab)

    #Complete dataframe creation with both model in long fata form
    df = pd.concat(frames, ignore_index= True)
    df = df.drop('condition', 1)
    df = df.rename(columns = {'model':'condition'})
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
            if cond == models[0] : i = 0
            elif cond == models[1] : i =1
            ax.fill_between(x, low, high, alpha=.2, color=colors_list[i])

    else: 
        pass

    #blu_patch = mlines.Line2D([], [], linewidth=1.2, color=colors_list[0], label='Without fire management' )
    green_patch = mlines.Line2D([], [], linewidth=1.2, color= colors_list[0] , label=models[0] )
    red_patch = mlines.Line2D([], [], linewidth=1.2, color=colors_list[1], label=models[1] )

    plt.legend(handles = [green_patch,red_patch],loc="best",ncol=1, shadow=True, fancybox=True)
    #plt.show()

    if buff==True :
        output_filename = os.path.join( output_path, '_'.join([ 'alfresco_numberoffires_cum_compared_buff', model , str(begin), str(end),model]) + '.png' )

    else :
        output_filename = os.path.join( output_path, '_'.join([ 'alfresco_numberoffires_cum_compared', model , str(begin), str(end),model ]) + '.png' )

    print "Writing %s to disk" %output_filename
    sns.despine()
    plt.savefig( output_filename )
    pdf.savefig()
    plt.close()

def compare_cab_vs_fs(model , year_range=(1950,2100)):
    #This graph shows the cumulative area burnt by fire size, managed and unmanaged model are compared on the same plot
    #Mainly based on Michael's code https://github.com/ua-snap/alfresco-calibration/blob/cavm/alfresco_postprocessing_plotting.py#L252
    sns.set(style="whitegrid")
    begin, end = year_range
    figsize = (14,8)

    fig, ax = plt.subplots(figsize=figsize) 
    graph_variable = 'all_fire_sizes'
    for color, model in zip(colors_list , models) :
        l = glob.glob( os.path.join( input_path,model,graph_variable, '*'.join([ 'alfresco',graph_variable.replace('_','') ,domain_name, model,'.csv' ]) ) )[0]

        mod_tab = pd.read_csv( l, index_col=0 )
        # wrangle it
        #Creates array of all fire size per rep
        all_fire_sizes_sorted = { col:np.sort([j for i in mod_tab[ col ] for j in ast.literal_eval(i) ]) for col in mod_tab }
        #cumsum it
        all_fire_sizes_cumsum = { k:np.cumsum( v ) for k,v in all_fire_sizes_sorted.iteritems() }
        #Create list of replicates names
        rep_names = all_fire_sizes_sorted.keys()

        # melt to long format [ the hard way ]
        afs_list = [ pd.DataFrame( data=[v, np.repeat(k, len(v))] ).T for k,v in all_fire_sizes_sorted.iteritems() ]
        afs_melted = pd.concat( afs_list, axis=0 )
        afs_melted.columns = ('fires', 'rep')

        cfs_list = [ pd.DataFrame( data=[np.cumsum(v), np.repeat(k, len(v))] ).T for k,v in all_fire_sizes_sorted.iteritems() ]
        cfs_melted =  pd.concat( cfs_list, axis=0 )
        cfs_melted.columns = ('cumsum', 'rep')

        combined = pd.concat( [afs_melted, cfs_melted['cumsum']], axis=1 )
        for i,j in [ i for i in combined.groupby('rep') ]: 
            plt.plot( j['fires'], j['cumsum'], color=color, alpha=0.5, lw=1 )

    #plt.plot( obs_melted['fires'], obs_melted['cumsum'], sns.xkcd_rgb[color[3]] )
    sns.set_style( 'whitegrid', {'ytick.major.size': 7, 'xtick.major.size': 7} )

    # build and display legend
    #blu_patch = mlines.Line2D([], [], linewidth=1.2, color=colors_list[0], label='Without fire management' )
    green_patch = mlines.Line2D([], [], linewidth=1.2, color= colors_list[0] , label=models[0] )
    red_patch = mlines.Line2D([], [], linewidth=1.2, color=colors_list[1], label=models[1])
    #Create label for axis
    plt.ylabel( 'Area burned in ('+'$\mathregular{km^2}$' + ')'  )
    plt.xlabel( 'Fire size ('+'$\mathregular{km^2}$' + ')' )

    fig.suptitle('Cumulative Area Burned vs. Fire Sizes %d-%d \n ALFRESCO, %s, CMIP3' \
                        % ( begin, end, model))
    plt.legend( handles=[green_patch,red_patch], frameon=False, loc=0 )


    sns.despine()

    output_filename = os.path.join( output_path, '_'.join([ 'alfresco_cab_vs_fs', model , str(begin), str(end),domain_name.replace(" ","") ]) + '.png' )
    print "Writing %s to disk" %output_filename
    plt.savefig( output_filename )
    pdf.savefig()
    plt.close()

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    import glob, os, json, ast
    import numpy as np
    import geopandas as gpd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import seaborn as sns
    from collections import OrderedDict
    from matplotlib.backends.backend_pdf import PdfPages

    
    data_path = '/atlas_scratch/jschroder/2017_IEM_birds_simple_polygon/'
    os.chdir(data_path)

    input_path = data_path
    visu = 'Visual_outputs_LCC'

    if not os.path.exists( os.path.join( data_path , visu ) ):
        os.mkdir( os.path.join(data_path,visu) )

    subdomains_path = '/workspace/Shared/Tech_Projects/ALF_bird_habitat/project_data/NA_TerrestrialProtectedAreas/NA_TPA_simplified.shp'



    models = ['cccma_cgcm3_1.sresa1b','mpi_echam5.sresa1b']
    #Order matters as the decade_plot takes metric[:-2] as input
    metrics = [ 'avg_fire_size' , 'number_of_fires' , 'total_area_burned' , 'veg_counts' , 'all_fire_sizes']

    veg_name_dict = {'BlackSpruce':'Black Spruce',
                'WhiteSpruce':'White Spruce',
                'Deciduous':'Deciduous',
                'ShrubTundra':'Shrub Tundra',
                'GraminoidTundra':'Graminoid Tundra',
                'Barrenlichen-moss':'Barren lichen-moss',
                'TemperateRainforest':'Temperate Rainforest'
                }

    shp = gpd.read_file( subdomains_path )
    id_name = zip(shp.OBJECTID , shp.MGT_AGENCY)
    id_name_dict = OrderedDict( id_name ) # this needs to be a helper function to get this from another dataset 

    storage = sns.color_palette('deep',7)
    colors_list = [storage[1],storage[2],sns.xkcd_rgb["charcoal"]]
    models = ['cccma_cgcm3_1.sresa1b','mpi_echam5.sresa1b']

    year_range = (1950,2100)


    for domain_num, domain_name in id_name_dict.iteritems() :

        output_path = os.path.join( visu , domain_name.replace(" ","_") ) #for production

        if not os.path.exists( output_path ):
            os.mkdir( output_path )

        pdf_output =  os.path.join( visu, '_'.join([ domain_name.replace(" ","_"),'plots','CMIP3' ]) + '.pdf' )

        with PdfPages(pdf_output) as pdf:

            for metric in metrics[:-2]:
                decade_plot(domain_name , metric, year_range)

            compare_area_burned(domain_name, 'total_area_burned',year_range)
            compare_vegcounts(domain_name, veg_name_dict,year_range)
            compare_firesize( domain_name,year_range)
            compare_numberoffires( domain_name,year_range)
            compare_cab_vs_fs(domain_name,year_range)


