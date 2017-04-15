
import pandas as pd
import numpy as np
import glob, os, ast, sys,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
# from alfresco_postprocessing import plot
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
pd.options.mode.chained_assignment = None  # default='warn'
import alfresco_postprocessing as ap

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
        return dict(dd)

def upcase( word ):
    _tmp = [i.title() for i in word.split('_')]
    _tmp = " ".join(_tmp)
    return _tmp

def get_veg_ratios( veg_dd, domain ,year_range = (1950,2100), group1=['White Spruce', 'Black Spruce'], group2=['Deciduous'] ):
    '''
    calculate ratios from lists of veg types.
    '''
    begin,end = year_range
    agg1 = sum([ veg_dd[ domain ][ i ].ix[begin:end] for i in group1 ])
    agg2 = sum([ veg_dd[ domain ][ i ].ix[begin:end]for i in group2 ])
    return agg1 / agg2

def fill_in(ax , df ,colors ,low_percentile = 5 , high_percentile = 95 , alpha = 0.2 ) :
    
    x = df.index.unique()

    ax.fill_between(x, df.groupby(df.index).apply(np.percentile, low_percentile ), \
    df.groupby(df.index).apply(np.percentile, high_percentile), alpha= alpha, color=colors)

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

def df_processing2(dictionnary , std_arg = False , cumsum_arg = False , *args):

    def _process_df(scen_arg , df , std_arg , cumsum_arg):

        if cumsum_arg == True :
            df = df.apply( np.cumsum, axis=0 )
        else : pass

        if std_arg == True :
            df['std'] = df.std(axis=1)
        else : pass
            
        return df

    _tmp = [_process_df( k , v , std_arg , cumsum_arg) for k , v in dictionnary.iteritems()]
    return _tmp[0]

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

def decade_plot(observed , output_path ,  model , graph_variable, year_range , *args):
    """  Takes a graphvariable/metric and create a bar plot by decade. The error values are calculated by calculating the variance amoung each reps
    for every year, then the error is calculated by taking the square root of the mean of those variance as shown here
    http://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation#26647"""

    for domain in observed.domains :

        begin, end = (1950,2009) 


        if graph_variable == 'avg_fire_size' :
            plot_title = 'Historical Average Size of Fire %d-%d \n %s ' % ( begin, end, underscore_fix(domain) )
            ylabel ='Average Fire Size ('+'$\mathregular{km^2}$' + ')' 

        elif graph_variable == 'number_of_fires' :
            plot_title = 'Historical Total Number of Fires %d-%d \n %s ' % ( begin, end, underscore_fix(domain) )
            ylabel = 'Number of Fires'

        elif graph_variable == 'total_area_burned' :
            plot_title = 'Historical Total Area Burned %d-%d \n %s ' % ( begin, end, underscore_fix(domain) )
            ylabel = 'Area Burned in ('+'$\mathregular{km^2}$' + ')'

        else : 'Error with Title'

        def std_calc(df):
            mean = df.mean(axis=1).groupby(df.index // 10 * 10).sum()
            error = df.var(axis=1).groupby(df.index // 10 * 10).apply(lambda d: np.sqrt(d.mean()))
            df = pd.concat([mean,error],axis=1)
            df.columns = ['mean','error']
            return df

        # Handling the historical, oserved data
        obs_domain = observed.__dict__[graph_variable][domain].ix[begin : 2016]
        # obs_domain['std'] = 0

        # # Building the two dataframes needed for the plotting
        # data = {scen_arg.scenario :std_calc(scen_arg.__dict__[graph_variable][domain].ix[begin : end]) for scen_arg in [scenario1]}
        # means = pd.concat([data['scenario1']['mean'],obs_domain['observed'].groupby(obs_domain.index // 10 * 10).sum()],axis=1)
        # error = pd.concat([data['scenario1']['error'],obs_domain['std'].groupby(obs_domain.index // 10 * 10).sum()],axis=1)


        #plotting
        if graph_variable == "avg_fire_size" :
            ax = obs_domain.groupby(obs_domain.index // 10 * 10).mean().plot(kind='bar', legend=False, color = [ observed.color], title=plot_title,  grid=False, width=0.8 )
        else :
            ax = obs_domain.groupby(obs_domain.index // 10 * 10).sum().plot(kind='bar', legend=False, color = [ observed.color], title=plot_title,  grid=False, width=0.8 )

        #Create label for axis
        plt.ylabel( ylabel )
        plt.xlabel( 'Years' )
        plt.ylim(ymin=0 ) 

        ax = ticks(ax , decade=True)
        
        plt.legend(handles = [ observed.patch],fontsize='medium',loc='best',borderaxespad=0.,ncol=1,frameon=False)

        output_filename = os.path.join( output_path, domain , '_'.join([ 'alfresco_barplot_decade', domain,graph_variable, model , str(begin), str(end)]) + '.png' )

        plt.savefig( output_filename )

        plt.close()


def launcher(  ) :


    from collections import defaultdict
    model = 'NCAR-CCSM4_rcp85'
    out = '/workspace/Shared/Users/jschroder/TMP/FireHistory/'

    json_obs = os.path.join(out , 'JSON' , 'Observed2.json' )

    observed = Scenario( json_obs, model, 'Observed', "Historical", '#B22222' )

    output_path = os.path.join( out , 'Plots' )




    if not os.path.exists( output_path ) :
        os.makedirs( output_path )

    for i in observed.domains :
        _tmp = os.path.join(output_path,i)
        if not os.path.exists( _tmp ) :
            os.makedirs( _tmp )

    for metric in observed.metrics :
        if metric not in [ 'veg_counts' , 'all_fire_sizes']:
            decade_plot( observed , output_path ,  model , metric, year_range)


