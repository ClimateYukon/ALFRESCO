def wrangling(f):
    #The Canadian data comes with epsg 3978 so this step reproject it to 3338 so both shapefile can be rasterized
    from shapely.ops import transform
    from functools import partial
    import pyproj
    
    if 'YEAR' in f['properties'] :
        project = partial(pyproj.transform , pyproj.Proj(init='epsg:3978') , pyproj.Proj(init='epsg:3338'))
        return mapping( transform( project, shape( f[ 'geometry' ] ) ) )    

    if 'FireYear' in f['properties'] :
        return f['geometry']  

def wrapper(year ) :
    #The shapefiles don't have the exact same field so here working with a list of list to be able to grab the years.
    l = [i for i in shp[0] if i['properties']['YEAR'] == year] + [i for i in shp[1] if i['properties']['FireYear'] == str(year)]

    ls = [wrangling(features) for features in l ]

    output_filename = os.path.join( output_path, '_'.join([ 'ALF_AK_CAN_FireHistory', str(year) + '.tif' ]))

    with rasterio.open( output_filename, 'w', **meta ) as out:
        new = rasterize( ( (i, 1) for i in ls ), fill=0, out_shape = rst.shape, transform=rst.transform )
        out.write_band( 1, new )

if __name__ == '__main__':
    import fiona, rasterio, os
    from rasterio.features import rasterize
    from shapely.geometry import shape, mapping
    from multiprocessing import Pool
    from functools import partial

    rst_fn = '/Data/Base_Data/ALFRESCO/ALFRESCO_Master_Dataset/ALFRESCO_Model_Input_Datasets/AK_CAN_Inputs/Fire/ALF_AK_CAN_FireHistory_1917.tif' #template
    output_path = '/workspace/Shared/Users/jschroder/FireHistory/'
    shp = '/workspace/Shared/Users/jschroder/TMP/FH_shp/FireAreaHistory.shp'
    shp2 = '/workspace/Shared/Users/jschroder/TMP/FH_shp/NFDB_poly_20160712.shp'

    if not os.path.exists( output_path  ):
        os.mkdir( output_path )

    rst = rasterio.open( rst_fn )
    meta = rst.meta
    meta.update( compress='lzw'  )

    year_range = range(1917,2017)
    shp = [[i for i in fiona.open(shp2,'r')] , [i for i in fiona.open(shp,'r')]]

    with Pool(32) as p :
        p.map(wrapper,year_range)
